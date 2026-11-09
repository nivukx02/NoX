package com.nivukx.nox.core.reconstruction

import android.app.ActivityManager
import android.content.Context
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES31
import android.os.Build
import com.nivukx.nox.core.model.GpuBackend
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.ceil

/**
 * Actual on-device compute backend.
 *
 * The engine prefers OpenGL ES 3.1 compute because it is broadly deployable on Android hardware.
 * Vulkan availability is exposed as a capability, but the runtime deliberately selects GLES for
 * this Kotlin-only implementation unless an explicit native Vulkan backend is supplied later.
 *
 * The compute kernel is a real SSBO-based normalization pass used by reconstruction preprocessing.
 */
class GpuAccelerationEngine(
    context: Context,
    private val gpuMinimumElements: Int = 256,
    private val maximumElements: Int = 2_000_000,
) : AutoCloseable {
    init {
        require(gpuMinimumElements >= 1)
        require(maximumElements >= gpuMinimumElements)
    }
    private val appContext = context.applicationContext

    val backend: GpuBackend = selectBackend()

    private var display: EGLDisplay? = null
    private var contextEgl: EGLContext? = null
    private var surface: EGLSurface? = null
    private var program = 0
    private var buffer = 0
    private var dispatchCount = 0L
    private var dispatchFailures = 0L
    private var cpuFallbackCount = 0L
    private var stagingReallocationCount = 0L
    private var lastDispatchMillis = 0.0
    private var lastError: String? = null
    private var stagingBytes: ByteBuffer? = null
    private var scaleLocation = -1
    private var biasLocation = -1
    private var countLocation = -1

    private val vertexShaderSource = """
        #version 310 es
        layout(local_size_x = 64) in;
        layout(std430, binding = 0) buffer Values {
            float values[];
        };
        uniform float scale;
        uniform float bias;
        uniform uint count;
        void main() {
            uint index = gl_GlobalInvocationID.x;
            if (index >= count) return;
            values[index] = max(0.0, values[index] * scale + bias);
        }
    """.trimIndent()

    @Synchronized
    fun normalize(
        values: FloatArray,
        scale: Float = 1.0f,
        bias: Float = 0.0f,
    ): FloatArray {
        if (
            values.isEmpty() ||
            values.size > maximumElements ||
            values.size < gpuMinimumElements ||
            backend != GpuBackend.OPENGL_ES_COMPUTE
        ) {
            cpuFallbackCount += 1L
            return values.copyOf().also {
                for (i in it.indices) {
                    it[i] = maxOf(0.0f, it[i] * scale + bias)
                }
            }
        }

        val start = System.nanoTime()
        return runCatching {
            ensureContext()
            dispatch(values, scale, bias)
        }.onSuccess {
            dispatchCount += 1L
            lastDispatchMillis =
                (System.nanoTime() - start).coerceAtLeast(0L) / 1e6
            lastError = null
        }.getOrElse { error ->
            dispatchFailures += 1L
            lastError = error.message ?: error.javaClass.simpleName
            values.copyOf().also { result ->
                for (i in result.indices) {
                    result[i] = maxOf(0.0f, result[i] * scale + bias)
                }
            }
        }
    }

    @Synchronized
    override fun close() {
        runCatching {
            if (
                display != null &&
                contextEgl != null &&
                surface != null
            ) {
                EGL14.eglMakeCurrent(
                    display,
                    surface,
                    surface,
                    contextEgl,
                )
            }

            if (program != 0) {
                GLES31.glDeleteProgram(program)
            }

            if (buffer != 0) {
                GLES31.glDeleteBuffers(
                    1,
                    intArrayOf(buffer),
                    0,
                )
            }
        }

        program = 0
        buffer = 0
        stagingBytes = null
        scaleLocation = -1
        biasLocation = -1
        countLocation = -1

        runCatching {
            if (display != null && surface != null) {
                EGL14.eglDestroySurface(
                    display,
                    surface,
                )
            }

            if (display != null && contextEgl != null) {
                EGL14.eglDestroyContext(
                    display,
                    contextEgl,
                )
            }

            if (display != null) {
                EGL14.eglTerminate(display)
            }
        }

        display = null
        contextEgl = null
        surface = null
    }

    @Synchronized
    fun telemetry(): Telemetry =
        Telemetry(
            backend = backend,
            dispatchCount = dispatchCount,
            dispatchFailures = dispatchFailures,
            cpuFallbackCount = cpuFallbackCount,
            stagingReallocationCount = stagingReallocationCount,
            lastDispatchMillis = lastDispatchMillis,
            lastError = lastError,
        )

    data class Telemetry(
        val backend: GpuBackend,
        val dispatchCount: Long,
        val dispatchFailures: Long,
        val cpuFallbackCount: Long = 0L,
        val stagingReallocationCount: Long = 0L,
        val lastDispatchMillis: Double,
        val lastError: String?,
    )

    private fun selectBackend(): GpuBackend {
        val supportsGl31 = runCatching {
            val am = appContext.getSystemService(
                Context.ACTIVITY_SERVICE,
            ) as ActivityManager
            val info = am.deviceConfigurationInfo
            info.reqGlEsVersion >= 0x00030001
        }.getOrDefault(false)

        if (supportsGl31) {
            return GpuBackend.OPENGL_ES_COMPUTE
        }

        return GpuBackend.CPU
    }

    private fun ensureContext() {
        if (display != null && contextEgl != null && surface != null) {
            return
        }

        check(Build.VERSION.SDK_INT >= 18)

        val dpy = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        check(dpy != EGL14.EGL_NO_DISPLAY)

        val version = IntArray(2)
        check(EGL14.eglInitialize(dpy, version, 0, version, 1))

        val attribs = intArrayOf(
            EGL14.EGL_RENDERABLE_TYPE,
            0x40,
            EGL14.EGL_SURFACE_TYPE,
            EGL14.EGL_PBUFFER_BIT,
            EGL14.EGL_NONE,
        )

        val configs = arrayOfNulls<EGLConfig>(1)
        val count = IntArray(1)

        check(
            EGL14.eglChooseConfig(
                dpy,
                attribs,
                0,
                configs,
                0,
                1,
                count,
                0,
            ),
        )

        val cfg = configs[0] ?: error("No EGL ES 3.1 config")

        val ctxAttribs = intArrayOf(
            EGL14.EGL_CONTEXT_CLIENT_VERSION,
            3,
            EGL14.EGL_NONE,
        )

        val ctx = EGL14.eglCreateContext(
            dpy,
            cfg,
            EGL14.EGL_NO_CONTEXT,
            ctxAttribs,
            0,
        )
        check(ctx != EGL14.EGL_NO_CONTEXT)

        stagingBytes = null

        val surfaceAttribs = intArrayOf(
            EGL14.EGL_WIDTH,
            1,
            EGL14.EGL_HEIGHT,
            1,
            EGL14.EGL_NONE,
        )

        val pbuffer = EGL14.eglCreatePbufferSurface(
            dpy,
            cfg,
            surfaceAttribs,
            0,
        )
        check(pbuffer != EGL14.EGL_NO_SURFACE)

        check(
            EGL14.eglMakeCurrent(
                dpy,
                pbuffer,
                pbuffer,
                ctx,
            ),
        )

        display = dpy
        contextEgl = ctx
        surface = pbuffer

        program = compileComputeProgram(vertexShaderSource)
        scaleLocation = GLES31.glGetUniformLocation(program, "scale")
        biasLocation = GLES31.glGetUniformLocation(program, "bias")
        countLocation = GLES31.glGetUniformLocation(program, "count")
    }

    private fun dispatch(
        values: FloatArray,
        scale: Float,
        bias: Float,
    ): FloatArray {
        val result = values.copyOf()

        val ids = IntArray(1)
        if (buffer == 0) {
            GLES31.glGenBuffers(1, ids, 0)
            buffer = ids[0]
        }

        GLES31.glBindBuffer(
            GLES31.GL_SHADER_STORAGE_BUFFER,
            buffer,
        )

        val requiredBytes = values.size * 4
        var staging = stagingBytes
        if (staging == null || staging.capacity() < requiredBytes) {
            staging = ByteBuffer
                .allocateDirect(requiredBytes)
                .order(ByteOrder.nativeOrder())
            stagingBytes = staging
            stagingReallocationCount += 1L
        } else {
            staging.clear()
        }

        val stagingFloatBuffer = staging
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        stagingFloatBuffer.clear()
        stagingFloatBuffer.put(values)
        stagingFloatBuffer.flip()

        GLES31.glBufferData(
            GLES31.GL_SHADER_STORAGE_BUFFER,
            requiredBytes,
            staging,
            GLES31.GL_DYNAMIC_COPY,
        )

        GLES31.glUseProgram(program)

        check(scaleLocation >= 0) { "Missing compute scale uniform" }
        check(biasLocation >= 0) { "Missing compute bias uniform" }
        check(countLocation >= 0) { "Missing compute count uniform" }

        GLES31.glUniform1f(scaleLocation, scale)
        GLES31.glUniform1f(biasLocation, bias)
        GLES31.glUniform1ui(
            countLocation,
            values.size,
        )

        GLES31.glBindBufferBase(
            GLES31.GL_SHADER_STORAGE_BUFFER,
            0,
            buffer,
        )

        val groups = ceil(values.size / 64.0).toInt()
        GLES31.glDispatchCompute(groups, 1, 1)
        checkNoGlError("compute dispatch")
        GLES31.glMemoryBarrier(
            GLES31.GL_SHADER_STORAGE_BARRIER_BIT,
        )

        val readback = GLES31.glMapBufferRange(
            GLES31.GL_SHADER_STORAGE_BUFFER,
            0,
            values.size * 4,
            GLES31.GL_MAP_READ_BIT,
        )

        if (readback is ByteBuffer) {
            val output = readback
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()

            output.get(result)
            GLES31.glUnmapBuffer(
                GLES31.GL_SHADER_STORAGE_BUFFER,
            )
        } else if (readback != null) {
            GLES31.glUnmapBuffer(
                GLES31.GL_SHADER_STORAGE_BUFFER,
            )
        }

        GLES31.glBindBuffer(
            GLES31.GL_SHADER_STORAGE_BUFFER,
            0,
        )

        return result
    }

    private fun checkNoGlError(
        operation: String,
    ) {
        val error = GLES31.glGetError()
        check(error == GLES31.GL_NO_ERROR) {
            "$operation failed with GL error=0x" +
                Integer.toHexString(error)
        }
    }

    private fun compileComputeProgram(source: String): Int {
        val shader = GLES31.glCreateShader(
            GLES31.GL_COMPUTE_SHADER,
        )

        GLES31.glShaderSource(shader, source)
        GLES31.glCompileShader(shader)

        val compiled = IntArray(1)
        GLES31.glGetShaderiv(
            shader,
            GLES31.GL_COMPILE_STATUS,
            compiled,
            0,
        )

        if (compiled[0] == 0) {
            val log = GLES31.glGetShaderInfoLog(shader)
            GLES31.glDeleteShader(shader)
            error("Compute shader compilation failed: $log")
        }

        val programId = GLES31.glCreateProgram()
        GLES31.glAttachShader(programId, shader)
        GLES31.glLinkProgram(programId)

        val linked = IntArray(1)
        GLES31.glGetProgramiv(
            programId,
            GLES31.GL_LINK_STATUS,
            linked,
            0,
        )

        GLES31.glDeleteShader(shader)

        if (linked[0] == 0) {
            val log = GLES31.glGetProgramInfoLog(
                programId,
            )
            GLES31.glDeleteProgram(programId)
            error("Compute program linking failed: $log")
        }

        return programId
    }
}
