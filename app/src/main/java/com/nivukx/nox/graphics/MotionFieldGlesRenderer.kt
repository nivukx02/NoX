package com.nivukx.nox.graphics

import android.graphics.SurfaceTexture
import android.opengl.GLES11Ext
import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.view.Surface
import com.nivukx.nox.core.model.MotionFieldActor
import com.nivukx.nox.core.model.MotionFieldGpuBackend
import com.nivukx.nox.core.model.MotionFieldGpuState
import com.nivukx.nox.core.model.MotionFieldPreviewFrame
import com.nivukx.nox.core.model.MotionFieldWorldJoint
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/**
 * GPU compositor for the live MotionField preview.
 *
 * Camera frames remain in a SurfaceTexture external texture. The renderer uploads only the small
 * motion-geometry stream and composites it over the latest camera frame on the GL render thread.
 */
class MotionFieldGlesRenderer(
    private val onCameraSurfaceChanged: (Surface?) -> Unit,
    private val requestRender: () -> Unit,
    private val onRendererFailure: (Throwable) -> Unit = {},
) : GLSurfaceView.Renderer {
    private val pendingFrame = AtomicReference<MotionFieldPreviewFrame?>(null)
    private val cameraFrameAvailable = AtomicBoolean(false)
    private val renderThreadReady = AtomicBoolean(false)
    @Volatile
    private var health = MotionFieldRendererHealth()

    private var surfaceTexture: SurfaceTexture? = null
    private var cameraSurface: Surface? = null
    private var cameraTextureId = 0
    private var cameraProgram = 0
    private var overlayProgram = 0
    private var cameraTransformLocation = -1
    private var cameraSamplerLocation = -1
    private var cameraPositionLocation = -1
    private var cameraTexCoordLocation = -1
    private var overlayPointSizeLocation = -1
    private var overlayPositionLocation = -1
    private var overlayColorLocation = -1
    private var overlayBufferId = 0
    private var viewportWidth = 1
    private var viewportHeight = 1
    private var lastRenderLatencyMillis = 0.0
    private var renderFrames = 0L
    private var droppedFrames = 0L
    private val textureTransform = FloatArray(16)
    private val cameraVertexBuffer = floatBuffer(
        floatArrayOf(
            -1f, -1f, 0f, 1f,
             1f, -1f, 1f, 1f,
            -1f,  1f, 0f, 0f,
             1f,  1f, 1f, 0f,
        ),
    )
    private val overlayCpuBuffer = ByteBuffer
        .allocateDirect(MAX_OVERLAY_VERTICES * FLOATS_PER_VERTEX * 4)
        .order(ByteOrder.nativeOrder())

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        renderThreadReady.set(false)
        runCatching {
            releaseCameraInputSurface()
            cameraProgram = linkProgram(
                MotionFieldShaders.CAMERA_VERTEX,
                MotionFieldShaders.CAMERA_FRAGMENT,
            )
            overlayProgram = linkProgram(
                MotionFieldShaders.OVERLAY_VERTEX,
                MotionFieldShaders.OVERLAY_FRAGMENT,
            )

            cameraTransformLocation =
                GLES30.glGetUniformLocation(cameraProgram, "uTextureTransform")
            cameraSamplerLocation =
                GLES30.glGetUniformLocation(cameraProgram, "uCameraTexture")
            cameraPositionLocation =
                GLES30.glGetAttribLocation(cameraProgram, "aPosition")
            cameraTexCoordLocation =
                GLES30.glGetAttribLocation(cameraProgram, "aTexCoord")
            overlayPointSizeLocation =
                GLES30.glGetUniformLocation(overlayProgram, "uPointSize")
            overlayPositionLocation =
                GLES30.glGetAttribLocation(overlayProgram, "aPosition")
            overlayColorLocation =
                GLES30.glGetAttribLocation(overlayProgram, "aColor")

            check(cameraPositionLocation >= 0) { "MotionField camera position attribute missing" }
            check(cameraTexCoordLocation >= 0) { "MotionField camera texture attribute missing" }
            check(overlayPositionLocation >= 0) { "MotionField overlay position attribute missing" }
            check(overlayColorLocation >= 0) { "MotionField overlay color attribute missing" }

            GLES30.glUseProgram(cameraProgram)
            if (cameraSamplerLocation >= 0) {
                GLES30.glUniform1i(cameraSamplerLocation, 0)
            }

            val textureIds = IntArray(1)
            GLES30.glGenTextures(1, textureIds, 0)
            cameraTextureId = textureIds[0]
            GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, cameraTextureId)
            GLES30.glTexParameteri(
                GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES30.GL_TEXTURE_MIN_FILTER,
                GLES30.GL_LINEAR,
            )
            GLES30.glTexParameteri(
                GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES30.GL_TEXTURE_MAG_FILTER,
                GLES30.GL_LINEAR,
            )
            GLES30.glTexParameteri(
                GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES30.GL_TEXTURE_WRAP_S,
                GLES30.GL_CLAMP_TO_EDGE,
            )
            GLES30.glTexParameteri(
                GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES30.GL_TEXTURE_WRAP_T,
                GLES30.GL_CLAMP_TO_EDGE,
            )

            val bufferIds = IntArray(1)
            GLES30.glGenBuffers(1, bufferIds, 0)
            overlayBufferId = bufferIds[0]

            surfaceTexture = SurfaceTexture(cameraTextureId).also { texture ->
                texture.setOnFrameAvailableListener {
                    cameraFrameAvailable.set(true)
                    requestRender()
                }
            }
            cameraSurface = Surface(surfaceTexture)
            onCameraSurfaceChanged(cameraSurface)

            GLES30.glDisable(GLES30.GL_DEPTH_TEST)
            GLES30.glEnable(GLES30.GL_BLEND)
            GLES30.glBlendFunc(GLES30.GL_SRC_ALPHA, GLES30.GL_ONE_MINUS_SRC_ALPHA)
            GLES30.glClearColor(0f, 0f, 0f, 1f)
            health = MotionFieldRendererHealth(lifecycle = MotionFieldRendererLifecycle.READY)
            renderThreadReady.set(true)
        }.onFailure { error ->
            health = MotionFieldRendererHealth(
                lifecycle = MotionFieldRendererLifecycle.FAILED,
                lastError = error.message ?: error.javaClass.simpleName,
            )
            runCatching { releaseCameraInputSurface() }
            runCatching { onRendererFailure(error) }
        }
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        viewportWidth = width.coerceAtLeast(1)
        viewportHeight = height.coerceAtLeast(1)
        GLES30.glViewport(0, 0, viewportWidth, viewportHeight)
        surfaceTexture?.setDefaultBufferSize(viewportWidth, viewportHeight)
    }

    override fun onDrawFrame(gl: GL10?) {
        val startNanos = System.nanoTime()
        val texture = surfaceTexture ?: return

        if (cameraFrameAvailable.getAndSet(false)) {
            runCatching {
                texture.updateTexImage()
                texture.getTransformMatrix(textureTransform)
            }.onFailure {
                droppedFrames += 1L
                health = health.copy(droppedFrames = droppedFrames)
                return
            }
        }

        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
        drawCameraTexture()

        pendingFrame.get()?.let(::drawMotionOverlay)

        renderFrames += 1L
        lastRenderLatencyMillis =
            (System.nanoTime() - startNanos).coerceAtLeast(0L) / 1e6
        health = health.copy(
            frameCount = renderFrames,
            droppedFrames = droppedFrames,
        )
    }

    fun submitFrame(frame: MotionFieldPreviewFrame) {
        pendingFrame.set(frame)
        if (renderThreadReady.get()) requestRender()
    }

    fun telemetry(): MotionFieldGpuState =
        MotionFieldGpuState(
            backend = MotionFieldGpuBackend.OPENGL_ES_RENDER,
            mode = pendingFrame.get()?.gpu?.mode
                ?: com.nivukx.nox.core.model.MotionFieldGpuMode.AUTO,
            renderLatencyMillis = lastRenderLatencyMillis,
            gpuUtilizationEstimate =
                (lastRenderLatencyMillis / DISPLAY_FRAME_BUDGET_MS)
                    .coerceIn(0.0, 1.0),
            droppedFrames = droppedFrames,
            warning =
                "GPU load is estimated from render-budget occupancy; driver utilization is not exposed by this backend.",
        )

    fun health(): MotionFieldRendererHealth = health

    fun release() {
        val hadResources =
            renderThreadReady.getAndSet(false) ||
                cameraTextureId != 0 ||
                overlayBufferId != 0 ||
                cameraProgram != 0 ||
                overlayProgram != 0 ||
                surfaceTexture != null ||
                cameraSurface != null
        if (!hadResources) {
            health = health.copy(lifecycle = MotionFieldRendererLifecycle.RELEASED)
            return
        }
        releaseCameraInputSurface()
        if (cameraTextureId != 0) {
            GLES30.glDeleteTextures(1, intArrayOf(cameraTextureId), 0)
            cameraTextureId = 0
        }
        if (overlayBufferId != 0) {
            GLES30.glDeleteBuffers(1, intArrayOf(overlayBufferId), 0)
            overlayBufferId = 0
        }
        if (cameraProgram != 0) {
            GLES30.glDeleteProgram(cameraProgram)
            cameraProgram = 0
        }
        if (overlayProgram != 0) {
            GLES30.glDeleteProgram(overlayProgram)
            overlayProgram = 0
        }
        pendingFrame.set(null)
        health = health.copy(lifecycle = MotionFieldRendererLifecycle.RELEASED)
    }

    private fun drawCameraTexture() {
        GLES30.glUseProgram(cameraProgram)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, cameraTextureId)
        GLES30.glUniformMatrix4fv(
            cameraTransformLocation,
            1,
            false,
            textureTransform,
            0,
        )

        cameraVertexBuffer.position(0)
        GLES30.glEnableVertexAttribArray(cameraPositionLocation)
        GLES30.glVertexAttribPointer(
            cameraPositionLocation,
            2,
            GLES30.GL_FLOAT,
            false,
            FLOATS_PER_CAMERA_VERTEX * 4,
            cameraVertexBuffer,
        )
        cameraVertexBuffer.position(2)
        GLES30.glEnableVertexAttribArray(cameraTexCoordLocation)
        GLES30.glVertexAttribPointer(
            cameraTexCoordLocation,
            2,
            GLES30.GL_FLOAT,
            false,
            FLOATS_PER_CAMERA_VERTEX * 4,
            cameraVertexBuffer,
        )
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)
        GLES30.glDisableVertexAttribArray(cameraPositionLocation)
        GLES30.glDisableVertexAttribArray(cameraTexCoordLocation)
    }

    private fun drawMotionOverlay(frame: MotionFieldPreviewFrame) {
        val buffer = overlayCpuBuffer.asFloatBuffer()
        buffer.clear()
        var lineVertexCount = 0
        var pointVertexCount = 0

        frame.actors.forEach { actor ->
            val jointMap = actor.joints.associateBy { it.landmarkId }
            CONNECTIONS.forEach { (a, b) ->
                val first = jointMap[a] ?: return@forEach
                val second = jointMap[b] ?: return@forEach
                appendVertex(
                    buffer,
                    first.x * 2.0 - 1.0,
                    1.0 - first.y * 2.0,
                    actor,
                    first.confidence,
                )
                appendVertex(
                    buffer,
                    second.x * 2.0 - 1.0,
                    1.0 - second.y * 2.0,
                    actor,
                    second.confidence,
                )
                lineVertexCount += 2
            }
        }

        if (frame.reconstructionJoints3d.isNotEmpty()) {
            val maxAbs = frame.reconstructionJoints3d.maxOf {
                maxOf(
                    kotlin.math.abs(it.xMeters),
                    kotlin.math.abs(it.yMeters),
                    kotlin.math.abs(it.zMeters),
                )
            }.coerceAtLeast(0.25)
            val worldMap = frame.reconstructionJoints3d.associateBy {
                it.landmarkId
            }
            CONNECTIONS.forEach { (a, b) ->
                val first = worldMap[a] ?: return@forEach
                val second = worldMap[b] ?: return@forEach
                val p1 = projectWorld(first, maxAbs)
                val p2 = projectWorld(second, maxAbs)
                appendWorldVertex(
                    buffer,
                    p1.first,
                    p1.second,
                    first.confidence,
                )
                appendWorldVertex(
                    buffer,
                    p2.first,
                    p2.second,
                    second.confidence,
                )
                lineVertexCount += 2
            }
        }

        frame.actors.forEach { actor ->
            actor.joints.forEach { joint ->
                appendVertex(
                    buffer,
                    joint.x * 2.0 - 1.0,
                    1.0 - joint.y * 2.0,
                    actor,
                    joint.confidence,
                )
                pointVertexCount += 1
            }
        }

        if (frame.reconstructionJoints3d.isNotEmpty()) {
            val maxAbs = frame.reconstructionJoints3d.maxOf {
                maxOf(
                    kotlin.math.abs(it.xMeters),
                    kotlin.math.abs(it.yMeters),
                    kotlin.math.abs(it.zMeters),
                )
            }.coerceAtLeast(0.25)
            frame.reconstructionJoints3d.forEach { joint ->
                val p = projectWorld(joint, maxAbs)
                appendWorldVertex(
                    buffer,
                    p.first,
                    p.second,
                    joint.confidence,
                )
                pointVertexCount += 1
            }
        }

        val totalVertices = lineVertexCount + pointVertexCount
        if (totalVertices == 0) return

        buffer.flip()
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, overlayBufferId)
        GLES30.glBufferData(
            GLES30.GL_ARRAY_BUFFER,
            totalVertices * FLOATS_PER_VERTEX * 4,
            buffer,
            GLES30.GL_DYNAMIC_DRAW,
        )
        GLES30.glUseProgram(overlayProgram)
        GLES30.glEnableVertexAttribArray(overlayPositionLocation)
        GLES30.glEnableVertexAttribArray(overlayColorLocation)
        GLES30.glVertexAttribPointer(
            overlayPositionLocation,
            2,
            GLES30.GL_FLOAT,
            false,
            FLOATS_PER_VERTEX * 4,
            0,
        )
        GLES30.glVertexAttribPointer(
            overlayColorLocation,
            4,
            GLES30.GL_FLOAT,
            false,
            FLOATS_PER_VERTEX * 4,
            2 * 4,
        )
        GLES30.glUniform1f(overlayPointSizeLocation, 7f)

        if (lineVertexCount > 0) {
            GLES30.glDrawArrays(GLES30.GL_LINES, 0, lineVertexCount)
        }
        if (pointVertexCount > 0) {
            GLES30.glDrawArrays(
                GLES30.GL_POINTS,
                lineVertexCount,
                pointVertexCount,
            )
        }

        GLES30.glDisableVertexAttribArray(overlayPositionLocation)
        GLES30.glDisableVertexAttribArray(overlayColorLocation)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0)
    }

    private fun appendVertex(
        target: FloatBuffer,
        x: Double,
        y: Double,
        actor: MotionFieldActor,
        confidence: Double,
    ) {
        target.put(x.toFloat())
        target.put(y.toFloat())
        if (actor.predicted) {
            target.put(1f).put(0.64f).put(0.35f)
        } else {
            target.put(0.43f).put(0.86f).put(1f)
        }
        target.put((0.28 + confidence.coerceIn(0.0, 1.0) * 0.72).toFloat())
    }

    private fun appendWorldVertex(
        target: FloatBuffer,
        x: Double,
        y: Double,
        confidence: Double,
    ) {
        target.put(x.toFloat())
        target.put(y.toFloat())
        target.put(0.43f).put(0.96f).put(0.74f)
        target.put((0.24 + confidence.coerceIn(0.0, 1.0) * 0.76).toFloat())
    }

    private fun projectWorld(joint: MotionFieldWorldJoint, scale: Double): Pair<Double, Double> {
        val insetLeft = 0.64
        val insetTop = -0.84
        val insetWidth = 0.32
        val insetHeight = 0.42
        val yaw = joint.zMeters * 0.18
        val x = (joint.xMeters - yaw) / scale
        val y = -joint.yMeters / scale
        return Pair(
            insetLeft + (x * 0.5 + 0.5) * insetWidth,
            insetTop + (y * 0.5 + 0.5) * insetHeight,
        )
    }

    private fun releaseCameraInputSurface() {
        onCameraSurfaceChanged(null)
        cameraSurface?.release()
        cameraSurface = null
        surfaceTexture?.setOnFrameAvailableListener(null)
        surfaceTexture?.release()
        surfaceTexture = null
        cameraFrameAvailable.set(false)
    }

    private fun linkProgram(vertexSource: String, fragmentSource: String): Int {
        val vertex = compileShader(GLES30.GL_VERTEX_SHADER, vertexSource)
        val fragment = compileShader(GLES30.GL_FRAGMENT_SHADER, fragmentSource)
        val program = GLES30.glCreateProgram()
        GLES30.glAttachShader(program, vertex)
        GLES30.glAttachShader(program, fragment)
        GLES30.glLinkProgram(program)
        val linked = IntArray(1)
        GLES30.glGetProgramiv(program, GLES30.GL_LINK_STATUS, linked, 0)
        GLES30.glDeleteShader(vertex)
        GLES30.glDeleteShader(fragment)
        check(linked[0] != 0) {
            "MotionField shader link failed: ${GLES30.glGetProgramInfoLog(program)}"
        }
        return program
    }

    private fun compileShader(type: Int, source: String): Int {
        val shader = GLES30.glCreateShader(type)
        check(shader != 0) { "MotionField shader allocation failed" }
        GLES30.glShaderSource(shader, source)
        GLES30.glCompileShader(shader)
        val compiled = IntArray(1)
        GLES30.glGetShaderiv(shader, GLES30.GL_COMPILE_STATUS, compiled, 0)
        if (compiled[0] == 0) {
            val log = GLES30.glGetShaderInfoLog(shader)
            GLES30.glDeleteShader(shader)
            error("MotionField shader compilation failed: $log")
        }
        return shader
    }

    private fun floatBuffer(values: FloatArray): FloatBuffer =
        ByteBuffer.allocateDirect(values.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .apply { put(values); flip() }

    companion object {
        private const val FLOATS_PER_CAMERA_VERTEX = 4
        private const val FLOATS_PER_VERTEX = 6
        private const val MAX_OVERLAY_VERTICES = 32768
        private const val DISPLAY_FRAME_BUDGET_MS = 16.6667
        private val CONNECTIONS = listOf(
            0 to 1, 1 to 2, 2 to 3, 3 to 7,
            0 to 4, 4 to 5, 5 to 6, 6 to 8,
            9 to 10,
            11 to 12, 11 to 13, 13 to 15, 15 to 17, 15 to 19, 15 to 21, 17 to 19,
            12 to 14, 14 to 16, 16 to 18, 16 to 20, 16 to 22, 18 to 20,
            11 to 23, 12 to 24, 23 to 24,
            23 to 25, 25 to 27, 27 to 29, 29 to 31,
            24 to 26, 26 to 28, 28 to 30, 30 to 32,
        )
    }
}