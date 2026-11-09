package com.nivukx.nox.graphics

import android.content.Context
import android.util.AttributeSet
import android.view.Surface
import android.opengl.GLSurfaceView
import com.nivukx.nox.core.model.MotionFieldPreviewFrame

class MotionFieldPreviewSurface @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : GLSurfaceView(context, attrs) {
    private var cameraSurfaceListener: ((Surface?) -> Unit)? = null
    private var rendererFailureListener: ((Throwable) -> Unit)? = null

    private val fieldRenderer = MotionFieldGlesRenderer(
        onCameraSurfaceChanged = { surface ->
            post { cameraSurfaceListener?.invoke(surface) }
        },
        requestRender = { requestRender() },
        onRendererFailure = { error ->
            post { rendererFailureListener?.invoke(error) }
        },
    )

    init {
        setEGLContextClientVersion(3)
        preserveEGLContextOnPause = true
        setRenderer(fieldRenderer)
        renderMode = RENDERMODE_WHEN_DIRTY
        setZOrderOnTop(false)
        setZOrderMediaOverlay(false)
    }

    fun setCameraSurfaceListener(listener: ((Surface?) -> Unit)?) {
        cameraSurfaceListener = listener
    }

    fun setRendererFailureListener(listener: ((Throwable) -> Unit)?) {
        rendererFailureListener = listener
    }

    fun submitFrame(frame: MotionFieldPreviewFrame) {
        fieldRenderer.submitFrame(frame)
    }

    fun rendererTelemetry() = fieldRenderer.telemetry()

    fun rendererHealth() = fieldRenderer.health()

    override fun onDetachedFromWindow() {
        runCatching {
            cameraSurfaceListener?.invoke(null)
            queueEvent { fieldRenderer.release() }
        }.onFailure { error ->
            rendererFailureListener?.invoke(error)
        }
        super.onDetachedFromWindow()
    }
}
