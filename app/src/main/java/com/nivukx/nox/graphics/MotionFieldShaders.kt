package com.nivukx.nox.graphics

internal object MotionFieldShaders {
    // CI gate trigger: source behavior unchanged.
    val CAMERA_VERTEX = """
        #version 300 es
        in vec2 aPosition;
        in vec2 aTexCoord;
        uniform mat4 uTextureTransform;
        out vec2 vTexCoord;
        void main() {
            gl_Position = vec4(aPosition, 0.0, 1.0);
            vTexCoord = (uTextureTransform * vec4(aTexCoord, 0.0, 1.0)).xy;
        }
    """.trimIndent()

    val CAMERA_FRAGMENT = """
        #version 300 es
        #extension GL_OES_EGL_image_external_essl3 : require
        precision mediump float;
        uniform samplerExternalOES uCameraTexture;
        in vec2 vTexCoord;
        out vec4 outColor;
        void main() {
            outColor = texture(uCameraTexture, vTexCoord);
        }
    """.trimIndent()

    val OVERLAY_VERTEX = """
        #version 300 es
        in vec2 aPosition;
        in vec4 aColor;
        uniform float uPointSize;
        out vec4 vColor;
        void main() {
            gl_Position = vec4(aPosition, 0.0, 1.0);
            gl_PointSize = uPointSize;
            vColor = aColor;
        }
    """.trimIndent()

    val OVERLAY_FRAGMENT = """
        #version 300 es
        precision mediump float;
        in vec4 vColor;
        out vec4 outColor;
        void main() {
            outColor = vColor;
        }
    """.trimIndent()
}
