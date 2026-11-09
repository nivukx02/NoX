package com.nivukx.nox

import java.io.File
import org.junit.Assert.assertTrue
import org.junit.Test

class AppStartupBrandingContractTest {
    @Test
    fun rootArtworkOrGeneratedFallbackProvidesTheAndroidLogo() {
        val artwork = File("../artwork_16k.png")
        val gradle = File("build.gradle.kts").readText()
        assertTrue(
            "missing source artwork must have a deterministic build-time fallback",
            (artwork.isFile && artwork.length() > 0L) ||
                gradle.contains("createFallbackNoxArtwork"),
        )
        assertTrue(
            "app build must copy artwork_16k.png into generated drawable resources",
            gradle.contains("artwork_16k.png"),
        )
        assertTrue(
            "generated Android resource must be named nox_logo.png",
            gradle.contains("""resolve("nox_logo.png")""") &&
                gradle.contains("""ImageIO.write(targetImage, "png", output)"""),
        )
    }

    @Test
    fun launcherUsesTheSameGeneratedLogo() {
        val manifest = File("src/main/AndroidManifest.xml").readText()
        assertTrue(
            "application icon must use the generated NoX artwork",
            manifest.contains("""android:icon="@drawable/nox_logo""""),
        )
        assertTrue(
            "round icon must use the generated NoX artwork",
            manifest.contains("""android:roundIcon="@drawable/nox_logo""""),
        )
    }

    @Test
    fun startupCannotBeginCameraCaptureBeforeThePreviewSurfaceExists() {
        val source = File("src/main/java/com/nivukx/nox/NoXViewModel.kt").readText()
        assertTrue(
            "camera startup must be gated by previewSurfaceReady",
            source.contains("if (!previewSurfaceReady) return"),
        )
        assertTrue(
            "camera permission must be checked before preview startup",
            source.contains("Manifest.permission.CAMERA"),
        )
    }

    @Test
    fun startupFailureHasAVisibleRecoverySurface() {
        val source = File("src/main/java/com/nivukx/nox/MainActivity.kt").readText()
        assertTrue(
            "MainActivity must expose a non-crashing startup recovery screen",
            source.contains("StartupRecoveryScreen"),
        )
        assertTrue(
            "UI must render before permission requests",
            source.indexOf("setContent") < source.indexOf("permissionLauncher.launch"),
        )
    }
}
