package com.nivukx.nox.ui.studio

import org.junit.Assert.assertTrue
import org.junit.Test

class StudioMaterialAdaptationEngineTest {
    private val engine = StudioMaterialAdaptationEngine()

    @Test
    fun reducesTransparencyWhenAccessibilityRequiresIt() {
        val normal = engine.evaluate(
            StudioMaterialAdaptationEngine.MaterialRole.SECONDARY_TELEMETRY,
            StudioMaterialAdaptationEngine.Input(
                underlyingLuminance = 0.50,
                contentContrast = 0.90,
            ),
        )
        val reduced = engine.evaluate(
            StudioMaterialAdaptationEngine.MaterialRole.SECONDARY_TELEMETRY,
            StudioMaterialAdaptationEngine.Input(
                underlyingLuminance = 0.50,
                contentContrast = 0.90,
                reduceTransparency = true,
            ),
        )
        assertTrue(reduced.materialAlpha >= normal.materialAlpha)
        assertTrue(reduced.warnings.contains("REDUCED_TRANSPARENCY_ACTIVE"))
    }

    @Test
    fun protectsLegibilityOverBrightUnderlay() {
        val policy = engine.evaluate(
            StudioMaterialAdaptationEngine.MaterialRole.PRIMARY_NAVIGATION,
            StudioMaterialAdaptationEngine.Input(
                underlyingLuminance = 0.95,
                contentContrast = 0.40,
            ),
        )
        assertTrue(policy.materialAlpha > 0.45)
        assertTrue(policy.contentContrastBoost > 0.0)
        assertTrue(policy.warnings.contains("HIGH_LUMINANCE_UNDERLAY"))
    }

    @Test
    fun transparencyAccessibilityModeRespectsConfiguredOpacityCeiling() {
        val compactEngine =
            StudioMaterialAdaptationEngine(
                StudioMaterialAdaptationEngine.Config(
                    minimumTransparency = 0.18,
                    maximumTransparency = 0.45,
                ),
            )

        val policy =
            compactEngine.evaluate(
                StudioMaterialAdaptationEngine.MaterialRole.PRIMARY_NAVIGATION,
                StudioMaterialAdaptationEngine.Input(
                    underlyingLuminance = 0.50,
                    contentContrast = 0.90,
                    reduceTransparency = true,
                ),
            )

        assertTrue(policy.materialAlpha <= 0.45)
        assertTrue(policy.materialAlpha >= 0.18)
    }

    @Test
    fun invalidMaterialPolicyThresholdsAreRejected() {
        kotlin.test.assertFailsWith<IllegalArgumentException> {
            StudioMaterialAdaptationEngine(
                StudioMaterialAdaptationEngine.Config(
                    lowLuminanceThreshold = 0.80,
                    highLuminanceThreshold = 0.20,
                ),
            )
        }
    }

    @Test
    fun reducedMotionLowersAnimationWithoutRemovingMaterial() {
        val policy = engine.evaluate(
            StudioMaterialAdaptationEngine.MaterialRole.TIMELINE,
            StudioMaterialAdaptationEngine.Input(
                underlyingLuminance = 0.45,
                contentContrast = 0.90,
                reduceMotion = true,
            ),
        )
        assertTrue(policy.animationScale < 0.50)
        assertTrue(policy.materialAlpha > 0.0)
        assertTrue(policy.preserveViewportPriority)
    }
}
