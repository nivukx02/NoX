package com.nivukx.nox.core.capture

import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.file.Files

class MotionSessionQualityAuditEngineTest {

    @Test
    fun missingProductionProvenanceLowersCoverage() {
        val dir = Files.createTempDirectory("nox-quality-provenance").toFile()
        try {
            File(dir, "manifest.json").writeText(
                """{"sessionId":"provenance","motionOnly":true,"mediaFramesPersisted":false,"videoEncoder":"none"}""",
            )
            File(dir, "reconstruction.ndjson").writeText(
                (1..10).joinToString("\n") { index ->
                    """{"timestampNanos":$index,"confidence":0.92}"""
                },
            )
            File(dir, "motion_uncertainty.ndjson").writeText(
                (1..10).joinToString("\n") { index ->
                    """{"timestampNanos":$index}"""
                },
            )
            File(dir, "production_policy.ndjson").writeText(
                (1..10).joinToString("\n") { index ->
                    """{"timestampNanos":$index}"""
                },
            )

            val report =
                MotionSessionQualityAuditEngine().audit(
                    dir,
                    MotionSessionQualityAuditEngine.Config(
                        minimumAuditableFrames = 2,
                        minimumProductionEvidenceCoverage = 0.90,
                    ),
                )

            assertTrue(report.productionEvidenceCoverage <= 0.50)
            assertTrue(
                report.warnings.any {
                    it.contains("production provenance")
                },
            )
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun cleanStructuredSessionProducesBoundedAudit() {
        val dir = Files.createTempDirectory("nox-quality").toFile()
        try {
            File(dir, "manifest.json").writeText(
                "{\"sessionId\":\"quality\",\"motionOnly\":true,\"mediaFramesPersisted\":false,\"videoEncoder\":\"none\"}",
            )
            File(dir, "reconstruction.ndjson").writeText(
                "{\"timestampNanos\":1,\"confidence\":0.90,\"reprojectionErrorPx\":1.2}\n" +
                    "{\"timestampNanos\":10000001,\"confidence\":0.88,\"reprojectionErrorPx\":1.4}\n",
            )

            val report = MotionSessionQualityAuditEngine().audit(
                dir,
                MotionSessionQualityAuditEngine.Config(
                    minimumAuditableFrames = 2,
                ),
            )

            assertTrue(report.qualityScore in 0.0..1.0)
            assertTrue(report.frameCount == 2L)
        } finally {
            dir.deleteRecursively()
        }
    }
}
