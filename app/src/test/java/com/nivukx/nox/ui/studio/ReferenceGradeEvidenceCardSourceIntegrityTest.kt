package com.nivukx.nox.ui.studio

import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.assertTrue
import org.junit.Test

class ReferenceGradeEvidenceCardSourceIntegrityTest {
    @Test
    fun referenceGradeCard_keeps_measured_and_modeled_precision_separate() {
        val sourcePath = Path.of(
            "src/main/java/com/nivukx/nox/ui/studio/ReferenceGradeEvidenceCard.kt",
        )
        assertTrue(sourcePath.toFile().exists())
        val source = Files.readString(sourcePath)
        assertTrue(source.contains("REFERENCE-GRADE EVIDENCE"))
        assertTrue(source.contains("Model P95"))
        assertTrue(source.contains("Measured P95"))
        assertTrue(source.contains("Measured P99"))
        assertTrue(source.contains("empirical benchmark"))
        assertTrue(source.contains("infrared-superiority"))
    }
}
