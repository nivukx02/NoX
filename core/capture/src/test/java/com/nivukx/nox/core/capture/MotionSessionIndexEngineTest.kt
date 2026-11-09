package com.nivukx.nox.core.capture

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.file.Files

class MotionSessionIndexEngineTest {

    @Test
    fun buildsTimestampedOffsetsAndSupportsRangeQuery() {
        val dir = Files.createTempDirectory("nox-index").toFile()
        try {
            File(dir, "poses.ndjson").writeText(
                """{"timestampNanos":100}
{"timestampNanos":200}
{"timestampNanos":300}
""",
            )

            val result = MotionSessionIndexEngine().build(dir)
            val query = MotionSessionIndexEngine().query(
                result = result,
                startTimestampNanos = 150L,
                endTimestampNanos = 250L,
                fileName = "poses.ndjson",
            )

            assertEquals(3L, result.totalRecords)
            assertEquals(1, query.size)
            assertEquals(200L, query.first().timestampNanos)
            assertTrue(
                File(dir, "session.index.json").isFile,
            )
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun indexWriteIsAtomicAtSessionBoundary() {
        val dir =
            Files.createTempDirectory(
                "nox-index-atomic",
            ).toFile()
        try {
            File(
                dir,
                "poses.ndjson",
            ).writeText(
                """{"timestampNanos":10}
{"timestampNanos":20}
""",
            )

            val result =
                MotionSessionIndexEngine().build(dir)

            val indexFile =
                File(
                    dir,
                    "session.index.json",
                )

            assertTrue(result.totalRecords == 2L)
            assertTrue(indexFile.isFile)
            assertTrue(
                dir.listFiles()
                    .orEmpty()
                    .none {
                        it.name ==
                            "session.index.json.tmp"
                    },
            )
            assertTrue(
                indexFile.readText()
                    .contains(
                        "\"schema\": 1",
                    ),
            )
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun strideReducesIndexDensityWithoutChangingRecordCount() {
        val dir = Files.createTempDirectory("nox-index-stride").toFile()
        try {
            File(dir, "poses.ndjson").writeText(
                (1L..10L).joinToString("\n") {
                    """{"timestampNanos":$it}"""
                } + "\n",
            )

            val result = MotionSessionIndexEngine().build(
                dir,
                MotionSessionIndexEngine.Policy(
                    strideRecords = 2,
                ),
            )

            assertEquals(10L, result.totalRecords)
            assertEquals(5L, result.files.single().indexedEntries)
        } finally {
            dir.deleteRecursively()
        }
    }
}
