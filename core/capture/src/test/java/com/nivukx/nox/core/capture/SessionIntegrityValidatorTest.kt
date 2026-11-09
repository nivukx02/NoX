package com.nivukx.nox.core.capture

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.DataOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.file.Files

class SessionIntegrityValidatorTest {

    @Test
    fun schema14RequiresProductionProvenanceStreams() {
        val session = Files
            .createTempDirectory("nox-schema14-integrity")
            .toFile()

        try {
            write(
                session,
                "manifest.json",
                JSONObject()
                    .put("schema", 14)
                    .put("calibration", JSONObject())
                    .put("motionCaptureProfile", JSONObject())
                    .put("devices", emptyList<Any>())
                    .put("motionOnly", true)
                    .put("mediaFramesPersisted", false)
                    .put("videoEncoder", "none")
                    .put(
                        "denseMesh",
                        JSONObject()
                            .put("file", "dense_mesh.bin")
                            .put("index", "dense_mesh_index.ndjson"),
                    )
                    .toString(),
            )
            write(
                session,
                "frames.ndjson",
                """{"sensorTimestampNanos":1000000000}""" + "\n",
            )
            write(session, "focus.ndjson", "")

            val report =
                SessionIntegrityValidator()
                    .validate(session)

            assertTrue(!report.valid)
            assertTrue(
                report.errors.any {
                    it.contains("schema 14 requires motion_uncertainty")
                },
            )
            assertTrue(
                report.errors.any {
                    it.contains("schema 14 requires production_policy")
                },
            )
            assertTrue(
                report.errors.any {
                    it.contains("schema 14 requires cross_modal_identity")
                },
            )
        } finally {
            session.deleteRecursively()
        }
    }

    @Test
    fun schema12FaceMeshSessionValidatesItsRecordedBlock() {
        val session = Files
            .createTempDirectory("nox-session-integrity")
            .toFile()

        try {
            writeManifest(session)
            write(
                session,
                "frames.ndjson",
                """{"sensorTimestampNanos":1000000000}""" + "\n",
            )
            write(session, "focus.ndjson", "")
            write(session, "dense_mesh.bin", ByteArray(0))
            write(session, "dense_mesh_index.ndjson", "")
            writeFaceMeshBinary(session)
            write(
                session,
                "face_mesh_index.ndjson",
                JSONObject()
                    .put("timestampNanos", 1L)
                    .put("offsetBytes", 0L)
                    .put("lengthBytes", 124L)
                    .put("trackingId", 7)
                    .put("points", 3)
                    .put("triangles", 1)
                    .put("confidence", 0.95)
                    .put("pointCoverage", 0.90)
                    .put("triangleCoverage", 0.90)
                    .toString() + "\n",
            )

            val report = SessionIntegrityValidator()
                .validate(session)

            assertTrue(
                "session errors: " + report.errors.joinToString(),
                report.valid,
            )
            assertEquals(12, report.manifestSchema)
            assertEquals(1L, report.faceMeshBlocks)
            assertEquals(124L, report.faceMeshBytes)
        } finally {
            session.deleteRecursively()
        }
    }

    private fun writeManifest(
        session: File,
    ) {
        write(
            session,
            "manifest.json",
            JSONObject()
                .put("schema", 12)
                .put("calibration", JSONObject())
                .put("motionCaptureProfile", JSONObject())
                .put("devices", emptyList<Any>())
                .put("motionOnly", true)
                .put("mediaFramesPersisted", false)
                .put("videoEncoder", "none")
                .put(
                    "denseMesh",
                    JSONObject()
                        .put("file", "dense_mesh.bin")
                        .put("index", "dense_mesh_index.ndjson"),
                )
                .toString(),
        )
    }

    private fun writeFaceMeshBinary(
        session: File,
    ) {
        val file = File(session, "face_mesh.bin")
        DataOutputStream(
            FileOutputStream(file),
        ).use { output ->
            output.writeInt(0x4E58464D)
            output.writeInt(1)
            output.writeLong(1L)
            output.writeInt(7)
            output.writeInt(640)
            output.writeInt(480)
            output.writeInt(0)
            output.writeFloat(10f)
            output.writeFloat(20f)
            output.writeFloat(600f)
            output.writeFloat(460f)
            output.writeFloat(0.95f)
            output.writeFloat(0.90f)
            output.writeInt(3)
            output.writeInt(1)

            writePoint(
                output,
                index = 0,
                x = 100f,
                y = 100f,
                z = 0f,
            )
            writePoint(
                output,
                index = 1,
                x = 120f,
                y = 100f,
                z = 1f,
            )
            writePoint(
                output,
                index = 2,
                x = 100f,
                y = 120f,
                z = 2f,
            )

            output.writeInt(0)
            output.writeInt(1)
            output.writeInt(2)
        }
    }

    private fun writePoint(
        output: DataOutputStream,
        index: Int,
        x: Float,
        y: Float,
        z: Float,
    ) {
        output.writeInt(index)
        output.writeFloat(x)
        output.writeFloat(y)
        output.writeFloat(z)
    }

    private fun write(
        session: File,
        name: String,
        text: String,
    ) {
        File(session, name).writeText(text)
    }

    private fun write(
        session: File,
        name: String,
        bytes: ByteArray,
    ) {
        File(session, name).writeBytes(bytes)
    }
}
