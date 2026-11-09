package com.nivukx.nox.core.ai

import kotlin.test.Test
import kotlin.test.assertEquals

class RecordedVideoMetadataTest {
    @Test
    fun explicitRotationOverrideTakesPrecedenceOverContainerMetadata() {
        assertEquals(
            180,
            resolveRecordedVideoRotation(
                overrideDegrees = 180,
                containerDegrees = 90,
            ),
        )
    }

    @Test
    fun containerRotationIsUsedWhenNoOverrideIsProvided() {
        assertEquals(
            270,
            resolveRecordedVideoRotation(
                overrideDegrees = null,
                containerDegrees = 270,
            ),
        )
    }

    @Test
    fun absentRotationMetadataDefaultsToZeroDegrees() {
        assertEquals(
            0,
            resolveRecordedVideoRotation(
                overrideDegrees = null,
                containerDegrees = null,
            ),
        )
    }

    @Test
    fun explicitZeroOverrideCanDisableContainerRotation() {
        assertEquals(
            0,
            resolveRecordedVideoRotation(
                overrideDegrees = 0,
                containerDegrees = 90,
            ),
        )
    }
}
