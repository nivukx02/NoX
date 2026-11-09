package com.nivukx.nox.core.capture

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class CaptureResourceLeakSentinelTest {

    @Test
    fun duplicateAcquireDoesNotOverwriteLiveLease() {
        val sentinel = CaptureResourceLeakSentinel()
        val id = CaptureResourceLeakSentinel.ResourceId(
            CaptureResourceLeakSentinel.ResourceType.NATIVE_BUFFER,
            "buffer-1",
        )

        assertTrue(sentinel.acquire(id, 1L, 1024L, "native"))
        assertFalse(sentinel.acquire(id, 2L, 2048L, "native"))

        val snapshot = sentinel.snapshot(3L)
        assertEquals(1, snapshot.activeResources)
        assertEquals(1024L, snapshot.residentBytes)
        assertTrue("duplicate-resource-acquisition" in snapshot.warnings)
    }

    @Test
    fun ownerBudgetRejectsUnboundedResidentGrowth() {
        val sentinel = CaptureResourceLeakSentinel(
            maximumBytesPerOwner = 1024L,
        )
        val accepted = sentinel.acquire(
            id = CaptureResourceLeakSentinel.ResourceId(
                CaptureResourceLeakSentinel.ResourceType.TSDF_BLOCK,
                "tsdf-a",
            ),
            timestampNanos = 1L,
            bytes = 1024L,
            owner = "tsdf",
        )
        val rejected = sentinel.acquire(
            id = CaptureResourceLeakSentinel.ResourceId(
                CaptureResourceLeakSentinel.ResourceType.TSDF_BLOCK,
                "tsdf-b",
            ),
            timestampNanos = 2L,
            bytes = 1L,
            owner = "tsdf",
        )

        assertTrue(accepted)
        assertFalse(rejected)

        val snapshot = sentinel.snapshot(3L)
        assertEquals(1, snapshot.activeResources)
        assertEquals(1024L, snapshot.residentBytes)
        assertTrue("resource-acquisition-rejected" in snapshot.warnings)
    }

    @Test
    fun releaseAllRecordsEveryReleasedLeaseInChurnAccounting() {
        val sentinel = CaptureResourceLeakSentinel()
        repeat(4) { index ->
            sentinel.acquire(
                id = CaptureResourceLeakSentinel.ResourceId(
                    CaptureResourceLeakSentinel.ResourceType.WORKER,
                    "worker-$index",
                ),
                timestampNanos = index.toLong() + 1L,
                bytes = 16L,
                owner = "workers",
            )
        }

        assertEquals(4, sentinel.releaseAll(10L))
        val snapshot = sentinel.snapshot(20L)

        assertEquals(0, snapshot.activeResources)
        assertEquals(4.0, snapshot.releasesPerMinute)
    }

    @Test
    fun typeAndOwnerBreakdownsRemainSeparated() {
        val sentinel = CaptureResourceLeakSentinel()
        sentinel.acquire(
            CaptureResourceLeakSentinel.ResourceId(
                CaptureResourceLeakSentinel.ResourceType.FACE_TRACK,
                "face-a",
            ),
            1L,
            200L,
            "face",
        )
        sentinel.acquire(
            CaptureResourceLeakSentinel.ResourceId(
                CaptureResourceLeakSentinel.ResourceType.EYE_TRACK,
                "eye-a",
            ),
            2L,
            300L,
            "face",
        )

        val snapshot = sentinel.snapshot(3_000_000_000L)
        assertEquals(2, snapshot.activeResources)
        assertEquals(500L, snapshot.byOwner["face"]?.residentBytes)
        assertEquals(200L, snapshot.byType[
            CaptureResourceLeakSentinel.ResourceType.FACE_TRACK
        ]?.residentBytes)
        assertEquals(300L, snapshot.byType[
            CaptureResourceLeakSentinel.ResourceType.EYE_TRACK
        ]?.residentBytes)
    }

    @Test
    fun staleLeasesAreExplicitlyReported() {
        val sentinel = CaptureResourceLeakSentinel(
            staleLeaseThresholdMilliseconds = 100L,
        )
        sentinel.acquire(
            CaptureResourceLeakSentinel.ResourceId(
                CaptureResourceLeakSentinel.ResourceType.GPU_SURFACE,
                "surface",
            ),
            0L,
            4096L,
            "renderer",
        )

        val snapshot = sentinel.snapshot(200_000_000L)
        assertEquals(1, snapshot.staleResources)
        assertTrue(snapshot.oldestLeaseAgeMilliseconds >= 200L)
        assertTrue("stale-resource-leases" in snapshot.warnings)
    }

    @Test
    fun overflowSafeCountersDoNotWrapResidentBytes() {
        val sentinel = CaptureResourceLeakSentinel(
            maximumBytesPerOwner = Long.MAX_VALUE,
        )
        val huge = Long.MAX_VALUE

        assertTrue(
            sentinel.acquire(
                CaptureResourceLeakSentinel.ResourceId(
                    CaptureResourceLeakSentinel.ResourceType.NATIVE_BUFFER,
                    "huge-a",
                ),
                1L,
                huge,
                "native",
            ),
        )
        assertFalse(
            sentinel.acquire(
                CaptureResourceLeakSentinel.ResourceId(
                    CaptureResourceLeakSentinel.ResourceType.NATIVE_BUFFER,
                    "huge-b",
                ),
                2L,
                1L,
                "native",
            ),
        )

        assertEquals(Long.MAX_VALUE, sentinel.snapshot(3L).residentBytes)
    }
}
