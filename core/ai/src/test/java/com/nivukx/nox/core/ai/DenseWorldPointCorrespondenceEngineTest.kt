package com.nivukx.nox.core.ai

import com.nivukx.nox.core.model.Vec2
import com.nivukx.nox.core.model.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DenseWorldPointCorrespondenceEngineTest {
    @Test
    fun worldContinuityReusesPersistentTrack() {
        val engine = DenseWorldPointCorrespondenceEngine()
        val first = engine.update(
            listOf(
                DenseWorldPointCorrespondenceEngine.Observation(1,10,1_000L,"a",Vec2(10.0,10.0),Vec3(0.0,0.0,1.0),Vec3(0.1,0.0,0.0),0.9),
            ),
            1_000L,
        )
        val id = first.tracks.single().persistentId
        val second = engine.update(
            listOf(
                DenseWorldPointCorrespondenceEngine.Observation(1,20,1_100L,"b",Vec2(11.0,10.0),Vec3(0.01,0.0,1.0),Vec3(0.1,0.0,0.0),0.92),
            ),
            1_100L,
        )
        assertEquals(id, second.tracks.single().persistentId)
        assertEquals(1, second.reusedTrackCount)
    }

    @Test
    fun distantWorldPointStartsNewTrack() {
        val engine = DenseWorldPointCorrespondenceEngine()
        engine.update(
            listOf(DenseWorldPointCorrespondenceEngine.Observation(1,1,1L,"a",Vec2(0.0,0.0),Vec3(0.0,0.0,0.0),null,0.9)),
            1L,
        )
        val second = engine.update(
            listOf(DenseWorldPointCorrespondenceEngine.Observation(1,2,2L,"b",Vec2(0.0,0.0),Vec3(10.0,0.0,0.0),null,0.9)),
            2L,
        )
        assertTrue(second.newTrackCount == 1)
    }
    @Test
    fun temporalContinuityReflectsActualGapBeforeTrackUpdate() {
        val engine = DenseWorldPointCorrespondenceEngine()
        engine.update(
            listOf(
                DenseWorldPointCorrespondenceEngine.Observation(
                    1, 1, 1_000L, "a", com.nivukx.nox.core.model.Vec2(0.0,0.0),
                    com.nivukx.nox.core.model.Vec3(0.0,0.0,1.0), null, 0.9,
                ),
            ),
            1_000L,
        )
        val result = engine.update(
            listOf(
                DenseWorldPointCorrespondenceEngine.Observation(
                    1, 2, 101_000_000L, "b", com.nivukx.nox.core.model.Vec2(0.1,0.1),
                    com.nivukx.nox.core.model.Vec3(0.01,0.0,1.0), null, 0.9,
                ),
            ),
            101_000_000L,
        )
        assertTrue(result.tracks.single().temporalContinuity < 0.75)
    }
}