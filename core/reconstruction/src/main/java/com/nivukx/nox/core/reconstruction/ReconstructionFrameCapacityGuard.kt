package com.nivukx.nox.core.reconstruction

import com.nivukx.nox.core.model.FrontierLandmarkMeasurement

/**
 * Performs stable first-seen de-duplication and eligibility filtering with O(k) retained memory.
 *
 * The consumer only needs the first maximumItems distinct eligible inputs plus proof of overflow.
 * Stopping after maximumItems + 1 unique eligible entries avoids materializing a full filtered or
 * de-duplicated copy of a potentially enormous source collection.
 */
internal object BoundedDistinctEligibleSelector {
    data class Selection<T>(
        val items: List<T>,
        val eligibleCountCappedAtOverflow: Int,
    )

    fun <T, K> select(
        items: Collection<T>,
        isEligible: (T) -> Boolean,
        keyOf: (T) -> K,
        maximumItems: Int,
    ): Selection<T> {
        require(maximumItems in 1 until Int.MAX_VALUE)

        val overflowCount = maximumItems + 1
        val selected = ArrayList<T>(minOf(maximumItems, items.size))
        val seen = HashSet<K>(minOf(overflowCount, items.size))
        var eligibleCount = 0

        for (item in items) {
            if (!isEligible(item)) continue
            if (!seen.add(keyOf(item))) continue

            if (eligibleCount < overflowCount) eligibleCount++
            if (selected.size < maximumItems) selected.add(item)
            if (eligibleCount >= overflowCount) break
        }

        return Selection(
            items = selected,
            eligibleCountCappedAtOverflow = eligibleCount,
        )
    }
}

/**
 * Balances an overloaded landmark budget across cameras without unbounded retained state.
 * Normal frames preserve the existing first-seen behavior. Overflow frames reserve per-camera
 * quotas using camera authority order, then reassign unused quota to cameras with more evidence.
 */
internal object BoundedMultiCameraLandmarkSelector {
    private data class Key(val landmarkId: Int, val cameraId: String, val timestampNanos: Long)

    fun select(
        items: Collection<FrontierLandmarkMeasurement>,
        timestampNanos: Long,
        maximumItems: Int,
        prioritizedCameraIds: List<String>,
    ): BoundedDistinctEligibleSelector.Selection<FrontierLandmarkMeasurement> {
        require(timestampNanos > 0L)
        require(maximumItems in 1 until Int.MAX_VALUE)
        val seen = HashSet<Key>(minOf(maximumItems + 1, items.size))
        val first = ArrayList<FrontierLandmarkMeasurement>(minOf(maximumItems, items.size))
        for (item in items) {
            if (item.observedAtNanos > timestampNanos) continue
            if (!seen.add(key(item))) continue
            if (first.size < maximumItems) first.add(item)
            if (seen.size > maximumItems) {
                first.clear()
                seen.clear()
                return selectOverflow(items, timestampNanos, maximumItems, prioritizedCameraIds)
            }
        }
        return BoundedDistinctEligibleSelector.Selection(first, seen.size)
    }

    private fun selectOverflow(
        items: Collection<FrontierLandmarkMeasurement>,
        timestampNanos: Long,
        maximumItems: Int,
        prioritizedCameraIds: List<String>,
    ): BoundedDistinctEligibleSelector.Selection<FrontierLandmarkMeasurement> {
        val cameraIds = prioritizedCameraIds.distinct()
            .take(ReconstructionFrameCapacityGuard.MAX_CAMERA_INPUTS)
        if (cameraIds.isEmpty()) {
            return BoundedDistinctEligibleSelector.select(
                items = items,
                isEligible = { it.observedAtNanos <= timestampNanos },
                keyOf = ::key,
                maximumItems = maximumItems,
            )
        }
        val base = maximumItems / cameraIds.size
        val remainder = maximumItems % cameraIds.size
        val quotas = LinkedHashMap<String, Int>(cameraIds.size)
        cameraIds.forEachIndexed { index, id -> quotas[id] = base + if (index < remainder) 1 else 0 }

        val admitted = ArrayList<FrontierLandmarkMeasurement>(maximumItems)
        val keys = HashSet<Key>(minOf(maximumItems, items.size))
        val counts = HashMap<String, Int>(cameraIds.size)
        for (item in items) {
            if (item.observedAtNanos > timestampNanos) continue
            val quota = quotas[item.cameraId] ?: continue
            if ((counts[item.cameraId] ?: 0) >= quota) continue
            if (!keys.add(key(item))) continue
            admitted.add(item)
            counts[item.cameraId] = (counts[item.cameraId] ?: 0) + 1
        }
        if (admitted.size < maximumItems) {
            for (item in items) {
                if (admitted.size >= maximumItems) break
                if (item.observedAtNanos > timestampNanos || item.cameraId !in quotas) continue
                if (keys.add(key(item))) admitted.add(item)
            }
        }
        return BoundedDistinctEligibleSelector.Selection(admitted, maximumItems + 1)
    }

    private fun key(item: FrontierLandmarkMeasurement) =
        Key(item.landmarkId, item.cameraId, item.observedAtNanos)
}

/**
 * Selects the best eligible surface observations with O(k) retained memory.
 *
 * Sorting a whole frame and applying take(k) afterwards is still O(n) memory. The bounded heap
 * below preserves the existing full evidence-weight ranking using confidence, visibility, temporal and photometric consistency, motion boundaries, and supporting camera count while allowing only the
 * requested number of candidates to remain resident. Equal scores keep the earliest input first,
 * matching a stable descending sort. Overflow counting saturates at max + 1 because the capacity
 * guard needs to know only whether the eligible stream exceeded its admitted size.
 */
internal object BoundedSurfaceObservationSelector {
    data class Selection(
        val observations: List<ProbabilisticDynamicSurfaceFusionEngine.SurfaceObservation>,
        val eligibleCountCappedAtOverflow: Int,
    )

    private data class RankedObservation(
        val inputIndex: Int,
        val observation: ProbabilisticDynamicSurfaceFusionEngine.SurfaceObservation,
        val score: Double,
    )

    fun select(
        observations: Collection<ProbabilisticDynamicSurfaceFusionEngine.SurfaceObservation>,
        timestampNanos: Long,
        maximumObservations: Int,
    ): Selection {
        require(timestampNanos >= 0L)
        require(maximumObservations in 1 until Int.MAX_VALUE)

        val retained = java.util.PriorityQueue<RankedObservation>(
            compareBy<RankedObservation> { it.score }
                .thenByDescending { it.inputIndex },
        )
        val overflowCount = maximumObservations + 1
        var eligibleCount = 0
        var inputIndex = 0

        observations.forEach { observation ->
            val index = inputIndex++
            if (observation.timestampNanos <= timestampNanos) {
                if (eligibleCount < overflowCount) eligibleCount++
                val candidate = RankedObservation(
                    inputIndex = index,
                    observation = observation,
                    score = observation.evidenceWeight,
                )
                if (retained.size < maximumObservations) {
                    retained.add(candidate)
                } else {
                    val lowest = retained.peek()
                    if (candidate.score > lowest.score) {
                        retained.poll()
                        retained.add(candidate)
                    }
                }
            }
        }

        val selected = retained.toList()
            .sortedWith(
                compareByDescending<RankedObservation> { it.score }
                    .thenBy { it.inputIndex },
            )
            .map { it.observation }

        return Selection(
            observations = selected,
            eligibleCountCappedAtOverflow = eligibleCount,
        )
    }
}

/**
 * Reports inputs that the reconstruction composition root bounds before expensive processing.
 *
 * Capacity is an implementation limit, not a promise that attached hardware can sustain the
 * requested workload. Any overflow is surfaced on the resulting frame and prevents it from being
 * reported as an unqualified successful measurement.
 */
internal object ReconstructionFrameCapacityGuard {
    const val MAX_CAMERA_INPUTS: Int = 128
    const val MAX_ACTOR_INPUTS: Int = 64

    enum class Stream {
        CAMERAS,
        ACTORS,
        LANDMARKS,
        SURFACE_OBSERVATIONS,
    }

    data class Result(
        val exceededStreams: Set<Stream>,
        val warnings: List<String>,
    ) {
        val isComplete: Boolean
            get() = exceededStreams.isEmpty()
    }

    fun inspect(
        cameraInputCount: Int,
        actorInputCount: Int,
        uniqueEligibleLandmarkCount: Int,
        eligibleSurfaceObservationCount: Int,
        maximumLandmarks: Int,
        maximumSurfaceObservations: Int,
    ): Result {
        require(cameraInputCount >= 0)
        require(actorInputCount >= 0)
        require(uniqueEligibleLandmarkCount >= 0)
        require(eligibleSurfaceObservationCount >= 0)
        require(maximumLandmarks >= 1)
        require(maximumSurfaceObservations >= 1)

        val exceeded =
            buildSet {
                if (cameraInputCount > MAX_CAMERA_INPUTS) add(Stream.CAMERAS)
                if (actorInputCount > MAX_ACTOR_INPUTS) add(Stream.ACTORS)
                if (uniqueEligibleLandmarkCount > maximumLandmarks) add(Stream.LANDMARKS)
                if (eligibleSurfaceObservationCount > maximumSurfaceObservations) {
                    add(Stream.SURFACE_OBSERVATIONS)
                }
            }

        val warnings =
            buildList {
                if (Stream.CAMERAS in exceeded) {
                    add(
                        "capacity-camera-input-truncated: unique-candidates-at-least=$cameraInputCount, " +
                            "maximum=$MAX_CAMERA_INPUTS",
                    )
                }
                if (Stream.ACTORS in exceeded) {
                    add(
                        "capacity-actor-input-truncated: unique-candidates-at-least=$actorInputCount, " +
                            "maximum=$MAX_ACTOR_INPUTS",
                    )
                }
                if (Stream.LANDMARKS in exceeded) {
                    add(
                        "capacity-landmark-input-truncated: at least " +
                            "${maximumLandmarks + 1} unique eligible landmarks; " +
                            "maximum=$maximumLandmarks",
                    )
                }
                if (Stream.SURFACE_OBSERVATIONS in exceeded) {
                    add(
                        "capacity-surface-observation-input-truncated: at least " +
                            "${maximumSurfaceObservations + 1} eligible observations; " +
                            "maximum=$maximumSurfaceObservations",
                    )
                }
            }

        return Result(
            exceededStreams = exceeded,
            warnings = warnings,
        )
    }
}
