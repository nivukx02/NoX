package com.nivukx.nox.core.sync

import com.nivukx.nox.core.model.DeviceId
import java.util.LinkedHashMap

/**
 * Bounded multi-camera timing-beacon observation store.
 *
 * Keeps only recent, detector-qualified observations per camera and hands the complete bounded
 * collection to VisualTimingBeaconCalibrationEngine. The store is transport-neutral, so local
 * Camera2 observations and observations received from Nearby/Wi-Fi/recorded sessions can share the
 * same calibration path.
 */
class DistributedTimingBeaconObservationBuffer(
    private val config: Config = Config(),
) {

    data class Config(
        val maximumDevices: Int = 128,
        val maximumObservationsPerDevice: Int = 128,
        val maximumTotalObservations: Int = 8_192,
        val maximumAgeNanos: Long = 2_000_000_000L,
        val minimumStoredDecodeConfidence: Double = 0.55,
    ) {
        init {
            require(maximumDevices in 1..128)
            require(maximumObservationsPerDevice in 3..512)
            require(maximumTotalObservations >= maximumObservationsPerDevice)
            require(maximumTotalObservations <= 65_536)
            require(maximumAgeNanos > 0L)
            require(
                minimumStoredDecodeConfidence.isFinite() &&
                    minimumStoredDecodeConfidence in 0.0..1.0,
            )
        }
    }

    private val observations =
        LinkedHashMap<DeviceId, ArrayDeque<VisualTimingBeaconCalibrationEngine.Observation>>()

    @Synchronized
    fun add(
        observation: VisualTimingBeaconCalibrationEngine.Observation,
    ): Boolean {
        if (
            observation.decodeConfidence <
                config.minimumStoredDecodeConfidence
        ) {
            return false
        }

        val deque =
            observations[observation.deviceId]
                ?: createDeviceBuffer().also {
                    observations[observation.deviceId] = it
                }

        if (
            deque.isNotEmpty() &&
            observation.deviceTimestampNanos <
                deque.last().deviceTimestampNanos
        ) {
            /*
             * A late packet is still useful when it belongs to a recent capture window. Insert
             * deterministically rather than throwing away a valid beacon because network ordering
             * differs from sensor ordering.
             */
            val existing =
                deque.toList()
            val merged =
                (
                    existing + observation
                )
                    .sortedWith(
                        compareBy<
                            VisualTimingBeaconCalibrationEngine.Observation
                        > {
                            it.deviceTimestampNanos
                        }.thenBy {
                            it.beaconTimestampNanos
                        },
                    )
                    .takeLast(
                        config.maximumObservationsPerDevice,
                    )

            observations[observation.deviceId] =
                ArrayDeque(merged)
        } else {
            deque.addLast(observation)
            while (
                deque.size >
                    config.maximumObservationsPerDevice
            ) {
                deque.removeFirst()
            }
        }

        enforceGlobalCapacity()
        return true
    }

    @Synchronized
    fun expire(
        nowNanos: Long,
    ) {
        require(nowNanos >= 0L)

        val cutoff =
            nowNanos -
                config.maximumAgeNanos

        val iterator =
            observations.entries.iterator()

        while (iterator.hasNext()) {
            val entry = iterator.next()
            val deque = entry.value

            while (
                deque.isNotEmpty() &&
                deque.first().deviceTimestampNanos <
                    cutoff
            ) {
                deque.removeFirst()
            }

            if (deque.isEmpty()) {
                iterator.remove()
            }
        }
    }

    @Synchronized
    fun snapshot(
        nowNanos: Long? = null,
    ): List<VisualTimingBeaconCalibrationEngine.Observation> {
        nowNanos?.let(::expire)

        return observations.values
            .asSequence()
            .flatMap { it.asSequence() }
            .toList()
    }

    @Synchronized
    fun calibrate(
        engine: VisualTimingBeaconCalibrationEngine =
            VisualTimingBeaconCalibrationEngine(),
        nowNanos: Long? = null,
    ): VisualTimingBeaconCalibrationEngine.CalibrationResult =
        engine.calibrate(
            observations = snapshot(nowNanos),
        )

    @Synchronized
    fun clear() {
        observations.clear()
    }

    @Synchronized
    fun clear(
        deviceId: DeviceId,
    ) {
        observations.remove(deviceId)
    }

    @Synchronized
    fun deviceCount(): Int =
        observations.size

    @Synchronized
    fun observationCount(
        deviceId: DeviceId,
    ): Int =
        observations[deviceId]?.size ?: 0

    @Synchronized
    fun totalObservationCount(): Int =
        observations.values.sumOf { it.size }

    private fun createDeviceBuffer():
        ArrayDeque<VisualTimingBeaconCalibrationEngine.Observation> {
        if (
            observations.size >=
                config.maximumDevices
        ) {
            observations.entries
                .firstOrNull()
                ?.key
                ?.let(observations::remove)
        }

        return ArrayDeque<
            VisualTimingBeaconCalibrationEngine.Observation
        >().also {
            /*
             * The caller inserts the resulting deque immediately after creation.
             * Keeping creation isolated makes capacity enforcement deterministic.
             */
        }
    }

    private fun enforceGlobalCapacity() {
        while (
            totalObservationCount() >
                config.maximumTotalObservations
        ) {
            val victim =
                observations.entries
                    .minByOrNull { (_, deque) ->
                        deque.firstOrNull()
                            ?.deviceTimestampNanos
                            ?: Long.MAX_VALUE
                    }
                    ?.key
                    ?: break

            val deque =
                observations[victim]
                    ?: break

            if (deque.isNotEmpty()) {
                deque.removeFirst()
            }

            if (deque.isEmpty()) {
                observations.remove(victim)
            }
        }
    }
}
