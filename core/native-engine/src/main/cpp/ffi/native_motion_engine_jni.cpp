#if defined(__ANDROID__)
#include <jni.h>
#include <android/hardware_buffer_jni.h>
#include <cstdint>
#include <vector>
#include <cmath>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "ffi/native_engine_abi.h"
#include "runtime/hardware_buffer_lease_registry.h"
#include "runtime/native_engine_runtime.h"
#include "synchronization/synchronization_estimator.h"

namespace {

std::mutex g_hardware_buffer_registry_mutex;
std::unordered_map<
    jlong,
    std::shared_ptr<
        nox::native::runtime::HardwareBufferLeaseRegistry>>
    g_hardware_buffer_registries;

std::shared_ptr<
    nox::native::runtime::HardwareBufferLeaseRegistry>
get_hardware_buffer_registry(
    jlong handle) noexcept {
    std::scoped_lock lock(
        g_hardware_buffer_registry_mutex);
    const auto iterator =
        g_hardware_buffer_registries.find(handle);
    return iterator == g_hardware_buffer_registries.end()
        ? nullptr
        : iterator->second;
}

jlongArray hardware_buffer_diagnostics_to_array(
    JNIEnv* env,
    nox::native::runtime::HardwareBufferLeaseRegistry* registry) {
    constexpr jsize kFieldCount = 6;
    jlongArray out = env->NewLongArray(kFieldCount);
    if (out == nullptr) return nullptr;

    const jlong values[kFieldCount] = {
        registry == nullptr
            ? 0L
            : static_cast<jlong>(registry->active_count()),
        registry == nullptr
            ? 0L
            : static_cast<jlong>(registry->resident_bytes()),
        registry == nullptr
            ? 0L
            : static_cast<jlong>(registry->acquire_failures()),
        registry == nullptr
            ? 0L
            : static_cast<jlong>(registry->invalid_release_count()),
        registry == nullptr
            ? 0L
            : static_cast<jlong>(registry->maximum_active_leases()),
        registry == nullptr
            ? 0L
            : static_cast<jlong>(registry->maximum_resident_bytes()),
    };
    env->SetLongArrayRegion(
        out,
        0,
        kFieldCount,
        values);
    return out;
}

jlongArray diagnostics_to_array(JNIEnv* env, const nox::native::NativeDiagnostics& d) {
    constexpr jsize kFieldCount = 43;
    jlongArray out = env->NewLongArray(kFieldCount);
    if (out == nullptr) return nullptr;

    const jlong values[kFieldCount] = {
        static_cast<jlong>(d.processed_observations),
        static_cast<jlong>(d.dropped_observations),
        static_cast<jlong>(d.capture_overflow_events),
        static_cast<jlong>(d.queue_depth),
        static_cast<jlong>(d.maximum_queue_depth),
        static_cast<jlong>(d.active_camera_count),
        static_cast<jlong>(d.worker_count),
        static_cast<jlong>(d.processing_epoch),
        static_cast<jlong>(d.active_buffer_blocks),
        static_cast<jlong>(d.free_buffer_blocks),
        static_cast<jlong>(d.workload_scale_ppm),
        static_cast<jlong>(d.queue_utilization_ppm),
        static_cast<jlong>(d.shedding),
        static_cast<jlong>(d.thermal_constrained),
        static_cast<jlong>(d.worker_processed_observations),
        static_cast<jlong>(d.worker_accepted_ratio_ppm),
        static_cast<jlong>(d.worker_quality_ppm),
        static_cast<jlong>(d.worker_temporal_gap_ppm),
        static_cast<jlong>(d.worker_queue_age_microseconds),
        static_cast<jlong>(d.worker_processing_microseconds),
        static_cast<jlong>(d.gpu_lease_active_count),
        static_cast<jlong>(d.gpu_lease_in_flight_count),
        static_cast<jlong>(d.gpu_lease_watchdog_aborts),
        static_cast<jlong>(d.gpu_lease_stale_generation_releases),
        static_cast<jlong>(d.gpu_lease_invalid_releases),
        static_cast<jlong>(d.gpu_lease_admission_rejected),
        static_cast<jlong>(d.gpu_lease_oldest_age_milliseconds),
        static_cast<jlong>(d.memory_lease_active_count),
        static_cast<jlong>(d.memory_lease_stale_count),
        static_cast<jlong>(d.memory_lease_invalid_releases),
        static_cast<jlong>(d.memory_lease_double_releases),
        static_cast<jlong>(d.memory_lease_forced_reclaims),
        static_cast<jlong>(d.memory_lease_bytes_active),
        static_cast<jlong>(d.memory_lease_oldest_age_milliseconds),
        static_cast<jlong>(d.worker_velocity_micropixels_per_second_ppm),
        static_cast<jlong>(d.worker_acceleration_micropixels_per_second2_ppm),
        static_cast<jlong>(d.worker_jerk_micropixels_per_second3_ppm),
        static_cast<jlong>(d.worker_p95_innovation_millipixels_ppm),
        static_cast<jlong>(d.worker_motion_direction_ppm),
        static_cast<jlong>(d.worker_micro_motion_ppm),
        static_cast<jlong>(d.worker_dynamic_stability_ppm),
        static_cast<jlong>(d.worker_exposure_motion_blur_risk_ppm),
        static_cast<jlong>(d.worker_subpixel_motion_retention_ppm),
    };

    env->SetLongArrayRegion(out, 0, kFieldCount, values);
    return out;
}

} // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_nivukx_nox_core_nativeengine_NativeMotionEngine_nativeCreate(
    JNIEnv*, jclass, jint worker_count, jint observation_capacity,
    jint deterministic_seed, jlong frame_budget_ns, jint validation_level) noexcept {
    const nox::native::NativeRuntimeConfig config{
        static_cast<std::uint32_t>(worker_count),
        static_cast<std::uint32_t>(observation_capacity),
        static_cast<std::uint32_t>(deterministic_seed),
        static_cast<std::uint64_t>(frame_budget_ns),
        static_cast<std::uint8_t>(validation_level),
    };

    auto* runtime =
        native_runtime_create(&config);
    if (runtime == nullptr) {
        return 0;
    }

    const auto handle =
        reinterpret_cast<jlong>(runtime);

    try {
        auto registry =
            std::make_shared<
                nox::native::runtime::HardwareBufferLeaseRegistry>();
        std::scoped_lock lock(
            g_hardware_buffer_registry_mutex);
        g_hardware_buffer_registries.emplace(
            handle,
            std::move(registry));
    } catch (...) {
        native_runtime_destroy(
            reinterpret_cast<nox::native::NativeRuntimeHandle>(
                handle));
        return 0;
    }

    return handle;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_nivukx_nox_core_nativeengine_NativeMotionEngine_nativeStart(
    JNIEnv*, jclass, jlong handle) noexcept {
    return native_runtime_start(reinterpret_cast<nox::native::NativeRuntimeHandle>(handle))
        ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_nivukx_nox_core_nativeengine_NativeMotionEngine_nativeIngest(
    JNIEnv* env, jclass, jlong handle, jint camera_id, jlong frame_id,
    jlong sensor_timestamp_nanos, jlong monotonic_timestamp_nanos,
    jlong exposure_duration_nanos, jlong rolling_shutter_skew_nanos,
    jlong nominal_frame_interval_nanos,
    jintArray keypoint_ids, jfloatArray keypoint_xy,
    jfloatArray keypoint_confidence) noexcept {
    if(
        handle==0 ||
        camera_id<0 ||
        frame_id<0 ||
        sensor_timestamp_nanos<0 ||
        monotonic_timestamp_nanos<0 ||
        exposure_duration_nanos<0 ||
        rolling_shutter_skew_nanos<0 ||
        nominal_frame_interval_nanos<0) {
        return nullptr;
    }

    const jsize confidence_count =
        keypoint_confidence == nullptr
            ? 0
            : env->GetArrayLength(keypoint_confidence);
    const jsize xy_count =
        keypoint_xy == nullptr
            ? 0
            : env->GetArrayLength(keypoint_xy);
    const jsize id_count =
        keypoint_ids == nullptr
            ? 0
            : env->GetArrayLength(keypoint_ids);

    if (
        confidence_count < 0 ||
        xy_count != confidence_count * 2 ||
        confidence_count > 128 ||
        !(id_count == 0 || id_count == confidence_count)) {
        return nullptr;
    }

    jint* ids =
        (keypoint_ids == nullptr || id_count == 0)
            ? nullptr
            : env->GetIntArrayElements(
                keypoint_ids,
                nullptr);

    jfloat* xy = keypoint_xy == nullptr
        ? nullptr
        : env->GetFloatArrayElements(keypoint_xy, nullptr);
    jfloat* confidence = keypoint_confidence == nullptr
        ? nullptr
        : env->GetFloatArrayElements(keypoint_confidence, nullptr);
    if (
        (id_count > 0 && ids == nullptr) ||
        (keypoint_xy != nullptr && xy == nullptr) ||
        (keypoint_confidence != nullptr && confidence == nullptr)) {
        if (ids != nullptr) {
            env->ReleaseIntArrayElements(
                keypoint_ids,
                ids,
                JNI_ABORT);
        }
        if (xy != nullptr) {
            env->ReleaseFloatArrayElements(
                keypoint_xy,
                xy,
                JNI_ABORT);
        }
        if (confidence != nullptr) {
            env->ReleaseFloatArrayElements(
                keypoint_confidence,
                confidence,
                JNI_ABORT);
        }
        return nullptr;
    }

    const nox::native::NativeCapturePacket packet{
        static_cast<std::uint32_t>(camera_id),
        static_cast<std::uint64_t>(frame_id),
        static_cast<std::uint64_t>(sensor_timestamp_nanos),
        static_cast<std::uint64_t>(monotonic_timestamp_nanos),
        static_cast<std::uint64_t>(exposure_duration_nanos),
        static_cast<std::uint64_t>(rolling_shutter_skew_nanos),
        static_cast<std::uint32_t>(confidence_count),
        reinterpret_cast<const std::uint32_t*>(ids),
        xy,
        confidence,
        static_cast<std::uint64_t>(
            nominal_frame_interval_nanos),
    };

    const auto result = native_runtime_ingest(
        reinterpret_cast<nox::native::NativeRuntimeHandle>(handle), &packet);

    if (ids != nullptr) {
        env->ReleaseIntArrayElements(
            keypoint_ids,
            ids,
            JNI_ABORT);
    }
    if (xy != nullptr) {
        env->ReleaseFloatArrayElements(
            keypoint_xy,
            xy,
            JNI_ABORT);
    }
    if (confidence != nullptr) {
        env->ReleaseFloatArrayElements(keypoint_confidence, confidence, JNI_ABORT);
    }

    jlongArray out = env->NewLongArray(8);
    if (out == nullptr) return nullptr;
    const jlong values[8] = {
        result.accepted ? 1L : 0L,
        result.duplicate ? 1L : 0L,
        result.out_of_order ? 1L : 0L,
        result.gap ? 1L : 0L,
        result.overflow ? 1L : 0L,
        static_cast<jlong>(result.gap_nanos),
        static_cast<jlong>(result.queue_depth),
        static_cast<jlong>(result.epoch),
    };
    env->SetLongArrayRegion(out, 0, 8, values);
    return out;
}

extern "C" JNIEXPORT void JNICALL
Java_com_nivukx_nox_core_nativeengine_NativeMotionEngine_nativeSetEnvironment(
    JNIEnv*, jclass, jlong handle, jint thermal_status, jdouble memory_pressure,
    jdouble gpu_headroom, jdouble npu_headroom, jboolean battery_saver) noexcept {
    if (handle == 0) return;
    static_cast<nox::native::NativeEngineRuntime*>(
        reinterpret_cast<nox::native::NativeRuntimeHandle>(handle)
    )->setEnvironment(
        thermal_status,
        memory_pressure,
        gpu_headroom,
        npu_headroom,
        battery_saver == JNI_TRUE);
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_nivukx_nox_core_nativeengine_NativeMotionEngine_nativeEstimateSynchronization(
    JNIEnv* env, jclass, jlong,
    jlongArray reference_timestamps, jlongArray remote_timestamps,
    jdoubleArray reference_events, jdoubleArray remote_events,
    jdouble event_sample_interval_nanos) noexcept {
    if(
        reference_timestamps==nullptr ||
        remote_timestamps==nullptr) {
        return nullptr;
    }
    const jsize count=env->GetArrayLength(reference_timestamps);
    if(count!=env->GetArrayLength(remote_timestamps) || count<3) return nullptr;
    std::vector<jlong> reference(static_cast<std::size_t>(count));
    std::vector<jlong> remote(static_cast<std::size_t>(count));
    env->GetLongArrayRegion(reference_timestamps,0,count,reference.data());
    env->GetLongArrayRegion(remote_timestamps,0,count,remote.data());
    std::vector<std::uint64_t> ref64(static_cast<std::size_t>(count));
    std::vector<std::uint64_t> remote64(static_cast<std::size_t>(count));
    for(jsize i=0;i<count;++i){
        if(reference[i]<0 || remote[i]<0) return nullptr;
        ref64[i]=static_cast<std::uint64_t>(reference[i]);
        remote64[i]=static_cast<std::uint64_t>(remote[i]);
    }
    std::vector<double> refEvents; std::vector<double> remoteEvents;
    if(reference_events!=nullptr && remote_events!=nullptr){
        const jsize eventCount=env->GetArrayLength(reference_events);
        if(eventCount!=env->GetArrayLength(remote_events)) return nullptr;
        refEvents.resize(static_cast<std::size_t>(eventCount));
        remoteEvents.resize(static_cast<std::size_t>(eventCount));
        env->GetDoubleArrayRegion(reference_events,0,eventCount,refEvents.data());
        env->GetDoubleArrayRegion(remote_events,0,eventCount,remoteEvents.data());
    }
    if(!std::isfinite(event_sample_interval_nanos) ||
       event_sample_interval_nanos<=0.0) {
        return nullptr;
    }

    const auto result=nox::native::sync::SynchronizationEstimator{}.estimate(
        ref64,remote64,refEvents,remoteEvents,event_sample_interval_nanos);
    constexpr jsize kSynchronizationFieldCount = 17;
    jdoubleArray out =
        env->NewDoubleArray(
            kSynchronizationFieldCount);
    if (out == nullptr) return nullptr;

    const jdouble values[
        kSynchronizationFieldCount] = {
        result.offset_nanos,
        result.drift_ppm,
        result.jitter_nanos,
        result.latency_nanos,
        result.subframe_delta_nanos,
        result.subframe_confidence,
        result.subframe_sharpness,
        result.temporal_confidence,
        result.residual_p95_nanos,
        result.residualMaxNanos,
        result.outlier_ratio,
        result.clockSupportRatio,
        result.offsetUncertaintyNanos,
        result.driftUncertaintyPpm,
        static_cast<double>(
            result.clockIterations),
        static_cast<double>(
            result.sampleCount),
        result.valid
            ? 1.0
            : 0.0,
    };

    env->SetDoubleArrayRegion(
        out,
        0,
        kSynchronizationFieldCount,
        values);
    return out;
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_nivukx_nox_core_nativeengine_NativeMotionEngine_nativeAdmitHardwareBuffer(
    JNIEnv* env,
    jclass,
    jlong handle,
    jobject hardware_buffer) noexcept {
    if (handle == 0 || hardware_buffer == nullptr) {
        return nullptr;
    }

    auto* native_buffer =
        AHardwareBuffer_fromHardwareBuffer(
            env,
            hardware_buffer);
    if (native_buffer == nullptr) {
        return nullptr;
    }

    auto registry =
        get_hardware_buffer_registry(handle);
    if (registry == nullptr) {
        return nullptr;
    }

    nox::native::runtime::HardwareBufferLeaseInfo info{};
    const auto lease_id =
        registry->acquire(
            native_buffer,
            &info);
    if (lease_id == 0) {
        return nullptr;
    }

    constexpr jsize kFieldCount = 7;
    jlongArray out =
        env->NewLongArray(kFieldCount);
    if (out == nullptr) {
        registry->release(lease_id);
        return nullptr;
    }

    const jlong values[kFieldCount] = {
        static_cast<jlong>(info.lease_id),
        static_cast<jlong>(info.width),
        static_cast<jlong>(info.height),
        static_cast<jlong>(info.layers),
        static_cast<jlong>(info.format),
        static_cast<jlong>(info.usage),
        static_cast<jlong>(info.resident_bytes),
    };

    env->SetLongArrayRegion(
        out,
        0,
        kFieldCount,
        values);
    return out;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_nivukx_nox_core_nativeengine_NativeMotionEngine_nativeReleaseHardwareBuffer(
    JNIEnv*,
    jclass,
    jlong handle,
    jlong lease_id) noexcept {
    if (handle == 0 || lease_id <= 0) {
        return JNI_FALSE;
    }

    auto registry =
        get_hardware_buffer_registry(handle);
    if (registry == nullptr) {
        return JNI_FALSE;
    }

    return registry->release(
        static_cast<std::uint64_t>(lease_id))
        ? JNI_TRUE
        : JNI_FALSE;
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_nivukx_nox_core_nativeengine_NativeMotionEngine_nativeHardwareBufferDiagnostics(
    JNIEnv* env,
    jclass,
    jlong handle) noexcept {
    const auto registry =
        get_hardware_buffer_registry(handle);
    return hardware_buffer_diagnostics_to_array(
        env,
        registry.get());
}

extern "C" JNIEXPORT void JNICALL
Java_com_nivukx_nox_core_nativeengine_NativeMotionEngine_nativeStop(
    JNIEnv*, jclass, jlong handle) noexcept {
    native_runtime_stop(reinterpret_cast<nox::native::NativeRuntimeHandle>(handle));
}

extern "C" JNIEXPORT void JNICALL
Java_com_nivukx_nox_core_nativeengine_NativeMotionEngine_nativeDestroy(
    JNIEnv*, jclass, jlong handle) noexcept {
    std::shared_ptr<
        nox::native::runtime::HardwareBufferLeaseRegistry>
        registry;
    {
        std::scoped_lock lock(
            g_hardware_buffer_registry_mutex);
        const auto iterator =
            g_hardware_buffer_registries.find(handle);
        if (iterator !=
            g_hardware_buffer_registries.end()) {
            registry = std::move(iterator->second);
            g_hardware_buffer_registries.erase(iterator);
        }
    }

    if (registry != nullptr) {
        registry->clear();
    }

    native_runtime_destroy(
        reinterpret_cast<nox::native::NativeRuntimeHandle>(
            handle));
}

extern "C" JNIEXPORT jint JNICALL
Java_com_nivukx_nox_core_nativeengine_NativeMotionEngine_nativeAbiVersion(
    JNIEnv*, jclass) noexcept {
    return static_cast<jint>(native_engine_abi_version());
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_nivukx_nox_core_nativeengine_NativeMotionEngine_nativeDiagnostics(
    JNIEnv* env, jclass, jlong handle) noexcept {
    return diagnostics_to_array(env, native_runtime_diagnostics(
        reinterpret_cast<nox::native::NativeRuntimeHandle>(handle)));
}

#endif
