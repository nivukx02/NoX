#if defined(__ANDROID__)
#include <jni.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "calibration/calibration_solver.h"

namespace {

constexpr std::size_t kMaxCameras = 128;
constexpr std::size_t kMaxObservations = 32'768;
constexpr std::size_t kIntrinsicStride = 11;
constexpr std::size_t kPoseStride = 7;
constexpr std::size_t kObservationStride = 6;
constexpr std::size_t kGlobalStride = 6;
constexpr std::size_t kCameraStride = 28;

bool finiteNonNegative(double value) noexcept {
    return std::isfinite(value) && value >= 0.0;
}

bool readString(
    JNIEnv* env,
    jstring value,
    std::string& output) {
    if (value == nullptr) return false;
    const char* chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) return false;
    output.assign(chars);
    env->ReleaseStringUTFChars(value, chars);
    return !output.empty() && output.size() <= 256;
}

jdoubleArray makeOutput(
    JNIEnv* env,
    const std::vector<double>& values) {
    if (values.size() >
        static_cast<std::size_t>(
            std::numeric_limits<jsize>::max())) {
        return nullptr;
    }
    auto* out = env->NewDoubleArray(
        static_cast<jsize>(values.size()));
    if (out == nullptr) return nullptr;
    if (!values.empty()) {
        env->SetDoubleArrayRegion(
            out,
            0,
            static_cast<jsize>(values.size()),
            values.data());
    }
    return out;
}

} // namespace

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_nivukx_nox_core_nativeengine_calibration_NativeCalibrationEngine_nativeSolve(
    JNIEnv* env,
    jclass,
    jobjectArray ids,
    jdoubleArray intrinsics,
    jdoubleArray poses,
    jbooleanArray fixed,
    jintArray observationCameraIndices,
    jdoubleArray observationData,
    jdouble huberDeltaPx,
    jint maxIterations,
    jint minimumObservationsPerCamera,
    jboolean refineIntrinsics,
    jboolean refineDistortion,
    jdouble focalPriorSigmaRatio,
    jdouble principalPointPriorSigmaRatio,
    jdouble distortionPriorSigma,
    jdouble refinementDamping,
    jint refinementIterations) noexcept {
    try {
        if (ids == nullptr ||
            intrinsics == nullptr ||
            poses == nullptr ||
            fixed == nullptr ||
            observationCameraIndices == nullptr ||
            observationData == nullptr) {
            return nullptr;
        }

        const jsize cameraCount =
            env->GetArrayLength(ids);
        if (cameraCount < 2 ||
            static_cast<std::size_t>(cameraCount) > kMaxCameras) {
            return nullptr;
        }

        if (env->GetArrayLength(intrinsics) !=
            static_cast<jsize>(
                static_cast<std::size_t>(cameraCount) *
                kIntrinsicStride)) {
            return nullptr;
        }
        if (env->GetArrayLength(poses) !=
            static_cast<jsize>(
                static_cast<std::size_t>(cameraCount) *
                kPoseStride)) {
            return nullptr;
        }
        if (env->GetArrayLength(fixed) != cameraCount) {
            return nullptr;
        }

        const jsize observationCount =
            env->GetArrayLength(observationCameraIndices);
        if (observationCount < 1 ||
            static_cast<std::size_t>(observationCount) >
                kMaxObservations) {
            return nullptr;
        }
        if (env->GetArrayLength(observationData) !=
            static_cast<jsize>(
                static_cast<std::size_t>(observationCount) *
                kObservationStride)) {
            return nullptr;
        }

        if (!std::isfinite(huberDeltaPx) ||
            huberDeltaPx <= 0.0 ||
            maxIterations < 1 ||
            maxIterations > 64 ||
            minimumObservationsPerCamera < 6 ||
            !std::isfinite(focalPriorSigmaRatio) ||
            !std::isfinite(principalPointPriorSigmaRatio) ||
            !std::isfinite(distortionPriorSigma) ||
            !std::isfinite(refinementDamping) ||
            !std::isfinite(static_cast<double>(refinementIterations)) ||
            refinementIterations < 1 ||
            refinementIterations > 64) {
            return nullptr;
        }

        std::vector<jdouble> intrinsicValues(
            static_cast<std::size_t>(cameraCount) *
            kIntrinsicStride);
        std::vector<jdouble> poseValues(
            static_cast<std::size_t>(cameraCount) *
            kPoseStride);
        std::vector<jboolean> fixedValues(
            static_cast<std::size_t>(cameraCount));
        std::vector<jint> observationCameras(
            static_cast<std::size_t>(observationCount));
        std::vector<jdouble> observationValues(
            static_cast<std::size_t>(observationCount) *
            kObservationStride);

        env->GetDoubleArrayRegion(
            intrinsics,
            0,
            env->GetArrayLength(intrinsics),
            intrinsicValues.data());
        env->GetDoubleArrayRegion(
            poses,
            0,
            env->GetArrayLength(poses),
            poseValues.data());
        env->GetBooleanArrayRegion(
            fixed,
            0,
            cameraCount,
            fixedValues.data());
        env->GetIntArrayRegion(
            observationCameraIndices,
            0,
            observationCount,
            observationCameras.data());
        env->GetDoubleArrayRegion(
            observationData,
            0,
            env->GetArrayLength(observationData),
            observationValues.data());
        if (env->ExceptionCheck()) return nullptr;

        nox::native::calibration::CalibrationProblem problem;
        problem.huberDeltaPx = huberDeltaPx;
        problem.maxIterations =
            static_cast<std::uint32_t>(maxIterations);
        problem.refineIntrinsics =
            refineIntrinsics == JNI_TRUE;
        problem.refineDistortion =
            refineDistortion == JNI_TRUE;
        problem.focalPriorSigmaRatio =
            focalPriorSigmaRatio;
        problem.principalPointPriorSigmaRatio =
            principalPointPriorSigmaRatio;
        problem.distortionPriorSigma =
            distortionPriorSigma;
        problem.refinementDamping =
            refinementDamping;
        problem.refinementIterations =
            static_cast<std::uint32_t>(refinementIterations);

        problem.cameras.reserve(
            static_cast<std::size_t>(cameraCount));
        std::size_t fixedCount = 0;

        for (jsize index = 0; index < cameraCount; ++index) {
            auto idObject =
                static_cast<jstring>(
                    env->GetObjectArrayElement(ids, index));
            std::string cameraId;
            const bool validId =
                readString(env, idObject, cameraId);
            env->DeleteLocalRef(idObject);
            if (!validId) return nullptr;

            nox::native::calibration::CameraSeed seed;
            seed.camera_id = cameraId;

            const std::size_t i =
                static_cast<std::size_t>(index) *
                kIntrinsicStride;
            auto& k = seed.intrinsics;
            k.fx = intrinsicValues[i];
            k.fy = intrinsicValues[i + 1];
            k.cx = intrinsicValues[i + 2];
            k.cy = intrinsicValues[i + 3];
            k.distortion.k1 = intrinsicValues[i + 4];
            k.distortion.k2 = intrinsicValues[i + 5];
            k.distortion.p1 = intrinsicValues[i + 6];
            k.distortion.p2 = intrinsicValues[i + 7];
            k.distortion.k3 = intrinsicValues[i + 8];
            const double width = intrinsicValues[i + 9];
            const double height = intrinsicValues[i + 10];
            if (!std::isfinite(k.fx) ||
                !std::isfinite(k.fy) ||
                k.fx <= 1.0 ||
                k.fy <= 1.0 ||
                !std::isfinite(k.cx) ||
                !std::isfinite(k.cy) ||
                !std::isfinite(width) ||
                !std::isfinite(height) ||
                width < 1.0 ||
                height < 1.0 ||
                width > 50'000.0 ||
                height > 50'000.0) {
                return nullptr;
            }
            k.width = static_cast<std::uint32_t>(width);
            k.height = static_cast<std::uint32_t>(height);

            const std::size_t p =
                static_cast<std::size_t>(index) *
                kPoseStride;
            seed.world_to_camera.rotation = {
                poseValues[p],
                poseValues[p + 1],
                poseValues[p + 2],
                poseValues[p + 3],
            };
            seed.world_to_camera.translation = {
                poseValues[p + 4],
                poseValues[p + 5],
                poseValues[p + 6],
            };
            if (!std::isfinite(seed.world_to_camera.rotation.w) ||
                !std::isfinite(seed.world_to_camera.rotation.x) ||
                !std::isfinite(seed.world_to_camera.rotation.y) ||
                !std::isfinite(seed.world_to_camera.rotation.z) ||
                !std::isfinite(seed.world_to_camera.translation.x) ||
                !std::isfinite(seed.world_to_camera.translation.y) ||
                !std::isfinite(seed.world_to_camera.translation.z)) {
                return nullptr;
            }

            if (fixedValues[index] == JNI_TRUE) {
                ++fixedCount;
            }
            problem.cameras.push_back(std::move(seed));
        }

        if (fixedCount != 1) return nullptr;

        for (jsize observationIndex = 0;
             observationIndex < observationCount;
             ++observationIndex) {
            const jint cameraIndex =
                observationCameras[observationIndex];
            if (cameraIndex < 0 ||
                cameraIndex >= cameraCount) {
                return nullptr;
            }

            const std::size_t o =
                static_cast<std::size_t>(observationIndex) *
                kObservationStride;
            const double wx = observationValues[o];
            const double wy = observationValues[o + 1];
            const double wz = observationValues[o + 2];
            const double px = observationValues[o + 3];
            const double py = observationValues[o + 4];
            const double weight = observationValues[o + 5];
            if (!std::isfinite(wx) ||
                !std::isfinite(wy) ||
                !std::isfinite(wz) ||
                !std::isfinite(px) ||
                !std::isfinite(py) ||
                !finiteNonNegative(weight)) {
                return nullptr;
            }

            problem.cameras[
                    static_cast<std::size_t>(cameraIndex)]
                .observations.push_back(
                    nox::native::calibration::CalibrationObservation{
                        {wx, wy, wz},
                        {px, py},
                        weight,
                    });
        }

        for (const auto& camera : problem.cameras) {
            if (camera.observations.size() <
                static_cast<std::size_t>(
                    std::max(minimumObservationsPerCamera, 0))) {
                return nullptr;
            }
        }

        const auto result =
            nox::native::calibration::CalibrationSolver{}.solve(problem);

        std::vector<double> output;
        output.reserve(
            kGlobalStride +
            problem.cameras.size() * kCameraStride);
        output.push_back(result.residual_rms_px);
        output.push_back(result.confidence);
        output.push_back(result.image_coverage);
        output.push_back(result.depth_coverage);
        output.push_back(result.observation_coverage);
        output.push_back(result.valid ? 1.0 : 0.0);

        for (const auto& seed : problem.cameras) {
            const auto iterator =
                std::find_if(
                    result.cameras.begin(),
                    result.cameras.end(),
                    [&](const auto& camera) {
                        return camera.camera_id ==
                            seed.camera_id;
                    });
            if (iterator == result.cameras.end()) {
                return nullptr;
            }

            const auto& camera = *iterator;
            output.push_back(camera.reprojection_rms_px);
            output.push_back(camera.conditioning);
            output.push_back(camera.confidence);
            output.push_back(camera.initial_reprojection_rms_px);
            output.push_back(camera.intrinsic_refinement_improvement_px);
            output.push_back(camera.imageCoverage);
            output.push_back(camera.depthCoverage);
            output.push_back(camera.observationCoverage);
            output.push_back(camera.intrinsics_refined ? 1.0 : 0.0);
            output.push_back(static_cast<double>(
                camera.intrinsic_jacobian_fallback_count));
            output.push_back(static_cast<double>(
                camera.translation_jacobian_fallback_count));
            output.push_back(camera.valid ? 1.0 : 0.0);
            output.push_back(camera.intrinsics.fx);
            output.push_back(camera.intrinsics.fy);
            output.push_back(camera.intrinsics.cx);
            output.push_back(camera.intrinsics.cy);
            output.push_back(camera.intrinsics.distortion.k1);
            output.push_back(camera.intrinsics.distortion.k2);
            output.push_back(camera.intrinsics.distortion.p1);
            output.push_back(camera.intrinsics.distortion.p2);
            output.push_back(camera.intrinsics.distortion.k3);
            output.push_back(camera.world_to_camera.rotation.w);
            output.push_back(camera.world_to_camera.rotation.x);
            output.push_back(camera.world_to_camera.rotation.y);
            output.push_back(camera.world_to_camera.rotation.z);
            output.push_back(camera.world_to_camera.translation.x);
            output.push_back(camera.world_to_camera.translation.y);
            output.push_back(camera.world_to_camera.translation.z);
        }

        return makeOutput(env, output);
    } catch (...) {
        return nullptr;
    }
}

#endif