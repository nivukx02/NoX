#if defined(__ANDROID__)
#include <jni.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include <vector>
#include "scene/sparse_tsdf_native_session.h"

namespace {

constexpr std::size_t kMaximumIndexedMeshTriangles = 650'000U;
constexpr std::size_t kMaximumMeshPayloadBytes = 64U * 1024U * 1024U;

std::size_t clampMeshTriangleBudget(
    jint requested) noexcept {
    if(requested<=0) return 0;
    return std::min<std::size_t>(
        static_cast<std::size_t>(requested),
        kMaximumIndexedMeshTriangles);
}

using nox::native::scene::SparseTsdfNativeSession;
using nox::native::math::Vec3d;

jfloatArray encode_indexed_mesh(
    JNIEnv* env,
    const nox::native::scene::TsdfMesh& mesh) {
    constexpr float kPayloadVersion = 2.0F;
    const std::size_t vertexCount = mesh.vertices.size();
    const std::size_t triangleCount = mesh.triangles.size();
    const std::size_t indexCount = triangleCount * 3U;
    if (vertexCount > 4'000'000 ||
        triangleCount > 1'333'333 ||
        indexCount > 3'999'999) {
        return nullptr;
    }

    const std::size_t payloadSize =
        4U + vertexCount * 7U + indexCount;
    if (
        payloadSize >
        static_cast<std::size_t>(
            std::numeric_limits<jsize>::max()) ||
        payloadSize * sizeof(jfloat) >
            kMaximumMeshPayloadBytes) {
        return nullptr;
    }

    jfloatArray output =
        env->NewFloatArray(static_cast<jsize>(payloadSize));
    if (output == nullptr) return nullptr;

    std::vector<jfloat> payload(payloadSize);
    payload[0] = kPayloadVersion;
    payload[1] = static_cast<jfloat>(vertexCount);
    payload[2] = static_cast<jfloat>(triangleCount);
    payload[3] = static_cast<jfloat>(mesh.meanConfidence);

    std::size_t cursor = 4U;
    for (std::size_t i = 0; i < vertexCount; ++i) {
        payload[cursor++] = static_cast<jfloat>(mesh.vertices[i].x);
        payload[cursor++] = static_cast<jfloat>(mesh.vertices[i].y);
        payload[cursor++] = static_cast<jfloat>(mesh.vertices[i].z);

        const auto normal = i < mesh.normals.size()
            ? mesh.normals[i]
            : Vec3d{0.0, 0.0, 1.0};
        payload[cursor++] = static_cast<jfloat>(normal.x);
        payload[cursor++] = static_cast<jfloat>(normal.y);
        payload[cursor++] = static_cast<jfloat>(normal.z);

        const double uncertainty =
            i < mesh.vertexUncertaintyMeters.size()
                ? mesh.vertexUncertaintyMeters[i]
                : 0.0;
        payload[cursor++] = static_cast<jfloat>(uncertainty);
    }

    for (const auto& triangle : mesh.triangles) {
        payload[cursor++] = static_cast<jfloat>(triangle[0]);
        payload[cursor++] = static_cast<jfloat>(triangle[1]);
        payload[cursor++] = static_cast<jfloat>(triangle[2]);
    }

    env->SetFloatArrayRegion(
        output,
        0,
        static_cast<jsize>(payload.size()),
        payload.data());
    return output;
}

jfloatArray encode_mesh(JNIEnv* env, const nox::native::scene::TsdfMesh& mesh) {
    const std::size_t vertexCount = mesh.vertices.size();
    const std::size_t triangleCount = mesh.triangles.size();
    if (vertexCount > 4'000'000 || triangleCount > 1'333'333) return nullptr;

    const std::size_t payloadSize = 3U + vertexCount * 7U;
    if (
        payloadSize >
            static_cast<std::size_t>(
                std::numeric_limits<jsize>::max()) ||
        payloadSize * sizeof(jfloat) >
            kMaximumMeshPayloadBytes) {
        return nullptr;
    }

    jfloatArray output = env->NewFloatArray(static_cast<jsize>(payloadSize));
    if (output == nullptr) return nullptr;

    std::vector<jfloat> payload(payloadSize);
    payload[0] = static_cast<jfloat>(vertexCount);
    payload[1] = static_cast<jfloat>(triangleCount);
    payload[2] = static_cast<jfloat>(mesh.meanConfidence);

    std::size_t cursor = 3U;
    for (std::size_t i = 0; i < vertexCount; ++i) {
        payload[cursor++] = static_cast<jfloat>(mesh.vertices[i].x);
        payload[cursor++] = static_cast<jfloat>(mesh.vertices[i].y);
        payload[cursor++] = static_cast<jfloat>(mesh.vertices[i].z);
        const auto normal = i < mesh.normals.size()
            ? mesh.normals[i]
            : Vec3d{0.0,0.0,1.0};
        payload[cursor++] = static_cast<jfloat>(normal.x);
        payload[cursor++] = static_cast<jfloat>(normal.y);
        payload[cursor++] = static_cast<jfloat>(normal.z);
        const double uncertainty = i < mesh.vertexUncertaintyMeters.size()
            ? mesh.vertexUncertaintyMeters[i]
            : 0.0;
        payload[cursor++] = static_cast<jfloat>(uncertainty);
    }

    env->SetFloatArrayRegion(
        output,
        0,
        static_cast<jsize>(payload.size()),
        payload.data());
    return output;
}

} // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_nivukx_nox_core_nativeengine_NativeSparseTsdfEngine_nativeCreate(
    JNIEnv*, jclass,
    jdouble voxelSize,
    jdouble truncationDistance,
    jfloat maxWeight,
    jint blockResolution) noexcept {
    try {
        const nox::native::scene::SparseTsdfConfig config{
            voxelSize,
            truncationDistance,
            maxWeight,
            blockResolution,
        };
        return reinterpret_cast<jlong>(
            new SparseTsdfNativeSession(config));
    } catch (...) {
        return 0;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_nivukx_nox_core_nativeengine_NativeSparseTsdfEngine_nativeReset(
    JNIEnv*, jclass, jlong handle) noexcept {
    if (handle == 0) return;
    try {
        reinterpret_cast<SparseTsdfNativeSession*>(handle)->reset();
    } catch (...) {
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_nivukx_nox_core_nativeengine_NativeSparseTsdfEngine_nativeReleaseMemory(
    JNIEnv*, jclass, jlong handle) noexcept {
    if(handle==0)return;
    try {
        reinterpret_cast<SparseTsdfNativeSession*>(
            handle)->releaseMemory();
    } catch (...) {
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_nivukx_nox_core_nativeengine_NativeSparseTsdfEngine_nativeIntegrateSurfacePoint(
    JNIEnv*, jclass, jlong handle,
    jdouble ox, jdouble oy, jdouble oz,
    jdouble dx, jdouble dy, jdouble dz,
    jdouble depth, jdouble confidence) noexcept {
    if (handle == 0) return;
    try {
        reinterpret_cast<SparseTsdfNativeSession*>(handle)->integrateSurfacePoint(
            {ox,oy,oz},
            {dx,dy,dz},
            depth,
            confidence);
    } catch (...) {
    }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_nivukx_nox_core_nativeengine_NativeSparseTsdfEngine_nativeExtractIndexed(
    JNIEnv* env, jclass, jlong handle,
    jfloat isoLevel, jint maxTriangles) noexcept {
    const auto triangleBudget =
        clampMeshTriangleBudget(maxTriangles);
    if (handle == 0 || triangleBudget == 0) return nullptr;
    try {
        const auto mesh =
            reinterpret_cast<SparseTsdfNativeSession*>(handle)->extract(
                isoLevel,
                triangleBudget);
        return encode_indexed_mesh(env, mesh);
    } catch (...) {
        return nullptr;
    }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_nivukx_nox_core_nativeengine_NativeSparseTsdfEngine_nativeExtract(
    JNIEnv* env, jclass, jlong handle,
    jfloat isoLevel, jint maxTriangles) noexcept {
    const auto triangleBudget =
        clampMeshTriangleBudget(maxTriangles);
    if (handle == 0 || triangleBudget == 0) return nullptr;
    try {
        const auto mesh =
            reinterpret_cast<SparseTsdfNativeSession*>(handle)->extract(
                isoLevel,
                triangleBudget);
        return encode_mesh(env, mesh);
    } catch (...) {
        return nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_nivukx_nox_core_nativeengine_NativeSparseTsdfEngine_nativeIntegrateDepthFrame(
    JNIEnv* env, jclass, jlong handle,
    jfloatArray depth,
    jfloatArray depthConfidence,
    jint width, jint height,
    jdouble fx, jdouble fy, jdouble cx, jdouble cy,
    jdouble qx, jdouble qy, jdouble qz, jdouble qw,
    jdouble tx, jdouble ty, jdouble tz,
    jint stride, jdouble confidence,
    jint depthConvention) noexcept {
    if (handle == 0 || depth == nullptr || width <= 0 || height <= 0) return;
    if (depthConvention < 0 || depthConvention > 1) return;
    const std::size_t sampleWidth = static_cast<std::size_t>(width);
    const std::size_t sampleHeight = static_cast<std::size_t>(height);
    if (sampleWidth > std::numeric_limits<std::size_t>::max() / sampleHeight) return;
    const std::size_t sampleCount = sampleWidth * sampleHeight;
    if (sampleCount > static_cast<std::size_t>(std::numeric_limits<jsize>::max())) return;
    const jsize count = env->GetArrayLength(depth);
    if (count != static_cast<jsize>(sampleCount)) return;
    if (depthConfidence != nullptr && env->GetArrayLength(depthConfidence) != count) return;

    jfloat* values = nullptr;
    jfloat* confidenceValues = nullptr;
    try {
        values = env->GetFloatArrayElements(depth, nullptr);
        if (values == nullptr) return;
        if (depthConfidence != nullptr) {
            confidenceValues = env->GetFloatArrayElements(depthConfidence, nullptr);
            if (confidenceValues == nullptr) {
                env->ReleaseFloatArrayElements(depth, values, JNI_ABORT);
                values = nullptr;
                return;
            }
        }
        const nox::native::math::SE3d cameraToWorld{
            nox::native::math::Quaterniond{qw, qx, qy, qz}.normalized(),
            {tx, ty, tz},
        };
        reinterpret_cast<SparseTsdfNativeSession*>(handle)->integrateDepthFrame(
            values, static_cast<std::size_t>(width), static_cast<std::size_t>(height),
            fx, fy, cx, cy, cameraToWorld,
            static_cast<std::size_t>(std::max(stride, 1)), confidence,
            depthConvention == 1
                ? nox::native::scene::DepthSampleConvention::OpticalAxisZ
                : nox::native::scene::DepthSampleConvention::EuclideanRay,
            confidenceValues);
        env->ReleaseFloatArrayElements(depth, values, JNI_ABORT);
        values = nullptr;
        if (confidenceValues != nullptr) {
            env->ReleaseFloatArrayElements(depthConfidence, confidenceValues, JNI_ABORT);
            confidenceValues = nullptr;
        }
    } catch (...) {
        if (values != nullptr) env->ReleaseFloatArrayElements(depth, values, JNI_ABORT);
        if (confidenceValues != nullptr) env->ReleaseFloatArrayElements(depthConfidence, confidenceValues, JNI_ABORT);
    }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_nivukx_nox_core_nativeengine_NativeSparseTsdfEngine_nativeIntegrateDepthFrameAdaptive(
    JNIEnv* env, jclass, jlong handle,
    jfloatArray depth,
    jfloatArray depthConfidence,
    jint width, jint height,
    jdouble fx, jdouble fy, jdouble cx, jdouble cy,
    jdouble qx, jdouble qy, jdouble qz, jdouble qw,
    jdouble tx, jdouble ty, jdouble tz,
    jint stride, jdouble confidence,
    jint depthConvention,
    jdouble motionEnergy, jdouble uncertainty, jdouble cameraCoverage,
    jdouble reconstructionConfidence, jdouble thermalHeadroom, jdouble gpuHeadroom,
    jdouble depthUncertaintyMeters, jdouble temporalConsistency,
    jdouble appearanceConsistency, jdouble contactConsistency,
    jdouble dynamicProbability) noexcept {
    jfloat* values = nullptr;
    jfloat* confidenceValues = nullptr;
    try {
        if (handle == 0 || depth == nullptr || width <= 0 || height <= 0) return nullptr;
        if (depthConvention < 0 || depthConvention > 1) return nullptr;
        const std::size_t sampleWidth = static_cast<std::size_t>(width);
        const std::size_t sampleHeight = static_cast<std::size_t>(height);
        if (sampleWidth > std::numeric_limits<std::size_t>::max() / sampleHeight) return nullptr;
        const std::size_t sampleCount = sampleWidth * sampleHeight;
        if (sampleCount > static_cast<std::size_t>(std::numeric_limits<jsize>::max())) return nullptr;
        const jsize count = env->GetArrayLength(depth);
        if (count != static_cast<jsize>(sampleCount)) return nullptr;
        if (depthConfidence != nullptr && env->GetArrayLength(depthConfidence) != count) return nullptr;

        values = env->GetFloatArrayElements(depth, nullptr);
        if (values == nullptr) return nullptr;
        if (depthConfidence != nullptr) {
            confidenceValues = env->GetFloatArrayElements(depthConfidence, nullptr);
            if (confidenceValues == nullptr) {
                env->ReleaseFloatArrayElements(depth, values, JNI_ABORT);
                values = nullptr;
                return nullptr;
            }
        }
        const nox::native::math::SE3d cameraToWorld{
            nox::native::math::Quaterniond{qw, qx, qy, qz}.normalized(),
            {tx, ty, tz},
        };
        const auto decision =
            reinterpret_cast<SparseTsdfNativeSession*>(handle)->integrateDepthFrameAdaptive(
                values, static_cast<std::size_t>(width), static_cast<std::size_t>(height),
                fx, fy, cx, cy, cameraToWorld,
                static_cast<std::size_t>(std::max(stride, 1)), confidence,
                depthConvention == 1
                    ? nox::native::scene::DepthSampleConvention::OpticalAxisZ
                    : nox::native::scene::DepthSampleConvention::EuclideanRay,
                {
                    motionEnergy, uncertainty, cameraCoverage, reconstructionConfidence,
                    thermalHeadroom, gpuHeadroom, depthUncertaintyMeters, temporalConsistency,
                    appearanceConsistency, contactConsistency, dynamicProbability,
                },
                confidenceValues);
        env->ReleaseFloatArrayElements(depth, values, JNI_ABORT);
        values = nullptr;
        if (confidenceValues != nullptr) {
            env->ReleaseFloatArrayElements(depthConfidence, confidenceValues, JNI_ABORT);
            confidenceValues = nullptr;
        }

        jfloatArray out = env->NewFloatArray(8);
        if (out == nullptr) return nullptr;
        const jfloat payload[8] = {
            static_cast<jfloat>(decision.pixelStride), static_cast<jfloat>(decision.observationWeightScale),
            static_cast<jfloat>(decision.computePressure), static_cast<jfloat>(decision.detailLevel),
            decision.preserveFineDetail ? 1.0F : 0.0F, decision.shedDetail ? 1.0F : 0.0F,
            decision.memoryBound ? 1.0F : 0.0F, decision.reason[0] == 0 ? 0.0F : 1.0F,
        };
        env->SetFloatArrayRegion(out, 0, 8, payload);
        return out;
    } catch (...) {
        if (values != nullptr) env->ReleaseFloatArrayElements(depth, values, JNI_ABORT);
        if (confidenceValues != nullptr) env->ReleaseFloatArrayElements(depthConfidence, confidenceValues, JNI_ABORT);
        return nullptr;
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_nivukx_nox_core_nativeengine_NativeSparseTsdfEngine_nativeIntegrateGaussianBatch(
    JNIEnv* env,
    jclass,
    jlong handle,
    jdoubleArray packedObservations,
    jlongArray timestampsNanos) noexcept {
    if (
        handle == 0 ||
        packedObservations == nullptr ||
        timestampsNanos == nullptr) {
        return 0;
    }

    try {
        const jsize packedLength =
            env->GetArrayLength(
                packedObservations);
        const jsize timestampCount =
            env->GetArrayLength(
                timestampsNanos);

        constexpr jsize kStride = 17;
        if (
            timestampCount <= 0 ||
            packedLength !=
                timestampCount *
                    kStride) {
            return 0;
        }

        std::vector<jdouble> packed(
            static_cast<std::size_t>(
                packedLength));
        std::vector<jlong> timestamps(
            static_cast<std::size_t>(
                timestampCount));

        env->GetDoubleArrayRegion(
            packedObservations,
            0,
            packedLength,
            packed.data());

        env->GetLongArrayRegion(
            timestampsNanos,
            0,
            timestampCount,
            timestamps.data());

        std::vector<nox::native::scene::GaussianSurfaceObservation>
            observations;
        observations.reserve(
            static_cast<std::size_t>(
                timestampCount));

        for (
            jsize index = 0;
            index < timestampCount;
            ++index) {
            const jsize base =
                index *
                kStride;

            const double px =
                packed[
                    static_cast<std::size_t>(
                        base)];
            const double py =
                packed[
                    static_cast<std::size_t>(
                        base + 1)];
            const double pz =
                packed[
                    static_cast<std::size_t>(
                        base + 2)];

            const double nx =
                packed[
                    static_cast<std::size_t>(
                        base + 3)];
            const double ny =
                packed[
                    static_cast<std::size_t>(
                        base + 4)];
            const double nz =
                packed[
                    static_cast<std::size_t>(
                        base + 5)];

            const double vx =
                packed[
                    static_cast<std::size_t>(
                        base + 6)];
            const double vy =
                packed[
                    static_cast<std::size_t>(
                        base + 7)];
            const double vz =
                packed[
                    static_cast<std::size_t>(
                        base + 8)];

            const double varX =
                packed[
                    static_cast<std::size_t>(
                        base + 9)];
            const double varY =
                packed[
                    static_cast<std::size_t>(
                        base + 10)];
            const double varZ =
                packed[
                    static_cast<std::size_t>(
                        base + 11)];

            const double confidence =
                packed[
                    static_cast<std::size_t>(
                        base + 12)];

            const double dynamicProbability =
                packed[
                    static_cast<std::size_t>(
                        base + 13)];

            const double temporalConsistency =
                packed[
                    static_cast<std::size_t>(
                        base + 14)];

            const double visibilityConfidence =
                packed[
                    static_cast<std::size_t>(
                        base + 15)];

            const double photometricConsistency =
                packed[
                    static_cast<std::size_t>(
                        base + 16)];

            if (
                !std::isfinite(px) ||
                !std::isfinite(py) ||
                !std::isfinite(pz) ||
                !std::isfinite(nx) ||
                !std::isfinite(ny) ||
                !std::isfinite(nz) ||
                !std::isfinite(vx) ||
                !std::isfinite(vy) ||
                !std::isfinite(vz) ||
                !std::isfinite(varX) ||
                !std::isfinite(varY) ||
                !std::isfinite(varZ) ||
                !std::isfinite(confidence) ||
                !std::isfinite(dynamicProbability) ||
                !std::isfinite(temporalConsistency) ||
                !std::isfinite(visibilityConfidence) ||
                !std::isfinite(photometricConsistency) ||
                visibilityConfidence < 0.0 ||
                visibilityConfidence > 1.0 ||
                photometricConsistency < 0.0 ||
                photometricConsistency > 1.0 ||
                timestamps[
                    static_cast<std::size_t>(
                        index)] <= 0 ||
                varX <= 0.0 ||
                varY <= 0.0 ||
                varZ <= 0.0) {
                continue;
            }

            observations.push_back({
                {
                    px,
                    py,
                    pz,
                },
                {
                    nx,
                    ny,
                    nz,
                },
                {
                    vx,
                    vy,
                    vz,
                },
                {
                    varX,
                    varY,
                    varZ,
                },
                confidence,
                dynamicProbability,
                temporalConsistency,
                visibilityConfidence,
                photometricConsistency,
                static_cast<std::uint64_t>(
                    timestamps[
                        static_cast<std::size_t>(
                            index)]),
            });
        }

        if (observations.empty()) {
            return 0;
        }

        return static_cast<jlong>(
            reinterpret_cast<SparseTsdfNativeSession*>(
                handle)->integrateGaussianObservations(
                    observations));
    } catch (...) {
        return 0;
    }
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_nivukx_nox_core_nativeengine_NativeSparseTsdfEngine_nativeGaussianStats(
    JNIEnv* env,
    jclass,
    jlong handle) noexcept {
    if (handle == 0) return nullptr;

    try {
        const auto stats =
            reinterpret_cast<SparseTsdfNativeSession*>(
                handle)->gaussianSurfaceStats();

        constexpr jsize fieldCount = 9;
        const jdouble values[fieldCount] = {
            static_cast<jdouble>(
                stats.primitiveCount),
            static_cast<jdouble>(
                stats.fusedObservations),
            static_cast<jdouble>(
                stats.rejectedObservations),
            static_cast<jdouble>(
                stats.dynamicPrimitives),
            stats.meanConfidence,
            stats.meanUncertaintyMeters,
            stats.meanDynamicProbability,
            stats.temporalStability,
            stats.trustedPrimitiveRatio,
        };

        auto output =
            env->NewDoubleArray(
                fieldCount);
        if (output == nullptr) return nullptr;

        env->SetDoubleArrayRegion(
            output,
            0,
            fieldCount,
            values);
        return output;
    } catch (...) {
        return nullptr;
    }
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_nivukx_nox_core_nativeengine_NativeSparseTsdfEngine_nativeGaussianDeformationStats(
    JNIEnv* env,
    jclass,
    jlong handle) noexcept {
    if (handle == 0) return nullptr;

    try {
        const auto stats =
            reinterpret_cast<SparseTsdfNativeSession*>(
                handle)->gaussianDeformationStats();

        constexpr jsize fieldCount = 9;
        const jdouble values[fieldCount] = {
            static_cast<jdouble>(stats.nodeCount),
            static_cast<jdouble>(stats.bindingCount),
            static_cast<jdouble>(stats.rejectedBindings),
            stats.meanNodeConfidence,
            stats.meanBindingConfidence,
            stats.p95EdgeStrainRatio,
            stats.maximumEdgeStrainRatio,
            stats.temporalStability,
            stats.trustedBindingRatio,
        };

        auto output =
            env->NewDoubleArray(
                fieldCount);
        if (output == nullptr) return nullptr;

        env->SetDoubleArrayRegion(
            output,
            0,
            fieldCount,
            values);
        return output;
    } catch (...) {
        return nullptr;
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_nivukx_nox_core_nativeengine_NativeSparseTsdfEngine_nativeVoxelCount(
    JNIEnv*, jclass, jlong handle) noexcept {
    if (handle == 0) return 0;
    return static_cast<jlong>(
        reinterpret_cast<SparseTsdfNativeSession*>(handle)->voxelCount());
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_nivukx_nox_core_nativeengine_NativeSparseTsdfEngine_nativeResidentMemoryBytes(
    JNIEnv*, jclass, jlong handle) noexcept {
    if(handle==0)return 0;
    try {
        return static_cast<jlong>(
            reinterpret_cast<SparseTsdfNativeSession*>(
                handle)->residentMemoryBytes());
    } catch (...) {
        return 0;
    }
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_nivukx_nox_core_nativeengine_NativeSparseTsdfEngine_nativeVoxelUtilization(
    JNIEnv*, jclass, jlong handle) noexcept {
    if(handle==0)return 1.0;
    try {
        return reinterpret_cast<SparseTsdfNativeSession*>(
            handle)->voxelUtilization();
    } catch (...) {
        return 1.0;
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_nivukx_nox_core_nativeengine_NativeSparseTsdfEngine_nativeRejectedVoxelAllocations(
    JNIEnv*, jclass, jlong handle) noexcept {
    if(handle==0)return 0;
    try {
        return static_cast<jlong>(
            reinterpret_cast<SparseTsdfNativeSession*>(
                handle)->rejectedVoxelAllocations());
    } catch (...) {
        return 0;
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_nivukx_nox_core_nativeengine_NativeSparseTsdfEngine_nativeRejectedTriangleCount(
    JNIEnv*, jclass, jlong handle) noexcept {
    if(handle==0)return 0;
    try {
        return static_cast<jlong>(
            reinterpret_cast<SparseTsdfNativeSession*>(
                handle)->rejectedTriangleCount());
    } catch (...) {
        return 0;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_nivukx_nox_core_nativeengine_NativeSparseTsdfEngine_nativeDestroy(
    JNIEnv*, jclass, jlong handle) noexcept {
    delete reinterpret_cast<SparseTsdfNativeSession*>(handle);
}

#endif
