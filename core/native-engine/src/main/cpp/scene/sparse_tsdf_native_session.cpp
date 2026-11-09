#include "scene/sparse_tsdf_native_session.h"

#include <algorithm>
#include <cmath>

namespace nox::native::scene {

SparseTsdfNativeSession::SparseTsdfNativeSession(SparseTsdfConfig config)
    : volume_(config) {}

void SparseTsdfNativeSession::reset() {
    volume_.reset();
    gaussianSurface_.reset();
    gaussianDeformationGraph_.reset();
}

void SparseTsdfNativeSession::releaseMemory() noexcept {
    volume_.releaseMemory();
    gaussianSurface_.reset();
    gaussianDeformationGraph_.reset();
}

void SparseTsdfNativeSession::integrateSurfacePoint(
    const math::Vec3d& origin,
    const math::Vec3d& direction,
    double surfaceDepth,
    double observationWeight) {
    volume_.integrateSurfacePoint(
        origin,
        direction,
        surfaceDepth,
        observationWeight);
}

TsdfMesh SparseTsdfNativeSession::extract(
    float isoLevel,
    std::size_t maxTriangles) const {
    return volume_.extractSurface(isoLevel, maxTriangles);
}

bool SparseTsdfNativeSession::integrateGaussianObservation(
    const GaussianSurfaceObservation& observation) noexcept {
    return gaussianSurface_.integrate(
        observation);
}

std::size_t SparseTsdfNativeSession::integrateGaussianObservations(
    const std::vector<GaussianSurfaceObservation>& observations) noexcept {
    if (observations.empty()) {
        return 0U;
    }

    std::uint64_t latestTimestamp = 0ULL;
    for (const auto& observation :
         observations) {
        latestTimestamp =
            std::max(
                latestTimestamp,
                observation.timestampNanos);
    }

    if (latestTimestamp != 0ULL) {
        static_cast<void>(
            gaussianSurface_.predict(
                latestTimestamp));
    }

    std::size_t accepted = 0U;
    for (const auto& observation :
         observations) {
        if (
            gaussianSurface_.integrate(
                observation)) {
            ++accepted;
        }
    }

    if (accepted > 0U) {
        static_cast<void>(
            gaussianDeformationGraph_.buildCanonicalGraph(
                gaussianSurface_.snapshot()));
    }

    return accepted;
}

std::vector<GaussianSurfacePrimitive>
SparseTsdfNativeSession::gaussianSurfaceSnapshot() const {
    return gaussianSurface_.snapshot();
}

GaussianSurfaceStats
SparseTsdfNativeSession::gaussianSurfaceStats() const noexcept {
    return gaussianSurface_.stats();
}

bool SparseTsdfNativeSession::rebuildGaussianDeformationGraph() {
    return gaussianDeformationGraph_.buildCanonicalGraph(
        gaussianSurface_.snapshot());
}

bool SparseTsdfNativeSession::updateGaussianDeformationNode(
    std::uint32_t nodeId,
    const math::SE3d& transform,
    const Vec3& velocity,
    double confidence,
    double temporalStability) noexcept {
    return gaussianDeformationGraph_.updateNodePose(
        nodeId,
        transform,
        velocity,
        confidence,
        temporalStability);
}

std::vector<GaussianSurfacePrimitive>
SparseTsdfNativeSession::deformGaussianSurface() const {
    return gaussianDeformationGraph_.deform(
        gaussianSurface_.snapshot());
}

GaussianDeformationStats
SparseTsdfNativeSession::gaussianDeformationStats() const noexcept {
    return gaussianDeformationGraph_.stats();
}

void SparseTsdfNativeSession::integrateDepthFrame(
    const float* depth,
    std::size_t width,
    std::size_t height,
    double fx,
    double fy,
    double cx,
    double cy,
    const math::SE3d& cameraToWorld,
    std::size_t stride,
    double observationWeight,
    DepthSampleConvention convention,
    const float* sampleConfidence) {
    if(depth==nullptr||width==0||height==0||fx<=0.0||fy<=0.0)return;
    const std::size_t step=std::max<std::size_t>(stride,1);
    for(std::size_t y=0;y<height;y+=step){
        for(std::size_t x=0;x<width;x+=step){
            const std::size_t sampleIndex = y * width + x;
            const float measured=depth[sampleIndex];
            if(!std::isfinite(measured)||measured<=0.05F||measured>50.0F)continue;
            const double pixelConfidence =
                sampleConfidence == nullptr
                    ? 1.0
                    : static_cast<double>(sampleConfidence[sampleIndex]);
            if (!std::isfinite(pixelConfidence) || pixelConfidence <= 0.0) continue;
            const math::Vec3d rayCamera{
                (static_cast<double>(x)-cx)/fx,
                (static_cast<double>(y)-cy)/fy,
                1.0,
            };
            const double rayScale=std::sqrt(
                rayCamera.x*rayCamera.x+
                rayCamera.y*rayCamera.y+
                rayCamera.z*rayCamera.z);
            if(rayScale<1e-12)continue;

            const auto direction=
                cameraToWorld.rotation.rotate(rayCamera);

            double euclideanDepth=static_cast<double>(measured);
            if(convention==DepthSampleConvention::OpticalAxisZ){
                euclideanDepth=
                    static_cast<double>(measured)*rayScale;
            }

            if(!std::isfinite(euclideanDepth)||euclideanDepth<=0.0)continue;

            integrateSurfacePoint(
                cameraToWorld.translation,
                direction,
                euclideanDepth,
                observationWeight * std::clamp(pixelConfidence, 0.0, 1.0));
        }
    }
}

void SparseTsdfNativeSession::integrateDepthFrameHighEnd(
    const float* depth,
    std::size_t width,
    std::size_t height,
    double fx,
    double fy,
    double cx,
    double cy,
    const math::SE3d& cameraToWorld,
    std::size_t stride,
    double observationWeight,
    double depthUncertaintyMeters,
    double temporalConsistency,
    double appearanceConsistency,
    double contactConsistency,
    double dynamicProbability,
    DepthSampleConvention convention,
    const float* sampleConfidence) {
    if (
        depth == nullptr ||
        width == 0 ||
        height == 0 ||
        fx <= 0.0 ||
        fy <= 0.0) {
        return;
    }

    const std::size_t step =
        std::max<std::size_t>(
            stride,
            1U);

    for (
        std::size_t y = 0;
        y < height;
        y += step) {
        for (
            std::size_t x = 0;
            x < width;
            x += step) {
            const std::size_t sampleIndex = y * width + x;
            const float measured = depth[sampleIndex];

            if (
                !std::isfinite(measured) ||
                measured <= 0.05F ||
                measured > 50.0F) {
                continue;
            }
            const double pixelConfidence =
                sampleConfidence == nullptr
                    ? 1.0
                    : static_cast<double>(sampleConfidence[sampleIndex]);
            if (!std::isfinite(pixelConfidence) || pixelConfidence <= 0.0) {
                continue;
            }

            const math::Vec3d rayCamera{
                (
                    static_cast<double>(x) -
                    cx
                ) / fx,
                (
                    static_cast<double>(y) -
                    cy
                ) / fy,
                1.0,
            };

            const double rayScale =
                std::sqrt(
                    rayCamera.x * rayCamera.x +
                    rayCamera.y * rayCamera.y +
                    rayCamera.z * rayCamera.z);

            if (rayScale < 1e-12) {
                continue;
            }

            const auto direction =
                cameraToWorld.rotation.rotate(
                    rayCamera);

            double euclideanDepth =
                static_cast<double>(
                    measured);

            if (
                convention ==
                DepthSampleConvention::OpticalAxisZ) {
                euclideanDepth =
                    static_cast<double>(
                        measured) *
                    rayScale;
            }

            if (
                !std::isfinite(euclideanDepth) ||
                euclideanDepth <= 0.0) {
                continue;
            }

            volume_.integrateSurfacePointHighEnd(
                cameraToWorld.translation,
                direction,
                euclideanDepth,
                observationWeight * std::clamp(pixelConfidence, 0.0, 1.0),
                depthUncertaintyMeters,
                temporalConsistency,
                appearanceConsistency,
                contactConsistency,
                dynamicProbability);
        }
    }
}

TsdfDetailDecision SparseTsdfNativeSession::integrateDepthFrameAdaptive(
    const float* depth,
    std::size_t width,
    std::size_t height,
    double fx,
    double fy,
    double cx,
    double cy,
    const math::SE3d& cameraToWorld,
    std::size_t stride,
    double observationWeight,
    DepthSampleConvention convention,
    const TsdfDetailEvidence& evidence,
    const float* sampleConfidence) {
    TsdfDetailEvidence adaptiveEvidence = evidence;
    adaptiveEvidence.voxelUtilization = volume_.voxelUtilization();
    const auto decision =
        detailController_.evaluate(adaptiveEvidence);
    const std::size_t effectiveStride =
        std::max<std::size_t>(1, stride) *
        std::max<std::uint32_t>(1U, decision.pixelStride);
    integrateDepthFrameHighEnd(
        depth,
        width,
        height,
        fx,
        fy,
        cx,
        cy,
        cameraToWorld,
        effectiveStride,
        observationWeight *
            decision.observationWeightScale,
        evidence.depthUncertaintyMeters,
        evidence.temporalConsistency,
        evidence.appearanceConsistency,
        evidence.contactConsistency,
        evidence.dynamicProbability,
        convention,
        sampleConfidence);
    return decision;
}

std::size_t SparseTsdfNativeSession::voxelCount() const noexcept {
    return volume_.activeVoxelCount();
}

std::size_t SparseTsdfNativeSession::residentMemoryBytes() const noexcept {
    return volume_.residentMemoryBytes();
}

double SparseTsdfNativeSession::voxelUtilization() const noexcept {
    return volume_.voxelUtilization();
}

std::size_t
SparseTsdfNativeSession::rejectedVoxelAllocations() const noexcept {
    return volume_.rejectedVoxelAllocations();
}

std::size_t SparseTsdfNativeSession::rejectedTriangleCount() const noexcept {
    return volume_.rejectedTriangleCount();
}

} // namespace nox::native::scene
