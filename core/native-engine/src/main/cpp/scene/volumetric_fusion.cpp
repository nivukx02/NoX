#include "scene/volumetric_fusion.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace nox::native::scene {

namespace {

constexpr double kLogTwo = 0.69314718055994530942;

} // namespace

double VolumetricFusion::clamp01(
    double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(
        value,
        0.0,
        1.0);
}

std::uint64_t VolumetricFusion::key(
    std::int32_t x,
    std::int32_t y,
    std::int32_t z) noexcept {
    return
        (
            static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(x)) *
            2654435761ull
        ) ^
        (
            static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(y)) *
            2246822519ull
        ) ^
        (
            static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(z)) *
            3266489917ull
        );
}

void VolumetricFusion::integrate(
    std::int32_t x,
    std::int32_t y,
    std::int32_t z,
    float tsdf,
    float weight) noexcept {
    if (
        !std::isfinite(tsdf) ||
        !std::isfinite(weight) ||
        weight <= 0.0f) {
        rejectedObservations_ += 1U;
        return;
    }

    const auto voxelKey =
        key(x, y, z);
    auto& voxel =
        grid_[voxelKey];

    const double incomingWeight =
        std::clamp(
            static_cast<double>(weight),
            0.0,
            65'535.0);
    const double existingWeight =
        std::max(
            0.0,
            static_cast<double>(voxel.weight));
    const double totalWeight =
        std::min(
            65'535.0,
            existingWeight +
                incomingWeight);

    if (totalWeight <= 0.0) return;

    const double blended =
        (
            static_cast<double>(voxel.tsdf) *
                existingWeight +
            static_cast<double>(tsdf) *
                incomingWeight
        ) /
        totalWeight;

    voxel.tsdf =
        static_cast<float>(
            std::clamp(
                blended,
                -1.0,
                1.0));
    voxel.weight =
        static_cast<float>(
            totalWeight);
    voxel.variance =
        std::min(
            voxel.variance,
            0.25f);
    voxel.observationCount =
        std::min<std::uint64_t>(
            std::uint64_t{1'000'000'000ULL},
            voxel.observationCount + std::uint64_t{1ULL});
    integratedObservations_ += 1U;
}

bool VolumetricFusion::integrateHighEnd(
    std::int32_t x,
    std::int32_t y,
    std::int32_t z,
    const VoxelObservation& observation,
    const VolumetricFusionOptions& options) noexcept {
    const double inputConfidence =
        clamp01(
            observation.confidence);
    const double variance =
        std::clamp(
            static_cast<double>(
                observation.variance),
            static_cast<double>(
                options.varianceFloor),
            static_cast<double>(
                options.maximumVariance));

    const double temporal =
        clamp01(
            observation.temporalConsistency);
    const double appearance =
        clamp01(
            observation.appearanceConsistency);
    const double contact =
        clamp01(
            observation.contactConsistency);

    if (
        !std::isfinite(
            observation.tsdf) ||
        inputConfidence <
            options.minimumConfidence ||
        observation.timestampNanos == 0ULL) {
        rejectedObservations_ += 1U;
        return false;
    }

    auto& voxel =
        grid_[key(x, y, z)];

    if (
        grid_.size() >
        options.maximumVoxels) {
        evictWeakest(options);
    }

    const double priorDynamic =
        clamp01(
            voxel.dynamicProbability);

    const double physicalConsistency =
        temporal * 0.45 +
        appearance * 0.25 +
        contact * 0.30;

    const double innovation =
        std::abs(
            static_cast<double>(
                observation.tsdf) -
            static_cast<double>(
                voxel.tsdf));

    const double robustResidualWeight =
        voxel.weight <= 0.0F
            ? 1.0
            : std::exp(
                -innovation /
                (
                    std::sqrt(
                        variance) +
                        0.002
                ));

    const double dynamicProbability =
        clamp01(
            priorDynamic * 0.72 +
            (
                1.0 -
                physicalConsistency
            ) *
                0.20 +
            clamp01(
                innovation *
                1.5) *
                0.08);

    const double dynamicSuppression =
        1.0 -
            dynamicProbability *
            clamp01(
                options.dynamicSuppression);

    const double precisionWeight =
        inputConfidence *
        physicalConsistency *
        robustResidualWeight *
        std::clamp(
            0.001 /
                std::max(
                    variance,
                    static_cast<double>(
                        options.varianceFloor)),
            0.10,
            4.0) *
        dynamicSuppression;

    if (
        !std::isfinite(precisionWeight) ||
        precisionWeight <= 0.0) {
        rejectedObservations_ += 1U;
        voxel.dynamicProbability =
            static_cast<float>(
                std::max(
                    priorDynamic,
                    dynamicProbability));
        return false;
    }

    const double incomingWeight =
        std::clamp(
            precisionWeight,
            0.0,
            static_cast<double>(
                options.maximumWeight));

    const double priorWeight =
        std::clamp(
            static_cast<double>(
                voxel.weight),
            0.0,
            static_cast<double>(
                options.maximumWeight));

    const double totalWeight =
        std::min(
            static_cast<double>(
                options.maximumWeight),
            priorWeight +
                incomingWeight);

    if (totalWeight <= 1e-9) {
        rejectedObservations_ += 1U;
        return false;
    }

    const double priorVariance =
        std::max(
            static_cast<double>(
                voxel.variance),
            static_cast<double>(
                options.varianceFloor));

    const double inverseVariance =
        1.0 /
        variance;
    const double priorPrecision =
        priorWeight > 0.0
            ? priorWeight /
                priorVariance
            : 0.0;
    const double measurementPrecision =
        incomingWeight *
        inverseVariance;
    const double posteriorPrecision =
        priorPrecision +
        measurementPrecision;

    const double posteriorVariance =
        std::clamp(
            posteriorPrecision > 0.0
                ? 1.0 /
                    posteriorPrecision
                : variance,
            static_cast<double>(
                options.varianceFloor),
            static_cast<double>(
                options.maximumVariance));

    const double blended =
        (
            static_cast<double>(
                voxel.tsdf) *
                priorWeight +
            static_cast<double>(
                observation.tsdf) *
                incomingWeight
        ) /
        totalWeight;

    voxel.tsdf =
        static_cast<float>(
            std::clamp(
                blended,
                -1.0,
                1.0));
    voxel.weight =
        static_cast<float>(
            totalWeight);
    voxel.variance =
        static_cast<float>(
            posteriorVariance);
    voxel.dynamicProbability =
        static_cast<float>(
            dynamicProbability);
    voxel.observationCount =
        std::min<std::uint64_t>(
            std::uint64_t{1'000'000'000ULL},
            voxel.observationCount + std::uint64_t{1ULL});
    voxel.lastTimestampNanos =
        observation.timestampNanos;

    integratedObservations_ += 1U;
    return true;
}

Voxel VolumetricFusion::sample(
    std::int32_t x,
    std::int32_t y,
    std::int32_t z) const noexcept {
    const auto iterator =
        grid_.find(
            key(x, y, z));
    return iterator ==
            grid_.end()
        ? Voxel{}
        : iterator->second;
}

void VolumetricFusion::decay(
    std::uint64_t timestampNanos,
    const VolumetricFusionOptions& options) noexcept {
    if (timestampNanos == 0ULL) return;

    if (
        lastDecayTimestampNanos_ != 0ULL &&
        timestampNanos <=
            lastDecayTimestampNanos_) {
        return;
    }

    const double halfLifeNanos =
        std::max(
            1.0,
            static_cast<double>(
                options.temporalDecayHalfLifeSeconds) *
                1e9);

    for (auto iterator = grid_.begin();
         iterator != grid_.end();) {
        auto& voxel =
            iterator->second;

        if (
            voxel.lastTimestampNanos != 0ULL &&
            timestampNanos >
                voxel.lastTimestampNanos) {
            const double ageNanos =
                static_cast<double>(
                    timestampNanos -
                    voxel.lastTimestampNanos);
            const double decayFactor =
                std::exp(
                    -kLogTwo *
                    ageNanos /
                    halfLifeNanos);

            voxel.weight =
                static_cast<float>(
                    std::clamp(
                        static_cast<double>(
                            voxel.weight) *
                            decayFactor,
                        0.0,
                        static_cast<double>(
                            options.maximumWeight)));

            voxel.dynamicProbability =
                static_cast<float>(
                    clamp01(
                        static_cast<double>(
                            voxel.dynamicProbability) *
                        (
                            0.60 +
                            0.40 *
                                decayFactor
                        )));

            voxel.variance =
                static_cast<float>(
                    std::clamp(
                        static_cast<double>(
                            voxel.variance) +
                            (
                                1.0 -
                                decayFactor
                            ) *
                            0.002,
                        static_cast<double>(
                            options.varianceFloor),
                        static_cast<double>(
                            options.maximumVariance)));
        }

        const bool expired =
            voxel.weight <
                options.minimumConfidence &&
            voxel.dynamicProbability >
                0.92f;

        if (expired) {
            iterator =
                grid_.erase(iterator);
            evictedVoxels_ += 1U;
        } else {
            ++iterator;
        }
    }

    while (
        grid_.size() >
        options.maximumVoxels) {
        evictWeakest(options);
    }

    lastDecayTimestampNanos_ =
        timestampNanos;
}

void VolumetricFusion::evictWeakest(
    const VolumetricFusionOptions& options) noexcept {
    if (grid_.empty()) return;

    auto weakest =
        grid_.end();
    double weakestScore =
        std::numeric_limits<double>::infinity();

    for (auto iterator = grid_.begin();
         iterator != grid_.end();
         ++iterator) {
        const auto& voxel =
            iterator->second;

        const double retention =
            clamp01(
                (
                    std::log1p(
                        std::max(
                            0.0,
                            static_cast<double>(
                                voxel.weight))) /
                    std::log1p(
                        std::max(
                            1.0,
                            static_cast<double>(
                                options.maximumWeight))) *
                    0.44
                ) +
                (
                    1.0 -
                    clamp01(
                        voxel.dynamicProbability)
                ) *
                    0.28 +
                (
                    1.0 -
                    clamp01(
                        static_cast<double>(
                            voxel.variance) /
                            std::max(
                                1e-8,
                                static_cast<double>(
                                    options.maximumVariance)))
                ) *
                    0.18 +
                clamp01(
                    static_cast<double>(
                        voxel.observationCount) /
                        32.0) *
                    0.10
            );

        if (retention < weakestScore) {
            weakestScore =
                retention;
            weakest =
                iterator;
        }
    }

    if (weakest != grid_.end()) {
        grid_.erase(weakest);
        evictedVoxels_ += 1U;
    }
}

VolumetricFusionStats VolumetricFusion::stats(
    const VolumetricFusionOptions& options) const noexcept {
    VolumetricFusionStats result;
    result.voxelCount =
        grid_.size();
    result.integratedObservations =
        integratedObservations_;
    result.rejectedObservations =
        rejectedObservations_;
    result.evictedVoxels =
        evictedVoxels_;

    if (grid_.empty()) {
        return result;
    }

    double varianceSum = 0.0;
    double dynamicSum = 0.0;
    double confidenceSum = 0.0;

    for (const auto& entry :
         grid_) {
        const auto& voxel =
            entry.second;

        const double weight =
            clamp01(
                static_cast<double>(
                    voxel.weight) /
                std::max(
                    1.0,
                    static_cast<double>(
                        options.maximumWeight)));

        varianceSum +=
            std::clamp(
                static_cast<double>(
                    voxel.variance),
                0.0,
                static_cast<double>(
                    options.maximumVariance));
        dynamicSum +=
            clamp01(
                voxel.dynamicProbability);
        confidenceSum +=
            weight *
            (
                1.0 -
                0.70 *
                    clamp01(
                        voxel.dynamicProbability)
            );
    }

    const double count =
        static_cast<double>(
            grid_.size());

    result.meanVariance =
        varianceSum /
        count;
    result.meanDynamicProbability =
        dynamicSum /
        count;
    result.confidence =
        clamp01(
            confidenceSum /
            count);
    result.memoryUtilization =
        clamp01(
            count /
            std::max(
                1.0,
                static_cast<double>(
                    options.maximumVoxels)));

    return result;
}

void VolumetricFusion::clear() noexcept {
    grid_.clear();
    lastDecayTimestampNanos_ = 0ULL;
    integratedObservations_ = 0ULL;
    rejectedObservations_ = 0ULL;
    evictedVoxels_ = 0ULL;
}

} // namespace nox::native::scene
