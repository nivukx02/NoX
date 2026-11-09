#include "calibration/robust_bundle_adjustment.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <vector>

namespace nox::native::calibration {

namespace {

struct Accumulator {
    double x{0.0};
    double y{0.0};
    double z{0.0};
    double weight{0.0};
};

double huberWeight(
    double residual,
    double scale) noexcept {
    if (
        !std::isfinite(residual) ||
        residual < 0.0) {
        return 0.0;
    }

    const double threshold =
        std::max(
            scale * 2.5,
            1e-6);

    const double normalized =
        residual /
            std::max(
                threshold,
                1e-6);

    const double huber =
        normalized <= 1.0
            ? 1.0
            : 1.0 /
                normalized;

    const double cauchy =
        1.0 /
            (
                1.0 +
                normalized *
                normalized
            );

    const double tukeyRadius =
        normalized /
            3.0;

    const double tukey =
        tukeyRadius < 1.0
            ? (
                1.0 -
                    tukeyRadius *
                    tukeyRadius
                ) *
                (
                    1.0 -
                        tukeyRadius *
                        tukeyRadius
                )
            : 0.02;

    return std::clamp(
        0.34 * huber +
            0.36 * cauchy +
            0.30 * tukey,
        0.0,
        1.0);
}

} // namespace

double RobustBundleAdjustment::clamp01(
    double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(
        value,
        0.0,
        1.0);
}

double RobustBundleAdjustment::distance(
    const BundlePoint& a,
    const BundlePoint& b) noexcept {
    return std::hypot(
        std::hypot(
            a.x - b.x,
            a.y - b.y),
        a.z - b.z);
}

double RobustBundleAdjustment::percentile95(
    std::vector<double> values) noexcept {
    values.erase(
        std::remove_if(
            values.begin(),
            values.end(),
            [](double value) {
                return
                    !std::isfinite(value) ||
                    value < 0.0;
            }),
        values.end());

    if (
        values.empty()) {
        return 0.0;
    }

    std::sort(
        values.begin(),
        values.end());

    const double position =
        0.95 *
        static_cast<double>(
            values.size() - 1U);

    const auto lower =
        static_cast<std::size_t>(
            std::floor(
                position));

    const auto upper =
        std::min(
            values.size() - 1U,
            lower + 1U);

    const double alpha =
        position -
        static_cast<double>(
            lower);

    return values[lower] +
        (
            values[upper] -
            values[lower]
        ) *
        alpha;
}

BundleResult RobustBundleAdjustment::solve(
    std::vector<BundlePoint>& points,
    const std::vector<BundleObservation>& observations,
    std::uint32_t maxIterations) const {
    BundleResult result;

    if (
        points.empty() ||
        observations.empty()) {
        return result;
    }

    const std::uint32_t iterations =
        std::max(
            std::uint32_t{1},
            maxIterations);

    std::map<
        std::size_t,
        std::vector<const BundleObservation*>> grouped;

    for (const auto& observation :
         observations) {
        if (
            !std::isfinite(
                observation.observed.x) ||
            !std::isfinite(
                observation.observed.y) ||
            !std::isfinite(
                observation.observed.z) ||
            !std::isfinite(
                observation.weight) ||
            observation.weight <= 0.0) {
            ++result.rejectedObservations;
            continue;
        }

        const std::size_t index =
            observation.pointIndex !=
                static_cast<std::size_t>(-1)
                ? observation.pointIndex
                : static_cast<std::size_t>(
                    observation.camera);

        if (
            index >=
            points.size()) {
            ++result.rejectedObservations;
            continue;
        }

        grouped[index].push_back(
            &observation);
    }

    if (
        grouped.empty()) {
        return result;
    }

    const auto rmsForState =
        [&]() {
            double weightedSquared = 0.0;
            double weightedCount = 0.0;

            for (const auto& [
                     pointIndex,
                     group] :
                 grouped) {
                if (
                    pointIndex >=
                    points.size()) {
                    continue;
                }

                for (
                    const auto* observation :
                    group) {
                    const double weight =
                        std::max(
                            0.0,
                            observation->weight);

                    const double error =
                        distance(
                            points[pointIndex],
                            observation->observed);

                    weightedSquared +=
                        weight *
                        error *
                        error;

                    weightedCount +=
                        weight;
                }
            }

            return weightedCount <= 0.0
                ? std::numeric_limits<double>::infinity()
                : std::sqrt(
                    weightedSquared /
                    weightedCount);
        };

    result.rms =
        rmsForState();

    for (
        std::uint32_t iteration = 0;
        iteration < iterations;
        ++iteration) {
        double totalMovement = 0.0;
        std::size_t movementCount = 0U;

        for (const auto& [
                 pointIndex,
                 group] :
             grouped) {
            if (
                pointIndex >=
                points.size() ||
                points[pointIndex].fixed) {
                continue;
            }

            std::vector<double> residuals;
            residuals.reserve(
                group.size());

            for (
                const auto* observation :
                group) {
                residuals.push_back(
                    distance(
                        points[pointIndex],
                        observation->observed));
            }

            auto sortedResiduals =
                residuals;
            std::sort(
                sortedResiduals.begin(),
                sortedResiduals.end());
            const double residualMedian =
                sortedResiduals.empty()
                    ? 0.0
                    : sortedResiduals[
                        sortedResiduals.size() / 2U];

            std::vector<double> deviations;
            deviations.reserve(
                residuals.size());
            for(const double residual :
                residuals){
                deviations.push_back(
                    std::abs(
                        residual -
                        residualMedian));
            }

            std::sort(
                deviations.begin(),
                deviations.end());
            const double mad =
                deviations.empty()
                    ? 0.0
                    : deviations[
                        deviations.size() / 2U];

            const double residualScale =
                std::max(
                    1e-5,
                    1.4826 * mad);

            Accumulator accumulator;

            for (
                std::size_t observationIndex = 0;
                observationIndex < group.size();
                ++observationIndex) {
                const auto* observation =
                    group[observationIndex];

                const double residual =
                    residuals[
                        observationIndex];

                const double robust =
                    huberWeight(
                        residual,
                        residualScale);

                const double weight =
                    std::max(
                        0.0,
                        observation->weight) *
                    robust;

                if (
                    weight <= 0.0) {
                    continue;
                }

                accumulator.x +=
                    observation->observed.x *
                    weight;
                accumulator.y +=
                    observation->observed.y *
                    weight;
                accumulator.z +=
                    observation->observed.z *
                    weight;
                accumulator.weight +=
                    weight;
            }

            if (
                accumulator.weight <=
                0.0) {
                continue;
            }

            const BundlePoint target{
                accumulator.x /
                    accumulator.weight,
                accumulator.y /
                    accumulator.weight,
                accumulator.z /
                    accumulator.weight,
                false,
            };

            const double movement =
                distance(
                    points[pointIndex],
                    target);

            if (
                movement <
                1e-7) {
                continue;
            }

            constexpr double kDamping =
                0.72;

            points[pointIndex].x +=
                (
                    target.x -
                    points[pointIndex].x
                ) *
                kDamping;
            points[pointIndex].y +=
                (
                    target.y -
                    points[pointIndex].y
                ) *
                kDamping;
            points[pointIndex].z +=
                (
                    target.z -
                    points[pointIndex].z
                ) *
                kDamping;

            totalMovement +=
                movement;
            ++movementCount;
        }

        result.rms =
            rmsForState();
        result.iterations =
            iteration + 1U;

        if (
            movementCount == 0U ||
            totalMovement /
                static_cast<double>(
                    movementCount) <
                1e-7) {
            result.converged =
                true;
            break;
        }
    }

    std::vector<double> finalResiduals;
    finalResiduals.reserve(
        observations.size());

    result.acceptedObservations = 0U;
    result.rejectedObservations = 0U;
    result.robustCost = 0.0;

    double weightedError = 0.0;
    double weightedCount = 0.0;
    double confidenceNumerator = 0.0;
    double confidenceDenominator = 0.0;

    for (const auto& [
             pointIndex,
             group] :
         grouped) {
        if (
            pointIndex >=
            points.size()) {
            continue;
        }

        for (const auto* observation :
             group) {
            const double weight =
                std::max(
                    0.0,
                    observation->weight);

            const double residual =
                distance(
                    points[pointIndex],
                    observation->observed);

            if (
                residual >
                0.50) {
                ++result.rejectedObservations;
                continue;
            }

            ++result.acceptedObservations;
            finalResiduals.push_back(
                residual);

            weightedError +=
                weight *
                residual;
            weightedCount +=
                weight;

            result.robustCost +=
                weight *
                (
                    residual <= 0.01
                        ? 0.5 *
                            residual *
                            residual
                        : 0.01 *
                            (
                                residual -
                                0.005
                            )
                );

            confidenceNumerator +=
                weight *
                clamp01(
                    std::exp(
                        -residual /
                        0.01));
            confidenceDenominator +=
                weight;
        }
    }

    result.p95Error =
        percentile95(
            finalResiduals);

    result.maxError =
        finalResiduals.empty()
            ? 0.0
            : *std::max_element(
                finalResiduals.begin(),
                finalResiduals.end());

    result.spreadMeters =
        weightedCount <= 0.0
            ? 0.0
            : weightedError /
                weightedCount;

    result.rms =
        rmsForState();

    const double acceptanceRatio =
        observations.empty()
            ? 0.0
            : clamp01(
                static_cast<double>(
                    result.acceptedObservations) /
                static_cast<double>(
                    observations.size()));

    const double residualConfidence =
        clamp01(
            std::exp(
                -result.p95Error /
                0.02));

    result.confidence =
        clamp01(
            acceptanceRatio * 0.35 +
            residualConfidence * 0.45 +
            (
                confidenceDenominator <=
                    0.0
                    ? 0.0
                    : confidenceNumerator /
                        confidenceDenominator
            ) *
                0.20);

    result.valid =
        result.acceptedObservations >= 1U &&
        std::isfinite(result.rms) &&
        result.confidence >= 0.30;

    return result;
}

} // namespace nox::native::calibration
