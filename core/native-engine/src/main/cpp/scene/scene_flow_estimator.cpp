#include "scene/scene_flow_estimator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace nox::native::scene {

namespace {

double clamp01(double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(value, 0.0, 1.0);
}

double norm3(
    double x,
    double y,
    double z) noexcept {
    return std::hypot(
        std::hypot(x, y),
        z);
}

double percentile95(
    std::vector<double> values) {
    values.erase(
        std::remove_if(
            values.begin(),
            values.end(),
            [](double value) {
                return !std::isfinite(value) || value < 0.0;
            }),
        values.end());

    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());

    const double position =
        0.95 *
        static_cast<double>(
            values.size() - 1U);
    const auto lower =
        static_cast<std::size_t>(
            std::floor(position));
    const auto upper =
        std::min(
            values.size() - 1U,
            lower + 1U);
    const double alpha =
        position -
        static_cast<double>(lower);

    return values[lower] +
        (
            values[upper] -
            values[lower]
        ) *
        alpha;
}

double directionalConsistency(
    double vx,
    double vy,
    double vz,
    const Flow* prior) noexcept {
    if (prior == nullptr) return 1.0;

    const double currentNorm =
        norm3(vx, vy, vz);
    const double priorNorm =
        norm3(
            prior->dx,
            prior->dy,
            prior->dz);

    if (
        currentNorm < 1e-8 ||
        priorNorm < 1e-8) {
        return 1.0;
    }

    const double cosine =
        (
            vx * prior->dx +
            vy * prior->dy +
            vz * prior->dz
        ) /
        (
            currentNorm *
            priorNorm
        );

    return clamp01(
        0.5 +
            0.5 *
            std::clamp(
                cosine,
                -1.0,
                1.0));
}

} // namespace

Flow SceneFlowEstimator::estimate(
    double px,
    double py,
    double pz,
    double cx,
    double cy,
    double cz,
    double dt) const noexcept {
    if (
        !std::isfinite(dt) ||
        dt <= 0.0) {
        return {};
    }

    const double dx =
        cx - px;
    const double dy =
        cy - py;
    const double dz =
        cz - pz;

    const double speed =
        norm3(
            dx,
            dy,
            dz) /
        dt;

    return {
        dx / dt,
        dy / dt,
        dz / dt,
        clamp01(
            std::exp(
                -speed / 5.0)),
        speed,
        0.0,
        0.02,
        1.0,
        0.0,
        false,
    };
}

Flow SceneFlowEstimator::estimateHighEnd(
    const FlowSample& previous,
    const FlowSample& current,
    const Flow* priorVelocity,
    const SceneFlowOptions& options) const noexcept {
    Flow result;

    if (
        previous.timestampNanos == 0ULL ||
        current.timestampNanos <= previous.timestampNanos) {
        return result;
    }

    const double dt =
        static_cast<double>(
            current.timestampNanos -
            previous.timestampNanos) /
        1e9;

    if (
        !std::isfinite(dt) ||
        dt <= 1e-6 ||
        dt > 2.0) {
        return result;
    }

    const double dx =
        current.x -
        previous.x;
    const double dy =
        current.y -
        previous.y;
    const double dz =
        current.z -
        previous.z;

    if (
        !std::isfinite(dx) ||
        !std::isfinite(dy) ||
        !std::isfinite(dz)) {
        return result;
    }

    const double vx =
        dx / dt;
    const double vy =
        dy / dt;
    const double vz =
        dz / dt;

    const double speed =
        norm3(vx, vy, vz);

    const double direction =
        directionalConsistency(
            vx,
            vy,
            vz,
            priorVelocity);

    double acceleration = 0.0;
    if (priorVelocity != nullptr) {
        const double previousSpeed =
            norm3(
                priorVelocity->dx,
                priorVelocity->dy,
                priorVelocity->dz);
        acceleration =
            norm3(
                vx - priorVelocity->dx,
                vy - priorVelocity->dy,
                vz - priorVelocity->dz) /
            dt;
        if (
            !std::isfinite(previousSpeed) ||
            !std::isfinite(acceleration)) {
            return result;
        }
    }

    const double predictedX =
        previous.x +
        (
            priorVelocity != nullptr
                ? priorVelocity->dx * dt
                : vx * dt
        );
    const double predictedY =
        previous.y +
        (
            priorVelocity != nullptr
                ? priorVelocity->dy * dt
                : vy * dt
        );
    const double predictedZ =
        previous.z +
        (
            priorVelocity != nullptr
                ? priorVelocity->dz * dt
                : vz * dt
        );

    const double innovation =
        norm3(
            current.x - predictedX,
            current.y - predictedY,
            current.z - predictedZ);

    const double speedGate =
        options.maximumSpeedMetersPerSecond <= 0.0
            ? 0.0
            : clamp01(
                1.0 -
                    std::max(
                        0.0,
                        speed -
                            options.maximumSpeedMetersPerSecond) /
                    options.maximumSpeedMetersPerSecond);

    const double accelerationGate =
        options.maximumAccelerationMetersPerSecond2 <= 0.0
            ? 0.0
            : clamp01(
                1.0 -
                    std::max(
                        0.0,
                        acceleration -
                            options.maximumAccelerationMetersPerSecond2) /
                    options.maximumAccelerationMetersPerSecond2);

    const double innovationGate =
        options.maximumInnovationMeters <= 0.0
            ? 0.0
            : clamp01(
                1.0 -
                    std::max(
                        0.0,
                        innovation -
                            options.maximumInnovationMeters) /
                    options.maximumInnovationMeters);

    const double measurementConfidence =
        clamp01(current.confidence);

    const double confidence =
        clamp01(
            measurementConfidence *
            (
                speedGate * 0.24 +
                accelerationGate * 0.22 +
                innovationGate * 0.24 +
                direction * 0.30
            ));

    const bool outlier =
        confidence <
            options.minimumConfidence ||
        (
            options.outlierRejectionStrength > 0.0 &&
            innovation >
                options.maximumInnovationMeters *
                    (
                        1.0 +
                            options.outlierRejectionStrength
                    )
        );

    const double uncertainty =
        std::clamp(
            options.uncertaintyFloorMeters +
                (
                    1.0 -
                    confidence
                ) *
                (
                    options.uncertaintyCeilingMeters -
                    options.uncertaintyFloorMeters
                ) +
                innovation * 0.25,
            options.uncertaintyFloorMeters,
            options.uncertaintyCeilingMeters);

    result.dx = vx;
    result.dy = vy;
    result.dz = vz;
    result.speedMetersPerSecond = speed;
    result.accelerationMetersPerSecond2 =
        acceleration;
    result.uncertaintyMeters =
        uncertainty;
    result.directionalConsistency =
        direction;
    result.innovationMeters =
        innovation;
    result.confidence =
        outlier
            ? confidence * 0.20
            : confidence;
    result.outlier =
        outlier;

    return result;
}

SceneFlowSequenceResult SceneFlowEstimator::estimateSequence(
    const std::vector<FlowSample>& samples,
    const SceneFlowOptions& options) const {
    SceneFlowSequenceResult result;

    if (samples.size() < 2U) {
        return result;
    }

    result.samples.reserve(
        samples.size() - 1U);

    std::vector<double> speeds;
    std::vector<double> accelerations;
    std::vector<double> innovations;

    speeds.reserve(
        samples.size() - 1U);
    accelerations.reserve(
        samples.size() - 1U);
    innovations.reserve(
        samples.size() - 1U);

    Flow prior{};
    bool havePrior = false;

    double confidenceSum = 0.0;
    double uncertaintySum = 0.0;
    double stabilitySum = 0.0;

    for (std::size_t index = 1U;
         index < samples.size();
         ++index) {
        const Flow* priorPointer =
            havePrior
                ? &prior
                : nullptr;

        const Flow currentFlow =
            estimateHighEnd(
                samples[index - 1U],
                samples[index],
                priorPointer,
                options);

        result.samples.push_back(
            currentFlow);

        if (currentFlow.outlier) {
            result.rejectedSamples += 1U;
        } else {
            speeds.push_back(
                currentFlow.speedMetersPerSecond);
            accelerations.push_back(
                currentFlow.accelerationMetersPerSecond2);
            innovations.push_back(
                currentFlow.innovationMeters);

            confidenceSum +=
                currentFlow.confidence;
            uncertaintySum +=
                currentFlow.uncertaintyMeters;
            stabilitySum +=
                currentFlow.directionalConsistency;
        }

        if (!currentFlow.outlier) {
            prior = currentFlow;
            havePrior = true;
        }
    }

    const double validCount =
        static_cast<double>(
            result.samples.size() -
                result.rejectedSamples);

    if (validCount <= 0.0) {
        return result;
    }

    result.meanSpeedMetersPerSecond =
        std::accumulate(
            speeds.begin(),
            speeds.end(),
            0.0) /
        validCount;
    result.p95SpeedMetersPerSecond =
        percentile95(
            speeds);
    result.p95AccelerationMetersPerSecond2 =
        percentile95(
            accelerations);
    result.p95InnovationMeters =
        percentile95(
            innovations);
    result.meanUncertaintyMeters =
        uncertaintySum /
        validCount;
    result.temporalStability =
        clamp01(
            stabilitySum /
            validCount);
    result.confidence =
        clamp01(
            confidenceSum /
            validCount);

    return result;
}

} // namespace nox::native::scene
