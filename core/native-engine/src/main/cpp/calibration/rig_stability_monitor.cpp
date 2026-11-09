#include "calibration/rig_stability_monitor.h"

#include <algorithm>
#include <cmath>
#include <map>

namespace nox::native::calibration {

namespace {

double clamp01(
    double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(
        value,
        0.0,
        1.0);
}

} // namespace

RigStabilityMonitor::RigStabilityMonitor(
    RigStabilityConfig config) noexcept
    : config_(config) {
    config_.maximumTranslationDriftMeters =
        std::max(
            config_.maximumTranslationDriftMeters,
            1e-6);
    config_.maximumRotationDriftRadians =
        std::max(
            config_.maximumRotationDriftRadians,
            1e-6);
    config_.degradedTranslationDriftMeters =
        std::clamp(
            config_.degradedTranslationDriftMeters,
            1e-6,
            config_.maximumTranslationDriftMeters);
    config_.degradedRotationDriftRadians =
        std::clamp(
            config_.degradedRotationDriftRadians,
            1e-6,
            config_.maximumRotationDriftRadians);
    config_.minimumSamplesPerCamera =
        std::clamp<std::size_t>(
            config_.minimumSamplesPerCamera,
            2U,
            1024U);
    config_.minimumConfidence =
        clamp01(
            config_.minimumConfidence);
}

double RigStabilityMonitor::translationDelta(
    const math::SE3d& reference,
    const math::SE3d& current) noexcept {
    const auto delta =
        current.translation -
        reference.translation;

    return std::hypot(
        std::hypot(
            delta.x,
            delta.y),
        delta.z);
}

double RigStabilityMonitor::rotationDelta(
    const math::SE3d& reference,
    const math::SE3d& current) noexcept {
    const auto relative =
        reference.rotation.conjugate();

    const auto delta =
        math::Quaterniond::multiply(
            relative,
            current.rotation)
            .normalized();

    const double vectorMagnitude =
        std::hypot(
            std::hypot(
                delta.x,
                delta.y),
            delta.z);

    return
        2.0 *
        std::atan2(
            vectorMagnitude,
            std::abs(
                delta.w));
}

double RigStabilityMonitor::percentile95(
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

    if (values.empty()) return 0.0;

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

RigStabilityResult
RigStabilityMonitor::evaluate(
    const std::vector<RigPoseSample>& samples) const {
    RigStabilityResult result;

    std::map<
        std::size_t,
        std::vector<RigPoseSample>> grouped;

    for (const auto& sample :
         samples) {
        if (
            sample.timestampNanos == 0ULL ||
            !std::isfinite(
                sample.confidence) ||
            sample.confidence <
                config_.minimumConfidence) {
            continue;
        }

        grouped[
            sample.cameraId
        ].push_back(
            sample);
    }

    result.cameraCount =
        grouped.size();

    std::vector<double> translationDrift;
    std::vector<double> rotationDrift;

    double confidenceSum = 0.0;
    std::size_t validCameras = 0U;

    for (auto& [cameraId, cameraSamples] :
         grouped) {
        (void)cameraId;

        if (
            cameraSamples.size() <
            config_.minimumSamplesPerCamera) {
            continue;
        }

        std::sort(
            cameraSamples.begin(),
            cameraSamples.end(),
            [](const auto& a,
               const auto& b) {
                return
                    a.timestampNanos <
                    b.timestampNanos;
            });

        const auto& reference =
            cameraSamples.front();

        double cameraMaxTranslation = 0.0;
        double cameraMaxRotation = 0.0;
        double cameraConfidenceSum = 0.0;

        for (
            const auto& sample :
            cameraSamples) {
            cameraMaxTranslation =
                std::max(
                    cameraMaxTranslation,
                    translationDelta(
                        reference.worldToCamera,
                        sample.worldToCamera));

            cameraMaxRotation =
                std::max(
                    cameraMaxRotation,
                    rotationDelta(
                        reference.worldToCamera,
                        sample.worldToCamera));

            cameraConfidenceSum +=
                clamp01(
                    sample.confidence);
        }

        translationDrift.push_back(
            cameraMaxTranslation);
        rotationDrift.push_back(
            cameraMaxRotation);

        confidenceSum +=
            cameraConfidenceSum /
            static_cast<double>(
                cameraSamples.size());

        ++validCameras;
    }

    result.validCameraCount =
        validCameras;

    if (validCameras == 0U) {
        return result;
    }

    result.maximumTranslationDriftMeters =
        *std::max_element(
            translationDrift.begin(),
            translationDrift.end());

    result.p95TranslationDriftMeters =
        percentile95(
            translationDrift);

    result.maximumRotationDriftRadians =
        *std::max_element(
            rotationDrift.begin(),
            rotationDrift.end());

    result.p95RotationDriftRadians =
        percentile95(
            rotationDrift);

    const double meanConfidence =
        confidenceSum /
        static_cast<double>(
            validCameras);

    const double translationScore =
        std::exp(
            -result.p95TranslationDriftMeters /
            config_.maximumTranslationDriftMeters);

    const double rotationScore =
        std::exp(
            -result.p95RotationDriftRadians /
            config_.maximumRotationDriftRadians);

    result.confidence =
        clamp01(
            translationScore *
            rotationScore *
            meanConfidence);

    result.unstable =
        result.maximumTranslationDriftMeters >
            config_.maximumTranslationDriftMeters ||
        result.maximumRotationDriftRadians >
            config_.maximumRotationDriftRadians;

    result.degraded =
        !result.unstable &&
        (
            result.p95TranslationDriftMeters >
                config_.degradedTranslationDriftMeters ||
            result.p95RotationDriftRadians >
                config_.degradedRotationDriftRadians
        );

    result.stable =
        !result.unstable &&
        !result.degraded &&
        result.confidence >= 0.70;

    return result;
}

} // namespace nox::native::calibration
