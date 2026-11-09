#include "face/face_surface_alignment.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace nox::native::face {
namespace {

double clamp01(double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(value, 0.0, 1.0);
}

Vec3 sub(
    const Vec3& a,
    const Vec3& b) noexcept {
    return {
        a.x - b.x,
        a.y - b.y,
        a.z - b.z,
    };
}

Vec3 add(
    const Vec3& a,
    const Vec3& b) noexcept {
    return {
        a.x + b.x,
        a.y + b.y,
        a.z + b.z,
    };
}

Vec3 scale(
    const Vec3& value,
    double scalar) noexcept {
    return {
        value.x * scalar,
        value.y * scalar,
        value.z * scalar,
    };
}

double norm(
    const Vec3& value) noexcept {
    return std::sqrt(
        value.x * value.x +
        value.y * value.y +
        value.z * value.z);
}

Vec3 normalize(
    const Vec3& value) noexcept {
    const double length = norm(value);
    if (!std::isfinite(length) || length < 1e-12) {
        return {0.0, 0.0, 1.0};
    }
    return scale(
        value,
        1.0 / length);
}

Vec3 lerp(
    const Vec3& a,
    const Vec3& b,
    double t) noexcept {
    const double blend =
        std::clamp(
            t,
            0.0,
            1.0);
    return add(
        scale(a, 1.0 - blend),
        scale(b, blend));
}

double distance(
    const Vec3& a,
    const Vec3& b) noexcept {
    return norm(
        sub(a, b));
}

double robustScale(
    std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(
        values.begin(),
        values.end());

    const double median =
        values[values.size() / 2U];

    for (double& value : values) {
        value =
            std::abs(
                value - median);
    }

    std::sort(
        values.begin(),
        values.end());

    return std::max(
        1e-9,
        1.4826 *
            values[values.size() / 2U]);
}

}  // namespace

double FaceSurfaceAlignment::shapeConsistency(
    const FaceGeometryState& state) noexcept {
    if (state.landmarks.size() < 3U) {
        return 0.0;
    }

    std::vector<double> distances;
    distances.reserve(
        state.landmarks.size() *
        (state.landmarks.size() - 1U) /
        2U);

    double confidenceSum = 0.0;
    std::size_t confidencePairs = 0U;

    for (std::size_t i = 0U;
         i < state.landmarks.size();
         ++i) {
        for (std::size_t j = i + 1U;
             j < state.landmarks.size();
             ++j) {
            const auto& a =
                state.landmarks[i];
            const auto& b =
                state.landmarks[j];

            const double pairConfidence =
                std::sqrt(
                    clamp01(a.confidence) *
                    clamp01(b.confidence));

            if (pairConfidence <= 0.05) continue;

            const double d =
                distance(
                    a.position,
                    b.position);

            if (!std::isfinite(d) || d < 1e-7) {
                continue;
            }

            distances.push_back(d);
            confidenceSum +=
                pairConfidence;
            ++confidencePairs;
        }
    }

    if (distances.size() < 3U ||
        confidencePairs == 0U) {
        return 0.0;
    }

    std::sort(
        distances.begin(),
        distances.end());

    const double median =
        distances[distances.size() / 2U];

    std::vector<double> deviations;
    deviations.reserve(
        distances.size());

    for (const double value : distances) {
        deviations.push_back(
            std::abs(
                value - median));
    }

    const double mad =
        robustScale(
            deviations);

    const double meanAbsoluteDeviation =
        [&distances, median] {
            double total = 0.0;
            for (const double value : distances) {
                total +=
                    std::abs(
                        value - median);
            }
            return total /
                static_cast<double>(
                    distances.size());
        }();

    const double relativeDispersion =
        meanAbsoluteDeviation /
        std::max(
            median,
            1e-7);

    const double robustDispersion =
        mad /
        std::max(
            median,
            1e-7);

    const double geometricConsistency =
        std::exp(
            -std::clamp(
                relativeDispersion * 7.0 +
                    robustDispersion * 2.0,
                0.0,
                14.0));

    const double confidence =
        std::clamp(
            confidenceSum /
                static_cast<double>(
                    confidencePairs),
            0.0,
            1.0);

    return std::clamp(
        geometricConsistency *
            (
                0.55 +
                0.45 * confidence
            ),
        0.0,
        1.0);
}

FaceGeometryState FaceSurfaceAlignment::stabilize(
    const FaceGeometryState& previous,
    const FaceGeometryState& current,
    double alpha) noexcept {
    if (previous.landmarks.empty() ||
        current.landmarks.empty()) {
        return current;
    }

    const double requestedBlend =
        std::clamp(
            alpha,
            0.0,
            1.0);

    const double dtSeconds =
        current.timestamp_nanos >
                previous.timestamp_nanos
            ? std::clamp(
                static_cast<double>(
                    current.timestamp_nanos -
                        previous.timestamp_nanos) /
                    1e9,
                1e-4,
                0.25)
            : 1.0 / 60.0;

    std::unordered_map<
        std::uint32_t,
        const FaceLandmarkState*>
        previousById;
    previousById.reserve(
        previous.landmarks.size());

    for (const auto& landmark :
         previous.landmarks) {
        previousById[landmark.id] =
            &landmark;
    }

    FaceGeometryState output =
        current;

    double innovationSum = 0.0;
    double velocitySum = 0.0;
    double accelerationSum = 0.0;
    double jerkSum = 0.0;
    double expressionRetentionSum = 0.0;
    double dynamicStabilitySum = 0.0;
    double temporalConsistencySum = 0.0;
    std::size_t matched = 0U;
    std::vector<double> innovations;
    innovations.reserve(
        output.landmarks.size());

    for (auto& landmark :
         output.landmarks) {
        const auto iterator =
            previousById.find(
                landmark.id);

        if (iterator ==
            previousById.end()) {
            continue;
        }

        const FaceLandmarkState& old =
            *iterator->second;

        const Vec3 delta =
            sub(
                landmark.position,
                old.position);

        const double innovation =
            norm(delta);

        const double velocity =
            innovation /
            dtSeconds;

        const double uncertainty =
            std::max(
                1e-6,
                old.uncertainty +
                    landmark.uncertainty);

        const double innovationGate =
            std::exp(
                -0.5 *
                    std::pow(
                        innovation /
                            uncertainty,
                        2.0));

        const double highMotion =
            clamp01(
                innovation /
                    std::max(
                        0.0005,
                        uncertainty * 4.0));

        const double adaptiveBlend =
            std::clamp(
                requestedBlend *
                    (
                        0.35 +
                        0.65 *
                            std::sqrt(
                                clamp01(old.confidence) *
                                clamp01(landmark.confidence))
                    ) *
                    (
                        0.30 +
                        0.70 *
                            innovationGate
                    ),
                0.02,
                1.0);

        const Vec3 previousVelocity =
            {0.0, 0.0, 0.0};

        const double acceleration =
            velocity /
            dtSeconds;

        const double jerk =
            acceleration /
            dtSeconds;

        const double expressionRetention =
            std::clamp(
                (
                    0.65 +
                    0.35 *
                        innovationGate
                ) *
                (
                    1.0 -
                    0.18 *
                        highMotion
                ),
                0.0,
                1.0);

        const double temporalConfidence =
            std::clamp(
                0.48 *
                    clamp01(
                        old.confidence) +
                0.28 *
                    innovationGate +
                0.24 *
                    clamp01(
                        landmark.confidence),
                0.0,
                1.0);

        landmark.position =
            lerp(
                old.position,
                landmark.position,
                adaptiveBlend);

        landmark.uncertainty =
            std::clamp(
                old.uncertainty *
                    (
                        1.0 -
                        adaptiveBlend
                    ) +
                landmark.uncertainty *
                    adaptiveBlend +
                innovation *
                    (
                        1.0 -
                        innovationGate
                    ) *
                    0.20,
                0.0,
                1.0);

        landmark.confidence =
            clamp01(
                0.46 *
                    landmark.confidence +
                0.24 *
                    old.confidence +
                0.20 *
                    innovationGate +
                0.10 *
                    temporalConfidence);

        landmark.triangulation_conditioning =
            clamp01(
                0.60 *
                    landmark.triangulation_conditioning +
                0.25 *
                    old.triangulation_conditioning +
                0.15 *
                    innovationGate);

        landmark.supporting_pairs =
            std::max(
                landmark.supporting_pairs,
                old.supporting_pairs);

        const double velocityResidual =
            norm(
                sub(
                    scale(
                        delta,
                        1.0 /
                            dtSeconds),
                    previousVelocity));

        const double accelerationEvidence =
            std::exp(
                -std::clamp(
                    velocityResidual /
                        2.0,
                    0.0,
                    12.0));

        innovationSum +=
            innovation;
        velocitySum +=
            velocity;
        accelerationSum +=
            acceleration *
                accelerationEvidence;
        jerkSum +=
            jerk *
                accelerationEvidence;
        expressionRetentionSum +=
            expressionRetention;
        dynamicStabilitySum +=
            std::clamp(
                1.0 -
                    0.55 *
                        highMotion,
                0.0,
                1.0);
        temporalConsistencySum +=
            temporalConfidence;
        innovations.push_back(
            innovation);

        ++matched;
    }

    if (matched > 0U) {
        output.mean_innovation =
            innovationSum /
            static_cast<double>(
                matched);

        output.mean_velocity =
            velocitySum /
            static_cast<double>(
                matched);

        output.mean_acceleration =
            accelerationSum /
            static_cast<double>(
                matched);

        output.mean_jerk =
            jerkSum /
            static_cast<double>(
                matched);

        output.micro_expression_retention =
            expressionRetentionSum /
            static_cast<double>(
                matched);

        output.dynamic_stability =
            dynamicStabilitySum /
            static_cast<double>(
                matched);

        output.temporal_confidence =
            temporalConsistencySum /
            static_cast<double>(
                matched);

        if (!innovations.empty()) {
            std::sort(
                innovations.begin(),
                innovations.end());
            const std::size_t p95 =
                std::min(
                    innovations.size() - 1U,
                    static_cast<std::size_t>(
                        std::floor(
                            0.95 *
                            static_cast<double>(
                                innovations.size() - 1U))));
            const std::size_t p99 =
                std::min(
                    innovations.size() - 1U,
                    static_cast<std::size_t>(
                        std::floor(
                            0.99 *
                            static_cast<double>(
                                innovations.size() - 1U))));
            output.p95_innovation =
                innovations[p95];
            output.temporal_innovation_p95 =
                innovations[p95];
            output.temporal_innovation_max =
                innovations.back();
            output.temporal_velocity_meters_per_second =
                output.mean_velocity;
            output.temporal_consistency =
                std::clamp(
                    std::exp(
                        -output.p95_innovation /
                        0.002),
                    0.0,
                    1.0);
            if (p99 > p95) {
                output.temporal_consistency =
                    std::min(
                        output.temporal_consistency,
                        std::exp(
                            -innovations[p99] /
                            0.004));
            }
        }
    }

    const double matchedRatio =
        std::clamp(
            static_cast<double>(
                matched) /
                static_cast<double>(
                    std::max(
                        current.landmarks.size(),
                        std::size_t{1})),
            0.0,
            1.0);

    output.head_center =
        lerp(
            previous.head_center,
            current.head_center,
            std::clamp(
                requestedBlend *
                    (
                        0.35 +
                        0.65 *
                            matchedRatio
                    ),
                0.02,
                1.0));

    output.head_forward =
        normalize(
            lerp(
                previous.head_forward,
                current.head_forward,
                requestedBlend));

    output.confidence =
        std::clamp(
            0.50 *
                current.confidence +
            0.25 *
                previous.confidence +
            0.15 *
                matchedRatio +
            0.10 *
                output.temporal_confidence,
            0.0,
            1.0);

    output.mean_landmark_uncertainty =
        std::clamp(
            0.68 *
                current.mean_landmark_uncertainty +
            0.20 *
                previous.mean_landmark_uncertainty +
            0.12 *
                output.p95_innovation,
            0.0,
            1.0);

    output.conditioning_score =
        std::clamp(
            0.70 *
                current.conditioning_score +
            0.20 *
                previous.conditioning_score +
            0.10 *
                output.temporal_confidence,
            0.0,
            1.0);

    return output;
}

}  // namespace nox::native::face
