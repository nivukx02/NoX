#include "tracking/occlusion_state.h"

#include <algorithm>
#include <cmath>

namespace nox::native::occlusion {
namespace {

double clamp01(double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(value, 0.0, 1.0);
}

double evidenceWeight(
    const Observation& value) noexcept {
    const double confidence =
        clamp01(value.confidence);
    const double certainty =
        1.0 -
        clamp01(value.uncertainty);
    const double crossView =
        clamp01(value.crossViewSupport);
    const double temporal =
        clamp01(value.temporalStability);

    return clamp01(
        confidence * 0.36 +
        certainty * 0.30 +
        crossView * 0.20 +
        temporal * 0.14);
}

} // namespace

Observation OcclusionStateTracker::visible(
    double confidence,
    double uncertainty,
    double crossViewSupport,
    double contactProbability,
    double temporalStability) const noexcept {
    const double c = clamp01(confidence);
    const double u = clamp01(uncertainty);
    const double cross = clamp01(crossViewSupport);
    const double temporal =
        clamp01(temporalStability);

    const State state =
        c >= 0.50 && u <= 0.55
            ? State::Visible
            : State::PartiallyOccluded;

    return {
        state,
        c,
        u,
        cross,
        clamp01(contactProbability),
        temporal,
        0.0,
    };
}

Observation OcclusionStateTracker::partiallyOccluded(
    double confidence,
    double uncertainty,
    double crossViewSupport,
    double contactProbability,
    double temporalStability) const noexcept {
    return {
        State::PartiallyOccluded,
        clamp01(confidence),
        clamp01(uncertainty),
        clamp01(crossViewSupport),
        clamp01(contactProbability),
        clamp01(temporalStability),
        0.0,
    };
}

Observation OcclusionStateTracker::predict(
    double confidence,
    double uncertainty,
    double gapSeconds,
    double maxGapSeconds,
    double crossViewSupport,
    double contactProbability,
    double temporalStability) const noexcept {
    if (
        !std::isfinite(gapSeconds) ||
        !std::isfinite(maxGapSeconds) ||
        maxGapSeconds <= 0.0 ||
        gapSeconds < 0.0 ||
        gapSeconds > maxGapSeconds) {
        return {
            State::Unsupported,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            std::max(0.0, gapSeconds),
        };
    }

    const double ratio =
        clamp01(
            gapSeconds /
                maxGapSeconds);

    const double baseConfidence =
        clamp01(confidence);

    const double baseUncertainty =
        clamp01(uncertainty);

    const double crossView =
        clamp01(crossViewSupport);

    const double contact =
        clamp01(contactProbability);

    const double temporal =
        clamp01(temporalStability);

    const double predictionDecay =
        std::exp(
            -2.4 * ratio);

    const double uncertaintyGrowth =
        std::clamp(
            baseUncertainty * (1.0 - 0.28 * ratio) +
                ratio * 0.82,
            0.0,
            1.0);

    const double contactRetention =
        0.55 + 0.45 * contact;

    const double confidenceDecay =
        predictionDecay *
        (
            0.62 +
            0.20 * crossView +
            0.18 * temporal
        ) *
        contactRetention;

    return {
        State::Predicted,
        clamp01(
            baseConfidence *
                confidenceDecay),
        clamp01(uncertaintyGrowth),
        crossView * (1.0 - ratio * 0.65),
        contact * (1.0 - ratio * 0.40),
        temporal * (1.0 - ratio * 0.75),
        gapSeconds,
    };
}

Observation OcclusionStateTracker::fuse(
    const std::vector<Observation>& evidence) const noexcept {
    if (evidence.empty()) {
        return {};
    }

    double totalWeight = 0.0;
    double confidence = 0.0;
    double uncertainty = 0.0;
    double crossView = 0.0;
    double contact = 0.0;
    double temporal = 0.0;
    bool hasVisible = false;
    bool hasPartial = false;
    bool hasPrediction = false;

    for (const auto& observation : evidence) {
        const double weight =
            evidenceWeight(observation);

        if (weight <= 0.0) continue;

        totalWeight += weight;
        confidence +=
            observation.confidence * weight;
        uncertainty +=
            observation.uncertainty * weight;
        crossView +=
            observation.crossViewSupport * weight;
        contact +=
            observation.contactProbability * weight;
        temporal +=
            observation.temporalStability * weight;

        hasVisible |=
            observation.state ==
                State::Visible;
        hasPartial |=
            observation.state ==
                State::PartiallyOccluded;
        hasPrediction |=
            observation.state ==
                State::Predicted;
    }

    if (totalWeight <= 1e-9) {
        return {};
    }

    const double fusedConfidence =
        clamp01(
            confidence /
                totalWeight);
    const double fusedUncertainty =
        clamp01(
            uncertainty /
                totalWeight);

    State state =
        State::Unsupported;

    if (hasVisible && fusedConfidence >= 0.48) {
        state = State::Visible;
    } else if (
        hasPartial &&
        fusedConfidence >= 0.20) {
        state = State::PartiallyOccluded;
    } else if (
        hasPrediction &&
        fusedConfidence >= 0.08) {
        state = State::Predicted;
    }

    return {
        state,
        fusedConfidence,
        fusedUncertainty,
        clamp01(crossView / totalWeight),
        clamp01(contact / totalWeight),
        clamp01(temporal / totalWeight),
        0.0,
    };
}

} // namespace nox::native::occlusion
