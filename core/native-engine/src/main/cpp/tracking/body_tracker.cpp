#include "tracking/body_tracker.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <algorithm>
#include <vector>
#include "tracking/motion_quality_fusion.h"

namespace nox::native::tracking {

namespace {

BodyTrackState buildState(
    std::uint64_t timestampNanos,
    std::uint64_t subjectId,
    const std::unordered_map<std::uint32_t, std::unique_ptr<JointTrack>>& tracks) {
    BodyTrackState out;
    out.timestamp_nanos = timestampNanos;
    out.subject_id = subjectId;
    if (tracks.empty()) return out;

    out.joints.reserve(tracks.size());
    double confidence = 0.0;
    double uncertainty = 0.0;
    double innovation = 0.0;
    double velocity = 0.0;
    double acceleration = 0.0;
    double jerk = 0.0;
    double direction = 0.0;
    double microMotion = 0.0;
    std::vector<double> innovations;
    innovations.reserve(tracks.size());
    std::size_t visible = 0;

    for (const auto& [_, track] : tracks) {
        const auto state = track->state();
        out.joints.push_back(state);
        confidence += state.confidence;
        uncertainty += state.uncertainty;
        innovation += state.innovationMagnitude;
        velocity += state.velocityMagnitude;
        acceleration += state.accelerationMagnitude;
        jerk += state.jerkMagnitude;
        direction +=
            std::clamp(
                std::exp(
                    -state.innovationMagnitude /
                    0.04),
                0.0,
                1.0);
        microMotion +=
            std::clamp(
                std::exp(
                    -std::abs(
                        state.velocityMagnitude -
                        0.08) /
                    0.30),
                0.0,
                1.0);
        innovations.push_back(
            std::max(
                0.0,
                state.innovationMagnitude));
        visible += state.visible ? 1u : 0u;
    }

    const double count = static_cast<double>(tracks.size());
    out.mean_confidence = confidence / count;
    out.mean_uncertainty = uncertainty / count;
    out.occlusion_ratio =
        1.0 - static_cast<double>(visible) / count;
    out.mean_innovation =
        innovation / count;
    out.mean_velocity =
        velocity / count;
    out.mean_acceleration =
        acceleration / count;
    out.mean_jerk =
        jerk / count;
    out.dynamic_stability =
        direction / count;
    out.micro_motion_retention =
        microMotion / count;
    if(!innovations.empty()) {
        std::sort(
            innovations.begin(),
            innovations.end());
        const double index =
            0.95 *
            static_cast<double>(
                innovations.size() - 1U);
        const auto lower =
            static_cast<std::size_t>(
                std::floor(index));
        const auto upper =
            std::min(
                innovations.size() - 1U,
                lower + 1U);
        const double alpha =
            index -
            static_cast<double>(lower);
        out.p95_innovation =
            innovations[lower] +
            (
                innovations[upper] -
                innovations[lower]
            ) *
            alpha;
    }
    out.healthy =
        out.mean_confidence >= 0.35 &&
        out.occlusion_ratio <= 0.40;
    return out;
}

} // namespace

BodyTrackState BodyTracker::update(
    const std::vector<JointObservation>& observations,
    std::uint64_t timestampNanos,
    double trustScale) {
    std::unordered_set<std::uint32_t> observedIds;
    observedIds.reserve(observations.size());

    for (const auto& observation : observations) {
        observedIds.insert(observation.id);
        auto it = tracks_.find(observation.id);
        if (it == tracks_.end()) {
            it = tracks_.emplace(
                observation.id,
                std::make_unique<JointTrack>(observation.id)
            ).first;
        }
        it->second->update(observation, timestampNanos, trustScale);
    }

    for (auto& [id, track] : tracks_) {
        if (!observedIds.contains(id)) {
            track->predict(timestampNanos, trustScale);
        }
    }

    last_timestamp_nanos_ = timestampNanos;
    return buildState(timestampNanos, subject_id_, tracks_);
}

BodyTrackState BodyTracker::predict(
    std::uint64_t timestampNanos,
    double trustScale) {
    for (auto& [_, track] : tracks_) {
        track->predict(timestampNanos, trustScale);
    }
    last_timestamp_nanos_ = timestampNanos;
    return buildState(timestampNanos, subject_id_, tracks_);
}

BodyTrackState BodyTracker::updateWithQuality(
    const std::vector<JointObservation>& observations,
    std::uint64_t timestampNanos,
    const QualityEvidence& evidence) {
    double motionBlurPenalty =
        std::clamp(
            evidence.motionBlurPenalty,
            0.0,
            1.0);

    if (evidence.autoEstimateMotionBlur) {
        const auto estimate =
            motionBlurEstimator_.estimate(
                evidence.exposureTimeNanos,
                evidence.cameraAngularVelocityRadiansPerSecond);

        if (evidence.motionBlurPenalty <= 0.0) {
            motionBlurPenalty = estimate.penalty;
        }
    }

    const auto quality = MotionQualityFusion{}.evaluate(
        {
            evidence.detectorConfidence,
            evidence.reprojectionRmsPx,
            evidence.positionUncertaintyMeters,
            evidence.syncConfidence,
            evidence.cameraCoverage,
            evidence.occlusionRatio,
            evidence.rollingShutterPenalty,
            motionBlurPenalty,
            evidence.depthConditioning,
            evidence.calibrationConfidence,
            evidence.cameraHealth,
            static_cast<double>(
                evidence.supportingViews),
            evidence.uncertaintyBudgetMeters,
        });
    return update(observations, timestampNanos, quality.trustScale);
}

BodyTrackState BodyTracker::updateCameraCompensated(
    const std::vector<JointObservation>& observations,
    std::uint64_t timestampNanos,
    const math::SE3d& cameraToWorld,
    double trustScale) {
    std::vector<JointObservation> worldObservations;
    worldObservations.reserve(observations.size());
    for(const auto& observation:observations){
        const math::Vec3d cameraPoint{
            observation.position.x,
            observation.position.y,
            observation.position.z,
        };
        const auto worldPoint=cameraToWorld.transform(cameraPoint);
        worldObservations.push_back({
            observation.id,
            {
                worldPoint.x,
                worldPoint.y,
                worldPoint.z,
            },
            observation.confidence,
        });
    }
    return update(worldObservations,timestampNanos,trustScale);
}

} // namespace nox::native::tracking
