#include "tracking/interaction_temporal_filter.h"
#include <algorithm>
#include <cmath>

namespace nox::native::tracking {

InteractionTemporalFilter::InteractionTemporalFilter(
    InteractionTemporalConfig config):config_(config){
    config_.timeConstantSeconds=std::max(config_.timeConstantSeconds,1e-4);
    config_.engageThreshold=std::clamp(config_.engageThreshold,0.0,1.0);
    config_.releaseThreshold=std::clamp(
        config_.releaseThreshold,0.0,config_.engageThreshold);
    config_.maxMissingFrames=std::max<std::uint32_t>(
        config_.maxMissingFrames,1);
    config_.velocitySmoothingAlpha=std::clamp(
        config_.velocitySmoothingAlpha,0.02,0.95);
    config_.contactAgeSmoothingAlpha=std::clamp(
        config_.contactAgeSmoothingAlpha,0.02,0.95);
    config_.impactApproachRate=
        std::max(config_.impactApproachRate,0.01);
    config_.impactScoreScale=
        std::max(config_.impactScoreScale,0.01);
    config_.maximumDeltaSeconds=
        std::max(config_.maximumDeltaSeconds,0.001);
}

void InteractionTemporalFilter::reset(){tracks_.clear();}

InteractionTemporalFilter::Key InteractionTemporalFilter::keyFor(
    const Interaction& v) noexcept {
    return {std::min(v.a,v.b),std::max(v.a,v.b)};
}

std::vector<Interaction> InteractionTemporalFilter::update(
    const std::vector<Interaction>& observations,double dt){
    const bool staleEvidence =
        !std::isfinite(dt) ||
        dt <= 0.0 ||
        dt > config_.maximumDeltaSeconds;
    const double safeDt =
        staleEvidence
            ? 0.0
            : std::max(dt,0.0);
    const double alpha=1.0-std::exp(
        -safeDt/config_.timeConstantSeconds);
    std::vector<Interaction> result;
    result.reserve(
        observations.size()+tracks_.size());

    std::map<Key,bool> observedKeys;
    for(const auto& v:observations){
        observedKeys[keyFor(v)]=true;
        auto& track=tracks_[keyFor(v)];

        const double distance =
            std::max(v.distance,0.0);
        const double approachRate =
            v.approachRate;
        const double confidence =
            std::clamp(v.contactProbability,0.0,1.0);

        if(!track.initialized){
            track.value=v;
            track.value.distance=distance;
            track.value.contactProbability=confidence;
            track.confidence=confidence;
            track.relativeVelocity=approachRate;
            track.peakApproachRate=
                std::max(0.0,-approachRate);
            track.contactAgeSeconds =
                v.interacting
                    ? safeDt
                    : 0.0;
            track.impactScore=
                std::clamp(
                    std::max(0.0,-approachRate) /
                        config_.impactApproachRate,
                    0.0,
                    1.0);
            track.distanceEma=distance;
            track.initialized=true;
        }else{
            track.value.distance +=
                (distance-track.value.distance)*alpha;
            track.value.approachRate +=
                (approachRate-track.value.approachRate)*alpha;

            track.confidence +=
                (confidence-track.confidence)*alpha;

            track.relativeVelocity +=
                (approachRate-track.relativeVelocity) *
                config_.velocitySmoothingAlpha;

            track.peakApproachRate =
                std::max(
                    track.peakApproachRate,
                    std::max(0.0,-approachRate));

            track.distanceEma +=
                (distance-track.distanceEma) *
                config_.velocitySmoothingAlpha;

            const double impact =
                std::clamp(
                    std::max(
                        0.0,
                        -approachRate) /
                        config_.impactApproachRate,
                    0.0,
                    1.0);

            track.impactScore +=
                (impact-track.impactScore) *
                config_.contactAgeSmoothingAlpha;

            if(
                track.confidence >=
                    config_.engageThreshold &&
                track.value.interacting){
                track.contactAgeSeconds += safeDt;
            }else if(track.contactAgeSeconds>0.0){
                track.contactAgeSeconds *=
                    std::exp(
                        -safeDt/
                            config_.timeConstantSeconds);
            }

            track.value.contactProbability=
                std::clamp(
                    track.confidence,
                    0.0,
                    1.0);
            track.value.a=v.a;
            track.value.b=v.b;
            track.value.distance=
                std::max(
                    track.distanceEma,
                    0.0);
            track.value.approachRate=
                track.relativeVelocity;
        }

        track.missingFrames=0;
        track.value.interacting=track.value.interacting
            ? track.confidence>=config_.releaseThreshold
            : track.confidence>=config_.engageThreshold;

        track.value.contactAgeSeconds=
            track.contactAgeSeconds;
        track.value.peakApproachRate=
            track.peakApproachRate;
        track.value.impactScore=
            track.impactScore;
        track.value.interactionConfidence=
            std::clamp(
                0.55*track.confidence+
                0.20*std::clamp(
                    track.contactAgeSeconds/
                        0.50,
                    0.0,
                    1.0)+
                0.15*track.impactScore+
                0.10*std::clamp(
                    1.0-
                        std::abs(
                            track.relativeVelocity),
                    0.0,
                    1.0),
                0.0,
                1.0);
        track.value.staleEvidenceRejected=
            staleEvidence;

        result.push_back(track.value);
    }

    for(auto it=tracks_.begin();it!=tracks_.end();){
        if(observedKeys.contains(it->first)){
            ++it;
            continue;
        }
        auto& track=it->second;
        ++track.missingFrames;
        track.confidence*=std::exp(
            -safeDt/config_.timeConstantSeconds);
        track.contactAgeSeconds*=
            std::exp(
                -safeDt/
                    config_.timeConstantSeconds);
        track.impactScore*=
            std::exp(
                -safeDt/
                    config_.timeConstantSeconds);
        track.value.contactProbability=track.confidence;
        track.value.contactAgeSeconds=
            track.contactAgeSeconds;
        track.value.peakApproachRate=
            track.peakApproachRate;
        track.value.impactScore=
            track.impactScore;
        track.value.interactionConfidence=
            std::clamp(
                0.65*track.confidence+
                0.20*std::clamp(
                    track.contactAgeSeconds/
                        0.50,
                    0.0,
                    1.0)+
                0.15*track.impactScore,
                0.0,
                1.0);
        track.value.staleEvidenceRejected=
            staleEvidence;
        track.value.interacting=
            track.value.interacting
                ? track.confidence>=config_.releaseThreshold
                : false;
        if(track.missingFrames>config_.maxMissingFrames){
            it=tracks_.erase(it);
            continue;
        }
        result.push_back(track.value);
        ++it;
    }

    return result;
}

} // namespace nox::native::tracking
