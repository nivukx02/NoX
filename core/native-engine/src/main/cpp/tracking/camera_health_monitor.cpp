#include "tracking/camera_health_monitor.h"

#include <algorithm>
#include <cmath>

namespace nox::native::tracking {
namespace {

double clamp01(double value) noexcept {
    return std::clamp(value, 0.0, 1.0);
}

double inverseThreshold(double value, double threshold) noexcept {
    if(!std::isfinite(value) || threshold<=0.0) return 0.0;
    return 1.0-clamp01(value/threshold);
}

CameraHealthState stateForScore(double score) noexcept {
    if(score>=0.80) return CameraHealthState::Healthy;
    if(score>=0.60) return CameraHealthState::Degraded;
    if(score>=0.40) return CameraHealthState::Recoverable;
    return CameraHealthState::Quarantined;
}

int rank(CameraHealthState state) noexcept {
    return static_cast<int>(state);
}

} // namespace

CameraHealthAssessment CameraHealthMonitor::assess(
    const CameraHealthSample& sample) const noexcept {
    CameraHealthAssessment output;
    output.cameraId=sample.cameraId;

    const double cadenceScore =
        (sample.expectedFrameIntervalMs>0.0 &&
         std::isfinite(sample.observedFrameIntervalMs))
            ? inverseThreshold(
                std::abs(
                    sample.observedFrameIntervalMs/
                    sample.expectedFrameIntervalMs-1.0),
                0.25)
            : 0.0;

    const double packetScore=inverseThreshold(
        sample.packetLossRatio,0.20);
    const double syncScore=inverseThreshold(
        std::abs(sample.syncResidualMs),5.0);
    const double reprojectionScore=inverseThreshold(
        sample.reprojectionRmsPx,4.0);
    const double thermalScore=clamp01(sample.thermalScore);
    const double coverageScore=clamp01(sample.coverage);
    const double confidenceScore=clamp01(sample.confidence);
    const double queueScore=1.0-clamp01(sample.queuePressure);
    const double calibrationScore=sample.calibrationStable?1.0:0.0;

    output.score=std::clamp(
        0.15*packetScore+
        0.15*syncScore+
        0.20*reprojectionScore+
        0.10*thermalScore+
        0.15*coverageScore+
        0.15*confidenceScore+
        0.05*queueScore+
        0.05*cadenceScore+
        0.05*calibrationScore,
        0.0,1.0);

    if(sample.packetLossRatio>0.10)
        output.reasons|=kPacketLoss;
    if(std::abs(sample.syncResidualMs)>2.0)
        output.reasons|=kSyncResidual;
    if(sample.reprojectionRmsPx>2.0)
        output.reasons|=kReprojectionResidual;
    if(sample.thermalScore<0.70)
        output.reasons|=kThermalPressure;
    if(sample.coverage<0.70)
        output.reasons|=kCoverageLoss;
    if(sample.confidence<0.70)
        output.reasons|=kConfidenceLoss;
    if(sample.queuePressure>0.70)
        output.reasons|=kQueuePressure;
    if(cadenceScore<0.60)
        output.reasons|=kFrameCadence;
    if(!sample.calibrationStable)
        output.reasons|=kCalibrationInstability;

    output.state=stateForScore(output.score);

    if(output.state==CameraHealthState::Quarantined){
        output.action=CameraRecoveryAction::Quarantine;
    }else if(sample.packetLossRatio>0.35){
        output.action=CameraRecoveryAction::Reconnect;
    }else if(std::abs(sample.syncResidualMs)>5.0){
        output.action=CameraRecoveryAction::Resynchronize;
    }else if(!sample.calibrationStable &&
             sample.reprojectionRmsPx>2.5){
        output.action=CameraRecoveryAction::Recalibrate;
    }else if(sample.queuePressure>0.90 ||
             sample.thermalScore<0.45){
        output.action=CameraRecoveryAction::ReduceLoad;
    }else{
        output.action=CameraRecoveryAction::None;
    }

    return output;
}

CameraHealthAssessment CameraHealthMonitor::update(
    const CameraHealthSample& sample) noexcept {
    auto& state=history_[sample.cameraId];
    const auto observation=assess(sample);
    const int currentRank=rank(state.state);
    const int observedRank=rank(observation.state);

    if(observedRank>currentRank){
        ++state.badStreak;
        state.goodStreak=0;
    }else if(observedRank<currentRank){
        ++state.goodStreak;
        state.badStreak=0;
    }else{
        state.badStreak=0;
        state.goodStreak=0;
    }

    constexpr std::uint32_t kDemotionSamples=3;
    constexpr std::uint32_t kPromotionSamples=5;

    if(observedRank>currentRank &&
       state.badStreak>=kDemotionSamples){
        state.state=observation.state;
    }

    if(observedRank<currentRank &&
       state.goodStreak>=kPromotionSamples){
        state.state=observation.state;
    }

    CameraHealthAssessment result=observation;
    result.state=state.state;
    result.badStreak=state.badStreak;
    result.goodStreak=state.goodStreak;

    if(result.state==CameraHealthState::Quarantined){
        result.action=CameraRecoveryAction::Quarantine;
    }

    return result;
}

void CameraHealthMonitor::reset(std::uint32_t cameraId) noexcept {
    history_.erase(cameraId);
}

void CameraHealthMonitor::clear() noexcept {
    history_.clear();
}

} // namespace nox::native::tracking
