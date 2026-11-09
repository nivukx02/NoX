#include "replay/frontier_replay_contract.h"
#include <algorithm>
#include <cmath>
#include <cstring>
namespace nox::native::replay {
namespace { bool finite(double value) noexcept { return std::isfinite(value); } }
FrontierReplayContract::FrontierReplayContract(FrontierReplayContractConfig config) : config_(config) {}
FrontierReplayQualification FrontierReplayContract::qualify(
    const ReplayManifest& manifest,
    const ReplayResult& result) const noexcept {
    FrontierReplayQualification output{};
    output.manifestValid =
        manifest.cameraCount >= config_.minimumCameras &&
        manifest.cameraCount <= config_.maximumCameras &&
        manifest.frameCount > 0U &&
        manifest.droppedFrames <= manifest.frameCount &&
        finite(manifest.clockDriftPpm) &&
        finite(manifest.calibrationResidualPx) &&
        finite(manifest.occlusionRatio) &&
        manifest.occlusionRatio >= 0.0 &&
        manifest.occlusionRatio <= config_.maximumOcclusionRatio;
    output.outputValid =
        result.valid &&
        finite(result.synchronizationConfidence) &&
        finite(result.calibrationConfidence) &&
        finite(result.recoveryConfidence);
    output.bounded =
        manifest.cameraCount <= config_.maximumCameras &&
        manifest.clockDriftPpm <= config_.maximumClockDriftPpm &&
        manifest.calibrationResidualPx <= config_.maximumCalibrationResidualPx;
    output.digest = stableDigest(manifest, result);
    if (!output.manifestValid) { output.blocker = "manifest-invalid-or-outside-bound"; return output; }
    if (!output.outputValid) { output.blocker = "replay-output-invalid"; return output; }
    if (!output.bounded) { output.blocker = "replay-evidence-exceeds-frontier-bound"; return output; }
    output.qualificationReady = true;
    return output;
}
std::uint64_t FrontierReplayContract::stableDigest(
    const ReplayManifest& manifest,
    const ReplayResult& result) noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    const auto mix = [&hash](std::uint64_t value) noexcept {
        hash ^= value;
        hash *= 1099511628211ULL;
    };
    const auto mixDouble = [&mix](double value) noexcept {
        std::uint64_t bits{0U};
        std::memcpy(&bits, &value, sizeof(bits));
        mix(bits);
    };
    mix(manifest.cameraCount);
    mix(manifest.frameCount);
    mixDouble(manifest.clockDriftPpm);
    mixDouble(manifest.calibrationResidualPx);
    mixDouble(manifest.occlusionRatio);
    mix(manifest.droppedFrames);
    mix(static_cast<std::uint64_t>(manifest.thermalStatus));
    mix(manifest.seed);
    mix(result.outputDigest);
    mixDouble(result.synchronizationConfidence);
    mixDouble(result.calibrationConfidence);
    mixDouble(result.recoveryConfidence);
    mix(result.droppedObservations);
    mix(result.valid ? 1U : 0U);
    return hash;
}
}  // namespace nox::native::replay
