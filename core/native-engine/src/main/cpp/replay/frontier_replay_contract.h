#pragma once
#include <cstdint>
#include <string>
#include "replay/replay_reader.h"
#include "replay/replay_runner.h"

namespace nox::native::replay {
struct FrontierReplayContractConfig {
    std::uint32_t maximumCameras{128U};
    std::uint32_t minimumCameras{2U};
    std::uint32_t maximumActors{64U};
    double maximumClockDriftPpm{150.0};
    double maximumCalibrationResidualPx{4.0};
    double maximumOcclusionRatio{1.0};
};
struct FrontierReplayQualification {
    bool manifestValid{false};
    bool outputValid{false};
    bool bounded{false};
    bool qualificationReady{false};
    std::uint64_t digest{0U};
    std::string blocker;
};
class FrontierReplayContract final {
public:
    explicit FrontierReplayContract(FrontierReplayContractConfig config = {});
    FrontierReplayQualification qualify(
        const ReplayManifest& manifest,
        const ReplayResult& result) const noexcept;
    static std::uint64_t stableDigest(
        const ReplayManifest& manifest,
        const ReplayResult& result) noexcept;
private:
    FrontierReplayContractConfig config_;
};
}  // namespace nox::native::replay
