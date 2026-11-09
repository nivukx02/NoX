#pragma once

#include <cstdint>
#include <unordered_map>

namespace nox::native::tracking {

enum class CameraHealthState : std::uint8_t {
    Healthy = 0,
    Degraded = 1,
    Recoverable = 2,
    Quarantined = 3,
};

enum class CameraRecoveryAction : std::uint8_t {
    None = 0,
    ReduceLoad = 1,
    Resynchronize = 2,
    Recalibrate = 3,
    Reconnect = 4,
    Quarantine = 5,
};

enum CameraHealthReason : std::uint32_t {
    kNone = 0,
    kPacketLoss = 1u << 0,
    kSyncResidual = 1u << 1,
    kReprojectionResidual = 1u << 2,
    kThermalPressure = 1u << 3,
    kCoverageLoss = 1u << 4,
    kConfidenceLoss = 1u << 5,
    kQueuePressure = 1u << 6,
    kFrameCadence = 1u << 7,
    kCalibrationInstability = 1u << 8,
};

struct CameraHealthSample {
    std::uint32_t cameraId{0};
    double expectedFrameIntervalMs{0};
    double observedFrameIntervalMs{0};
    double packetLossRatio{0};
    double syncResidualMs{0};
    double reprojectionRmsPx{0};
    double thermalScore{1};
    double coverage{1};
    double confidence{1};
    double queuePressure{0};
    bool calibrationStable{true};
};

struct CameraHealthAssessment {
    std::uint32_t cameraId{0};
    CameraHealthState state{CameraHealthState::Healthy};
    CameraRecoveryAction action{CameraRecoveryAction::None};
    std::uint32_t reasons{kNone};
    double score{0};
    std::uint32_t badStreak{0};
    std::uint32_t goodStreak{0};
};

class CameraHealthMonitor final {
public:
    CameraHealthAssessment assess(const CameraHealthSample&) const noexcept;

    CameraHealthAssessment update(
        const CameraHealthSample& sample) noexcept;

    void reset(std::uint32_t cameraId) noexcept;
    void clear() noexcept;

private:
    struct History {
        CameraHealthState state{CameraHealthState::Healthy};
        std::uint32_t badStreak{0};
        std::uint32_t goodStreak{0};
    };

    std::unordered_map<std::uint32_t, History> history_;
};

} // namespace nox::native::tracking
