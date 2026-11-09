#pragma once

#include <cstdint>
#include <vector>

#include "math/se3.h"

namespace nox::native::geometry3d {

struct PoseSample {
    std::uint64_t timestampNanos{0};
    math::SE3d pose{};
};

struct PoseTimelineResult {
    math::SE3d pose{};
    std::uint64_t temporalGapNanos{0};
    double interpolationAlpha{0.0};
    double linearSpeedMetersPerSecond{0.0};
    double angularSpeedRadiansPerSecond{0.0};
    double confidence{0.0};
    bool interpolated{false};
    bool extrapolated{false};
    bool clamped{false};
};

class PoseTimeline final {
public:
    void clear();

    bool addSample(
        const PoseSample&);

    [[nodiscard]] math::SE3d sample(
        std::uint64_t timestampNanos,
        std::uint64_t maxInterpolationGapNanos=50'000'000ULL) const;

    [[nodiscard]] PoseTimelineResult sampleHighEnd(
        std::uint64_t timestampNanos,
        std::uint64_t maxInterpolationGapNanos=50'000'000ULL) const;

    [[nodiscard]] math::SE3d sampleRow(
        std::uint64_t frameTimestampNanos,
        double normalizedRow,
        std::uint64_t readoutTimeNanos,
        std::uint64_t maxInterpolationGapNanos=50'000'000ULL) const;

    [[nodiscard]] PoseTimelineResult sampleRowHighEnd(
        std::uint64_t frameTimestampNanos,
        double normalizedRow,
        std::uint64_t readoutTimeNanos,
        std::uint64_t maxInterpolationGapNanos=50'000'000ULL) const;

    [[nodiscard]] std::size_t size() const noexcept {
        return samples_.size();
    }

private:
    std::vector<PoseSample> samples_;

    static double translationDistance(
        const math::SE3d&,
        const math::SE3d&) noexcept;

    static double angularDistance(
        const math::SE3d&,
        const math::SE3d&) noexcept;
};

} // namespace nox::native::geometry3d
