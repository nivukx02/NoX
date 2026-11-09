#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "face/face_track_state.h"

namespace nox::native::face {

struct FaceTemporalOptions {
    double maximumLandmarkVelocityMetersPerSecond{12.0};
    double maximumInnovationMeters{0.15};
    double innovationScaleMeters{0.015};
    double minimumTemporalConfidence{0.35};
    double maximumDtSeconds{0.25};
};

class FaceGeometryFusion final {
public:
    struct Config {
        std::size_t maxInputObservations{8192};
        std::size_t maxTrackedLandmarks{512};
        std::size_t maxObservationsPerLandmark{16};
        double minimumConfidence{0.05};
    };

    FaceGeometryFusion() noexcept;
    explicit FaceGeometryFusion(
        Config config) noexcept;

    FaceGeometryState solve(
        const std::vector<FaceLandmarkObservation>& observations,
        std::uint64_t timestampNanos) const;

    FaceGeometryState solveTemporal(
        const std::vector<FaceLandmarkObservation>& observations,
        std::uint64_t timestampNanos,
        const FaceGeometryState& previous,
        double dtSeconds,
        const FaceTemporalOptions& options = {}) const;

private:
    Config config_{};

    static bool finite(
        const FaceLandmarkObservation&) noexcept;

    static bool betterCandidate(
        const FaceLandmarkObservation& lhs,
        const FaceLandmarkObservation& rhs) noexcept;
};

} // namespace nox::native::face
