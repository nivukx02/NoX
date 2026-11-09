#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include "math/se3.h"

namespace nox::native::scene {

struct IcpCorrespondence {
    math::Vec3d source{};
    math::Vec3d target{};
    math::Vec3d normal{0.0,0.0,1.0};
    double weight{1.0};
};

struct IcpOptions {
    std::uint32_t iterations{20};
    double maxCorrespondenceDistance{0.20};
    double robustScale{0.03};
    double damping{1e-4};
    double convergence{1e-5};
};

struct IcpResult {
    math::SE3d pose{};
    double initialRms{0.0};
    double finalRms{0.0};
    double confidence{0.0};
    std::size_t inliers{0};
    std::uint32_t iterations{0};
    double conditionEstimate{0.0};
    bool converged{false};
};

class PointToPlaneIcp final {
public:
    IcpResult align(
        const std::vector<IcpCorrespondence>& correspondences,
        const math::SE3d& initial = {},
        const IcpOptions& options = {}) const;
};

} // namespace nox::native::scene
