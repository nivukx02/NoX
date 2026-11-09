#pragma once
#include <cstdint>
#include <vector>
namespace nox::native::tracking {
struct KeypointCandidate { std::uint32_t id{0}; double x{0},y{0},confidence{0}; std::uint32_t temporalBin{0}; };
class ConfidenceKeypointSampler final {
public: std::vector<KeypointCandidate> select(const std::vector<KeypointCandidate>&,
                                              std::size_t maxCount,std::size_t gridX=8,std::size_t gridY=6) const;
    std::vector<KeypointCandidate> selectSpatiotemporal(const std::vector<KeypointCandidate>&,
                                                        std::size_t maxCount,std::size_t gridX=8,std::size_t gridY=6,
                                                        std::size_t temporalBins=4) const;
};
}