#pragma once
#include <cstdint>
namespace nox::native::lighting {
struct FrameTelemetry { double meanLuma{0}; double variance{0}; double exposure{0}; double whitePoint{0}; std::uint64_t timestampNanos{0}; };
struct IlluminationStats { double meanLuma{0}; double variance{0}; double exposureMean{0}; double exposureVariance{0}; double changeScore{0}; std::uint64_t samples{0}; };
class IlluminationStatistics final {
public:
    IlluminationStats update(const FrameTelemetry&) noexcept;
    IlluminationStats snapshot() const noexcept { return state_; }
private:
    IlluminationStats state_{};
    double mean2_{0};
};
}