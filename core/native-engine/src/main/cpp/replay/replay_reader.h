#pragma once
#include <cstdint>
#include <string>
namespace nox::native::replay {
struct ReplayManifest { std::uint32_t cameraCount{0}; std::uint64_t frameCount{0}; double clockDriftPpm{0}; double calibrationResidualPx{0}; double occlusionRatio{0}; std::uint64_t droppedFrames{0}; int thermalStatus{0}; std::uint32_t seed{0}; };
class ReplayReader final { public: ReplayManifest parse(const std::string&) const; };
}