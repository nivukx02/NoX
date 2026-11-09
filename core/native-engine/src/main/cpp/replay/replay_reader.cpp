#include "replay/replay_reader.h"
#include <cstdlib>
#include <string>

namespace nox::native::replay {
namespace {
bool numberAfter(const std::string& text, const std::string& key, double& out) {
    const std::string quoted = std::string("\"") + key + "\"";
    const auto keyPosition = text.find(quoted);
    if (keyPosition == std::string::npos) return false;
    const auto colon = text.find(':', keyPosition);
    if (colon == std::string::npos) return false;
    char* end = nullptr;
    out = std::strtod(text.c_str() + colon + 1, &end);
    return end != text.c_str() + colon + 1;
}
}
ReplayManifest ReplayReader::parse(const std::string& json) const {
    ReplayManifest manifest;
    double value = 0.0;
    if (numberAfter(json, "cameraCount", value)) manifest.cameraCount = static_cast<std::uint32_t>(value);
    if (numberAfter(json, "frameCount", value)) manifest.frameCount = static_cast<std::uint64_t>(value);
    if (numberAfter(json, "clockDriftPpm", value)) manifest.clockDriftPpm = value;
    if (numberAfter(json, "calibrationResidualPx", value)) manifest.calibrationResidualPx = value;
    if (numberAfter(json, "occlusionRatio", value)) manifest.occlusionRatio = value;
    if (numberAfter(json, "droppedFrames", value)) manifest.droppedFrames = static_cast<std::uint64_t>(value);
    if (numberAfter(json, "thermalStatus", value)) manifest.thermalStatus = static_cast<int>(value);
    if (numberAfter(json, "seed", value)) manifest.seed = static_cast<std::uint32_t>(value);
    return manifest;
}
} // namespace nox::native::replay