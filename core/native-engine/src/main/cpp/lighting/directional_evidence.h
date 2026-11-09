#pragma once
#include "lighting/illumination_statistics.h"
namespace nox::native::lighting {
struct DirectionalCue { double x{0}, y{1}, z{0}; double confidence{0}; bool valid{false}; };
class DirectionalEvidence final {
public:
    DirectionalCue estimate(const FrameTelemetry&, const IlluminationStats&) const noexcept;
};
}