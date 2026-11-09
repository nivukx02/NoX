#pragma once
#include <cstdint>
#include "lighting/illumination_statistics.h"
namespace nox::native::lighting {
struct LightEvent { std::uint64_t timestampNanos{0}; double magnitude{0}; bool rising{false}; };
class LightEventDetector final {
public:
    LightEvent update(const FrameTelemetry&,const IlluminationStats&) noexcept;
private:
    double previousLuma_{0};
};
}