#include "lighting/light_event_detector.h"
#include <cmath>
namespace nox::native::lighting {
LightEvent LightEventDetector::update(const FrameTelemetry& frame,const IlluminationStats&) noexcept {
    const double delta=frame.meanLuma-previousLuma_;
    previousLuma_=frame.meanLuma;
    return{frame.timestampNanos,std::abs(delta),delta>0.0};
}
}