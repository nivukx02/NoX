#include "lighting/illumination_statistics.h"
#include <algorithm>
#include <cmath>
namespace nox::native::lighting {
IlluminationStats IlluminationStatistics::update(const FrameTelemetry& frame) noexcept {
    const double alpha=state_.samples==0?1.0:0.08;
    const double previous=state_.meanLuma;
    state_.meanLuma=previous*(1-alpha)+frame.meanLuma*alpha;
    state_.variance=state_.variance*(1-alpha)+frame.variance*alpha;
    const double delta=std::abs(frame.meanLuma-previous);
    state_.changeScore=std::clamp(0.92*state_.changeScore+0.08*delta,0.0,1.0);
    const double exposureDelta=frame.exposure-state_.exposureMean;
    state_.exposureMean+=alpha*exposureDelta;
    state_.exposureVariance=(1-alpha)*state_.exposureVariance+alpha*exposureDelta*exposureDelta;
    ++state_.samples;
    mean2_=frame.meanLuma;
    return state_;
}
}