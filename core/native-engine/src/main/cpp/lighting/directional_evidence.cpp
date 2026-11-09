#include "lighting/directional_evidence.h"
#include <algorithm>
namespace nox::native::lighting {
DirectionalCue DirectionalEvidence::estimate(const FrameTelemetry& frame,const IlluminationStats& stats) const noexcept {
    const double horizontal=std::clamp(frame.meanLuma-frame.whitePoint,-1.0,1.0);
    const double confidence=std::clamp((frame.variance>1e-6?0.65:0.15)*(1.0-std::min(stats.changeScore,1.0)),0.0,1.0);
    return{horizontal,1.0,std::clamp(-horizontal, -1.0,1.0),confidence,confidence>0.25};
}
}