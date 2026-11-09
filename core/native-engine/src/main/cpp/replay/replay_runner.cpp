#include "replay/replay_runner.h"
#include "diagnostics/evidence_trace.h"
#include "runtime/thermal_policy.h"
#include <algorithm>
#include <cmath>
namespace nox::native::replay {
namespace { std::uint64_t mix(std::uint64_t h,std::uint64_t v){h^=v+0x9e3779b97f4a7c15ull+(h<<6)+(h>>2);return h;} }
ReplayResult ReplayRunner::run(const std::string& fixture) const {
    const auto m=ReplayReader{}.parse(fixture);
    ReplayResult r;
    if(m.cameraCount==0 || m.frameCount==0)return r;
    const double sync=std::exp(-std::abs(m.clockDriftPpm)/250.0);
    const double calibration=std::exp(-m.calibrationResidualPx/4.0);
    const double recovery=std::clamp(1.0-0.85*m.occlusionRatio,0.0,1.0);
    const auto thermal=runtime::ThermalPolicy{}.evaluate(m.thermalStatus,0.0);
    const double dropPenalty=std::exp(-static_cast<double>(m.droppedFrames)/std::max<double>(1.0,m.frameCount));
    evidence::EvidenceTraceBuilder trace;
    trace.record(0,"camera-count",std::to_string(m.cameraCount));
    trace.record(1,"frame-count",std::to_string(m.frameCount));
    trace.record(2,"sync-confidence",std::to_string(sync));
    trace.record(3,"calibration-confidence",std::to_string(calibration));
    trace.record(4,"recovery-confidence",std::to_string(recovery));
    trace.record(5,"drop-penalty",std::to_string(dropPenalty));
    trace.record(6,"thermal-scale",std::to_string(thermal.workloadScale));
    r.outputDigest=mix(trace.finalize(),static_cast<std::uint64_t>(m.seed));
    r.synchronizationConfidence=sync;
    r.calibrationConfidence=calibration;
    r.recoveryConfidence=recovery;
    r.droppedObservations=m.droppedFrames;
    r.valid=sync>=0.50&&calibration>=0.50&&thermal.workloadScale>0.0;
    return r;
}
}