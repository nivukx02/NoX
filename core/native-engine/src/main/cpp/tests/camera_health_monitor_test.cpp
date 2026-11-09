#include "tracking/camera_health_monitor.h"

#include <cassert>
#include <cmath>

int main() {
    using namespace nox::native::tracking;

    CameraHealthMonitor monitor;

    const CameraHealthSample healthy{
        7, 33.333, 33.330, 0.01, 0.4, 0.7, 0.95,
        0.95, 0.95, 0.15, true
    };
    const auto h=monitor.assess(healthy);
    assert(h.state==CameraHealthState::Healthy);
    assert(h.action==CameraRecoveryAction::None);
    assert(h.score>0.80);

    const CameraHealthSample syncBad{
        7, 33.333, 33.333, 0.03, 8.0, 0.8, 0.90,
        0.90, 0.85, 0.10, true
    };
    const auto s=monitor.assess(syncBad);
    assert((s.reasons & kSyncResidual)!=0);
    assert(s.action==CameraRecoveryAction::Resynchronize);

    const CameraHealthSample calibrationBad{
        7, 33.333, 33.333, 0.02, 0.7, 3.2, 0.90,
        0.90, 0.55, 0.10, false
    };
    const auto c=monitor.assess(calibrationBad);
    assert((c.reasons & kCalibrationInstability)!=0);
    assert(c.action==CameraRecoveryAction::Recalibrate ||
           c.action==CameraRecoveryAction::Quarantine);

    const CameraHealthSample packetBad{
        7, 33.333, 33.333, 0.50, 0.5, 0.8, 0.90,
        0.30, 0.40, 0.20, true
    };
    const auto p=monitor.assess(packetBad);
    assert(p.action==CameraRecoveryAction::Reconnect ||
           p.action==CameraRecoveryAction::Quarantine);

    const CameraHealthSample degraded{
        9, 33.333, 34.8, 0.12, 3.2, 3.6, 0.55,
        0.42, 0.48, 0.72, true
    };
    auto d1=monitor.update(degraded);
    auto d2=monitor.update(degraded);
    assert(d1.state==CameraHealthState::Healthy ||
           d1.state==CameraHealthState::Degraded);
    assert(d2.state==CameraHealthState::Healthy ||
           d2.state==CameraHealthState::Degraded);

    const auto d3=monitor.update(degraded);
    assert(d3.state!=CameraHealthState::Healthy);

    monitor.reset(9);
    const auto resetHealthy=monitor.update(healthy);
    assert(resetHealthy.cameraId==7);
    assert(std::isfinite(resetHealthy.score));

    return 0;
}
