#include <cassert>
#include "tracking/loop_closure_drift.h"
#include "capture/rolling_shutter_compensator.h"
#include "lighting/exposure_vignette_model.h"
#include "memory/allocation_telemetry.h"

int main() {
    const auto drift=
        nox::native::tracking::LoopClosureDriftCorrector{}.solve({0.4,0.2,0.9,100});
    assert(drift.applied);
    assert(drift.confidence>.4);

    const auto rolling=
        nox::native::capture::RollingShutterCompensator{}.correct({1.0,1.0},0.6,1.0,0.01);
    assert(rolling.x!=1.0);

    const auto luma=
        nox::native::lighting::ExposureVignetteModel{}.correctedLuma(1.0,0.0,0.5,0.2);
    assert(luma>0.0);

    nox::native::memory::AllocationTelemetryRecorder recorder;
    recorder.recordAllocation(100);
    recorder.recordAllocation(50);
    recorder.recordFree(50);
    const auto telemetry=recorder.snapshot();
    assert(telemetry.allocations==2);
    assert(telemetry.frees==1);
    assert(telemetry.bytesAllocated==150);
    assert(telemetry.bytesFreed==50);
    return 0;
}
