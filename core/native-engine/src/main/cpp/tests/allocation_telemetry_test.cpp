#include <cassert>
#include <cstdint>
#include <thread>
#include <vector>

#include "memory/allocation_telemetry.h"

int main() {
    using nox::native::memory::AllocationTelemetryRecorder;

    AllocationTelemetryRecorder telemetry;
    constexpr std::uint32_t kThreads = 8;
    constexpr std::uint32_t kIterations = 1000;

    std::vector<std::thread> workers;
    workers.reserve(kThreads);

    for(std::uint32_t worker=0; worker<kThreads; ++worker) {
        workers.emplace_back([&telemetry]() {
            for(std::uint32_t iteration=0;
                iteration<kIterations;
                ++iteration) {
                telemetry.recordAllocation(64);
                telemetry.recordFree(32);
            }
        });
    }

    for(auto& worker:workers) {
        worker.join();
    }

    const auto snapshot=telemetry.snapshot();
    const auto expected=
        static_cast<std::uint64_t>(
            kThreads)*kIterations;

    assert(snapshot.allocations==expected);
    assert(snapshot.frees==expected);
    assert(snapshot.bytesAllocated==expected*64u);
    assert(snapshot.bytesFreed==expected*32u);
    assert(snapshot.netAllocations()==0);
    assert(snapshot.netBytes()==
           static_cast<std::int64_t>(expected*32u));



    telemetry.reset();
    telemetry.recordAllocation(128u);
    telemetry.recordAllocation(256u);
    auto live=telemetry.snapshot();
    assert(live.outstandingAllocations()==2u);
    assert(live.peakAllocations()==2u);

    telemetry.recordFree(128u);
    live=telemetry.snapshot();
    assert(live.outstandingAllocations()==1u);
    assert(live.peakAllocations()==2u);

    telemetry.recordFree(256u);
    live=telemetry.snapshot();
    assert(live.outstandingAllocations()==0u);
    assert(live.peakAllocations()==2u);


    telemetry.recordAllocation(1024u);
    assert(telemetry.snapshot().outstandingBytes()==1024u);
    assert(telemetry.snapshot().peakBytes()>=1024u);
    assert(telemetry.snapshot().peakAllocations() >= 1u);

    telemetry.recordAllocation(2048u);
    telemetry.recordFree(1024u);
    const auto peak=telemetry.snapshot();
    assert(peak.outstandingBytes()==2048u);
    assert(peak.peakBytes()>=3072u);

    telemetry.reset();
    const auto reset=telemetry.snapshot();
    assert(reset.allocations==0u);
    assert(reset.frees==0u);
    assert(reset.bytesAllocated==0u);
    assert(reset.bytesFreed==0u);
    assert(reset.outstandingBytes()==0u);
    assert(reset.peakBytes()==0u);
    assert(reset.peakAllocations()==0u);
    return 0;
}
