#include <cassert>
#include <string>

#include "diagnostics/evidence_trace.h"

int main() {
    using nox::native::evidence::EvidenceTraceBuilder;

    EvidenceTraceBuilder trace(
        EvidenceTraceBuilder::Config{2u, 8u});

    trace.record(1u, "capture", "accepted");
    trace.record(2u, "quality", "stable");
    trace.record(3u, "overflow", "abcdefghijklmnopqrstuvwxyz");

    assert(trace.events().size() == 2u);
    assert(trace.droppedEvents() == 1u);
    assert(trace.events().front().timestampNanos == 2u);
    assert(trace.events().back().timestampNanos == 3u);
    assert(trace.events().back().value.size() == 8u);

    const auto hashA = trace.finalize();
    trace.record(4u, "quality", "stable");
    const auto hashB = trace.finalize();
    assert(hashA != hashB);

    return 0;
}
