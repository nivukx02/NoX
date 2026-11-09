#include <cassert>
#include "synchronization/jitter_buffer.h"

int main() {
    using namespace nox::native::sync;

    JitterBuffer buffer(
        4,
        5'000'000ULL);

    (void)buffer.push({30,1.0,3});
    (void)buffer.push({10,1.0,1});
    (void)buffer.push({20,0.9,2});

    JitterSample sample{};
    const bool ok =
        buffer.pop(sample) &&
        sample.timestamp == 10 &&
        buffer.pop(sample) &&
        sample.timestamp == 20 &&
        buffer.pop(sample) &&
        sample.timestamp == 30;

    if (!ok) return 1;

    (void)buffer.push({40,1.0,4});
    (void)buffer.pop(sample);

    const bool lateRejected =
        !buffer.push({30,1.0,5});

    const bool duplicateRejected =
        !buffer.push({40,1.0,4});

    const auto stats =
        buffer.stats();


    JitterBuffer readyBuffer(
        8,
        5'000'000ULL);
    (void)readyBuffer.push({100,1.0,10});
    (void)readyBuffer.push({120,1.0,11});

    JitterSample readySample{};
    assert(
        !readyBuffer.popReady(
            readySample,
            110));
    assert(
        readyBuffer.popReady(
            readySample,
            120));
    assert(readySample.timestamp == 100);

    const auto readyStats =
        readyBuffer.stats();
    assert(readyStats.maximumQueueDepth >= 1U);
    assert(readyStats.watermarkBlockedCount >= 1U);
    assert(readyStats.temporalIntegrity >= 0.0);
    assert(readyStats.temporalIntegrity <= 1.0);

    return (
        lateRejected &&
        duplicateRejected &&
        stats.lateDropCount >= 1U &&
        stats.duplicateCount >= 1U &&
        stats.emittedCount >= 4U &&
        stats.orderingConfidence >= 0.0 &&
        stats.orderingConfidence <= 1.0
    ) ? 0 : 1;
}
