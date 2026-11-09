#include "memory/memory_lease_auditor.h"
#include <cassert>

int main() {
    nox::native::memory::MemoryLeaseAuditor auditor;
    const auto first = auditor.acquire(1024u, 1000u, 42u);
    assert(first.valid());
    auto stats = auditor.stats(1500u);
    assert(stats.active == 1u);
    assert(stats.bytesActive == 1024u);
    assert(auditor.release(first, 1600u));
    assert(!auditor.release(first, 1700u));

    const auto second = auditor.acquire(2048u, 2000u, 43u);
    assert(second.valid());
    assert(auditor.reclaimExpired(4500u, 1000u) == 1u);

    stats = auditor.stats(4500u);
    assert(stats.active == 0u);
    assert(stats.stale == 1u);
    assert(stats.forcedReclaims == 1u);
    return 0;
}
