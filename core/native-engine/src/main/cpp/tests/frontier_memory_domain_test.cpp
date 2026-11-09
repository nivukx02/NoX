#include <cassert>

#include "memory/frontier_memory_domain.h"

using nox::native::memory::FrontierMemoryDomain;
using nox::native::memory::MemoryDomainConfig;
using nox::native::memory::MemoryLease;

int main() {
    FrontierMemoryDomain domain{MemoryDomainConfig{1024U}};

    const auto first = domain.acquire(512U);
    const auto second = domain.acquire(512U);
    assert(first.valid);
    assert(second.valid);
    assert(!domain.canAcquire(1U));

    const auto rejected = domain.acquire(1U);
    assert(!rejected.valid);

    assert(domain.release(first));
    assert(domain.canAcquire(256U));
    assert(domain.release(second));

    assert(!domain.release(second));
    assert(!domain.release(MemoryLease{999999U, 512U, true}));

    const auto stats = domain.stats();
    assert(stats.active_bytes == 0U);
    assert(stats.peak_bytes == 1024U);
    assert(stats.rejected_acquires == 1U);
    assert(stats.invalid_releases == 2U);

    domain.reset();
    assert(domain.stats().peak_bytes == 0U);
    FrontierMemoryDomain resetDomain{MemoryDomainConfig{1024U}};
    const auto staleLease = resetDomain.acquire(512U);
    assert(staleLease.valid);
    resetDomain.reset();
    const resetStats = resetDomain.stats();
    assert(resetStats.active_bytes == 0U);
    assert(resetStats.active_leases == 0U);
    assert(resetDomain.release(staleLease) == false);
    const freshLease = resetDomain.acquire(1024U);
    assert(freshLease.valid);
    assert(freshLease.id != staleLease.id);
    assert(resetDomain.stats().active_bytes == 1024U);
    assert(resetDomain.release(freshLease));

    return 0;
}