#include "memory/memory_lease_auditor.h"
#include <algorithm>
#include <mutex>
namespace nox::native::memory {
LeaseHandle MemoryLeaseAuditor::acquire(
    std::uint64_t bytes,
    std::uint64_t nowNanos,
    std::uint32_t tag) noexcept {
    std::scoped_lock lock(mutex_);
    ++acquisitions_;
    for (std::size_t i = 0u; i < entries_.size(); ++i) {
        auto& entry = entries_[i];
        if (entry.active) continue;
        entry.active = true;
        entry.stale = false;
        entry.bytes = bytes;
        entry.acquiredNanos = nowNanos;
        entry.tag = tag;
        ++entry.generation;
        if (entry.generation == 0u) entry.generation = 1u;
        return {static_cast<std::uint32_t>(i), entry.generation};
    }
    return {};
}
bool MemoryLeaseAuditor::release(
    LeaseHandle handle,
    std::uint64_t /*nowNanos*/) noexcept {
    std::scoped_lock lock(mutex_);
    if (!handle.valid() || handle.slot >= entries_.size()) {
        ++invalidReleases_;
        return false;
    }
    auto& entry = entries_[handle.slot];
    if (entry.generation != handle.generation) {
        ++invalidReleases_;
        return false;
    }
    if (!entry.active) {
        ++doubleReleases_;
        return false;
    }
    entry.active = false;
    entry.stale = false;
    entry.bytes = 0u;
    entry.acquiredNanos = 0u;
    entry.tag = 0u;
    ++releases_;
    return true;
}
std::uint32_t MemoryLeaseAuditor::reclaimExpired(
    std::uint64_t nowNanos,
    std::uint64_t maximumAgeNanos) noexcept {
    std::scoped_lock lock(mutex_);
    std::uint32_t reclaimed = 0u;
    for (auto& entry : entries_) {
        if (!entry.active) continue;
        const bool expired =
            nowNanos >= entry.acquiredNanos &&
            (nowNanos - entry.acquiredNanos) > maximumAgeNanos;
        if (!expired) continue;
        entry.active = false;
        entry.stale = true;
        entry.bytes = 0u;
        entry.acquiredNanos = 0u;
        entry.tag = 0u;
        ++entry.generation;
        if (entry.generation == 0u) entry.generation = 1u;
        ++reclaimed;
        ++forcedReclaims_;
    }
    return reclaimed;
}
LeaseStats MemoryLeaseAuditor::stats(
    std::uint64_t nowNanos) const noexcept {
    std::scoped_lock lock(mutex_);
    LeaseStats out{};
    out.capacity = static_cast<std::uint32_t>(entries_.size());
    out.acquisitions = acquisitions_;
    out.releases = releases_;
    out.invalidReleases = invalidReleases_;
    out.doubleReleases = doubleReleases_;
    out.forcedReclaims = forcedReclaims_;
    for (const auto& entry : entries_) {
        if (entry.stale) ++out.stale;
        if (!entry.active) continue;
        ++out.active;
        out.bytesActive += entry.bytes;
        if (nowNanos >= entry.acquiredNanos) {
            out.oldestAgeNanos = std::max(
                out.oldestAgeNanos,
                nowNanos - entry.acquiredNanos);
        }
    }
    return out;
}
void MemoryLeaseAuditor::reset() noexcept {
    std::scoped_lock lock(mutex_);
    entries_ = {};
    acquisitions_ = 0u;
    releases_ = 0u;
    invalidReleases_ = 0u;
    doubleReleases_ = 0u;
    forcedReclaims_ = 0u;
}
} // namespace nox::native::memory
