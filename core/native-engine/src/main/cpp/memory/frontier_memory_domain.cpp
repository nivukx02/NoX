#include "memory/frontier_memory_domain.h"

#include <limits>

namespace nox::native::memory {

FrontierMemoryDomain::FrontierMemoryDomain(
    MemoryDomainConfig config)
    : config_(config) {
    if (config_.hard_bytes == 0U) {
        config_.hard_bytes = 1U;
    }
}

MemoryLease FrontierMemoryDomain::acquire(
    std::uint64_t bytes) noexcept {
    std::scoped_lock lock(mutex_);
    if (bytes == 0U ||
        bytes > config_.hard_bytes ||
        active_bytes_ > config_.hard_bytes - bytes) {
        ++rejected_acquires_;
        return {};
    }

    if (next_id_ == 0U ||
        next_id_ == std::numeric_limits<std::uint64_t>::max()) {
        ++rejected_acquires_;
        return {};
    }

    MemoryLease lease{
        next_id_++,
        bytes,
        true,
    };
    active_bytes_ += bytes;
    active_leases_ += 1U;
    active_lease_bytes_.emplace(lease.id, lease.bytes);
    peak_bytes_ = active_bytes_ > peak_bytes_
        ? active_bytes_
        : peak_bytes_;
    return lease;
}

bool FrontierMemoryDomain::release(MemoryLease lease) noexcept {
    std::scoped_lock lock(mutex_);
    if (!lease.valid ||
        lease.id == 0U ||
        lease.bytes == 0U ||
        active_bytes_ < lease.bytes ||
        active_leases_ == 0U ||
        active_lease_bytes_.find(lease.id) == active_lease_bytes_.end() ||
        active_lease_bytes_.at(lease.id) != lease.bytes) {
        ++invalid_releases_;
        return false;
    }

    active_bytes_ -= lease.bytes;
    active_leases_ -= 1U;
    active_lease_bytes_.erase(lease.id);
    return true;
}

bool FrontierMemoryDomain::canAcquire(
    std::uint64_t bytes) const noexcept {
    std::scoped_lock lock(mutex_);
    return bytes > 0U &&
        bytes <= config_.hard_bytes &&
        active_bytes_ <= config_.hard_bytes - bytes;
}

MemoryDomainStats FrontierMemoryDomain::stats() const noexcept {
    std::scoped_lock lock(mutex_);
    return {
        active_bytes_,
        peak_bytes_,
        rejected_acquires_,
        invalid_releases_,
        active_leases_,
    };
}

void FrontierMemoryDomain::reset() noexcept {
    std::scoped_lock lock(mutex_);
    active_bytes_ = 0U;
    peak_bytes_ = 0U;
    active_leases_ = 0U;
    rejected_acquires_ = 0U;
    invalid_releases_ = 0U;
    active_lease_bytes_.clear();
}

}  // namespace nox::native::memory
