#pragma once

#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace nox::native::memory {

struct MemoryDomainConfig {
    std::uint64_t hard_bytes{0};
};

struct MemoryLease {
    std::uint64_t id{0};
    std::uint64_t bytes{0};
    bool valid{false};
};

struct MemoryDomainStats {
    std::uint64_t active_bytes{0};
    std::uint64_t peak_bytes{0};
    std::uint64_t rejected_acquires{0};
    std::uint64_t invalid_releases{0};
    std::uint64_t active_leases{0};
};

class FrontierMemoryDomain final {
public:
    explicit FrontierMemoryDomain(MemoryDomainConfig config);

    MemoryLease acquire(std::uint64_t bytes) noexcept;
    bool release(MemoryLease lease) noexcept;
    bool canAcquire(std::uint64_t bytes) const noexcept;
    MemoryDomainStats stats() const noexcept;
    void reset() noexcept;

private:
    MemoryDomainConfig config_;
    mutable std::mutex mutex_;
    std::uint64_t next_id_{1};
    std::uint64_t active_bytes_{0};
    std::uint64_t peak_bytes_{0};
    std::uint64_t rejected_acquires_{0};
    std::uint64_t invalid_releases_{0};
    std::uint64_t active_leases_{0};
    std::unordered_map<std::uint64_t, std::uint64_t> active_lease_bytes_;
};

}  // namespace nox::native::memory
