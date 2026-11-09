#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace nox::native::memory {
struct LeaseHandle {
    std::uint32_t slot{0};
    std::uint32_t generation{0};
    [[nodiscard]] bool valid() const noexcept { return generation != 0u; }
};
struct LeaseStats {
    std::uint32_t active{0};
    std::uint32_t capacity{0};
    std::uint32_t stale{0};
    std::uint64_t acquisitions{0};
    std::uint64_t releases{0};
    std::uint64_t invalidReleases{0};
    std::uint64_t doubleReleases{0};
    std::uint64_t forcedReclaims{0};
    std::uint64_t bytesActive{0};
    std::uint64_t oldestAgeNanos{0};
};
class MemoryLeaseAuditor final {
public:
    static constexpr std::size_t kCapacity = 512u;
    [[nodiscard]] LeaseHandle acquire(std::uint64_t bytes, std::uint64_t nowNanos, std::uint32_t tag) noexcept;
    bool release(LeaseHandle handle, std::uint64_t nowNanos) noexcept;
    std::uint32_t reclaimExpired(std::uint64_t nowNanos, std::uint64_t maximumAgeNanos) noexcept;
    [[nodiscard]] LeaseStats stats(std::uint64_t nowNanos) const noexcept;
    void reset() noexcept;
private:
    struct Entry {
        std::uint32_t generation{1u};
        std::uint32_t tag{0u};
        std::uint64_t bytes{0u};
        std::uint64_t acquiredNanos{0u};
        bool active{false};
        bool stale{false};
    };
    mutable std::mutex mutex_;
    std::array<Entry, kCapacity> entries_{};
    std::uint64_t acquisitions_{0u};
    std::uint64_t releases_{0u};
    std::uint64_t invalidReleases_{0u};
    std::uint64_t doubleReleases_{0u};
    std::uint64_t forcedReclaims_{0u};
};
} // namespace nox::native::memory
