#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>

#include <android/hardware_buffer.h>

namespace nox::native::runtime {

struct HardwareBufferLeaseInfo {
    std::uint64_t lease_id{0};
    std::uint32_t width{0};
    std::uint32_t height{0};
    std::uint32_t layers{0};
    std::uint32_t format{0};
    std::uint64_t usage{0};
    std::size_t resident_bytes{0};
};

class HardwareBufferLeaseRegistry final {
public:
    static constexpr std::uint32_t kDefaultMaximumActiveLeases = 256u;
    static constexpr std::size_t kDefaultMaximumResidentBytes =
        768u * 1024u * 1024u;

    explicit HardwareBufferLeaseRegistry(
        std::uint32_t maximum_active_leases =
            kDefaultMaximumActiveLeases,
        std::size_t maximum_resident_bytes =
            kDefaultMaximumResidentBytes) noexcept;

    HardwareBufferLeaseRegistry(const HardwareBufferLeaseRegistry&) = delete;
    HardwareBufferLeaseRegistry& operator=(const HardwareBufferLeaseRegistry&) = delete;
    ~HardwareBufferLeaseRegistry();

    std::uint64_t acquire(
        AHardwareBuffer* buffer,
        HardwareBufferLeaseInfo* info) noexcept;

    bool release(std::uint64_t lease_id) noexcept;
    void clear() noexcept;

    [[nodiscard]] std::uint32_t active_count() const noexcept;
    [[nodiscard]] std::size_t resident_bytes() const noexcept;
    [[nodiscard]] std::uint64_t acquire_failures() const noexcept;
    [[nodiscard]] std::uint64_t invalid_release_count() const noexcept;
    [[nodiscard]] std::uint32_t maximum_active_leases() const noexcept;
    [[nodiscard]] std::size_t maximum_resident_bytes() const noexcept;

private:
    struct Lease {
        AHardwareBuffer* buffer{nullptr};
        HardwareBufferLeaseInfo info{};
    };

    mutable std::mutex mutex_;
    std::unordered_map<std::uint64_t, Lease> leases_;
    std::uint64_t next_lease_id_{1};
    std::size_t resident_bytes_{0};
    std::uint64_t acquire_failures_{0};
    std::uint64_t invalid_release_count_{0};
    std::uint32_t maximum_active_leases_{
        kDefaultMaximumActiveLeases};
    std::size_t maximum_resident_bytes_{
        kDefaultMaximumResidentBytes};
};

}  // namespace nox::native::runtime
