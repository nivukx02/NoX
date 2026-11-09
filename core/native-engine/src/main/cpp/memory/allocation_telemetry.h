#pragma once

#include <atomic>
#include <cstdint>

namespace nox::native::memory {

struct AllocationTelemetry {
    std::uint64_t allocations{0};
    std::uint64_t frees{0};
    std::uint64_t bytesAllocated{0};
    std::uint64_t bytesFreed{0};
    std::uint64_t outstandingBytesValue{0};
    std::uint64_t peakBytesValue{0};
    std::uint64_t outstandingAllocationsValue{0};
    std::uint64_t peakAllocationsValue{0};

    [[nodiscard]] std::int64_t netAllocations() const noexcept {
        return static_cast<std::int64_t>(allocations) -
            static_cast<std::int64_t>(frees);
    }

    [[nodiscard]] std::int64_t netBytes() const noexcept {
        return static_cast<std::int64_t>(bytesAllocated) -
            static_cast<std::int64_t>(bytesFreed);
    }

    [[nodiscard]] std::uint64_t outstandingBytes() const noexcept {
        return outstandingBytesValue;
    }

    [[nodiscard]] std::uint64_t peakBytes() const noexcept {
        return peakBytesValue;
    }

    [[nodiscard]] std::uint64_t outstandingAllocations() const noexcept {
        return outstandingAllocationsValue;
    }

    [[nodiscard]] std::uint64_t peakAllocations() const noexcept {
        return peakAllocationsValue;
    }
};

class AllocationTelemetryRecorder final {
public:
    void recordAllocation(std::uint64_t bytes) noexcept;
    void recordFree(std::uint64_t bytes) noexcept;
    void reset() noexcept;

    [[nodiscard]] AllocationTelemetry snapshot() const noexcept;

private:
    std::atomic<std::uint64_t> allocations_{0};
    std::atomic<std::uint64_t> frees_{0};
    std::atomic<std::uint64_t> bytesAllocated_{0};
    std::atomic<std::uint64_t> bytesFreed_{0};
    std::atomic<std::uint64_t> outstandingBytes_{0};
    std::atomic<std::uint64_t> peakBytes_{0};
    std::atomic<std::uint64_t> outstandingAllocations_{0};
    std::atomic<std::uint64_t> peakAllocations_{0};
};

} // namespace nox::native::memory
