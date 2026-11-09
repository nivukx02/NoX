#include "memory/allocation_telemetry.h"

namespace nox::native::memory {
namespace {

void updatePeak(
    std::atomic<std::uint64_t>& peak,
    std::uint64_t value) noexcept {
    auto current = peak.load(
        std::memory_order_relaxed);
    while (
        current < value &&
        !peak.compare_exchange_weak(
            current,
            value,
            std::memory_order_relaxed,
            std::memory_order_relaxed)) {
    }
}

void subtractFloorZero(
    std::atomic<std::uint64_t>& value,
    std::uint64_t amount) noexcept {
    auto current = value.load(
        std::memory_order_relaxed);
    for (;;) {
        const auto next =
            current >= amount
                ? current - amount
                : 0u;
        if (
            value.compare_exchange_weak(
                current,
                next,
                std::memory_order_relaxed,
                std::memory_order_relaxed)) {
            return;
        }
    }
}

} // namespace

void AllocationTelemetryRecorder::recordAllocation(
    std::uint64_t bytes) noexcept {
    allocations_.fetch_add(
        1,
        std::memory_order_relaxed);
    const auto outstandingAllocations =
        outstandingAllocations_.fetch_add(
            1,
            std::memory_order_relaxed) +
        1u;
    bytesAllocated_.fetch_add(
        bytes,
        std::memory_order_relaxed);
    const auto outstanding =
        outstandingBytes_.fetch_add(
            bytes,
            std::memory_order_relaxed) +
        bytes;
    updatePeak(
        peakBytes_,
        outstanding);
    updatePeak(
        peakAllocations_,
        outstandingAllocations);
}

void AllocationTelemetryRecorder::recordFree(
    std::uint64_t bytes) noexcept {
    frees_.fetch_add(
        1,
        std::memory_order_relaxed);
    bytesFreed_.fetch_add(
        bytes,
        std::memory_order_relaxed);
    subtractFloorZero(
        outstandingBytes_,
        bytes);
    subtractFloorZero(
        outstandingAllocations_,
        1u);
}

void AllocationTelemetryRecorder::reset() noexcept {
    allocations_.store(
        0,
        std::memory_order_relaxed);
    frees_.store(
        0,
        std::memory_order_relaxed);
    bytesAllocated_.store(
        0,
        std::memory_order_relaxed);
    bytesFreed_.store(
        0,
        std::memory_order_relaxed);
    outstandingBytes_.store(
        0,
        std::memory_order_relaxed);
    peakBytes_.store(
        0,
        std::memory_order_relaxed);
    outstandingAllocations_.store(
        0,
        std::memory_order_relaxed);
    peakAllocations_.store(
        0,
        std::memory_order_relaxed);
}

AllocationTelemetry
AllocationTelemetryRecorder::snapshot() const noexcept {
    return {
        allocations_.load(std::memory_order_relaxed),
        frees_.load(std::memory_order_relaxed),
        bytesAllocated_.load(std::memory_order_relaxed),
        bytesFreed_.load(std::memory_order_relaxed),
        outstandingBytes_.load(std::memory_order_relaxed),
        peakBytes_.load(std::memory_order_relaxed),
        outstandingAllocations_.load(std::memory_order_relaxed),
        peakAllocations_.load(std::memory_order_relaxed),
    };
}

} // namespace nox::native::memory
