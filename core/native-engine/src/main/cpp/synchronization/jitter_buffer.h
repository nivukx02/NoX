#pragma once

#include <cstdint>
#include <deque>
#include <cstddef>

namespace nox::native::sync {

struct JitterSample {
    std::uint64_t timestamp{0};
    double confidence{1.0};
    std::uint64_t sequence{0};
};

struct JitterBufferStats {
    std::uint64_t pushedCount{0};
    std::uint64_t emittedCount{0};
    std::uint64_t droppedCount{0};
    std::uint64_t lateDropCount{0};
    std::uint64_t duplicateCount{0};
    std::uint64_t overflowDropCount{0};
    std::uint64_t outOfOrderCount{0};
    std::uint64_t watermarkBlockedCount{0};
    std::uint64_t maximumQueueDepth{0};
    double interArrivalJitterNanos{0.0};
    double orderingConfidence{0.0};
    double reorderRatio{0.0};
    double temporalIntegrity{0.0};
};

class JitterBuffer final {
public:
    explicit JitterBuffer(
        // Distributed camera nodes can arrive with deeper burst/reorder windows.
        std::size_t capacity = 128,
        std::uint64_t maximumReorderNanos = 5'000'000ULL) noexcept
        : capacity_(capacity),
          maximumReorderNanos_(maximumReorderNanos) {}

    [[nodiscard]] bool push(
        const JitterSample&) noexcept;

    [[nodiscard]] bool pop(
        JitterSample&) noexcept;

    [[nodiscard]] bool popReady(
        JitterSample&,
        std::uint64_t watermarkTimestamp) noexcept;

    [[nodiscard]] std::uint64_t watermark() const noexcept {
        if (q_.empty()) return lastEmittedTimestamp_;
        const auto newest = q_.back().timestamp;
        return newest > maximumReorderNanos_
            ? newest - maximumReorderNanos_
            : 0ULL;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return q_.size();
    }

    [[nodiscard]] std::uint64_t droppedCount() const noexcept {
        return stats_.droppedCount;
    }

    [[nodiscard]] JitterBufferStats stats() const noexcept {
        return stats_;
    }

private:
    std::size_t capacity_{128};
    std::uint64_t maximumReorderNanos_{5'000'000ULL};
    std::deque<JitterSample> q_;
    JitterBufferStats stats_{};
    std::uint64_t lastEmittedTimestamp_{0};
    std::uint64_t lastSequence_{0};
    std::uint64_t lastInputTimestamp_{0};
    std::uint64_t lastInterArrivalNanos_{0};
};

} // namespace nox::native::sync
