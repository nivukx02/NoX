#include "synchronization/jitter_buffer.h"

#include <algorithm>
#include <cmath>

namespace nox::native::sync {

namespace {

void refreshConfidence(
    JitterBufferStats& stats) noexcept {
    const double pushed =
        static_cast<double>(
            stats.pushedCount);

    if (
        pushed <=
        0.0) {
        stats.orderingConfidence =
            0.0;
        stats.reorderRatio =
            0.0;
        stats.temporalIntegrity =
            0.0;
        return;
    }

    const double dropRatio =
        std::clamp(
            static_cast<double>(
                stats.droppedCount) /
            pushed,
            0.0,
            1.0);

    stats.reorderRatio =
        std::clamp(
            static_cast<double>(
                stats.outOfOrderCount) /
                pushed,
            0.0,
            1.0);

    const double emissionRatio =
        std::clamp(
            static_cast<double>(
                stats.emittedCount) /
                pushed,
            0.0,
            1.0);

    stats.orderingConfidence =
        std::clamp(
            (
                1.0 -
                dropRatio
            ) *
            (
                1.0 -
                0.70 *
                    stats.reorderRatio
            ),
            0.0,
            1.0);

    stats.temporalIntegrity =
        std::clamp(
            emissionRatio *
                (
                    1.0 -
                    std::min(
                        0.90,
                        stats.interArrivalJitterNanos /
                            5'000'000.0)
                ) *
                stats.orderingConfidence,
            0.0,
            1.0);
}

} // namespace

bool JitterBuffer::push(
    const JitterSample& sample) noexcept {
    if (
        capacity_ == 0U ||
        sample.timestamp == 0ULL ||
        !std::isfinite(
            sample.confidence) ||
        sample.confidence < 0.0 ||
        sample.confidence > 1.0) {
        return false;
    }

    ++stats_.pushedCount;

    if (
        lastSequence_ != 0ULL &&
        sample.sequence != 0ULL &&
        sample.sequence ==
            lastSequence_) {
        ++stats_.duplicateCount;
        ++stats_.droppedCount;
        refreshConfidence(
            stats_);
        return false;
    }

    for (const auto& existing :
         q_) {
        if (
            (
                sample.sequence != 0ULL &&
                existing.sequence ==
                    sample.sequence
            ) ||
            (
                sample.sequence == 0ULL &&
                existing.timestamp ==
                    sample.timestamp
            )) {
            ++stats_.duplicateCount;
            ++stats_.droppedCount;
            refreshConfidence(
                stats_);
            return false;
        }
    }

    if (
        lastInputTimestamp_ != 0ULL &&
        sample.timestamp <
            lastInputTimestamp_) {
        ++stats_.outOfOrderCount;
    }

    if (
        lastEmittedTimestamp_ != 0ULL &&
        sample.timestamp <=
            lastEmittedTimestamp_) {
        ++stats_.lateDropCount;
        ++stats_.droppedCount;
        refreshConfidence(
            stats_);
        return false;
    }

    if (
        lastInputTimestamp_ != 0ULL &&
        sample.timestamp >
            lastInputTimestamp_) {
        const std::uint64_t interval =
            sample.timestamp -
            lastInputTimestamp_;

        if (
            lastInterArrivalNanos_ !=
            0ULL) {
            const double deviation =
                std::abs(
                    static_cast<double>(
                        interval) -
                    static_cast<double>(
                        lastInterArrivalNanos_));

            stats_.interArrivalJitterNanos =
                stats_.interArrivalJitterNanos *
                    0.875 +
                deviation *
                    0.125;
        }

        lastInterArrivalNanos_ =
            interval;
    }

    lastInputTimestamp_ =
        std::max(
            lastInputTimestamp_,
            sample.timestamp);

    if (
        q_.size() >=
        capacity_) {
        ++stats_.overflowDropCount;
        ++stats_.droppedCount;
        refreshConfidence(
            stats_);
        return false;
    }

    const auto position =
        std::upper_bound(
            q_.begin(),
            q_.end(),
            sample.timestamp,
            [](std::uint64_t timestamp,
               const JitterSample& value) {
                return
                    timestamp <
                    value.timestamp;
            });

    q_.insert(
        position,
        sample);

    stats_.maximumQueueDepth =
        std::max(
            stats_.maximumQueueDepth,
            static_cast<std::uint64_t>(
                q_.size()));

    refreshConfidence(
        stats_);

    return true;
}

bool JitterBuffer::pop(
    JitterSample& sample) noexcept {
    if (
        q_.empty()) {
        return false;
    }

    sample =
        q_.front();

    q_.pop_front();

    lastEmittedTimestamp_ =
        sample.timestamp;

    lastSequence_ =
        sample.sequence;

    ++stats_.emittedCount;

    refreshConfidence(
        stats_);

    return true;
}

bool JitterBuffer::popReady(
    JitterSample& sample,
    std::uint64_t watermarkTimestamp) noexcept {
    if (
        q_.empty()) {
        return false;
    }

    const std::uint64_t newestBuffered =
        q_.back().timestamp;

    if (
        watermarkTimestamp <
            newestBuffered ||
        q_.front().timestamp >
            watermarkTimestamp) {
        ++stats_.watermarkBlockedCount;
        refreshConfidence(
            stats_);
        return false;
    }

    return pop(
        sample);
}

} // namespace nox::native::sync
