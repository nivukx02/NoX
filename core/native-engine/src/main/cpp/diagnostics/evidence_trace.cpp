#include "diagnostics/evidence_trace.h"

#include <algorithm>
#include <cstdint>
#include <utility>

namespace nox::native::evidence {

EvidenceTraceBuilder::EvidenceTraceBuilder() noexcept
    : EvidenceTraceBuilder(Config{}) {}

EvidenceTraceBuilder::EvidenceTraceBuilder(
    Config config) noexcept
    : config_(config) {
    config_.maximumEvents =
        std::max<std::size_t>(
            config_.maximumEvents,
            1u);
    config_.maximumStringBytes =
        std::max<std::size_t>(
            config_.maximumStringBytes,
            1u);
    events_.reserve(config_.maximumEvents);
}

void EvidenceTraceBuilder::record(
    std::uint64_t timestampNanos,
    const std::string& type,
    const std::string& value) {
    std::scoped_lock lock(mutex_);

    EvidenceEvent event{
        timestampNanos,
        bounded(type, config_.maximumStringBytes),
        bounded(value, config_.maximumStringBytes),
    };

    if (events_.size() >= config_.maximumEvents) {
        events_.erase(events_.begin());
        ++droppedEvents_;
    }

    events_.push_back(std::move(event));
}

std::uint64_t EvidenceTraceBuilder::finalize() const noexcept {
    std::vector<EvidenceEvent> snapshot;
    std::uint64_t dropped = 0;
    {
        std::scoped_lock lock(mutex_);
        snapshot = events_;
        dropped = droppedEvents_;
    }

    std::uint64_t hash = 1469598103934665603ull;

    auto mixBytes = [&hash](const std::string& value) {
        for (const unsigned char c : value) {
            hash ^= c;
            hash *= 1099511628211ull;
        }
    };

    hash ^= dropped;
    hash *= 1099511628211ull;

    for (const auto& event : snapshot) {
        hash ^= event.timestampNanos;
        hash *= 1099511628211ull;
        mixBytes(event.type);
        mixBytes(event.value);
    }

    return hash;
}

const std::vector<EvidenceEvent>&
EvidenceTraceBuilder::events() const noexcept {
    std::scoped_lock lock(mutex_);
    readSnapshot_ = events_;
    return readSnapshot_;
}

std::vector<EvidenceEvent>
EvidenceTraceBuilder::snapshotEvents() const {
    std::scoped_lock lock(mutex_);
    return events_;
}

std::uint64_t EvidenceTraceBuilder::droppedEvents() const noexcept {
    std::scoped_lock lock(mutex_);
    return droppedEvents_;
}

std::string EvidenceTraceBuilder::bounded(
    const std::string& value,
    std::size_t maximumBytes) {
    if (value.size() <= maximumBytes) return value;
    return value.substr(0, maximumBytes);
}

} // namespace nox::native::evidence
