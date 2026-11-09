#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace nox::native::evidence {

struct EvidenceEvent {
    std::uint64_t timestampNanos{0};
    std::string type;
    std::string value;
};

class EvidenceTraceBuilder final {
public:
    struct Config {
        std::size_t maximumEvents{4096};
        std::size_t maximumStringBytes{256};
    };

    EvidenceTraceBuilder() noexcept;
    explicit EvidenceTraceBuilder(Config config) noexcept;

    void record(
        std::uint64_t timestampNanos,
        const std::string& type,
        const std::string& value);

    [[nodiscard]] std::uint64_t finalize() const noexcept;

    [[nodiscard]] const std::vector<EvidenceEvent>& events() const noexcept;

    [[nodiscard]] std::vector<EvidenceEvent> snapshotEvents() const;

    [[nodiscard]] std::uint64_t droppedEvents() const noexcept;

private:
    static std::string bounded(
        const std::string& value,
        std::size_t maximumBytes);

    Config config_{};
    mutable std::mutex mutex_;
    std::vector<EvidenceEvent> events_;
    mutable std::vector<EvidenceEvent> readSnapshot_;
    std::uint64_t droppedEvents_{0};
};

} // namespace nox::native::evidence
