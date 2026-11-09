#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nox::native::runtime {

struct SessionRecord {
    std::uint64_t sequence{0};
    std::uint64_t timestampNanos{0};
    std::uint32_t cameraId{0};
    std::uint64_t payloadHash{0};
    std::uint32_t payloadBytes{0};
    std::uint32_t flags{0};
};

struct SessionIntegrityResult {
    std::size_t recordCount{0};
    std::size_t validRecordCount{0};
    std::size_t sequenceGapCount{0};
    std::size_t duplicateSequenceCount{0};
    std::size_t timestampViolationCount{0};
    std::size_t hashViolationCount{0};
    std::uint64_t lastGoodSequence{0};
    std::uint64_t firstCorruptSequence{0};
    double continuityScore{0.0};
    double confidence{0.0};
    bool recoverable{false};
    bool valid{false};
};

class SessionIntegrityJournal final {
public:
    [[nodiscard]] static std::uint64_t hashPayload(
        const std::uint8_t* data,
        std::size_t bytes) noexcept;

    [[nodiscard]] bool append(
        const SessionRecord& record) noexcept;

    [[nodiscard]] bool appendPayload(
        std::uint64_t sequence,
        std::uint64_t timestampNanos,
        std::uint32_t cameraId,
        const std::uint8_t* data,
        std::size_t bytes,
        std::uint32_t flags = 0U) noexcept;

    [[nodiscard]] SessionIntegrityResult validate() const noexcept;

    void reset() noexcept;

    [[nodiscard]] const std::vector<SessionRecord>& records() const noexcept {
        return records_;
    }

private:
    std::vector<SessionRecord> records_;
};

} // namespace nox::native::runtime
