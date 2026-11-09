#include "runtime/session_integrity_journal.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace nox::native::runtime {

std::uint64_t SessionIntegrityJournal::hashPayload(
    const std::uint8_t* data,
    std::size_t bytes) noexcept {
    std::uint64_t hash =
        1469598103934665603ULL;

    if (
        data == nullptr &&
        bytes != 0U) {
        return 0ULL;
    }

    for (
        std::size_t index = 0;
        index < bytes;
        ++index) {
        hash ^=
            static_cast<std::uint64_t>(
                data[index]);
        hash *=
            1099511628211ULL;
    }

    return hash;
}

bool SessionIntegrityJournal::append(
    const SessionRecord& record) noexcept {
    if (
        record.sequence == 0ULL ||
        record.timestampNanos == 0ULL) {
        return false;
    }

    if (!records_.empty()) {
        const auto& previous =
            records_.back();

        if (
            record.sequence <=
            previous.sequence ||
            record.timestampNanos <
                previous.timestampNanos) {
            return false;
        }
    }

    records_.push_back(
        record);
    return true;
}

bool SessionIntegrityJournal::appendPayload(
    std::uint64_t sequence,
    std::uint64_t timestampNanos,
    std::uint32_t cameraId,
    const std::uint8_t* data,
    std::size_t bytes,
    std::uint32_t flags) noexcept {
    if (
        data == nullptr &&
        bytes != 0U) {
        return false;
    }

    const auto hash =
        hashPayload(
            data,
            bytes);

    return append({
        sequence,
        timestampNanos,
        cameraId,
        hash,
        static_cast<std::uint32_t>(
            std::min<std::size_t>(
                bytes,
                0xffffffffULL)),
        flags,
    });
}

SessionIntegrityResult
SessionIntegrityJournal::validate() const noexcept {
    SessionIntegrityResult result;
    result.recordCount =
        records_.size();

    if (
        records_.empty()) {
        result.valid = true;
        result.recoverable = false;
        result.confidence = 0.0;
        return result;
    }

    std::uint64_t previousSequence = 0ULL;
    std::uint64_t previousTimestamp = 0ULL;

    for (
        const auto& record :
        records_) {
        bool validRecord = true;

        if (
            record.sequence == 0ULL ||
            record.timestampNanos == 0ULL) {
            validRecord = false;
            ++result.timestampViolationCount;
        }

        if (
            previousSequence != 0ULL) {
            if (
                record.sequence ==
                previousSequence) {
                ++result.duplicateSequenceCount;
                validRecord = false;
            } else if (
                record.sequence >
                previousSequence + 1ULL) {
                ++result.sequenceGapCount;
            }

            if (
                record.timestampNanos <
                previousTimestamp) {
                ++result.timestampViolationCount;
                validRecord = false;
            }
        }

        if (validRecord) {
            ++result.validRecordCount;
            result.lastGoodSequence =
                record.sequence;
        } else if (
            result.firstCorruptSequence ==
            0ULL) {
            result.firstCorruptSequence =
                record.sequence;
        }

        previousSequence =
            std::max(
                previousSequence,
                record.sequence);
        previousTimestamp =
            std::max(
                previousTimestamp,
                record.timestampNanos);
    }

    const double count =
        static_cast<double>(
            std::max<std::size_t>(
                result.recordCount,
                1U));

    const double validityRatio =
        static_cast<double>(
            result.validRecordCount) /
        count;

    const double gapPenalty =
        static_cast<double>(
            result.sequenceGapCount) /
        count;

    const double violationPenalty =
        static_cast<double>(
            result.timestampViolationCount +
            result.duplicateSequenceCount +
            result.hashViolationCount) /
        count;

    result.continuityScore =
        std::clamp(
            validityRatio -
                gapPenalty * 0.25 -
                violationPenalty * 0.75,
            0.0,
            1.0);

    result.confidence =
        std::clamp(
            result.continuityScore *
            (
                result.firstCorruptSequence == 0ULL
                    ? 1.0
                    : 0.75
            ),
            0.0,
            1.0);

    result.recoverable =
        result.validRecordCount > 0U &&
        result.lastGoodSequence > 0ULL;

    result.valid =
        result.sequenceGapCount == 0U &&
        result.duplicateSequenceCount == 0U &&
        result.timestampViolationCount == 0U &&
        result.hashViolationCount == 0U;

    return result;
}

void SessionIntegrityJournal::reset() noexcept {
    records_.clear();
}

} // namespace nox::native::runtime
