#include <cassert>
#include <cstdint>
#include <vector>

#include "runtime/session_integrity_journal.h"

int main() {
    using namespace nox::native::runtime;

    SessionIntegrityJournal journal;

    const std::vector<std::uint8_t> payload{
        1,2,3,4,5,6,7,8,
    };

    assert(
        journal.appendPayload(
            1,
            1'000'000'000ULL,
            0,
            payload.data(),
            payload.size()));

    assert(
        journal.appendPayload(
            2,
            1'008'000'000ULL,
            0,
            payload.data(),
            payload.size()));

    const auto valid =
        journal.validate();
    assert(valid.valid);
    assert(valid.recoverable);
    assert(valid.lastGoodSequence == 2ULL);
    assert(valid.continuityScore > 0.99);

    assert(
        !journal.append({
            2,
            1'016'000'000ULL,
            0,
            0,
            8,
            0,
        }));

    assert(
        SessionIntegrityJournal::hashPayload(
            payload.data(),
            payload.size()) != 0ULL);

    journal.reset();
    assert(journal.records().empty());

    return 0;
}
