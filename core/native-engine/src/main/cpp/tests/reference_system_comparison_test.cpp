#include <cassert>

#include "validation/reference_system_comparison.h"

int main() {
    using namespace nox::native::validation;

    MotionAccuracyReport nox{};
    MotionAccuracyReport reference{};

    nox.p95ErrorMeters = 0.0005;
    nox.p99ErrorMeters = 0.0009;
    nox.dropoutRate = 0.01;
    nox.temporalOffsetP95Millis = 0.7;
    nox.jitterRmsMeters = 0.0002;

    reference.p95ErrorMeters = 0.0010;
    reference.p99ErrorMeters = 0.0018;
    reference.dropoutRate = 0.01;
    reference.temporalOffsetP95Millis = 0.9;
    reference.jitterRmsMeters = 0.0004;

    const auto better =
        ReferenceSystemComparisonEngine{}.compare(
            nox,
            reference,
            5,
            5,
            2'000,
            0.0002);

    assert(
        better.winner ==
            ReferenceWinner::NOX);
    assert(
        better.relativeP95Advantage > 0.30);
    assert(
        better.absoluteQualificationEvidenceAvailable);

    const auto insufficient =
        ReferenceSystemComparisonEngine{}.compare(
            nox,
            reference,
            1,
            1,
            10,
            0.0002);

    assert(
        insufficient.winner ==
            ReferenceWinner::INDETERMINATE);
    assert(
        !insufficient.statisticallyComparable);

    reference.p95ErrorMeters = 0.0002;
    reference.p99ErrorMeters = 0.0004;

    const auto referenceWins =
        ReferenceSystemComparisonEngine{}.compare(
            nox,
            reference,
            6,
            6,
            2'000,
            0.0002);

    assert(
        referenceWins.winner ==
            ReferenceWinner::REFERENCE);

    {
        std::vector<PairedReferenceSample> paired;
        paired.reserve(600);

        for (int i = 0; i < 600; ++i) {
            paired.push_back({
                0.0012,
                0.0007,
            });
        }

        const auto disagreement =
            ReferenceSystemComparisonEngine{}.comparePaired(
                nox,
                reference,
                5,
                5,
                2'000,
                0.0002,
                paired,
                0x123456789abcdef0ULL);

        assert(
            disagreement.winner ==
                ReferenceWinner::INDETERMINATE);
        assert(
            disagreement.pairedComparisonAvailable);
    }

    return 0;
}
