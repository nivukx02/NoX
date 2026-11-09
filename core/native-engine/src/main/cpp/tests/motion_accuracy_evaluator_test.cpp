#include <cassert>
#include <cmath>
#include <vector>

#include "validation/motion_accuracy_evaluator.h"

int main() {
    using namespace nox::native::validation;

    std::vector<GroundTruthSample> truth;
    std::vector<EstimateSample> estimate;

    for (std::uint64_t frame=0; frame<120; ++frame) {
        const auto timestamp =
            frame*8'333'333ull;
        const double x=
            0.5*static_cast<double>(frame)/60.0;
        const double y=
            0.1*std::sin(
                static_cast<double>(frame)*0.08);
        const double z=2.0;

        truth.push_back({
            timestamp,
            0,
            x,
            y,
            z,
        });

        estimate.push_back({
            timestamp+2'000'000ull,
            0,
            x+0.0004,
            y-0.0003,
            z+0.0002,
        });
    }

    const auto report =
        MotionAccuracyEvaluator{}.evaluate(
            truth,
            estimate);

    assert(report.matchedSamples==120u);
    assert(report.dropoutRate<1e-9);
    assert(report.p95ErrorMeters<0.001);
    assert(report.p99ErrorMeters<0.001);
    assert(report.temporalOffsetP95Millis<3.0);
    assert(report.scaleRatioMedian>0.999);
    assert(report.scaleRatioMedian<1.001);
    assert(report.passesPrecisionBudget(
        1.0,
        1.0,
        0.01,
        5.0));

    const auto interpolatedReport =
        MotionAccuracyEvaluator{
            MotionAccuracyEvaluator::Config{
                .maxMatchDeltaNanos = 20'000'000ull,
                .maximumSupportedLagNanos = 100'000'000ull,
                .maximumInterpolationGapNanos = 30'000'000ull,
            }
        }.evaluateInterpolated(
            truth,
            estimate);

    assert(interpolatedReport.matchedSamples >= 118U);
    assert(interpolatedReport.dropoutRate < 0.03);
    assert(interpolatedReport.temporalOffsetP95Millis == 0.0);
    assert(interpolatedReport.p95ErrorMeters < 0.001);


    std::vector<EstimateSample> duplicate;
    duplicate.push_back(estimate.front());
    duplicate.push_back(estimate.front());
    const auto duplicateReport =
        MotionAccuracyEvaluator{}.evaluate(
            std::vector<GroundTruthSample>{
                truth.front(),
                truth[1],
            },
            duplicate);
    assert(duplicateReport.matchedSamples==1u);
    assert(duplicateReport.dropoutRate>0.40);

    auto sparseEstimate=estimate;
    sparseEstimate.erase(
        sparseEstimate.begin()+40,
        sparseEstimate.begin()+60);
    const auto sparseReport =
        MotionAccuracyEvaluator{}.evaluate(
            truth,
            sparseEstimate);
    assert(sparseReport.dropoutRate>0.10);
    assert(sparseReport.dropoutRate<0.20);

    return 0;
}
