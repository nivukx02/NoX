#include "validation/motion_accuracy_evaluator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <unordered_map>

namespace nox::native::validation {
namespace {

struct Match {
    const GroundTruthSample* truth{nullptr};
    const EstimateSample* estimate{nullptr};
    double error{0.0};
    double timestampOffsetMillis{0.0};
};

double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(),values.end());
    return values[values.size()/2U];
}

double percentileSorted(
    const std::vector<double>& sorted,
    double p) {
    if (sorted.empty()) return 0.0;
    const double clamped=std::clamp(p,0.0,1.0);
    const double index=
        clamped *
        static_cast<double>(sorted.size()-1U);
    const auto lower=
        static_cast<std::size_t>(std::floor(index));
    const auto upper=
        static_cast<std::size_t>(std::ceil(index));
    if (lower==upper) {
        return sorted[lower];
    }
    const double alpha=
        index-static_cast<double>(lower);
    return sorted[lower]*(1.0-alpha)+
        sorted[upper]*alpha;
}

} // namespace

MotionAccuracyEvaluator::MotionAccuracyEvaluator() noexcept
    : MotionAccuracyEvaluator(Config{}) {}

MotionAccuracyEvaluator::MotionAccuracyEvaluator(
    Config config) noexcept
    : config_(config) {
    config_.maxMatchDeltaNanos=
        std::clamp<std::uint64_t>(
            config_.maxMatchDeltaNanos,
            1'000'000ull,
            500'000'000ull);
    config_.maximumSupportedLagNanos=
        std::max(
            config_.maximumSupportedLagNanos,
            config_.maxMatchDeltaNanos);
    config_.maximumInterpolationGapNanos =
        std::clamp<std::uint64_t>(
            config_.maximumInterpolationGapNanos,
            2'000'000ull,
            500'000'000ull);
}

double MotionAccuracyEvaluator::distance(
    const GroundTruthSample& truth,
    const EstimateSample& estimate) noexcept {
    const double dx=
        truth.x-estimate.x;
    const double dy=
        truth.y-estimate.y;
    const double dz=
        truth.z-estimate.z;
    return std::sqrt(
        dx*dx+
        dy*dy+
        dz*dz);
}

double MotionAccuracyEvaluator::percentile(
    std::vector<double> values,
    double p) noexcept {
    std::sort(values.begin(),values.end());
    return percentileSorted(values,p);
}

MotionAccuracyReport MotionAccuracyEvaluator::evaluate(
    const std::vector<GroundTruthSample>& groundTruth,
    const std::vector<EstimateSample>& estimates) const {
    MotionAccuracyReport report;
    report.groundTruthSamples=groundTruth.size();
    report.estimateSamples=estimates.size();

    if (groundTruth.empty()) {
        report.dropoutRate=0.0;
        return report;
    }

    std::vector<GroundTruthSample> truth=groundTruth;
    std::vector<EstimateSample> predicted=estimates;

    std::sort(
        truth.begin(),
        truth.end(),
        [](const auto& a,const auto& b){
            if(a.jointId!=b.jointId) {
                return a.jointId<b.jointId;
            }
            return a.timestampNanos<b.timestampNanos;
        });

    std::sort(
        predicted.begin(),
        predicted.end(),
        [](const auto& a,const auto& b){
            if(a.jointId!=b.jointId) {
                return a.jointId<b.jointId;
            }
            return a.timestampNanos<b.timestampNanos;
        });

    std::unordered_map<std::uint32_t,std::vector<EstimateSample>> byJoint;
    byJoint.reserve(predicted.size());

    for(const auto& sample:predicted) {
        if(
            !std::isfinite(sample.x) ||
            !std::isfinite(sample.y) ||
            !std::isfinite(sample.z)) {
            continue;
        }
        byJoint[sample.jointId].push_back(sample);
    }

    std::vector<Match> matches;
    matches.reserve(truth.size());

    std::unordered_map<std::uint32_t,std::size_t> cursors;
    std::unordered_map<std::uint32_t,std::vector<bool>> consumed;

    for(const auto& sample:truth) {
        const auto iterator=
            byJoint.find(sample.jointId);
        if(iterator==byJoint.end()) {
            ++report.unmatchedGroundTruthSamples;
            continue;
        }

        const auto& candidates=iterator->second;
        auto& cursor=cursors[sample.jointId];
        auto& used=consumed[sample.jointId];
        if(used.empty()) used.assign(candidates.size(),false);

        while(
            cursor+1U<candidates.size() &&
            candidates[cursor+1U].timestampNanos<=
                sample.timestampNanos) {
            ++cursor;
        }

        std::size_t bestIndex=candidates.size();
        std::uint64_t bestDelta=
            std::numeric_limits<std::uint64_t>::max();

        const std::size_t begin=
            cursor>2U?cursor-2U:0U;
        const std::size_t end=
            std::min(
                candidates.size(),
                cursor+3U);

        for(std::size_t index=begin;
            index<end;
            ++index) {
            if(used[index]) continue;
            const auto t=candidates[index].timestampNanos;
            const auto delta=
                t>sample.timestampNanos
                    ? t-sample.timestampNanos
                    : sample.timestampNanos-t;
            if(delta<bestDelta) {
                bestDelta=delta;
                bestIndex=index;
            }
        }

        if(bestIndex==candidates.size() ||
            bestDelta > config_.maxMatchDeltaNanos) {
            ++report.unmatchedGroundTruthSamples;
            continue;
        }

        used[bestIndex]=true;
        if(bestIndex>=cursor) cursor=bestIndex;
        const auto& estimate=candidates[bestIndex];

        Match match;
        match.truth=&sample;
        match.estimate=&estimate;
        match.error=distance(sample,estimate);
        match.timestampOffsetMillis=
            (
                static_cast<double>(
                    estimate.timestampNanos) -
                static_cast<double>(
                    sample.timestampNanos)
            ) / 1e6;
        matches.push_back(match);
    }

    report.matchedSamples=matches.size();
    report.dropoutRate=
        1.0-
        static_cast<double>(report.matchedSamples) /
        static_cast<double>(
            std::max<std::uint64_t>(
                report.groundTruthSamples,
                1U));

    if(matches.empty()) return report;

    std::vector<double> errors;
    std::vector<double> offsets;
    errors.reserve(matches.size());
    offsets.reserve(matches.size());

    double squaredSum=0.0;
    double absoluteSum=0.0;

    std::map<std::uint32_t,std::vector<double>> jointErrors;

    for(const auto& match:matches) {
        errors.push_back(match.error);
        offsets.push_back(
            std::abs(match.timestampOffsetMillis));
        squaredSum +=
            match.error*match.error;
        absoluteSum += match.error;
        jointErrors[match.truth->jointId]
            .push_back(match.error);
    }

    report.meanAbsoluteErrorMeters=
        absoluteSum/
        static_cast<double>(matches.size());

    report.rmseMeters=
        std::sqrt(
            squaredSum/
            static_cast<double>(matches.size()));

    report.p50ErrorMeters=
        percentile(errors,0.50);
    report.p95ErrorMeters=
        percentile(errors,0.95);
    report.p99ErrorMeters=
        percentile(errors,0.99);
    report.maxErrorMeters=
        *std::max_element(
            errors.begin(),
            errors.end());

    report.temporalOffsetMedianMillis=
        median(offsets);
    report.temporalOffsetP95Millis=
        percentile(offsets,0.95);

    double jitterSquared=0.0;
    std::size_t jitterCount=0;

    for(std::size_t index=1; index<matches.size(); ++index) {
        const auto& previous=matches[index-1U];
        const auto& current=matches[index];
        if(previous.truth->jointId!=current.truth->jointId) continue;

        const double delta=current.error-previous.error;
        jitterSquared += delta*delta;
        ++jitterCount;
    }

    report.jitterRmsMeters=
        jitterCount==0
            ? 0.0
            : std::sqrt(
                jitterSquared/
                static_cast<double>(jitterCount));

    std::vector<double> scaleRatios;
    std::unordered_map<std::uint32_t,
                       std::vector<std::pair<double,double>>>
        scaleSamples;

    for(const auto& match:matches) {
        const double truthRadius=
            std::sqrt(
                match.truth->x*match.truth->x+
                match.truth->y*match.truth->y+
                match.truth->z*match.truth->z);
        const double estimateRadius=
            std::sqrt(
                match.estimate->x*match.estimate->x+
                match.estimate->y*match.estimate->y+
                match.estimate->z*match.estimate->z);
        if(truthRadius>1e-6 &&
           estimateRadius>1e-6) {
            scaleRatios.push_back(
                estimateRadius/truthRadius);
            scaleSamples[match.truth->jointId].push_back({
                static_cast<double>(
                    match.truth->timestampNanos),
                estimateRadius/truthRadius,
            });
        }
    }

    report.scaleRatioMedian=
        scaleRatios.empty()
            ? 1.0
            : median(scaleRatios);

    std::vector<double> driftRates;
    for(const auto& [_,samples]:scaleSamples) {
        if(samples.size()<2U) continue;
        const auto& first=samples.front();
        const auto& last=samples.back();
        const double minutes=
            (
                last.first-first.first
            ) / 60e9;
        if(minutes<=1e-6) continue;
        driftRates.push_back(
            (
                last.second-first.second
            ) / minutes);
    }

    report.scaleDriftPerMinute=
        driftRates.empty()
            ? 0.0
            : median(driftRates);

    for(const auto& [jointId,jointValues]:jointErrors) {
        JointAccuracy joint;
        joint.jointId=jointId;
        joint.matchedSamples=jointValues.size();
        joint.meanErrorMeters=
            std::accumulate(
                jointValues.begin(),
                jointValues.end(),
                0.0) /
            static_cast<double>(
                jointValues.size());
        joint.p95ErrorMeters=
            percentile(
                jointValues,
                0.95);
        joint.maxErrorMeters=
            *std::max_element(
                jointValues.begin(),
                jointValues.end());
        report.joints.push_back(joint);
    }

    std::sort(
        report.joints.begin(),
        report.joints.end(),
        [](const JointAccuracy& a,
           const JointAccuracy& b) {
            if(a.p95ErrorMeters!=b.p95ErrorMeters) {
                return a.p95ErrorMeters>b.p95ErrorMeters;
            }
            return a.jointId<b.jointId;
        });

    return report;
}

bool MotionAccuracyReport::passesPrecisionBudget(
    double p95Millimeters,
    double jitterMillimeters,
    double maximumDropoutRatio,
    double maximumLagMillis) const noexcept {
    return p95ErrorMeters*1000.0<=
            p95Millimeters &&
        jitterRmsMeters*1000.0<=
            jitterMillimeters &&
        dropoutRate<=
            maximumDropoutRatio &&
        temporalOffsetP95Millis<=
            maximumLagMillis;
}


MotionAccuracyReport MotionAccuracyEvaluator::evaluateInterpolated(
    const std::vector<GroundTruthSample>& groundTruth,
    const std::vector<EstimateSample>& estimates) const {
    if (
        groundTruth.empty() ||
        estimates.empty()) {
        return evaluate(
            groundTruth,
            estimates);
    }

    std::unordered_map<
        std::uint32_t,
        std::vector<EstimateSample>> byJoint;

    byJoint.reserve(
        estimates.size());

    for (const auto& estimate :
         estimates) {
        if (
            !std::isfinite(
                estimate.x) ||
            !std::isfinite(
                estimate.y) ||
            !std::isfinite(
                estimate.z)) {
            continue;
        }

        byJoint[
            estimate.jointId
        ].push_back(
            estimate);
    }

    for (auto& [_, samples] :
         byJoint) {
        std::sort(
            samples.begin(),
            samples.end(),
            [](const auto& a,
               const auto& b) {
                return
                    a.timestampNanos <
                    b.timestampNanos;
            });
    }

    std::vector<EstimateSample>
        interpolated;
    interpolated.reserve(
        groundTruth.size());

    for (const auto& truth :
         groundTruth) {
        const auto iterator =
            byJoint.find(
                truth.jointId);

        if (
            iterator ==
            byJoint.end()) {
            continue;
        }

        const auto& samples =
            iterator->second;

        if (samples.size() < 2U) {
            continue;
        }

        auto upper =
            std::lower_bound(
                samples.begin(),
                samples.end(),
                truth.timestampNanos,
                [](const EstimateSample& sample,
                   std::uint64_t timestamp) {
                    return
                        sample.timestampNanos <
                        timestamp;
                });

        if (
            upper ==
            samples.begin() ||
            upper ==
            samples.end()) {
            continue;
        }

        const auto& after =
            *upper;
        const auto& before =
            *(upper - 1U);

        if (
            truth.timestampNanos <
                before.timestampNanos ||
            truth.timestampNanos >
                after.timestampNanos) {
            continue;
        }

        const std::uint64_t gap =
            after.timestampNanos -
            before.timestampNanos;

        if (
            gap == 0ULL ||
            gap >
                config_.maximumInterpolationGapNanos) {
            continue;
        }

        const double alpha =
            static_cast<double>(
                truth.timestampNanos -
                before.timestampNanos) /
            static_cast<double>(
                gap);

        interpolated.push_back({
            truth.timestampNanos,
            truth.jointId,
            before.x +
                (
                    after.x -
                    before.x
                ) *
                alpha,
            before.y +
                (
                    after.y -
                    before.y
                ) *
                alpha,
            before.z +
                (
                    after.z -
                    before.z
                ) *
                alpha,
        });
    }

    MotionAccuracyReport result =
        evaluate(
            groundTruth,
            interpolated);

    if (
        result.matchedSamples > 0U) {
        result.temporalOffsetMedianMillis =
            0.0;
        result.temporalOffsetP95Millis =
            0.0;
    }

    return result;
}

} // namespace nox::native::validation
