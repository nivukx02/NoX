#include "calibration/rig_observability_analyzer.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace nox::native::calibration {

namespace {

constexpr double kHalfPi =
    1.57079632679489661923;

class Dsu final {
public:
    explicit Dsu(
        std::size_t size)
        : parent_(size),
          size_(size, 1U) {
        std::iota(
            parent_.begin(),
            parent_.end(),
            0U);
    }

    std::size_t find(
        std::size_t value) {
        if (
            parent_[value] !=
            value) {
            parent_[value] =
                find(
                    parent_[value]);
        }
        return parent_[value];
    }

    void unite(
        std::size_t a,
        std::size_t b) {
        a = find(a);
        b = find(b);
        if (a == b) return;

        if (size_[a] < size_[b]) {
            std::swap(a, b);
        }

        parent_[b] = a;
        size_[a] +=
            size_[b];
    }

private:
    std::vector<std::size_t>
        parent_;
    std::vector<std::size_t>
        size_;
};

double vectorNorm(
    const math::Vec3d& value) noexcept {
    return std::hypot(
        std::hypot(
            value.x,
            value.y),
        value.z);
}

double angleDiversity(
    const std::vector<math::Vec3d>& baselines) {
    if (
        baselines.size() < 2U) {
        return baselines.empty()
            ? 0.0
            : 0.25;
    }

    double weightedSum = 0.0;
    double weightSum = 0.0;

    for (
        std::size_t i = 0;
        i < baselines.size();
        ++i) {
        const double firstNorm =
            vectorNorm(
                baselines[i]);

        if (
            firstNorm <= 1e-9) {
            continue;
        }

        for (
            std::size_t j = i + 1U;
            j < baselines.size();
            ++j) {
            const double secondNorm =
                vectorNorm(
                    baselines[j]);

            if (
                secondNorm <= 1e-9) {
                continue;
            }

            const double cosine =
                std::clamp(
                    (
                        baselines[i].x *
                            baselines[j].x +
                        baselines[i].y *
                            baselines[j].y +
                        baselines[i].z *
                            baselines[j].z
                    ) /
                        (
                            firstNorm *
                            secondNorm
                        ),
                    -1.0,
                    1.0);

            const double angle =
                std::acos(
                    std::abs(
                        cosine));

            const double weight =
                std::min(
                    firstNorm,
                    secondNorm);

            weightedSum +=
                (
                    angle /
                    kHalfPi
                ) *
                weight;

            weightSum +=
                weight;
        }
    }

    return
        weightSum <= 0.0
            ? 0.0
            : std::clamp(
                weightedSum /
                    weightSum,
                0.0,
                1.0);
}

double leverageDiversity(
    const std::vector<math::Vec3d>& baselines) {
    if (baselines.empty()) {
        return 0.0;
    }

    double sum = 0.0;
    double sumSquare = 0.0;

    for (
        const auto& baseline :
        baselines) {
        const double length =
            vectorNorm(
                baseline);

        if (
            !std::isfinite(length) ||
            length <= 0.0) {
            continue;
        }

        const double normalized =
            std::clamp(
                length /
                    0.25,
                0.0,
                4.0);

        sum += normalized;
        sumSquare +=
            normalized *
            normalized;
    }

    if (
        sumSquare <= 1e-12) {
        return 0.0;
    }

    const double effectiveSampleSize =
        sum * sum /
        sumSquare;

    return std::clamp(
        effectiveSampleSize /
            static_cast<double>(
                std::max<std::size_t>(
                    1U,
                    baselines.size())),
        0.0,
        1.0);
}

} // namespace

double RigObservabilityAnalyzer::clamp01(
    double value) noexcept {
    if (!std::isfinite(value)) {
        return 0.0;
    }

    return std::clamp(
        value,
        0.0,
        1.0);
}

double RigObservabilityAnalyzer::percentile(
    std::vector<double> values,
    double p) noexcept {
    values.erase(
        std::remove_if(
            values.begin(),
            values.end(),
            [](double value) {
                return
                    !std::isfinite(value) ||
                    value < 0.0;
            }),
        values.end());

    if (values.empty()) {
        return 0.0;
    }

    std::sort(
        values.begin(),
        values.end());

    const double position =
        clamp01(p) *
        static_cast<double>(
            values.size() - 1U);

    const auto lower =
        static_cast<std::size_t>(
            std::floor(
                position));

    const auto upper =
        std::min(
            values.size() - 1U,
            lower + 1U);

    const double alpha =
        position -
        static_cast<double>(
            lower);

    return
        values[lower] +
        (
            values[upper] -
            values[lower]
        ) *
            alpha;
}

RigObservabilityResult
RigObservabilityAnalyzer::analyze(
    std::size_t cameraCount,
    const std::vector<GraphEdge>& edges) const {
    RigObservabilityResult result;

    if (
        cameraCount == 0U) {
        return result;
    }

    Dsu dsu(cameraCount);

    std::unordered_map<
        std::string,
        std::size_t> ids;

    ids.reserve(cameraCount);

    for (
        std::size_t index = 0;
        index < cameraCount;
        ++index) {
        ids.emplace(
            "C" +
                std::to_string(index),
            index);
    }

    std::vector<math::Vec3d>
        baselines;
    std::vector<double>
        baselineLengths;
    std::vector<double>
        edgeUncertainties;
    std::vector<std::size_t>
        degree(
            cameraCount,
            0U);

    baselines.reserve(edges.size());
    baselineLengths.reserve(edges.size());
    edgeUncertainties.reserve(edges.size());

    double weightedEdgeConfidence = 0.0;
    double edgeWeightSum = 0.0;

    for (
        const auto& edge :
        edges) {
        const auto from =
            ids.find(
                edge.from);
        const auto to =
            ids.find(
                edge.to);

        if (
            from == ids.end() ||
            to == ids.end() ||
            from->second ==
                to->second) {
            continue;
        }

        dsu.unite(
            from->second,
            to->second);

        ++degree[from->second];
        ++degree[to->second];

        const math::Vec3d baseline{
            edge.relative.x,
            edge.relative.y,
            edge.relative.z,
        };

        const double baselineLength =
            vectorNorm(
                baseline);

        if (
            std::isfinite(
                baselineLength) &&
            baselineLength > 1e-6) {
            baselines.push_back(
                baseline);
            baselineLengths.push_back(
                baselineLength);
        }

        const double translationSigma =
            std::max(
                1e-6,
                std::abs(
                    edge.translationSigma));

        const double rotationSigma =
            std::max(
                1e-6,
                std::abs(
                    edge.rotationSigma));

        const double combinedUncertainty =
            std::hypot(
                translationSigma,
                rotationSigma);

        edgeUncertainties.push_back(
            combinedUncertainty);

        const double confidence =
            clamp01(
                std::max(
                    0.0,
                    edge.weight) /
                    (
                        1.0 +
                        combinedUncertainty
                    ));

        weightedEdgeConfidence +=
            confidence;
        edgeWeightSum += 1.0;
        ++result.validEdgeCount;
    }

    for (
        std::size_t camera = 0;
        camera < cameraCount;
        ++camera) {
        if (
            degree[camera] == 0U) {
            ++result.isolatedCameraCount;
        } else if (
            degree[camera] < 2U) {
            ++result.weakCameraCount;
        }
    }

    std::size_t components =
        0U;

    for (
        std::size_t index = 0;
        index < cameraCount;
        ++index) {
        if (
            dsu.find(index) ==
            index) {
            ++components;
        }
    }

    result.connectedComponents =
        components;

    result.connectivity =
        components == 1U
            ? 1.0
            : clamp01(
                1.0 -
                static_cast<double>(
                    components - 1U) /
                    static_cast<double>(
                        cameraCount));

    result.minimumBaselineMeters =
        baselineLengths.empty()
            ? 0.0
            : *std::min_element(
                baselineLengths.begin(),
                baselineLengths.end());

    result.p10BaselineMeters =
        percentile(
            baselineLengths,
            0.10);

    result.p95BaselineMeters =
        percentile(
            baselineLengths,
            0.95);

    if (!baselineLengths.empty()) {
        result.baselineCoverage =
            clamp01(
                (
                    result.p10BaselineMeters /
                    0.08
                ) *
                    0.55 +
                (
                    result.p95BaselineMeters /
                    0.25
                ) *
                    0.45);
    }

    result.angularDiversity =
        angleDiversity(
            baselines);

    result.leverageDiversity =
        leverageDiversity(
            baselines);

    result.edgeConfidence =
        edgeWeightSum <= 0.0
            ? 0.0
            : clamp01(
                weightedEdgeConfidence /
                    edgeWeightSum);

    result.p95EdgeUncertainty =
        percentile(
            edgeUncertainties,
            0.95);

    result.redundancy =
        cameraCount <= 1U
            ? 1.0
            : result.validEdgeCount <=
                    cameraCount - 1U
                ? 0.0
                : clamp01(
                    static_cast<double>(
                        result.validEdgeCount -
                        (
                            cameraCount - 1U
                        )) /
                        static_cast<double>(
                            cameraCount));

    const double weakCameraPenalty =
        clamp01(
            1.0 -
            (
                static_cast<double>(
                    result.weakCameraCount) *
                    0.20 +
                static_cast<double>(
                    result.isolatedCameraCount) *
                    0.40
            ) /
                static_cast<double>(
                    cameraCount));

    const double uncertaintyScore =
        std::exp(
            -result.p95EdgeUncertainty /
            0.35);

    result.score =
        clamp01(
            result.connectivity *
                0.24 +
            result.baselineCoverage *
                0.18 +
            result.angularDiversity *
                0.18 +
            result.edgeConfidence *
                0.14 +
            result.redundancy *
                0.08 +
            result.leverageDiversity *
                0.08 +
            uncertaintyScore *
                0.05 +
            weakCameraPenalty *
                0.05);

    result.observable =
        result.connectedComponents == 1U &&
        result.isolatedCameraCount == 0U &&
        result.baselineCoverage >= 0.20 &&
        result.edgeConfidence >= 0.40 &&
        result.score >= 0.50;

    result.referenceGradeObservable =
        result.observable &&
        result.weakCameraCount <= 1U &&
        result.angularDiversity >= 0.55 &&
        result.leverageDiversity >= 0.55 &&
        result.p95EdgeUncertainty <= 0.20 &&
        result.baselineCoverage >= 0.55 &&
        result.edgeConfidence >= 0.75 &&
        result.score >= 0.78;

    return result;
}

} // namespace nox::native::calibration
