#include "geometry3d/imu_extrinsic_calibration.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace nox::native::imu {
namespace {
double clamp01(double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(value, 0.0, 1.0);
}
}

Quaternion ImuExtrinsicCalibration::normalizeSafe(
    const Quaternion& input) noexcept {
    const double n =
        std::sqrt(
            input.w * input.w +
            input.x * input.x +
            input.y * input.y +
            input.z * input.z);

    if (
        !std::isfinite(n) ||
        n < 1e-12) {
        return {};
    }

    return {
        input.w / n,
        input.x / n,
        input.y / n,
        input.z / n,
    };
}

Quaternion ImuExtrinsicCalibration::inverse(
    const Quaternion& input) noexcept {
    const auto q =
        normalizeSafe(input);

    return {
        q.w,
        -q.x,
        -q.y,
        -q.z,
    };
}

double ImuExtrinsicCalibration::angularResidual(
    const Quaternion& a,
    const Quaternion& b) noexcept {
    const auto qa =
        normalizeSafe(a);
    const auto qb =
        normalizeSafe(b);

    const double dot =
        std::abs(
            qa.w * qb.w +
            qa.x * qb.x +
            qa.y * qb.y +
            qa.z * qb.z);

    return
        2.0 *
        std::acos(
            std::clamp(
                dot,
                0.0,
                1.0));
}

ExtrinsicResult ImuExtrinsicCalibration::solve(
    const Quaternion& vision,
    const Quaternion& imu) const noexcept {
    const auto relative =
        multiply(
            normalizeSafe(vision),
            inverse(imu));

    const double residual =
        std::sqrt(
            std::max(
                0.0,
                1.0 -
                    normalizeSafe(
                        relative).w *
                    normalizeSafe(
                        relative).w));

    const double angle =
        2.0 *
        std::asin(
            std::clamp(
                residual,
                0.0,
                1.0));

    return {
        normalizeSafe(relative),
        angle,
        angle,
        std::clamp(
            std::exp(
                -angle * 4.0),
            0.0,
            1.0),
        angle < 0.25
            ? 1.0
            : 0.0,
        angle < 0.25
            ? 1U
            : 0U,
        angle < 0.25
            ? 0U
            : 1U,
        angle < 0.25,
    };
}

ExtrinsicResult ImuExtrinsicCalibration::solveBatch(
    const std::vector<ExtrinsicSample>& samples,
    double maximumResidualRadians) const noexcept {
    ExtrinsicResult result;

    if (samples.empty()) {
        return result;
    }

    const double maximumResidual =
        std::max(
            1e-4,
            maximumResidualRadians);

    Quaternion estimate{};
    bool initialized = false;
    double totalWeight = 0.0;

    std::vector<Quaternion> relatives;
    std::vector<double> weights;

    relatives.reserve(
        samples.size());
    weights.reserve(
        samples.size());

    for (const auto& sample :
         samples) {
        const double confidence =
            std::clamp(
                sample.weight,
                0.0,
                1.0) *
            std::clamp(
                sample.synchronizationConfidence,
                0.0,
                1.0);

        if (
            confidence <= 0.01) {
            ++result.rejectedSamples;
            continue;
        }

        const Quaternion relative =
            normalizeSafe(
                multiply(
                    normalizeSafe(
                        sample.vision),
                    inverse(
                        sample.imu)));

        if (!initialized) {
            estimate =
                relative;
            initialized = true;
        }

        Quaternion aligned =
            relative;

        const double hemisphere =
            aligned.w * estimate.w +
            aligned.x * estimate.x +
            aligned.y * estimate.y +
            aligned.z * estimate.z;

        if (hemisphere < 0.0) {
            aligned.w *= -1.0;
            aligned.x *= -1.0;
            aligned.y *= -1.0;
            aligned.z *= -1.0;
        }

        relatives.push_back(
            aligned);
        weights.push_back(
            confidence);
        totalWeight +=
            confidence;
    }

    if (!initialized || totalWeight <= 0.0) {
        return result;
    }

    for (std::uint32_t iteration = 0;
         iteration < 8U;
         ++iteration) {
        double w = 0.0;
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        double robustWeightSum = 0.0;

        for (
            std::size_t index = 0;
            index < relatives.size();
            ++index) {
            const double angle =
                angularResidual(
                    estimate,
                    relatives[index]);

            const double robust =
                angle <=
                    maximumResidual
                    ? 1.0
                    : maximumResidual /
                        std::max(
                            angle,
                            maximumResidual);

            const double weight =
                weights[index] *
                robust;

            w +=
                relatives[index].w *
                weight;
            x +=
                relatives[index].x *
                weight;
            y +=
                relatives[index].y *
                weight;
            z +=
                relatives[index].z *
                weight;
            robustWeightSum +=
                weight;
        }

        if (
            robustWeightSum <= 0.0) {
            break;
        }

        const Quaternion next =
            normalizeSafe({
                w,
                x,
                y,
                z,
            });

        const double step =
            angularResidual(
                estimate,
                next);

        estimate =
            next;

        if (step < 1e-9) {
            break;
        }
    }

    std::vector<double> residuals;
    residuals.reserve(
        relatives.size());

    double residualSum = 0.0;
    double confidenceSum = 0.0;

    for (
        std::size_t index = 0;
        index < relatives.size();
        ++index) {
        const double angle =
            angularResidual(
                estimate,
                relatives[index]);

        const bool accepted =
            angle <=
            maximumResidual;

        if (accepted) {
            residuals.push_back(
                angle);
            ++result.acceptedSamples;
            residualSum +=
                angle *
                weights[index];

            confidenceSum +=
                weights[index];
        } else {
            ++result.rejectedSamples;
        }
    }

    if (
        result.acceptedSamples == 0U) {
        return result;
    }

    std::sort(
        residuals.begin(),
        residuals.end());

    const double p95Index =
        0.95 *
        static_cast<double>(
            residuals.size() - 1U);

    const auto p95Lower =
        static_cast<std::size_t>(
            std::floor(p95Index));

    const auto p95Upper =
        std::min(
            residuals.size() - 1U,
            p95Lower + 1U);

    const double p95Alpha =
        p95Index -
        static_cast<double>(
            p95Lower);

    result.rotation =
        estimate;

    result.residual =
        confidenceSum <= 0.0
            ? 0.0
            : residualSum /
                confidenceSum;

    result.p95Residual =
        residuals[p95Lower] +
        (
            residuals[p95Upper] -
            residuals[p95Lower]
        ) *
        p95Alpha;

    result.sampleConsensus =
        clamp01(
            static_cast<double>(
                result.acceptedSamples) /
            static_cast<double>(
                relatives.size()));

    result.confidence =
        clamp01(
            result.sampleConsensus *
            0.45 +
            std::exp(
                -result.p95Residual *
                3.5) *
                0.35 +
            std::exp(
                -result.residual *
                4.0) *
                0.20);

    result.valid =
        result.acceptedSamples >= 3U &&
        result.sampleConsensus >= 0.70 &&
        result.p95Residual <=
            maximumResidual &&
        result.confidence >= 0.65;

    return result;
}

} // namespace nox::native::imu
