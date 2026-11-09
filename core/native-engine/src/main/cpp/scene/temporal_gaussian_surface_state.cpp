#include "scene/temporal_gaussian_surface_state.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace nox::native::scene {

namespace {

using Mat3 = std::array<double,9>;

Mat3 covarianceMatrix(
    const std::array<double,6>& covariance) noexcept {
    return {
        covariance[0], covariance[3], covariance[4],
        covariance[3], covariance[1], covariance[5],
        covariance[4], covariance[5], covariance[2],
    };
}

Mat3 addMat3(
    const Mat3& a,
    const Mat3& b) noexcept {
    Mat3 out{};
    for (std::size_t i = 0; i < 9U; ++i) {
        out[i] = a[i] + b[i];
    }
    return out;
}

Mat3 multiplyMat3(
    const Mat3& a,
    const Mat3& b) noexcept {
    Mat3 out{};
    for (std::size_t row = 0; row < 3U; ++row) {
        for (std::size_t column = 0; column < 3U; ++column) {
            double value = 0.0;
            for (std::size_t k = 0; k < 3U; ++k) {
                value +=
                    a[row * 3U + k] *
                    b[k * 3U + column];
            }
            out[row * 3U + column] = value;
        }
    }
    return out;
}

bool inverseMat3(
    const Mat3& input,
    Mat3& inverse) noexcept {
    const double a = input[0];
    const double b = input[1];
    const double c = input[2];
    const double d = input[3];
    const double e = input[4];
    const double f = input[5];
    const double g = input[6];
    const double h = input[7];
    const double i = input[8];

    const double determinant =
        a * (e * i - f * h) -
        b * (d * i - f * g) +
        c * (d * h - e * g);

    if (
        !std::isfinite(determinant) ||
        std::abs(determinant) < 1e-18) {
        return false;
    }

    const double invDet = 1.0 / determinant;
    inverse = {
        (e * i - f * h) * invDet,
        (c * h - b * i) * invDet,
        (b * f - c * e) * invDet,
        (f * g - d * i) * invDet,
        (a * i - c * g) * invDet,
        (c * d - a * f) * invDet,
        (d * h - e * g) * invDet,
        (b * g - a * h) * invDet,
        (a * e - b * d) * invDet,
    };

    for (double value : inverse) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

Vec3 covarianceWeightedInnovation(
    const std::array<double,6>& priorCovariance,
    const std::array<double,3>& measurementVariance,
    double priorWeight,
    double measurementWeight,
    const Vec3& innovation,
    double minimumVariance,
    double maximumVariance) noexcept {
    const double safePriorWeight =
        std::max(priorWeight, 1e-6);
    const double safeMeasurementWeight =
        std::max(measurementWeight, 1e-6);

    Mat3 prior =
        covarianceMatrix(
            priorCovariance);
    prior = Mat3{
        safeVariance(
            prior[0],
            minimumVariance,
            maximumVariance) /
            safePriorWeight,
        prior[1] / safePriorWeight,
        prior[2] / safePriorWeight,
        prior[3] / safePriorWeight,
        safeVariance(
            prior[4],
            minimumVariance,
            maximumVariance) /
            safePriorWeight,
        prior[5] / safePriorWeight,
        prior[6] / safePriorWeight,
        prior[7] / safePriorWeight,
        safeVariance(
            prior[8],
            minimumVariance,
            maximumVariance) /
            safePriorWeight,
    };

    Mat3 measurement{
        safeVariance(
            measurementVariance[0],
            minimumVariance,
            maximumVariance) /
            safeMeasurementWeight,
        0.0,
        0.0,
        0.0,
        safeVariance(
            measurementVariance[1],
            minimumVariance,
            maximumVariance) /
            safeMeasurementWeight,
        0.0,
        0.0,
        0.0,
        safeVariance(
            measurementVariance[2],
            minimumVariance,
            maximumVariance) /
            safeMeasurementWeight,
    };

    Mat3 innovationCovariance =
        addMat3(
            prior,
            measurement);

    Mat3 inverseInnovation{};
    if (!inverseMat3(
            innovationCovariance,
            inverseInnovation)) {
        return innovation;
    }

    const Mat3 gain =
        multiplyMat3(
            prior,
            inverseInnovation);

    return {
        gain[0] * innovation.x +
            gain[1] * innovation.y +
            gain[2] * innovation.z,
        gain[3] * innovation.x +
            gain[4] * innovation.y +
            gain[5] * innovation.z,
        gain[6] * innovation.x +
            gain[7] * innovation.y +
            gain[8] * innovation.z,
    };
}

double safeVariance(
    double value,
    double floor,
    double ceiling) noexcept {
    if (!std::isfinite(value)) {
        return floor;
    }
    return std::clamp(
        value,
        floor,
        ceiling);
}

Mat3 josephPosteriorCovariance(
    const std::array<double,6>& priorCovariance,
    const std::array<double,3>& measurementVariance,
    double priorWeight,
    double measurementWeight,
    double minimumVariance,
    double maximumVariance) noexcept {
    const double safePriorWeight =
        std::max(priorWeight, 1e-6);
    const double safeMeasurementWeight =
        std::max(measurementWeight, 1e-6);

    Mat3 prior =
        covarianceMatrix(
            priorCovariance);

    for (std::size_t i = 0; i < 3U; ++i) {
        prior[i * 3U + i] =
            safeVariance(
                prior[i * 3U + i],
                minimumVariance,
                maximumVariance) /
            safePriorWeight;
    }

    for (std::size_t i = 0; i < 3U; ++i) {
        for (std::size_t j = 0; j < 3U; ++j) {
            if (i != j) {
                const std::size_t index = i * 3U + j;
                prior[index] =
                    (
                        std::isfinite(prior[index])
                            ? prior[index]
                            : 0.0
                    ) /
                    safePriorWeight;
            }
        }
    }

    Mat3 measurement{};
    measurement[0] =
        safeVariance(
            measurementVariance[0],
            minimumVariance,
            maximumVariance) /
        safeMeasurementWeight;
    measurement[4] =
        safeVariance(
            measurementVariance[1],
            minimumVariance,
            maximumVariance) /
        safeMeasurementWeight;
    measurement[8] =
        safeVariance(
            measurementVariance[2],
            minimumVariance,
            maximumVariance) /
        safeMeasurementWeight;

    const Mat3 innovationCovariance =
        addMat3(
            prior,
            measurement);

    Mat3 inverseInnovation{};
    if (!inverseMat3(
            innovationCovariance,
            inverseInnovation)) {
        return prior;
    }

    const Mat3 gain =
        multiplyMat3(
            prior,
            inverseInnovation);

    Mat3 identity{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    };

    Mat3 residualTransform{};
    for (std::size_t row = 0; row < 3U; ++row) {
        for (std::size_t column = 0; column < 3U; ++column) {
            residualTransform[row * 3U + column] =
                identity[row * 3U + column] -
                gain[row * 3U + column];
        }
    }

    const Mat3 residualTransformTranspose{
        residualTransform[0],
        residualTransform[3],
        residualTransform[6],
        residualTransform[1],
        residualTransform[4],
        residualTransform[7],
        residualTransform[2],
        residualTransform[5],
        residualTransform[8],
    };

    const Mat3 gainTranspose{
        gain[0],
        gain[3],
        gain[6],
        gain[1],
        gain[4],
        gain[7],
        gain[2],
        gain[5],
        gain[8],
    };

    const Mat3 first =
        multiplyMat3(
            multiplyMat3(
                residualTransform,
                prior),
            residualTransformTranspose);

    const Mat3 second =
        multiplyMat3(
            multiplyMat3(
                gain,
                measurement),
            gainTranspose);

    return addMat3(first, second);
}

std::array<double,6> toSymmetricCovariance(
    const Mat3& matrix,
    double minimumVariance,
    double maximumVariance) noexcept {
    return {
        safeVariance(
            matrix[0],
            minimumVariance,
            maximumVariance),
        safeVariance(
            matrix[4],
            minimumVariance,
            maximumVariance),
        safeVariance(
            matrix[8],
            minimumVariance,
            maximumVariance),
        std::isfinite(matrix[1]) ? matrix[1] : 0.0,
        std::isfinite(matrix[2]) ? matrix[2] : 0.0,
        std::isfinite(matrix[5]) ? matrix[5] : 0.0,
    };
}

} // namespace

TemporalGaussianSurfaceState::TemporalGaussianSurfaceState() noexcept
    : TemporalGaussianSurfaceState(GaussianSurfaceOptions{}) {}

TemporalGaussianSurfaceState::TemporalGaussianSurfaceState(
    GaussianSurfaceOptions options)
    : options_(options) {
    options_.cellSizeMeters =
        std::max(
            options_.cellSizeMeters,
            1e-4);
    options_.maximumAssociationMahalanobis =
        std::max(
            options_.maximumAssociationMahalanobis,
            1.0);
    options_.maximumSpatialResidualMeters =
        std::max(
            options_.maximumSpatialResidualMeters,
            options_.cellSizeMeters);
    options_.minimumNormalAssociation =
        std::clamp(
            std::isfinite(options_.minimumNormalAssociation)
                ? options_.minimumNormalAssociation
                : 0.20,
            0.0,
            1.0);
    options_.normalAssociationWeight =
        std::clamp(
            std::isfinite(options_.normalAssociationWeight)
                ? options_.normalAssociationWeight
                : 0.55,
            0.0,
            4.0);
    options_.minimumObservationConfidence =
        clamp01(
            options_.minimumObservationConfidence);
    options_.dynamicSuppression =
        clamp01(
            options_.dynamicSuppression);
    options_.temporalHalfLifeSeconds =
        std::max(
            options_.temporalHalfLifeSeconds,
            0.01);
    options_.velocityDecay =
        std::clamp(
            options_.velocityDecay,
            0.0,
            1.0);
    options_.confidenceDecay =
        std::clamp(
            options_.confidenceDecay,
            0.0,
            1.0);
    options_.maximumVelocityMetersPerSecond =
        std::max(
            std::isfinite(
                options_.maximumVelocityMetersPerSecond)
                ? options_.maximumVelocityMetersPerSecond
                : 25.0,
            0.1);
    options_.minimumVarianceMeters2 =
        std::max(
            options_.minimumVarianceMeters2,
            1e-12);
    options_.maximumVarianceMeters2 =
        std::max(
            options_.maximumVarianceMeters2,
            options_.minimumVarianceMeters2);
    options_.innovationVarianceInflation =
        std::clamp(
            std::isfinite(options_.innovationVarianceInflation)
                ? options_.innovationVarianceInflation
                : 0.35,
            0.0,
            4.0);
    options_.crossCovarianceRetention =
        std::clamp(
            std::isfinite(options_.crossCovarianceRetention)
                ? options_.crossCovarianceRetention
                : 0.88,
            0.0,
            1.0);
    options_.maximumCorrelationMagnitude =
        std::clamp(
            std::isfinite(options_.maximumCorrelationMagnitude)
                ? options_.maximumCorrelationMagnitude
                : 0.96,
            0.0,
            0.999);
    options_.maximumPrimitives =
        std::max<std::size_t>(
            options_.maximumPrimitives,
            1024U);
    options_.minimumEvictionUtility =
        std::clamp(
            options_.minimumEvictionUtility,
            0.0,
            1.0);
}

void TemporalGaussianSurfaceState::reset() noexcept {
    primitives_.clear();
    primitiveCells_.clear();
    cells_.clear();
    fusedObservationCount_ = 0ULL;
    rejectedObservationCount_ = 0ULL;
}

double TemporalGaussianSurfaceState::clamp01(
    double value) noexcept {
    if (!std::isfinite(value)) {
        return 0.0;
    }

    return std::clamp(
        value,
        0.0,
        1.0);
}

double TemporalGaussianSurfaceState::norm(
    const Vec3& value) noexcept {
    return std::hypot(
        std::hypot(
            value.x,
            value.y),
        value.z);
}

Vec3 TemporalGaussianSurfaceState::normalize(
    const Vec3& value) noexcept {
    const double magnitude =
        norm(value);

    if (
        !std::isfinite(magnitude) ||
        magnitude < 1e-12) {
        return {
            0.0,
            0.0,
            1.0,
        };
    }

    return {
        value.x / magnitude,
        value.y / magnitude,
        value.z / magnitude,
    };
}

double TemporalGaussianSurfaceState::dot(
    const Vec3& a,
    const Vec3& b) noexcept {
    return
        a.x * b.x +
        a.y * b.y +
        a.z * b.z;
}

std::size_t TemporalGaussianSurfaceState::CellHash::operator()(
    const std::array<std::int64_t,3>& key) const noexcept {
    std::uint64_t x =
        static_cast<std::uint64_t>(
            key[0]);
    std::uint64_t y =
        static_cast<std::uint64_t>(
            key[1]);
    std::uint64_t z =
        static_cast<std::uint64_t>(
            key[2]);

    x ^= x >> 30U;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27U;
    y ^= y >> 30U;
    y *= 0x94d049bb133111ebULL;
    y ^= y >> 31U;
    z ^= z >> 30U;
    z *= 0x9e3779b97f4a7c15ULL;
    z ^= z >> 27U;

    return static_cast<std::size_t>(
        x ^
        (
            y +
            0x9e3779b97f4a7c15ULL +
            (x << 6U) +
            (x >> 2U)
        ) ^
        (
            z +
            0xc2b2ae3d27d4eb4fULL +
            (y << 7U) +
            (y >> 3U)
        ));
}

std::array<std::int64_t,3>
TemporalGaussianSurfaceState::cellFor(
    const Vec3& position) const noexcept {
    const double size =
        std::max(
            options_.cellSizeMeters,
            1e-6);

    return {
        static_cast<std::int64_t>(
            std::floor(
                position.x / size)),
        static_cast<std::int64_t>(
            std::floor(
                position.y / size)),
        static_cast<std::int64_t>(
            std::floor(
                position.z / size)),
    };
}

double TemporalGaussianSurfaceState::mahalanobis(
    const GaussianSurfacePrimitive& primitive,
    const GaussianSurfaceObservation& observation) const noexcept {
    const double residual[3]{
        observation.position.x - primitive.mean.x,
        observation.position.y - primitive.mean.y,
        observation.position.z - primitive.mean.z,
    };

    std::array<double,6> covariance =
        primitive.covariance;

    covariance[0] =
        safeVariance(
            covariance[0],
            options_.minimumVarianceMeters2,
            options_.maximumVarianceMeters2) +
        safeVariance(
            observation.variance[0],
            options_.minimumVarianceMeters2,
            options_.maximumVarianceMeters2);
    covariance[1] =
        safeVariance(
            covariance[1],
            options_.minimumVarianceMeters2,
            options_.maximumVarianceMeters2) +
        safeVariance(
            observation.variance[1],
            options_.minimumVarianceMeters2,
            options_.maximumVarianceMeters2);
    covariance[2] =
        safeVariance(
            covariance[2],
            options_.minimumVarianceMeters2,
            options_.maximumVarianceMeters2) +
        safeVariance(
            observation.variance[2],
            options_.minimumVarianceMeters2,
            options_.maximumVarianceMeters2);

    stabilizeCovariance(covariance);

    const double a = covariance[0];
    const double b = covariance[1];
    const double c = covariance[2];
    const double d = covariance[3];
    const double e = covariance[4];
    const double f = covariance[5];

    const double determinant =
        a * (b * c - f * f) -
        d * (d * c - e * f) +
        e * (d * f - b * e);

    if (
        !std::isfinite(determinant) ||
        determinant <= 1e-18) {
        return
            residual[0] * residual[0] /
                std::max(a, 1e-12) +
            residual[1] * residual[1] /
                std::max(b, 1e-12) +
            residual[2] * residual[2] /
                std::max(c, 1e-12);
    }

    const double inv00 =
        (b * c - f * f) / determinant;
    const double inv01 =
        (e * f - d * c) / determinant;
    const double inv02 =
        (d * f - e * b) / determinant;
    const double inv11 =
        (a * c - e * e) / determinant;
    const double inv12 =
        (d * e - a * f) / determinant;
    const double inv22 =
        (a * b - d * d) / determinant;

    const double q =
        residual[0] *
            (
                inv00 * residual[0] +
                inv01 * residual[1] +
                inv02 * residual[2]) +
        residual[1] *
            (
                inv01 * residual[0] +
                inv11 * residual[1] +
                inv12 * residual[2]) +
        residual[2] *
            (
                inv02 * residual[0] +
                inv12 * residual[1] +
                inv22 * residual[2]);

    return std::max(
        0.0,
        std::isfinite(q) ? q : 0.0);
}

void TemporalGaussianSurfaceState::stabilizeCovariance(
    std::array<double,6>& covariance) const noexcept {
    for (std::size_t axis = 0; axis < 3U; ++axis) {
        covariance[axis] =
            safeVariance(
                covariance[axis],
                options_.minimumVarianceMeters2,
                options_.maximumVarianceMeters2);
    }

    const double maximumCorrelation =
        options_.maximumCorrelationMagnitude;

    const auto clampPair =
        [&covariance, maximumCorrelation](
            std::size_t covarianceIndex,
            std::size_t first,
            std::size_t second) {
            const double limit =
                maximumCorrelation *
                std::sqrt(
                    std::max(
                        covariance[first] *
                            covariance[second],
                        1e-24));
            covariance[covarianceIndex] =
                std::clamp(
                    std::isfinite(
                        covariance[covarianceIndex])
                        ? covariance[covarianceIndex]
                        : 0.0,
                    -limit,
                    limit);
        };

    clampPair(3U, 0U, 1U);
    clampPair(4U, 0U, 2U);
    clampPair(5U, 1U, 2U);

    for (int iteration = 0; iteration < 8; ++iteration) {
        const double a = covariance[0];
        const double b = covariance[1];
        const double c = covariance[2];
        const double d = covariance[3];
        const double e = covariance[4];
        const double f = covariance[5];

        const double determinant =
            a * (b * c - f * f) -
            d * (d * c - e * f) +
            e * (d * f - b * e);

        if (
            std::isfinite(determinant) &&
            determinant > 1e-18) {
            return;
        }

        covariance[3] *= 0.72;
        covariance[4] *= 0.72;
        covariance[5] *= 0.72;
    }

    covariance[3] = 0.0;
    covariance[4] = 0.0;
    covariance[5] = 0.0;
}

std::size_t TemporalGaussianSurfaceState::findCandidate(
    const GaussianSurfaceObservation& observation) const noexcept {
    const auto baseCell = cellFor(observation.position);
    std::size_t best = primitives_.size();
    double bestMahalanobis = options_.maximumAssociationMahalanobis;

    const auto inspectRadius =
        [&](std::int64_t searchRadius) {
            for (std::int64_t dz = -searchRadius; dz <= searchRadius; ++dz) {
                for (std::int64_t dy = -searchRadius; dy <= searchRadius; ++dy) {
                    for (std::int64_t dx = -searchRadius; dx <= searchRadius; ++dx) {
                        const std::array<std::int64_t,3> cell{
                            baseCell[0] + dx,
                            baseCell[1] + dy,
                            baseCell[2] + dz,
                        };
                        const auto iterator = cells_.find(cell);
                        if (iterator == cells_.end()) continue;

                        for (const auto index : iterator->second) {
                            if (
                                index >= primitives_.size() ||
                                index >= primitiveCells_.size() ||
                                primitiveCells_[index] != cell
                            ) {
                                continue;
                            }
                            const auto& primitive = primitives_[index];
                            const Vec3 delta{
                                observation.position.x - primitive.mean.x,
                                observation.position.y - primitive.mean.y,
                                observation.position.z - primitive.mean.z,
                            };
                            const double euclidean = norm(delta);
                            if (!std::isfinite(euclidean) ||
                                euclidean > options_.maximumSpatialResidualMeters) {
                                continue;
                            }

                            const Vec3 observationNormal =
                                normalize(observation.normal);
                            const Vec3 primitiveNormal =
                                normalize(primitive.normal);
                            const double normalAgreement =
                                std::abs(
                                    dot(
                                        observationNormal,
                                        primitiveNormal));

                            if (
                                !std::isfinite(normalAgreement) ||
                                normalAgreement <
                                    options_.minimumNormalAssociation
                            ) {
                                continue;
                            }

                            const double spatialScore =
                                mahalanobis(primitive, observation);
                            if (!std::isfinite(spatialScore)) {
                                continue;
                            }

                            const double normalPenalty =
                                options_.normalAssociationWeight *
                                (1.0 - normalAgreement) *
                                (1.0 + 0.35 *
                                    (1.0 - primitive.temporalStability));
                            const double score =
                                spatialScore + normalPenalty;

                            if (score < bestMahalanobis) {
                                bestMahalanobis = score;
                                best = index;
                            }
                        }
                    }
                }
            }
        };

    inspectRadius(1);
    if (best == primitives_.size()) {
        const auto fullRadius = std::max<std::int64_t>(
            1,
            static_cast<std::int64_t>(
                std::ceil(
                    options_.maximumSpatialResidualMeters /
                    std::max(options_.cellSizeMeters, 1e-6))));
        if (fullRadius > 1) inspectRadius(fullRadius);
    }

    return best;
}
double TemporalGaussianSurfaceState::primitiveUtility(
    const GaussianSurfacePrimitive& primitive) const noexcept {
    const double covarianceTrace =
        std::max(
            primitive.covariance[0],
            std::max(
                primitive.covariance[1],
                primitive.covariance[2]));

    const double uncertaintyScore =
        1.0 /
        (
            1.0 +
            covarianceTrace /
                std::max(
                    options_.maximumVarianceMeters2,
                    options_.minimumVarianceMeters2 * 10.0)
        );

    const double dynamicScore =
        1.0 -
        clamp01(primitive.dynamicProbability);

    const double supportScore =
        1.0 -
        std::exp(
            -static_cast<double>(
                std::min<std::uint64_t>(
                    primitive.observationCount,
                    64ULL)
            ) / 8.0);

    return clamp01(
        clamp01(primitive.confidence) * 0.42 +
        clamp01(primitive.opacity) * 0.18 +
        clamp01(primitive.temporalStability) * 0.14 +
        uncertaintyScore * 0.14 +
        dynamicScore * 0.08 +
        supportScore * 0.04);
}

bool TemporalGaussianSurfaceState::evictLowestUtility() noexcept {
    if (primitives_.empty()) return false;

    std::size_t victim = primitives_.size();
    double victimUtility = std::numeric_limits<double>::infinity();

    for (std::size_t index = 0; index < primitives_.size(); ++index) {
        const double utility =
            primitiveUtility(primitives_[index]);
        if (
            utility < victimUtility &&
            std::isfinite(utility)) {
            victimUtility = utility;
            victim = index;
        }
    }

    if (
        victim == primitives_.size() ||
        victimUtility > options_.minimumEvictionUtility) {
        return false;
    }

    const auto victimCell = primitiveCells_[victim];
    const std::size_t lastIndex =
        primitives_.size() - 1U;

    if (victim != lastIndex) {
        primitives_[victim] =
            std::move(primitives_[lastIndex]);
        primitiveCells_[victim] =
            primitiveCells_[lastIndex];

        const auto movedCell =
            primitiveCells_[victim];

        auto& movedBucket =
            cells_[movedCell];
        for (auto& entry : movedBucket) {
            if (entry == lastIndex) {
                entry = victim;
                break;
            }
        }
    }

    // Remove exactly the evicted index from its old cell. When the moved element
    // originated from the same cell, the bucket temporarily contains the replacement
    // index twice, so erasing every match would also delete the surviving primitive.
    auto victimBucketIterator =
        cells_.find(victimCell);
    if (victimBucketIterator != cells_.end()) {
        auto& bucket = victimBucketIterator->second;
        const auto entry =
            std::find(
                bucket.begin(),
                bucket.end(),
                victim);
        if (entry != bucket.end()) {
            bucket.erase(entry);
        }
        if (bucket.empty()) {
            cells_.erase(victimBucketIterator);
        }
    }

    primitives_.pop_back();
    primitiveCells_.pop_back();
    return true;
}

void TemporalGaussianSurfaceState::fusePrimitive(
    GaussianSurfacePrimitive& primitive,
    const GaussianSurfaceObservation& observation) noexcept {
    const double measurementConfidence =
        clamp01(
            observation.confidence);

    const double dynamic =
        clamp01(
            observation.dynamicProbability);

    const double temporal =
        clamp01(
            observation.temporalConsistency);

    const Vec3 innovation{
        observation.position.x - primitive.mean.x,
        observation.position.y - primitive.mean.y,
        observation.position.z - primitive.mean.z,
    };

    const double innovationScale =
        std::clamp(
            options_.innovationVarianceInflation,
            0.0,
            4.0);

    const double priorWeight =
        clamp01(
            primitive.confidence) +
        0.10;

    const double visibility =
        clamp01(
            observation.visibilityConfidence);

    const double photometric =
        clamp01(
            observation.photometricConsistency);

    const double normalConfidence =
        clamp01(
            observation.normalConfidence);

    const double visibilityWeight =
        0.30 +
        0.70 * visibility;

    const double photometricWeight =
        0.35 +
        0.65 * photometric;

    const double measurementWeight =
        measurementConfidence *
        (
            0.40 +
            0.60 * temporal
        ) *
        visibilityWeight *
        photometricWeight *
        (
            0.60 +
                0.40 * normalConfidence
        ) *
        (
            1.0 -
            options_.dynamicSuppression *
                dynamic
        );

    const double totalWeight =
        priorWeight +
        measurementWeight;

    if (
        measurementWeight <= 0.0 ||
        totalWeight <= 0.0) {
        return;
    }

    const meanInnovation =
        covarianceWeightedInnovation(
            primitive.covariance,
            observation.variance,
            priorWeight,
            measurementWeight,
            innovation,
            options_.minimumVarianceMeters2,
            options_.maximumVarianceMeters2);

    /*
     * Full-covariance update: cross-axis uncertainty can rotate the correction direction instead of
     * forcing each Cartesian component to move independently. If the matrix becomes numerically
     * singular, the helper returns the raw innovation and the existing bounded covariance logic
     * remains the safety fallback.
     */
    primitive.mean = {
        primitive.mean.x +
            meanInnovation.x,
        primitive.mean.y +
            meanInnovation.y,
        primitive.mean.z +
            meanInnovation.z,
    };

    const auto finiteVec3 =
        [](const Vec3& value) noexcept {
            if (
                !std::isfinite(value.x) ||
                !std::isfinite(value.y) ||
                !std::isfinite(value.z)
            ) {
                return Vec3{
                    0.0,
                    0.0,
                    0.0,
                };
            }
            return value;
        };

    const Vec3 safeObservationAcceleration =
        finiteVec3(observation.acceleration);
    const Vec3 safeObservationJerk =
        finiteVec3(observation.jerk);

    const Vec3 observedVelocity{
        std::isfinite(observation.velocity.x)
            ? observation.velocity.x
            : 0.0,
        std::isfinite(observation.velocity.y)
            ? observation.velocity.y
            : 0.0,
        std::isfinite(observation.velocity.z)
            ? observation.velocity.z
            : 0.0,
    };
    const double observedVelocityMagnitude =
        norm(observedVelocity);
    const Vec3 boundedObservedVelocity =
        observedVelocityMagnitude >
                options_.maximumVelocityMetersPerSecond
            ? observedVelocity *
                (
                    options_.maximumVelocityMetersPerSecond /
                    std::max(
                        observedVelocityMagnitude,
                        1e-9)
                )
            : observedVelocity;

    const Vec3 previousVelocity =
        primitive.velocity;

    primitive.velocity = {
        (
            previousVelocity.x *
                priorWeight *
                options_.velocityDecay +
            boundedObservedVelocity.x *
                measurementWeight
        ) /
            std::max(
                priorWeight *
                    options_.velocityDecay +
                    measurementWeight,
                1e-9),
        (
            previousVelocity.y *
                priorWeight *
                options_.velocityDecay +
            boundedObservedVelocity.y *
                measurementWeight
        ) /
            std::max(
                priorWeight *
                    options_.velocityDecay +
                    measurementWeight,
                1e-9),
        (
            previousVelocity.z *
                priorWeight *
                options_.velocityDecay +
            boundedObservedVelocity.z *
                measurementWeight
        ) /
            std::max(
                priorWeight *
                    options_.velocityDecay +
                    measurementWeight,
                1e-9),
    };

    const double dt =
        primitive.lastTimestampNanos > 0ULL &&
            observation.timestampNanos >
                primitive.lastTimestampNanos
            ? std::clamp(
                static_cast<double>(
                    observation.timestampNanos -
                        primitive.lastTimestampNanos) /
                    1e9,
                1.0 / 240.0,
                1.0)
            : 0.0;

    const Vec3 measuredAcceleration =
        dt > 0.0
            ? Vec3{
                (
                    primitive.velocity.x -
                        previousVelocity.x
                    ) / dt,
                (
                    primitive.velocity.y -
                        previousVelocity.y
                    ) / dt,
                (
                    primitive.velocity.z -
                        previousVelocity.z
                    ) / dt,
            }
            : safeObservationAcceleration;

    const double measuredAccelerationMagnitude =
        norm(measuredAcceleration);
    const Vec3 boundedAcceleration =
        measuredAccelerationMagnitude >
                options_.maximumAccelerationMetersPerSecond2
            ? measuredAcceleration *
                (
                    options_.maximumAccelerationMetersPerSecond2 /
                        std::max(
                            measuredAccelerationMagnitude,
                            1e-9)
                )
            : measuredAcceleration;

    const Vec3 previousAcceleration =
        primitive.acceleration;

    primitive.acceleration = {
        previousAcceleration.x *
                options_.accelerationDecay *
                (1.0 - measurementWeight / std::max(totalWeight, 1e-9)) +
            boundedAcceleration.x *
                (measurementWeight / std::max(totalWeight, 1e-9)),
        previousAcceleration.y *
                options_.accelerationDecay *
                (1.0 - measurementWeight / std::max(totalWeight, 1e-9)) +
            boundedAcceleration.y *
                (measurementWeight / std::max(totalWeight, 1e-9)),
        previousAcceleration.z *
                options_.accelerationDecay *
                (1.0 - measurementWeight / std::max(totalWeight, 1e-9)) +
            boundedAcceleration.z *
                (measurementWeight / std::max(totalWeight, 1e-9)),
    };

    const Vec3 measuredJerk =
        dt > 0.0
            ? (
                primitive.acceleration -
                    previousAcceleration
                ) / dt
            : safeObservationJerk;

    const double measuredJerkMagnitude =
        norm(measuredJerk);
    const Vec3 boundedJerk =
        measuredJerkMagnitude >
                options_.maximumJerkMetersPerSecond3
            ? measuredJerk *
                (
                    options_.maximumJerkMetersPerSecond3 /
                        std::max(
                            measuredJerkMagnitude,
                            1e-9)
                )
            : measuredJerk;

    primitive.jerk = {
        primitive.jerk.x *
                options_.jerkDecay +
            boundedJerk.x *
                (1.0 - options_.jerkDecay),
        primitive.jerk.y *
                options_.jerkDecay +
            boundedJerk.y *
                (1.0 - options_.jerkDecay),
        primitive.jerk.z *
                options_.jerkDecay +
            boundedJerk.z *
                (1.0 - options_.jerkDecay),
    };

    const Vec3 observedNormal =
        normalize(
            observation.normal);

    const normalMeasurementWeight =
        measurementWeight *
        (
            0.25 +
                0.75 * normalConfidence
        );

    primitive.normal =
        normalize({
            primitive.normal.x *
                priorWeight +
            observedNormal.x *
                normalMeasurementWeight,
            primitive.normal.y *
                priorWeight +
            observedNormal.y *
                normalMeasurementWeight,
            primitive.normal.z *
                priorWeight +
            observedNormal.z *
                normalMeasurementWeight,
        });

    const joseph =
        josephPosteriorCovariance(
            primitive.covariance,
            observation.variance,
            priorWeight,
            measurementWeight,
            options_.minimumVarianceMeters2,
            options_.maximumVarianceMeters2);

    auto posterior =
        toSymmetricCovariance(
            joseph,
            options_.minimumVarianceMeters2,
            options_.maximumVarianceMeters2);

    const crossWeight =
        measurementWeight /
            std::max(
                totalWeight,
                1e-9);

    /*
     * Joseph form preserves a positive-semidefinite posterior under correlated prior covariance and
     * diagonal measurement noise. The innovation outer-product term is deliberately additive: it is
     * a model-mismatch uncertainty floor, not an invented measurement.
     */
    posterior[0] +=
        innovationScale *
        crossWeight *
        innovation.x *
        innovation.x;
    posterior[1] +=
        innovationScale *
        crossWeight *
        innovation.y *
        innovation.y;
    posterior[2] +=
        innovationScale *
        crossWeight *
        innovation.z *
        innovation.z;

    const innovationXY =
        innovation.x *
        innovation.y;
    const innovationXZ =
        innovation.x *
        innovation.z;
    const innovationYZ =
        innovation.y *
        innovation.z;

    posterior[3] =
        options_.crossCovarianceRetention *
            primitive.covariance[3] +
        (
            1.0 -
            options_.crossCovarianceRetention
        ) *
            (
                posterior[3] +
                innovationScale *
                    crossWeight *
                    innovationXY
            );

    posterior[4] =
        options_.crossCovarianceRetention *
            primitive.covariance[4] +
        (
            1.0 -
            options_.crossCovarianceRetention
        ) *
            (
                posterior[4] +
                innovationScale *
                    crossWeight *
                    innovationXZ
            );

    posterior[5] =
        options_.crossCovarianceRetention *
            primitive.covariance[5] +
        (
            1.0 -
            options_.crossCovarianceRetention
        ) *
            (
                posterior[5] +
                innovationScale *
                    crossWeight *
                    innovationYZ
            );

    primitive.covariance =
        posterior;

    stabilizeCovariance(primitive);

    primitive.confidence =
        clamp01(
            primitive.confidence *
                0.55 +
            measurementConfidence *
                0.45);

    primitive.dynamicProbability =
        clamp01(
            primitive.dynamicProbability *
                0.62 +
            dynamic *
                0.23 +
            (
                1.0 -
                temporal
            ) *
                0.15);

    primitive.temporalStability =
        clamp01(
            primitive.temporalStability *
                0.60 +
            temporal *
                0.40);

    const double opacityEvidence =
        measurementConfidence *
        temporal *
        visibility *
        photometric *
        (
            1.0 -
            options_.dynamicSuppression *
                dynamic *
                0.70
        );

    const double occlusionDecay =
        0.76 +
        0.18 * visibility;

    primitive.opacity =
        clamp01(
            primitive.opacity *
                occlusionDecay +
            opacityEvidence *
                0.26);

    primitive.observationCount =
        std::min<std::uint64_t>(
            std::uint64_t{1'000'000'000ULL},
            primitive.observationCount + std::uint64_t{1ULL});

    primitive.lastTimestampNanos =
        std::max(
            primitive.lastTimestampNanos,
            observation.timestampNanos);
}

bool TemporalGaussianSurfaceState::integrate(
    const GaussianSurfaceObservation& observation) noexcept {
    if (
        !std::isfinite(
            observation.position.x) ||
        !std::isfinite(
            observation.position.y) ||
        !std::isfinite(
            observation.position.z) ||
        !std::isfinite(
            observation.confidence) ||
        !std::isfinite(
            observation.normalConfidence) ||
        observation.normalConfidence < 0.0 ||
        observation.normalConfidence > 1.0 ||
        observation.timestampNanos == 0ULL ||
        clamp01(
            observation.confidence) <
            options_.minimumObservationConfidence) {
        ++rejectedObservationCount_;
        return false;
    }

    const auto cell =
        cellFor(
            observation.position);

    const auto candidate =
        findCandidate(
            observation);

    if (
        candidate ==
        primitives_.size()) {
        if (
            primitives_.size() >=
            options_.maximumPrimitives) {
            prune(
                observation.timestampNanos);
        }

        if (
            primitives_.size() >=
            options_.maximumPrimitives) {
            if (!evictLowestUtility()) {
                ++rejectedObservationCount_;
                return false;
            }
        }

        GaussianSurfacePrimitive primitive;
        primitive.mean =
            observation.position;
        primitive.normal =
            normalize(
                observation.normal);
        primitive.velocity =
            observation.velocity;

        primitive.covariance = {
            safeVariance(
                observation.variance[0],
                options_.minimumVarianceMeters2,
                options_.maximumVarianceMeters2),
            safeVariance(
                observation.variance[1],
                options_.minimumVarianceMeters2,
                options_.maximumVarianceMeters2),
            safeVariance(
                observation.variance[2],
                options_.minimumVarianceMeters2,
                options_.maximumVarianceMeters2),
        };

        primitive.confidence =
            clamp01(
                observation.confidence);
        primitive.dynamicProbability =
            clamp01(
                observation.dynamicProbability);
        primitive.temporalStability =
            clamp01(
                observation.temporalConsistency);
        primitive.opacity =
            primitive.confidence *
            primitive.temporalStability;
        primitive.observationCount = 1ULL;
        primitive.lastTimestampNanos =
            observation.timestampNanos;

        primitives_.push_back(
            primitive);
        primitiveCells_.push_back(
            cell);

        cells_[cell].push_back(
            primitives_.size() - 1U);
        ++fusedObservationCount_;

        return true;
    }

    auto& primitive =
        primitives_[candidate];

    if (
        primitive.lastTimestampNanos != 0ULL &&
        observation.timestampNanos <
            primitive.lastTimestampNanos) {
        ++rejectedObservationCount_;
        return false;
    }

    const double associationScore =
        mahalanobis(
            primitive,
            observation);

    const double normalizedAssociation =
        std::sqrt(
            std::max(
                0.0,
                associationScore));

    const double cauchy =
        1.0 /
            (
                1.0 +
                (
                    normalizedAssociation /
                        2.0
                ) *
                (
                    normalizedAssociation /
                        2.0
                )
            );

    const double tukeyRadius =
        normalizedAssociation /
            4.0;

    const double tukey =
        tukeyRadius < 1.0
            ? (
                (
                    1.0 -
                        tukeyRadius *
                        tukeyRadius
                ) *
                (
                    1.0 -
                        tukeyRadius *
                        tukeyRadius
                )
            )
            : 0.02;

    GaussianSurfaceObservation robustObservation =
        observation;

    robustObservation.confidence =
        clamp01(
            observation.confidence *
                (
                    0.35 * cauchy +
                        0.65 * tukey
                ));

    fusePrimitive(
        primitive,
        robustObservation);
    ++fusedObservationCount_;

    const auto newCell =
        cellFor(
            primitive.mean);

    if (
        candidate <
            primitiveCells_.size() &&
        primitiveCells_[candidate] !=
            newCell) {
        auto& oldBucket =
            cells_[
                primitiveCells_[candidate]
            ];

        oldBucket.erase(
            std::remove(
                oldBucket.begin(),
                oldBucket.end(),
                candidate),
            oldBucket.end());

        if (oldBucket.empty()) {
            cells_.erase(
                primitiveCells_[candidate]);
        }

        primitiveCells_[candidate] =
            newCell;

        cells_[newCell].push_back(
            candidate);
    }

    return true;
}

bool TemporalGaussianSurfaceState::predict(
    std::uint64_t timestampNanos) noexcept {
    if (
        timestampNanos == 0ULL ||
        primitives_.empty()) {
        return false;
    }

    bool predictedAny = false;

    const double velocityLog =
        std::log(
            std::max(
                options_.velocityDecay,
                1e-9));
    const double confidenceLog =
        std::log(
            std::max(
                options_.confidenceDecay,
                1e-9));
    const double stabilityLog =
        std::log(
            0.985);

    for (auto& primitive :
         primitives_) {
        if (
            primitive.lastTimestampNanos == 0ULL ||
            timestampNanos <=
                primitive.lastTimestampNanos) {
            continue;
        }

        const double dt =
            std::clamp(
                static_cast<double>(
                    timestampNanos -
                    primitive.lastTimestampNanos) /
                    1e9,
                0.0,
                1.0);

        const double dtSquared =
            dt * dt;
        const double dtCubed =
            dtSquared * dt;

        primitive.mean.x +=
            primitive.velocity.x *
                dt +
            0.5 *
                primitive.acceleration.x *
                dtSquared +
            (1.0 / 6.0) *
                primitive.jerk.x *
                dtCubed;
        primitive.mean.y +=
            primitive.velocity.y *
                dt +
            0.5 *
                primitive.acceleration.y *
                dtSquared +
            (1.0 / 6.0) *
                primitive.jerk.y *
                dtCubed;
        primitive.mean.z +=
            primitive.velocity.z *
                dt +
            0.5 *
                primitive.acceleration.z *
                dtSquared +
            (1.0 / 6.0) *
                primitive.jerk.z *
                dtCubed;

        const double frameEquivalent =
            dt * 60.0;
        const double velocityDecay =
            std::exp(
                velocityLog *
                frameEquivalent);
        const double confidenceDecay =
            std::exp(
                confidenceLog *
                frameEquivalent);
        const double stabilityDecay =
            std::exp(
                stabilityLog *
                frameEquivalent);

        primitive.velocity.x *=
            velocityDecay;
        primitive.velocity.y *=
            velocityDecay;
        primitive.velocity.z *=
            velocityDecay;

        const double accelerationDecay =
            std::pow(
                std::clamp(
                    options_.accelerationDecay,
                    0.0,
                    1.0),
                frameEquivalent);
        primitive.acceleration.x *=
            accelerationDecay;
        primitive.acceleration.y *=
            accelerationDecay;
        primitive.acceleration.z *=
            accelerationDecay;

        const double jerkDecay =
            std::pow(
                std::clamp(
                    options_.jerkDecay,
                    0.0,
                    1.0),
                frameEquivalent);
        primitive.jerk.x *= jerkDecay;
        primitive.jerk.y *= jerkDecay;
        primitive.jerk.z *= jerkDecay;

        primitive.confidence *=
            confidenceDecay;

        primitive.temporalStability *=
            stabilityDecay;

        primitive.opacity =
            clamp01(
                primitive.confidence *
                primitive.temporalStability);

        for (double& variance :
             primitive.covariance) {
            variance =
                std::clamp(
                    variance +
                        dt *
                            (
                                0.0002 +
                                0.002 *
                                    norm(
                                        primitive.velocity)
                            ),
                    options_.minimumVarianceMeters2,
                    options_.maximumVarianceMeters2);
        }

        primitive.lastTimestampNanos =
            timestampNanos;

        predictedAny = true;
    }

    prune(
        timestampNanos);

    cells_.clear();
    primitiveCells_.clear();
    primitiveCells_.reserve(
        primitives_.size());
    for (
        std::size_t index = 0;
        index < primitives_.size();
        ++index) {
        const auto cell =
            cellFor(
                primitives_[index].mean);
        primitiveCells_.push_back(
            cell);
        cells_[cell].push_back(
            index);
    }

    return predictedAny;
}

void TemporalGaussianSurfaceState::prune(
    std::uint64_t timestampNanos) noexcept {
    if (
        primitives_.empty()) {
        return;
    }

    const double halfLifeSeconds =
        std::max(
            options_.temporalHalfLifeSeconds,
            0.01);

    std::vector<GaussianSurfacePrimitive> survivors;
    survivors.reserve(
        primitives_.size());

    for (const auto& primitive :
         primitives_) {
        const double ageSeconds =
            primitive.lastTimestampNanos == 0ULL ||
            timestampNanos <=
                primitive.lastTimestampNanos
                ? 0.0
                : static_cast<double>(
                    timestampNanos -
                    primitive.lastTimestampNanos) /
                    1e9;

        const double ageDecay =
            std::exp(
                -0.6931471805599453 *
                ageSeconds /
                halfLifeSeconds);

        const bool retain =
            primitive.confidence >=
                options_.minimumObservationConfidence *
                    0.25 &&
            (
                primitive.temporalStability *
                    ageDecay >
                    0.05
            );

        if (retain) {
            survivors.push_back(
                primitive);
        }
    }

    primitives_.swap(
        survivors);

    primitiveCells_.clear();
    primitiveCells_.reserve(
        primitives_.size());

    if (
        primitives_.size() <=
        options_.maximumPrimitives) {
        cells_.clear();
        primitiveCells_.clear();
        primitiveCells_.reserve(
            primitives_.size());

        for (
            std::size_t index = 0;
            index < primitives_.size();
            ++index) {
            const auto cell =
                cellFor(
                    primitives_[index].mean);
            primitiveCells_.push_back(
                cell);
            cells_[cell].push_back(
                index);
        }
        return;
    }

    std::sort(
        primitives_.begin(),
        primitives_.end(),
        [](const GaussianSurfacePrimitive& a,
           const GaussianSurfacePrimitive& b) {
            const double scoreA =
                a.confidence *
                (
                    0.40 +
                    0.60 *
                        a.temporalStability
                ) *
                (
                    1.0 -
                    0.70 *
                        a.dynamicProbability
                );
            const double scoreB =
                b.confidence *
                (
                    0.40 +
                    0.60 *
                        b.temporalStability
                ) *
                (
                    1.0 -
                    0.70 *
                        b.dynamicProbability
                );

            if (scoreA != scoreB) {
                return scoreA > scoreB;
            }

            return
                a.observationCount >
                b.observationCount;
        });

    primitives_.resize(
        options_.maximumPrimitives);

    cells_.clear();
    primitiveCells_.clear();
    primitiveCells_.reserve(
        primitives_.size());

    for (
        std::size_t index = 0;
        index < primitives_.size();
        ++index) {
        const auto cell =
            cellFor(
                primitives_[index].mean);
        primitiveCells_.push_back(
            cell);
        cells_[cell].push_back(
            index);
    }
}

std::vector<GaussianSurfacePrimitive>
TemporalGaussianSurfaceState::snapshot() const {
    return primitives_;
}

GaussianSurfaceStats
TemporalGaussianSurfaceState::stats() const noexcept {
    GaussianSurfaceStats result;

    result.primitiveCount =
        primitives_.size();
    result.fusedObservations =
        static_cast<std::size_t>(
            fusedObservationCount_);
    result.rejectedObservations =
        static_cast<std::size_t>(
            rejectedObservationCount_);

    if (
        primitives_.empty()) {
        return result;
    }

    double confidenceSum = 0.0;
    double uncertaintySum = 0.0;
    double dynamicSum = 0.0;
    double stabilitySum = 0.0;
    std::size_t trusted = 0U;

    for (const auto& primitive :
         primitives_) {
        confidenceSum +=
            clamp01(
                primitive.confidence);

        const double uncertainty =
            std::sqrt(
                std::max(
                    0.0,
                    (
                        primitive.covariance[0] +
                        primitive.covariance[1] +
                        primitive.covariance[2]
                    ) /
                    3.0));

        uncertaintySum +=
            uncertainty;

        dynamicSum +=
            clamp01(
                primitive.dynamicProbability);

        stabilitySum +=
            clamp01(
                primitive.temporalStability);

        if (
            primitive.confidence >= 0.70 &&
            uncertainty <= 0.010 &&
            primitive.dynamicProbability <= 0.35 &&
            primitive.temporalStability >= 0.75) {
            ++trusted;
        }

        if (
            primitive.dynamicProbability >=
            0.55) {
            ++result.dynamicPrimitives;
        }
    }

    const double count =
        static_cast<double>(
            primitives_.size());

    result.meanConfidence =
        confidenceSum /
        count;
    result.meanUncertaintyMeters =
        uncertaintySum /
        count;
    result.meanDynamicProbability =
        dynamicSum /
        count;
    result.temporalStability =
        stabilitySum /
        count;
    result.trustedPrimitiveRatio =
        static_cast<double>(
            trusted) /
        count;

    return result;
}

} // namespace nox::native::scene
