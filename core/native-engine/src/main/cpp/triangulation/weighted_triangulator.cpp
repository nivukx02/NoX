#include "triangulation/weighted_triangulator.h"
#include "triangulation/reprojection_consensus.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace nox::native::triangulation {

namespace {
double dot(Vec3 a, Vec3 b) noexcept {
    return a.x*b.x+a.y*b.y+a.z*b.z;
}

bool validDirection(Vec3 v) noexcept {
    const double n2=dot(v,v);
    return std::isfinite(n2) && n2>1e-16;
}

Vec3 normalize(Vec3 v) noexcept {
    const double n=std::sqrt(dot(v,v));
    if(!std::isfinite(n) || n<1e-8) {
        return {0.0,0.0,0.0};
    }
    return {v.x/n,v.y/n,v.z/n};
}

bool solve3(double matrix[3][4], double x[3]) {
    double matrixScale = 0.0;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            matrixScale = std::max(
                matrixScale,
                std::abs(matrix[row][column]));
        }
    }
    const double pivotFloor =
        std::max(
            1e-14,
            matrixScale * 1e-10);

    for(int column=0;column<3;++column){
        int pivot=column;
        for(int row=column+1;row<3;++row){
            if(std::abs(matrix[row][column])>
               std::abs(matrix[pivot][column])){
                pivot=row;
            }
        }
        if(std::abs(matrix[pivot][column])<pivotFloor)return false;
        if(pivot!=column){
            for(int k=column;k<4;++k){
                std::swap(matrix[pivot][k],matrix[column][k]);
            }
        }
        for(int row=column+1;row<3;++row){
            const double factor=
                matrix[row][column]/matrix[column][column];
            for(int k=column;k<4;++k){
                matrix[row][k]-=factor*matrix[column][k];
            }
        }
    }
    for(int row=2;row>=0;--row){
        double value=matrix[row][3];
        for(int k=row+1;k<3;++k){
            value-=matrix[row][k]*x[k];
        }
        x[row]=value/matrix[row][row];
    }
    return true;
}

bool invert3(
    const double input[3][3],
    double inverse[3][3]) noexcept {
    double inputScale = 0.0;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            inputScale = std::max(
                inputScale,
                std::abs(input[row][col]));
        }
    }
    const double pivotFloor =
        std::max(
            1e-14,
            inputScale * 1e-10);

    double augmented[3][7]{};
    for(int row=0;row<3;++row){
        for(int col=0;col<3;++col){
            augmented[row][col]=input[row][col];
        }
        augmented[row][row+3]=1.0;
    }

    for(int column=0;column<3;++column){
        int pivot=column;
        for(int row=column+1;row<3;++row){
            if(std::abs(augmented[row][column])>
               std::abs(augmented[pivot][column])){
                pivot=row;
            }
        }
        if(std::abs(augmented[pivot][column])<pivotFloor)return false;
        if(pivot!=column){
            for(int col=0;col<6;++col){
                std::swap(augmented[pivot][col],augmented[column][col]);
            }
        }
        const double divisor=augmented[column][column];
        for(int col=0;col<6;++col){
            augmented[column][col]/=divisor;
        }
        for(int row=0;row<3;++row){
            if(row==column)continue;
            const double factor=augmented[row][column];
            for(int col=0;col<6;++col){
                augmented[row][col]-=factor*augmented[column][col];
            }
        }
    }

    for(int row=0;row<3;++row){
        for(int col=0;col<3;++col){
            inverse[row][col]=augmented[row][col+3];
        }
    }
    return true;
}
}


namespace {

double rayWeight(const RayObservation& ray) noexcept {
    if (!std::isfinite(ray.weight) ||
        !std::isfinite(ray.uncertaintyScale) ||
        !std::isfinite(ray.rollingShutterConfidence) ||
        !std::isfinite(ray.metricScaleConfidence) ||
        !std::isfinite(ray.temporalAlignmentConfidence) ||
        !std::isfinite(ray.crossViewIndependence) ||
        !std::isfinite(ray.depthConditioningConfidence) ||
        ray.weight <= 0.0) {
        return 0.0;
    }

    const double shutterTrust =
        std::clamp(
            ray.rollingShutterConfidence,
            0.05,
            1.0);
    const double scaleTrust =
        std::clamp(
            ray.metricScaleConfidence,
            0.05,
            1.0);
    const double temporalTrust =
        std::clamp(
            ray.temporalAlignmentConfidence,
            0.05,
            1.0);
    const double independenceTrust =
        std::clamp(
            ray.crossViewIndependence,
            0.05,
            1.0);
    const double conditioningTrust =
        std::clamp(
            ray.depthConditioningConfidence,
            0.05,
            1.0);

    return ray.weight *
        shutterTrust *
        scaleTrust *
        temporalTrust *
        independenceTrust *
        conditioningTrust /
        std::max(
            ray.uncertaintyScale * ray.uncertaintyScale,
            0.05 * 0.05);
}

double rayResidual(
    const Vec3& point,
    const RayObservation& ray) noexcept {
    const auto direction = normalize(ray.direction);
    const Vec3 q{
        point.x - ray.origin.x,
        point.y - ray.origin.y,
        point.z - ray.origin.z,
    };
    const double axial = dot(q, direction);
    const Vec3 perpendicular{
        q.x - direction.x * axial,
        q.y - direction.y * axial,
        q.z - direction.z * axial,
    };
    return std::sqrt(
        std::max(
            0.0,
            dot(perpendicular, perpendicular)));
}

double huberWeight(
    double residual,
    double scale) noexcept {
    if (!std::isfinite(residual)) {
        return 0.0;
    }
    const double safeScale =
        std::max(scale, 1e-6);
    const double threshold =
        safeScale * 2.5;
    if (residual <= threshold) {
        return 1.0;
    }
    return threshold /
        std::max(residual, threshold);
}

double tukeyWeight(
    double residual,
    double scale) noexcept {
    if (!std::isfinite(residual)) {
        return 0.0;
    }
    const double safeScale =
        std::max(scale, 1e-6);
    const double normalized =
        residual /
        (safeScale * 4.685);
    if (normalized >= 1.0) {
        return 0.0;
    }
    const double oneMinus =
        1.0 - normalized * normalized;
    return oneMinus * oneMinus;
}

double adaptiveRobustWeight(
    double residual,
    double scale) noexcept {
    const double huber =
        huberWeight(residual, scale);
    const double tukey =
        tukeyWeight(residual, scale);
    if (!std::isfinite(huber) ||
        !std::isfinite(tukey)) {
        return 0.0;
    }
    return std::clamp(
        0.70 * huber + 0.30 * tukey,
        0.02,
        1.0);
}

bool solvePoint(
    const std::vector<RayObservation>& rays,
    const std::vector<double>& robustWeights,
    double point[3],
    double H[3][3],
    double& totalWeight) {
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            H[row][col] = 0.0;
        }
        point[row] = 0.0;
    }
    double rhs[3]{};
    totalWeight = 0.0;

    for (std::size_t index = 0;
         index < rays.size();
         ++index) {
        const auto& ray = rays[index];
        const auto direction = normalize(ray.direction);
        const double baseWeight = rayWeight(ray);
        const double robust =
            index < robustWeights.size()
                ? std::clamp(
                    robustWeights[index],
                    0.0,
                    1.0)
                : 1.0;
        const double weight =
            baseWeight * robust;

        if (weight <= 0.0) {
            continue;
        }

        const double P[3][3] = {
            {
                1.0 - direction.x * direction.x,
                -direction.x * direction.y,
                -direction.x * direction.z,
            },
            {
                -direction.y * direction.x,
                1.0 - direction.y * direction.y,
                -direction.y * direction.z,
            },
            {
                -direction.z * direction.x,
                -direction.z * direction.y,
                1.0 - direction.z * direction.z,
            },
        };

        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                H[row][col] +=
                    weight * P[row][col];
            }
            rhs[row] += weight * (
                P[row][0] * ray.origin.x +
                P[row][1] * ray.origin.y +
                P[row][2] * ray.origin.z);
        }
        totalWeight += weight;
    }

    const double trace =
        H[0][0] +
        H[1][1] +
        H[2][2];

    if (!std::isfinite(trace) ||
        trace < 1e-8 ||
        totalWeight <= 0.0) {
        return false;
    }

    double augmented[3][4] = {
        {H[0][0], H[0][1], H[0][2], rhs[0]},
        {H[1][0], H[1][1], H[1][2], rhs[1]},
        {H[2][0], H[2][1], H[2][2], rhs[2]},
    };

    double x[3]{};
    if (!solve3(augmented, x)) {
        return false;
    }

    point[0] = x[0];
    point[1] = x[1];
    point[2] = x[2];
    return true;
}

struct AngularGeometry {
    double minimumRadians{0.0};
    double meanRadians{0.0};
    double baselineMeters{0.0};
};

AngularGeometry angularGeometry(
    const std::vector<RayObservation>& rays) noexcept {
    constexpr double kPi = 3.14159265358979323846;
    double minimum = std::numeric_limits<double>::infinity();
    double sum = 0.0;
    double maximumBaseline = 0.0;
    std::size_t count = 0;

    for (std::size_t i = 0; i < rays.size(); ++i) {
        const auto a = normalize(rays[i].direction);

        for (std::size_t j = i + 1; j < rays.size(); ++j) {
            const auto b = normalize(rays[j].direction);
            const double cosine =
                std::clamp(
                    dot(a, b),
                    -1.0,
                    1.0);

            const double angle =
                std::acos(cosine);

            if(angle <= 1e-4) continue;

            const Vec3 delta{
                rays[i].origin.x - rays[j].origin.x,
                rays[i].origin.y - rays[j].origin.y,
                rays[i].origin.z - rays[j].origin.z,
            };

            const double baseline =
                std::sqrt(
                    std::max(
                        0.0,
                        dot(delta, delta)));

            minimum =
                std::min(
                    minimum,
                    angle);
            sum += angle;
            maximumBaseline =
                std::max(
                    maximumBaseline,
                    baseline);
            ++count;
        }
    }

    return {
        std::isfinite(minimum)
            ? std::clamp(minimum, 0.0, kPi)
            : 0.0,
        count == 0
            ? 0.0
            : std::clamp(
                sum /
                    static_cast<double>(count),
                0.0,
                kPi),
        maximumBaseline,
    };
}

double metricScaleConfidence(
    const std::vector<RayObservation>& rays,
    const AngularGeometry& geometry) noexcept {
    if (rays.empty()) return 0.0;

    double total = 0.0;
    double weight = 0.0;

    for (const auto& ray : rays) {
        const double w =
            std::clamp(
                ray.weight *
                    ray.metricScaleConfidence *
                    ray.rollingShutterConfidence,
                0.0,
                1.0);

        total += w;
        weight += 1.0;
    }

    const double mean =
        weight <= 0.0
            ? 0.0
            : total / weight;

    const double baselineScore =
        std::clamp(
            geometry.baselineMeters / 0.50,
            0.0,
            1.0);

    const double parallaxScore =
        std::clamp(
            geometry.minimumRadians / 0.12,
            0.0,
            1.0);

    return std::clamp(
        mean * 0.44 +
            baselineScore * 0.26 +
            parallaxScore * 0.30,
        0.0,
        1.0);
}

double angularDiversity(
    const std::vector<RayObservation>& rays) noexcept {
    double maximum = 0.0;
    for (std::size_t i = 0; i < rays.size(); ++i) {
        const auto a = normalize(rays[i].direction);
        for (std::size_t j = i + 1; j < rays.size(); ++j) {
            const auto b = normalize(rays[j].direction);
            maximum = std::max(
                maximum,
                std::sqrt(
                    std::max(
                        0.0,
                        1.0 -
                            dot(a, b) *
                            dot(a, b))));
        }
    }
    return std::clamp(maximum, 0.0, 1.0);
}

double median(
    std::vector<double> values) noexcept {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const std::size_t middle =
        values.size() / 2U;
    if ((values.size() & 1U) != 0U) {
        return values[middle];
    }
    return (
        values[middle - 1U] +
        values[middle]
    ) * 0.5;
}

} // namespace

TriangulationResult WeightedTriangulator::solve(
    const std::vector<RayObservation>& inputRays) const {
    TriangulationResult output;

    bool rejectedInvalidInput = false;

    std::vector<RayObservation> rays;
    rays.reserve(inputRays.size());

    for (const auto& ray : inputRays) {
        if (!validDirection(ray.direction)) {
            rejectedInvalidInput = true;
            continue;
        }
        const auto direction = normalize(ray.direction);
        const double baseWeight = rayWeight(ray);
        if (baseWeight <= 0.0) {
            continue;
        }
        if (!std::isfinite(ray.origin.x) ||
            !std::isfinite(ray.origin.y) ||
            !std::isfinite(ray.origin.z) ||
            !std::isfinite(direction.x) ||
            !std::isfinite(direction.y) ||
            !std::isfinite(direction.z)) {
            rejectedInvalidInput = true;
            continue;
        }
        rays.push_back(ray);
    }

    const auto geometry =
        angularGeometry(rays);

    std::unordered_map<std::uint32_t, std::size_t> cameraUseCount;
    cameraUseCount.reserve(rays.size());
    bool cameraIdentityKnown = false;
    for (const auto& ray : rays) {
        ++cameraUseCount[ray.cameraId];
        if (ray.cameraId != 0U) {
            cameraIdentityKnown = true;
        }
    }

    const std::size_t distinctKnownCameras =
        cameraIdentityKnown
            ? std::count_if(
                cameraUseCount.begin(),
                cameraUseCount.end(),
                [](const auto& entry) {
                    return entry.first != 0U &&
                        entry.second > 0U;
                })
            : 0U;

    const double geometricConditioningConfidence =
        std::clamp(
            geometry.minimumRadians / 0.10,
            0.05,
            1.0) *
        std::clamp(
            geometry.baselineMeters / 0.40,
            0.05,
            1.0);

    for (auto& ray : rays) {
        const auto iterator =
            cameraUseCount.find(ray.cameraId);
        const std::size_t count =
            iterator == cameraUseCount.end()
                ? 1U
                : iterator->second;
        ray.crossViewIndependence =
            std::clamp(
                1.0 /
                    std::sqrt(
                        static_cast<double>(
                            std::max<std::size_t>(
                                count,
                                1U))),
                0.35,
                1.0);
        ray.depthConditioningConfidence =
            geometricConditioningConfidence;
    }

    output.minimumAngularSeparationRadians =
        geometry.minimumRadians;
    output.meanAngularSeparationRadians =
        geometry.meanRadians;
    output.baselineMeters =
        geometry.baselineMeters;
    output.metricScaleConfidence =
        metricScaleConfidence(
            rays,
            geometry);

    output.angularDiversity =
        angularDiversity(rays);
    output.cameraIdentityUnknown =
        !cameraIdentityKnown;
    output.distinctCameraCount =
        static_cast<std::uint32_t>(
            cameraIdentityKnown
                ? distinctKnownCameras
                : 1U);
    output.viewIndependenceScore =
        cameraIdentityKnown
            ? std::clamp(
                static_cast<double>(distinctKnownCameras) /
                    static_cast<double>(
                        std::min<std::size_t>(
                            rays.size(),
                            4U)),
                0.0,
                1.0)
            : 0.50;

    if (rays.size() < 2 ||
        output.angularDiversity < 1e-3 ||
        output.meanAngularSeparationRadians <= 1e-4) {
        return output;
    }

    std::vector<double> robustWeights(
        rays.size(),
        1.0);

    double point[3]{};
    double H[3][3]{};
    double totalWeight = 0.0;

    for (int iteration = 0;
         iteration < 6;
         ++iteration) {
        if (!solvePoint(
                rays,
                robustWeights,
                point,
                H,
                totalWeight)) {
            return output;
        }

        std::vector<double> residuals;
        residuals.reserve(rays.size());
        for (const auto& ray : rays) {
            residuals.push_back(
                rayResidual(
                    {point[0], point[1], point[2]},
                    ray));
        }

        const double residualMedian =
            median(residuals);

        std::vector<double> deviations;
        deviations.reserve(residuals.size());
        for (const double residual : residuals) {
            deviations.push_back(
                std::abs(
                    residual -
                    residualMedian));
        }

        const double mad =
            median(deviations);
        const double robustScale =
            std::max(
                1e-5,
                1.4826 * mad +
                    residualMedian * 0.25);

        for (std::size_t index = 0;
             index < rays.size();
             ++index) {
            robustWeights[index] =
                adaptiveRobustWeight(
                    residuals[index],
                    robustScale);
        }
    }

    if (!solvePoint(
            rays,
            robustWeights,
            point,
            H,
            totalWeight)) {
        return output;
    }

    output.point = {
        point[0],
        point[1],
        point[2],
    };

    std::vector<double> residuals;
    residuals.reserve(rays.size());

    double weightedSquaredResidual = 0.0;
    std::size_t inlierCount = 0;

    for (std::size_t index = 0;
         index < rays.size();
         ++index) {
        const auto& ray = rays[index];
        const double residual =
            rayResidual(
                output.point,
                ray);
        residuals.push_back(residual);

        const double weight =
            rayWeight(ray) *
            std::clamp(
                robustWeights[index],
                0.0,
                1.0);

        // Each observation contributes exactly once to the fallback weighted residual budget.
        weightedSquaredResidual +=
            weight *
            residual *
            residual;

        if (robustWeights[index] >= 0.20) {
            ++inlierCount;
        }
    }

    output.contributingRays =
        static_cast<std::uint32_t>(
            inlierCount);
    output.rejectedRays =
        static_cast<std::uint32_t>(
            rays.size() -
            inlierCount);

    output.robustInlierRatio =
        static_cast<double>(inlierCount) /
        static_cast<double>(rays.size());

    double robustSquaredResidual = 0.0;
    double robustWeightSum = 0.0;
    for (std::size_t index = 0;
         index < rays.size();
         ++index) {
        const double robust =
            std::clamp(
                robustWeights[index],
                0.0,
                1.0);
        if (robust < 0.20) {
            continue;
        }
        const double baseWeight =
            rayWeight(rays[index]);
        robustSquaredResidual +=
            baseWeight *
            robust *
            residuals[index] *
            residuals[index];
        robustWeightSum +=
            baseWeight *
            robust;
    }

    output.reprojectionRms =
        robustWeightSum > 1e-9
            ? std::sqrt(
                robustSquaredResidual /
                robustWeightSum)
            : std::sqrt(
                weightedSquaredResidual /
                std::max(totalWeight, 1e-9));

    output.residualScale =
        median(residuals);

    const ReprojectionConsensus reprojectionConsensus;
    const auto consensus =
        reprojectionConsensus.evaluate(
            residuals,
            3.0);

    output.reprojectionConsensusConfidence =
        consensus.confidence;
    output.reprojectionConsensusMedian =
        consensus.median;
    output.reprojectionConsensusMad =
        consensus.mad;

    const double trace =
        H[0][0] +
        H[1][1] +
        H[2][2];

    output.conditioning =
        std::clamp(
            output.angularDiversity *
                std::sqrt(
                    std::max(
                        trace /
                            std::max(
                                totalWeight,
                                1e-9),
                        0.0)),
            0.0,
            1.0);

    const double robustObservationVariance =
        robustWeightSum > 1e-9
            ? robustSquaredResidual /
                std::max(
                    robustWeightSum -
                        3.0,
                    1.0)
            : weightedSquaredResidual /
                std::max(
                    totalWeight -
                        3.0,
                    1.0);

    const double observationVariance =
        std::max(
            1e-10,
            robustObservationVariance);

    double inverseH[3][3]{};
    if (invert3(H, inverseH)) {
        output.covarianceMatrix = {
            inverseH[0][0] * observationVariance,
            inverseH[0][1] * observationVariance,
            inverseH[0][2] * observationVariance,
            inverseH[1][0] * observationVariance,
            inverseH[1][1] * observationVariance,
            inverseH[1][2] * observationVariance,
            inverseH[2][0] * observationVariance,
            inverseH[2][1] * observationVariance,
            inverseH[2][2] * observationVariance,
        };

        output.covarianceTrace =
            output.covarianceMatrix[0] +
            output.covarianceMatrix[4] +
            output.covarianceMatrix[8];

        output.covarianceP95 =
            std::sqrt(
                std::max(
                    0.0,
                    output.covarianceTrace)) *
            1.96;

        output.covariance =
            output.covarianceTrace /
            3.0;
    }

    std::size_t cheiralityCount = 0;
    for (const auto& ray : rays) {
        const auto direction =
            normalize(ray.direction);
        const Vec3 fromOrigin{
            output.point.x - ray.origin.x,
            output.point.y - ray.origin.y,
            output.point.z - ray.origin.z,
        };
        const double axial =
            dot(fromOrigin, direction);

        if (axial > 1e-5) {
            ++cheiralityCount;
        }
    }

    output.cheiralityConfidence =
        static_cast<double>(cheiralityCount) /
        static_cast<double>(rays.size());

    const double residualConfidence =
        std::exp(
            -output.reprojectionRms /
            0.025);

    const double covarianceConfidence =
        std::clamp(
            1.0 /
                (1.0 +
                    output.covarianceP95 *
                    4.0),
            0.0,
            1.0);

    output.depthConditioning =
        std::clamp(
            output.minimumAngularSeparationRadians /
                0.18,
            0.0,
            1.0) *
        std::clamp(
            output.baselineMeters /
                0.50,
            0.0,
            1.0);

    const double evidenceConfidence =
        residualConfidence *
        std::pow(
            output.conditioning,
            0.65) *
        std::pow(
            output.robustInlierRatio,
            0.70) *
        std::pow(
            output.cheiralityConfidence,
            0.70) *
        std::pow(
            output.metricScaleConfidence,
            0.55) *
        std::pow(
            std::max(
                output.reprojectionConsensusConfidence,
                1e-6),
            0.55) *
        std::pow(
            std::max(
                output.depthConditioning,
                1e-6),
            0.45) *
        std::pow(
            std::max(
                output.viewIndependenceScore,
                1e-6),
            0.35) *
        covarianceConfidence;

    output.confidence =
        std::clamp(
            std::max(
                evidenceConfidence,
                0.18 *
                    std::sqrt(
                        std::clamp(
                            output.robustInlierRatio *
                                output.cheiralityConfidence,
                            0.0,
                            1.0))),
            0.0,
            1.0);

    const bool robustConsensusSatisfied =
        output.reprojectionConsensusConfidence >= 0.55 ||
        (
            output.rejectedRays > 0U &&
            output.robustInlierRatio >= 0.70 &&
            output.reprojectionRms < 0.10
        );

    output.valid =
        output.contributingRays >= 2 &&
        output.robustInlierRatio >=
            (rays.size() >= 3 ? 0.50 : 0.95) &&
        output.angularDiversity >= 0.05 &&
        output.conditioning > 1e-5 &&
        !rejectedInvalidInput &&
        output.cheiralityConfidence >= 0.999 &&
        output.reprojectionRms < 0.10 &&
        output.metricScaleConfidence >= 0.20 &&
        robustConsensusSatisfied &&
        output.depthConditioning >= 0.001 &&
        (!cameraIdentityKnown ||
            output.distinctCameraCount >= 2U) &&
        output.confidence >= 0.05;


    return output;
}
} // namespace nox::native::triangulation
