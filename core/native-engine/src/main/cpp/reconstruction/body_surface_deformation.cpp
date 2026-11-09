#include "reconstruction/body_surface_deformation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace nox::native::reconstruction {

namespace {

struct Accumulator {
    double x{0.0};
    double y{0.0};
    double z{0.0};
    double weight{0.0};
};

} // namespace

double BodySurfaceDeformation::clamp01(double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(value, 0.0, 1.0);
}

bool BodySurfaceDeformation::finite(double value) noexcept {
    return std::isfinite(value);
}

double BodySurfaceDeformation::distance(
    const Vertex& a,
    const Vertex& b) noexcept {
    return std::hypot(
        std::hypot(a.x - b.x, a.y - b.y),
        a.z - b.z);
}

double BodySurfaceDeformation::percentile95(
    std::vector<double> values) noexcept {
    values.erase(
        std::remove_if(
            values.begin(),
            values.end(),
            [](double value) {
                return !std::isfinite(value) || value < 0.0;
            }),
        values.end());
    if (values.empty()) return 0.0;

    std::sort(values.begin(), values.end());
    const double position =
        0.95 *
        static_cast<double>(values.size() - 1U);
    const auto lower =
        static_cast<std::size_t>(std::floor(position));
    const auto upper =
        std::min(values.size() - 1U, lower + 1U);
    const double fraction =
        position - static_cast<double>(lower);

    return values[lower] +
        (
            values[upper] -
            values[lower]
        ) *
        fraction;
}

std::vector<Vertex> BodySurfaceDeformation::apply(
    const std::vector<Vertex>& base,
    const std::vector<Displacement>& displacement) const {
    std::vector<Vertex> output = base;
    const std::size_t count =
        std::min(base.size(), displacement.size());

    for (std::size_t i = 0; i < count; ++i) {
        const auto& delta = displacement[i];
        const double confidence =
            clamp01(delta.confidence);
        const double uncertainty =
            std::max(
                0.0005,
                delta.uncertaintyMeters);
        const double precisionWeight =
            1.0 /
            (
                uncertainty *
                uncertainty
            );
        const double normalizedWeight =
            confidence *
            std::clamp(
                precisionWeight / 10'000.0,
                0.10,
                1.0);

        output[i].x +=
            delta.x *
            normalizedWeight;
        output[i].y +=
            delta.y *
            normalizedWeight;
        output[i].z +=
            delta.z *
            normalizedWeight;
    }

    return output;
}

DeformationResult BodySurfaceDeformation::solve(
    const std::vector<Vertex>& base,
    const std::vector<Displacement>& displacement,
    const std::vector<SurfaceEdge>& edges,
    const std::vector<ContactAnchor>& anchors,
    const DeformationOptions& options) const {
    DeformationResult result;

    if (
        base.empty() ||
        displacement.empty() ||
        base.size() != displacement.size() ||
        base.size() > options.maximumVertices) {
        result.vertices = base;
        return result;
    }

    const double dataWeight =
        std::max(0.0, options.dataWeight);
    const double smoothnessWeight =
        std::max(0.0, options.smoothnessWeight);
    const double contactWeight =
        std::max(0.0, options.contactWeight);
    const double strainWeight =
        std::max(0.0, options.strainWeight);
    const double damping =
        std::clamp(options.damping, 0.05, 0.98);
    const double maximumDisplacement =
        std::max(
            options.maximumDisplacementMeters,
            0.0005);
    const double maximumEdgeStrain =
        std::max(
            options.maximumEdgeStrain,
            0.001);

    result.vertices = apply(base, displacement);

    std::vector<double> restLengths;
    restLengths.reserve(edges.size());
    for (const auto& edge : edges) {
        if (
            edge.a >= base.size() ||
            edge.b >= base.size() ||
            edge.a == edge.b) {
            restLengths.push_back(0.0);
            continue;
        }
        restLengths.push_back(
            distance(
                base[edge.a],
                base[edge.b]));
    }

    std::vector<std::vector<std::pair<std::size_t, double>>> adjacency(
        base.size());
    for (std::size_t edgeIndex = 0;
         edgeIndex < edges.size();
         ++edgeIndex) {
        const auto& edge = edges[edgeIndex];
        if (
            edge.a >= base.size() ||
            edge.b >= base.size() ||
            edge.a == edge.b ||
            !finite(restLengths[edgeIndex]) ||
            restLengths[edgeIndex] <= 1e-8) {
            continue;
        }

        const double confidence =
            clamp01(edge.confidence);

        adjacency[edge.a].push_back(
            {edgeIndex, confidence});
        adjacency[edge.b].push_back(
            {edgeIndex, confidence});
    }

    double previousObjective =
        std::numeric_limits<double>::infinity();

    for (
        std::uint32_t iteration = 0;
        iteration < std::max<std::uint32_t>(
            1U,
            options.iterations);
        ++iteration) {
        std::vector<Vertex> proposal =
            result.vertices;

        double objective = 0.0;
        double movement = 0.0;

        for (std::size_t index = 0;
             index < result.vertices.size();
             ++index) {
            const auto& current =
                result.vertices[index];
            const auto& delta =
                displacement[index];

            const double confidence =
                clamp01(delta.confidence);
            const double uncertainty =
                std::max(
                    0.0005,
                    delta.uncertaintyMeters);
            const double precision =
                1.0 /
                (
                    uncertainty *
                    uncertainty
                );
            const double confidenceWeight =
                dataWeight *
                confidence *
                std::clamp(
                    precision / 10'000.0,
                    0.05,
                    1.0);

            Accumulator force{};

            const Vertex target{
                base[index].x + delta.x,
                base[index].y + delta.y,
                base[index].z + delta.z,
            };

            force.x +=
                confidenceWeight *
                (target.x - current.x);
            force.y +=
                confidenceWeight *
                (target.y - current.y);
            force.z +=
                confidenceWeight *
                (target.z - current.z);
            force.weight += confidenceWeight;

            for (const auto& relation :
                 adjacency[index]) {
                const auto edgeIndex =
                    relation.first;
                const double edgeConfidence =
                    relation.second;
                const auto& edge =
                    edges[edgeIndex];

                const std::size_t neighbor =
                    edge.a == index
                        ? edge.b
                        : edge.a;

                if (neighbor >= result.vertices.size()) {
                    continue;
                }

                const auto& neighborVertex =
                    result.vertices[neighbor];
                const Vertex restEndpoint =
                    base[neighbor];

                const double currentLength =
                    distance(
                        current,
                        neighborVertex);
                const double restLength =
                    restLengths[edgeIndex];

                if (
                    !finite(currentLength) ||
                    !finite(restLength) ||
                    restLength <= 1e-8) {
                    continue;
                }

                const double lengthResidual =
                    currentLength - restLength;
                const double normalizedStrain =
                    lengthResidual /
                    restLength;

                const double strainMagnitude =
                    std::abs(normalizedStrain);

                double regularization =
                    smoothnessWeight *
                    edgeConfidence;

                if (
                    strainMagnitude >
                    maximumEdgeStrain) {
                    regularization +=
                        strainWeight *
                        std::clamp(
                            (
                                strainMagnitude -
                                maximumEdgeStrain
                            ) /
                            maximumEdgeStrain,
                            0.0,
                            4.0);
                }

                const double safeLength =
                    std::max(
                        currentLength,
                        1e-8);

                force.x +=
                    regularization *
                    (
                        neighborVertex.x -
                        current.x
                    ) *
                    (
                        lengthResidual /
                        safeLength
                    );
                force.y +=
                    regularization *
                    (
                        neighborVertex.y -
                        current.y
                    ) *
                    (
                        lengthResidual /
                        safeLength
                    );
                force.z +=
                    regularization *
                    (
                        neighborVertex.z -
                        current.z
                    ) *
                    (
                        lengthResidual /
                        safeLength
                    );
                force.weight += regularization;

                objective +=
                    regularization *
                    lengthResidual *
                    lengthResidual;

                const double baseLength =
                    distance(
                        base[index],
                        restEndpoint);
                (void)baseLength;
            }

            if (
                force.weight >
                1e-8) {
                const double step =
                    damping /
                    force.weight;
                proposal[index].x =
                    current.x +
                    std::clamp(
                        force.x * step,
                        -maximumDisplacement,
                        maximumDisplacement);
                proposal[index].y =
                    current.y +
                    std::clamp(
                        force.y * step,
                        -maximumDisplacement,
                        maximumDisplacement);
                proposal[index].z =
                    current.z +
                    std::clamp(
                        force.z * step,
                        -maximumDisplacement,
                        maximumDisplacement);
            }
        }

        for (const auto& anchor : anchors) {
            if (anchor.vertex >= proposal.size()) {
                continue;
            }

            const double confidence =
                clamp01(anchor.confidence);
            const double stiffness =
                std::max(0.0, anchor.stiffness);
            const double weight =
                contactWeight *
                confidence *
                stiffness;

            if (weight <= 0.0) continue;

            auto& vertex =
                proposal[anchor.vertex];

            vertex.x +=
                (
                    anchor.x -
                    vertex.x
                ) *
                std::clamp(
                    weight * damping,
                    0.0,
                    0.85);
            vertex.y +=
                (
                    anchor.y -
                    vertex.y
                ) *
                std::clamp(
                    weight * damping,
                    0.0,
                    0.85);
            vertex.z +=
                (
                    anchor.z -
                    vertex.z
                ) *
                std::clamp(
                    weight * damping,
                    0.0,
                    0.85);
        }

        for (std::size_t index = 0;
             index < proposal.size();
             ++index) {
            const double dx =
                proposal[index].x -
                base[index].x;
            const double dy =
                proposal[index].y -
                base[index].y;
            const double dz =
                proposal[index].z -
                base[index].z;

            const double displacementMagnitude =
                std::hypot(
                    std::hypot(dx, dy),
                    dz);

            if (
                displacementMagnitude >
                maximumDisplacement) {
                const double scale =
                    maximumDisplacement /
                    displacementMagnitude;
                proposal[index].x =
                    base[index].x +
                    dx * scale;
                proposal[index].y =
                    base[index].y +
                    dy * scale;
                proposal[index].z =
                    base[index].z +
                    dz * scale;
            }

            movement +=
                distance(
                    result.vertices[index],
                    proposal[index]);

            const double confidence =
                clamp01(
                    displacement[index].confidence);
            objective +=
                dataWeight *
                confidence *
                (
                    dx * dx +
                    dy * dy +
                    dz * dz
                );
        }

        result.vertices.swap(proposal);
        result.iterations = iteration + 1U;

        const double safeObjective =
            finite(objective)
                ? objective
                : std::numeric_limits<double>::infinity();

        if (
            movement <=
            options.convergenceToleranceMeters ||
            safeObjective >= previousObjective) {
            result.converged =
                movement <=
                options.convergenceToleranceMeters;
            result.objective =
                safeObjective;
            break;
        }

        previousObjective = safeObjective;
        result.objective =
            safeObjective;
    }

    std::vector<double> displacementMagnitudes;
    displacementMagnitudes.reserve(
        result.vertices.size());

    double displacementSum = 0.0;
    double confidenceSum = 0.0;
    for (std::size_t index = 0;
         index < result.vertices.size();
         ++index) {
        const double magnitude =
            distance(
                base[index],
                result.vertices[index]);
        if (finite(magnitude)) {
            displacementMagnitudes.push_back(
                magnitude);
            displacementSum += magnitude;
        }
        confidenceSum +=
            clamp01(
                displacement[index].confidence);
    }

    double edgeStrainSum = 0.0;
    double edgeStrainWeight = 0.0;
    double maxEdgeStrain = 0.0;

    for (std::size_t edgeIndex = 0;
         edgeIndex < edges.size();
         ++edgeIndex) {
        const auto& edge =
            edges[edgeIndex];
        if (
            edge.a >= result.vertices.size() ||
            edge.b >= result.vertices.size() ||
            edge.a == edge.b) {
            continue;
        }

        const double restLength =
            restLengths[edgeIndex];
        if (
            !finite(restLength) ||
            restLength <= 1e-8) {
            continue;
        }

        const double currentLength =
            distance(
                result.vertices[edge.a],
                result.vertices[edge.b]);

        if (!finite(currentLength)) continue;

        const double strain =
            std::abs(
                currentLength -
                restLength) /
            restLength;

        edgeStrainSum +=
            strain *
            clamp01(edge.confidence);
        edgeStrainWeight +=
            clamp01(edge.confidence);
        maxEdgeStrain =
            std::max(
                maxEdgeStrain,
                strain);
    }

    const double count =
        static_cast<double>(
            result.vertices.size());

    result.meanDisplacementMeters =
        count <= 0.0
            ? 0.0
            : displacementSum / count;
    result.p95DisplacementMeters =
        percentile95(
            displacementMagnitudes);
    result.maxDisplacementMeters =
        displacementMagnitudes.empty()
            ? 0.0
            : *std::max_element(
                displacementMagnitudes.begin(),
                displacementMagnitudes.end());
    result.meanEdgeStrain =
        edgeStrainWeight <= 0.0
            ? 0.0
            : edgeStrainSum /
                edgeStrainWeight;
    result.maxEdgeStrain =
        maxEdgeStrain;
    result.confidence =
        count <= 0.0
            ? 0.0
            : clamp01(
                confidenceSum / count);

    const bool strainAccepted =
        maxEdgeStrain <=
        maximumEdgeStrain * 1.5;
    const bool displacementAccepted =
        result.p95DisplacementMeters <=
        maximumDisplacement;
    const bool finiteResult =
        std::all_of(
            result.vertices.begin(),
            result.vertices.end(),
            [](const Vertex& vertex) {
                return
                    std::isfinite(vertex.x) &&
                    std::isfinite(vertex.y) &&
                    std::isfinite(vertex.z);
            });

    result.accepted =
        finiteResult &&
        strainAccepted &&
        displacementAccepted &&
        result.confidence >= 0.10 &&
        (result.converged || result.iterations >= 1U);

    return result;
}

} // namespace nox::native::reconstruction
