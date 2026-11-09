#include "calibration/global_camera_graph_optimizer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include "math/robust_loss.h"
#include "calibration/minimum_spanning_camera_tree.h"
#include "calibration/graph_cycle_validator.h"

namespace nox::native::calibration {
namespace {
math::SE3d toSE3(const GraphPose& pose) {
    return {
        pose.rotation,
        {pose.x, pose.y, pose.z},
    };
}

GraphPose fromSE3(const math::SE3d& pose) {
    return {
        pose.translation.x,
        pose.translation.y,
        pose.translation.z,
        pose.rotation,
    };
}
}

GraphOptimizationResult GlobalCameraGraphOptimizer::solve(
    const std::vector<std::string>& cameraIds,
    const std::vector<GraphEdge>& edges,
    std::size_t iterations,
    double step) const {
    if (
        cameraIds.empty() ||
        cameraIds.size() >
            GlobalCameraGraphOptimizer::kMaximumCameras ||
        edges.size() >
            GlobalCameraGraphOptimizer::kMaximumEdges
    ) {
        return {};
    }

    std::unordered_map<std::string, std::size_t> ids;
    ids.reserve(cameraIds.size());

    for (std::size_t index = 0; index < cameraIds.size(); ++index) {
        if (
            cameraIds[index].empty() ||
            !ids.emplace(
                cameraIds[index],
                index).second
        ) {
            return {};
        }
    }

    GraphOptimizationResult output;
    output.poses.resize(cameraIds.size());

    if (edges.empty()) {
        return output;
    }

    for (const auto& edge : edges) {
        if (
            ids.find(edge.from) == ids.end() ||
            ids.find(edge.to) == ids.end() ||
            edge.from == edge.to
        ) {
            continue;
        }
        if (
            !std::isfinite(edge.weight) ||
            edge.weight < 0.0 ||
            !std::isfinite(edge.translationSigma) ||
            edge.translationSigma <= 0.0 ||
            !std::isfinite(edge.rotationSigma) ||
            edge.rotationSigma <= 0.0
        ) {
            return {};
        }
    }

    const auto tree =
        MinimumSpanningCameraTree{}.solve(
            cameraIds,
            edges);

    output.initializationEdges =
        tree.selectedEdgeIndices.size();
    output.connected = tree.connected;
    output.disconnectedCameraCount =
        cameraIds.size() >
            output.initializationEdges
                ? cameraIds.size() -
                    output.initializationEdges -
                    1U
                : 0U;

    if (!tree.connected) {
        output.confidence = 0.0;
        output.converged = false;
        return output;
    }

    std::vector<std::vector<std::pair<std::size_t, std::size_t>>> adjacency(
        cameraIds.size());

    for (const auto edgeIndex : tree.selectedEdgeIndices) {
        const auto& edge = edges[edgeIndex];
        const auto from = ids.find(edge.from);
        const auto to = ids.find(edge.to);
        if (from == ids.end() || to == ids.end()) continue;
        adjacency[from->second].push_back(
            {to->second, edgeIndex});
        adjacency[to->second].push_back(
            {from->second, edgeIndex});
    }

    std::vector<bool> visited(
        cameraIds.size(),
        false);
    std::vector<std::size_t> queue;
    queue.reserve(cameraIds.size());

    visited[0] = true;
    queue.push_back(0);

    for (
        std::size_t cursor = 0;
        cursor < queue.size();
        ++cursor
    ) {
        const auto current = queue[cursor];
        const auto currentPose =
            toSE3(output.poses[current]);

        for (
            const auto& [next, edgeIndex] :
                adjacency[current]
        ) {
            if (visited[next]) continue;

            const auto& edge = edges[edgeIndex];
            const bool forward =
                edge.from == cameraIds[current];
            const auto relative =
                toSE3(edge.relative);

            output.poses[next] =
                fromSE3(
                    forward
                        ? currentPose.compose(relative)
                        : currentPose.compose(
                            relative.inverse()));

            visited[next] = true;
            queue.push_back(next);
        }
    }

    const auto initialCycle =
        GraphCycleValidator{}.evaluate(
            output.poses,
            cameraIds,
            edges);

    output.cycleConfidence =
        initialCycle.confidence;
    output.cycleFailures =
        initialCycle.inconsistentEdges;
    output.residual =
        std::hypot(
            initialCycle.rmsTranslation,
            initialCycle.rmsRotation);

    const std::size_t safeIterations =
        std::clamp(
            iterations,
            std::size_t{1},
            std::size_t{512});

    const double safeInitialStep =
        std::clamp(
            step,
            0.01,
            0.75);

    for (
        std::size_t iteration = 0;
        iteration < safeIterations;
        ++iteration
    ) {
        std::vector<math::Twist6d> accumulated(
            cameraIds.size());
        std::vector<double> accumulatedWeight(
            cameraIds.size(),
            0.0);

        double weightedSquared = 0.0;
        double totalWeight = 0.0;

        for (const auto& edge : edges) {
            const auto fromIt = ids.find(edge.from);
            const auto toIt = ids.find(edge.to);

            if (
                fromIt == ids.end() ||
                toIt == ids.end() ||
                fromIt->second == toIt->second
            ) {
                continue;
            }

            const auto fromIndex = fromIt->second;
            const auto toIndex = toIt->second;
            const auto fromPose =
                toSE3(output.poses[fromIndex]);
            const auto target =
                toSE3(output.poses[toIndex]);
            const auto relative =
                toSE3(edge.relative);

            const auto predicted =
                fromPose.compose(relative);
            const auto residual =
                predicted.inverse().compose(target);
            const auto twist = residual.log();

            const double translationNorm =
                std::sqrt(
                    twist.translation.x * twist.translation.x +
                    twist.translation.y * twist.translation.y +
                    twist.translation.z * twist.translation.z);

            const double rotationNorm =
                std::sqrt(
                    twist.rotation.x * twist.rotation.x +
                    twist.rotation.y * twist.rotation.y +
                    twist.rotation.z * twist.rotation.z);

            const double normalizedTranslation =
                translationNorm /
                std::max(edge.translationSigma, 0.001);
            const double normalizedRotation =
                rotationNorm /
                std::max(edge.rotationSigma, 0.001);
            const double normalizedResidual =
                std::hypot(
                    normalizedTranslation,
                    normalizedRotation);

            const double robustWeight =
                std::clamp(edge.weight, 0.0, 100.0) *
                math::RobustLoss::weight(
                    math::RobustLossKind::Huber,
                    normalizedResidual,
                    1.0);

            if (robustWeight <= 0.0) continue;

            weightedSquared +=
                robustWeight *
                normalizedResidual *
                normalizedResidual;
            totalWeight += robustWeight;

            const double translationDamping =
                std::clamp(
                    std::exp(-normalizedTranslation),
                    0.05,
                    1.0);
            const double rotationDamping =
                std::clamp(
                    std::exp(-normalizedRotation),
                    0.05,
                    1.0);

            const double stepScale =
                safeInitialStep *
                std::pow(
                    0.985,
                    static_cast<double>(iteration));

            const math::Twist6d correction{
                twist.rotation *
                    (-stepScale * robustWeight *
                        rotationDamping),
                twist.translation *
                    (-stepScale * robustWeight *
                        translationDamping),
            };

            const double normalizedWeight =
                std::clamp(
                    robustWeight,
                    0.0,
                    1.0);

            if (toIndex != 0u) {
                accumulated[toIndex].rotation =
                    accumulated[toIndex].rotation +
                    correction.rotation *
                        normalizedWeight;
                accumulated[toIndex].translation =
                    accumulated[toIndex].translation +
                    correction.translation *
                        normalizedWeight;
                accumulatedWeight[toIndex] +=
                    normalizedWeight;
            }

            if (fromIndex != 0u) {
                const math::Twist6d opposite{
                    correction.rotation * -0.5,
                    correction.translation * -0.5,
                };
                accumulated[fromIndex].rotation =
                    accumulated[fromIndex].rotation +
                    opposite.rotation *
                        normalizedWeight;
                accumulated[fromIndex].translation =
                    accumulated[fromIndex].translation +
                    opposite.translation *
                        normalizedWeight;
                accumulatedWeight[fromIndex] +=
                    normalizedWeight;
            }
        }

        output.residual =
            totalWeight > 0.0
                ? std::sqrt(
                    weightedSquared /
                    totalWeight)
                : std::numeric_limits<double>::infinity();

        for (
            std::size_t camera = 1u;
            camera < cameraIds.size();
            ++camera
        ) {
            const double weight =
                accumulatedWeight[camera];
            if (weight <= 1e-9) continue;

            const double inverseWeight =
                1.0 / weight;
            const math::Twist6d correction{
                accumulated[camera].rotation *
                    inverseWeight,
                accumulated[camera].translation *
                    inverseWeight,
            };

            const auto current =
                toSE3(output.poses[camera]);

            output.poses[camera] =
                fromSE3(
                    current.compose(
                        math::SE3d::exp(correction)));
        }

        output.poses[0] = {};

        if (output.residual < 1e-6) {
            output.converged = true;
            break;
        }
    }

    const auto cycle =
        GraphCycleValidator{}.evaluate(
            output.poses,
            cameraIds,
            edges);

    output.cycleConfidence =
        std::max(
            initialCycle.confidence,
            cycle.confidence);
    output.cycleFailures =
        cycle.inconsistentEdges;
    output.confidence =
        std::clamp(
            std::min(
                std::exp(-output.residual),
                cycle.confidence),
            0.0,
            1.0);
    output.converged =
        output.converged ||
        (
            output.residual < 0.02 &&
            cycle.consistent
        );
    return output;
}

GraphOptimizationResult GlobalCameraGraphOptimizer::solve(
    std::size_t cameraCount,
    const std::vector<GraphEdge>& edges,
    std::size_t iterations,
    double step) const {
    if (
        cameraCount == 0 ||
        cameraCount >
            GlobalCameraGraphOptimizer::kMaximumCameras
    ) {
        return {};
    }

    std::vector<std::string> cameraIds;
    cameraIds.reserve(cameraCount);

    for (
        std::size_t index = 0;
        index < cameraCount;
        ++index
    ) {
        cameraIds.push_back(
            "C" + std::to_string(index));
    }

    return solve(
        cameraIds,
        edges,
        iterations,
        step);
}

} // namespace nox::native::calibration
