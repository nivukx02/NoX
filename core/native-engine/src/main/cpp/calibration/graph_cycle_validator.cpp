#include "calibration/graph_cycle_validator.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>

namespace nox::native::calibration {
namespace {

math::SE3d poseAsSe3(
    const GraphPose& pose) {
    return {
        pose.rotation,
        {pose.x, pose.y, pose.z},
    };
}

GraphCycleResult evaluateWithIds(
    const std::vector<GraphPose>& poses,
    const std::vector<std::string>& cameraIds,
    const std::vector<GraphEdge>& edges) {
    GraphCycleResult result;

    if (
        poses.empty() ||
        poses.size() != cameraIds.size() ||
        edges.empty()
    ) {
        return result;
    }

    std::unordered_map<std::string, std::size_t> ids;
    ids.reserve(cameraIds.size());

    for (
        std::size_t index = 0;
        index < cameraIds.size();
        ++index
    ) {
        if (
            cameraIds[index].empty() ||
            !ids.emplace(
                cameraIds[index],
                index).second
        ) {
            return result;
        }
    }

    double translationSquared = 0.0;
    double rotationSquared = 0.0;
    std::vector<double> translationErrors;
    std::vector<double> rotationErrors;

    translationErrors.reserve(edges.size());
    rotationErrors.reserve(edges.size());

    for (const auto& edge : edges) {
        const auto from =
            ids.find(edge.from);
        const auto to =
            ids.find(edge.to);

        if (
            from == ids.end() ||
            to == ids.end() ||
            from->second == to->second
        ) {
            continue;
        }

        if (
            !std::isfinite(edge.translationSigma) ||
            edge.translationSigma <= 0.0 ||
            !std::isfinite(edge.rotationSigma) ||
            edge.rotationSigma <= 0.0
        ) {
            continue;
        }

        const auto fromPose =
            poseAsSe3(
                poses[from->second]);
        const auto toPose =
            poseAsSe3(
                poses[to->second]);
        const auto measured =
            poseAsSe3(edge.relative);

        const auto predicted =
            fromPose.inverse().compose(
                toPose);
        const auto residual =
            measured.inverse().compose(
                predicted).log();

        const double translationError =
            std::sqrt(
                residual.translation.x *
                    residual.translation.x +
                residual.translation.y *
                    residual.translation.y +
                residual.translation.z *
                    residual.translation.z);

        const double rotationError =
            std::sqrt(
                residual.rotation.x *
                    residual.rotation.x +
                residual.rotation.y *
                    residual.rotation.y +
                residual.rotation.z *
                    residual.rotation.z);

        if (
            !std::isfinite(translationError) ||
            !std::isfinite(rotationError)
        ) {
            ++result.inconsistentEdges;
            ++result.cyclesChecked;
            continue;
        }

        translationSquared +=
            translationError *
            translationError;
        rotationSquared +=
            rotationError *
            rotationError;

        translationErrors.push_back(
            translationError);
        rotationErrors.push_back(
            rotationError);

        ++result.cyclesChecked;

        const double sigma =
            std::max(
                0.05,
                std::hypot(
                    edge.translationSigma,
                    edge.rotationSigma));

        if (
            std::hypot(
                translationError,
                rotationError) >
            3.0 * sigma
        ) {
            ++result.inconsistentEdges;
        }
    }

    if (result.cyclesChecked == 0U) {
        return result;
    }

    result.rmsTranslation =
        std::sqrt(
            translationSquared /
                static_cast<double>(
                    result.cyclesChecked));

    result.rmsRotation =
        std::sqrt(
            rotationSquared /
                static_cast<double>(
                    result.cyclesChecked));

    auto robustMedian = [](
        std::vector<double> values) {
        if (values.empty()) {
            return 0.0;
        }

        std::sort(
            values.begin(),
            values.end());

        const std::size_t middle =
            values.size() / 2U;

        if (
            values.size() % 2U == 0U &&
            middle > 0U
        ) {
            return (
                values[middle - 1U] +
                values[middle]
            ) * 0.5;
        }

        return values[middle];
    };

    const double robustTranslation =
        robustMedian(
            translationErrors);
    const double robustRotation =
        robustMedian(
            rotationErrors);

    result.confidence =
        std::clamp(
            std::exp(
                -robustTranslation / 0.05 -
                robustRotation / 0.05),
            0.0,
            1.0);

    result.consistent =
        result.inconsistentEdges == 0U &&
        result.confidence >= 0.55;

    return result;
}

} // namespace

GraphCycleResult
GraphCycleValidator::evaluate(
    const std::vector<GraphPose>& poses,
    const std::vector<GraphEdge>& edges) const {
    std::vector<std::string> cameraIds;
    cameraIds.reserve(poses.size());

    for (
        std::size_t index = 0;
        index < poses.size();
        ++index
    ) {
        cameraIds.push_back(
            "C" +
            std::to_string(index));
    }

    return evaluateWithIds(
        poses,
        cameraIds,
        edges);
}

GraphCycleResult
GraphCycleValidator::evaluate(
    const std::vector<GraphPose>& poses,
    const std::vector<std::string>& cameraIds,
    const std::vector<GraphEdge>& edges) const {
    return evaluateWithIds(
        poses,
        cameraIds,
        edges);
}

} // namespace nox::native::calibration
