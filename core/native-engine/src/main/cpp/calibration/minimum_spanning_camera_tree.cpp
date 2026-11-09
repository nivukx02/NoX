#include "calibration/minimum_spanning_camera_tree.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>

namespace nox::native::calibration {
namespace {

class DisjointSet final {
public:
    explicit DisjointSet(std::size_t size)
        : parent_(size),
          rank_(size, 0U) {
        std::iota(
            parent_.begin(),
            parent_.end(),
            0U);
    }

    std::size_t find(
        std::size_t value) {
        if (parent_[value] == value) {
            return value;
        }
        parent_[value] =
            find(parent_[value]);
        return parent_[value];
    }

    bool unite(
        std::size_t a,
        std::size_t b) {
        a = find(a);
        b = find(b);

        if (a == b) {
            return false;
        }

        if (rank_[a] < rank_[b]) {
            std::swap(a, b);
        }

        parent_[b] = a;

        if (rank_[a] == rank_[b]) {
            ++rank_[a];
        }

        return true;
    }

private:
    std::vector<std::size_t> parent_;
    std::vector<unsigned> rank_;
};

double edgeCost(
    const GraphEdge& edge) {
    const double sigma =
        std::max(
            0.05,
            std::hypot(
                edge.translationSigma,
                edge.rotationSigma));

    const double weight =
        std::max(
            edge.weight,
            1e-6);

    return sigma / weight;
}

bool validEdge(
    const GraphEdge& edge) {
    return !edge.from.empty() &&
        !edge.to.empty() &&
        edge.from != edge.to &&
        std::isfinite(edge.weight) &&
        edge.weight >= 0.0 &&
        std::isfinite(edge.translationSigma) &&
        edge.translationSigma > 0.0 &&
        std::isfinite(edge.rotationSigma) &&
        edge.rotationSigma > 0.0;
}

} // namespace

CameraTreeResult
MinimumSpanningCameraTree::solve(
    const std::vector<std::string>& cameraIds,
    const std::vector<GraphEdge>& edges) const {
    CameraTreeResult result;

    if (cameraIds.empty()) {
        result.connected = false;
        return result;
    }

    if (cameraIds.size() == 1U) {
        result.connected = true;
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
            result.connected = false;
            return result;
        }
    }

    std::vector<std::size_t> ordered;
    ordered.reserve(edges.size());

    for (
        std::size_t index = 0;
        index < edges.size();
        ++index
    ) {
        const auto& edge = edges[index];

        if (!validEdge(edge)) {
            continue;
        }

        if (
            ids.find(edge.from) ==
                ids.end() ||
            ids.find(edge.to) ==
                ids.end()
        ) {
            continue;
        }

        ordered.push_back(index);
    }

    std::sort(
        ordered.begin(),
        ordered.end(),
        [&edges](
            std::size_t left,
            std::size_t right) {
            const double leftCost =
                edgeCost(edges[left]);
            const double rightCost =
                edgeCost(edges[right]);

            if (leftCost != rightCost) {
                return leftCost < rightCost;
            }

            if (
                edges[left].from !=
                edges[right].from
            ) {
                return edges[left].from <
                    edges[right].from;
            }

            if (
                edges[left].to !=
                edges[right].to
            ) {
                return edges[left].to <
                    edges[right].to;
            }

            return left < right;
        });

    DisjointSet dsu(cameraIds.size());

    for (
        const std::size_t edgeIndex :
            ordered
    ) {
        const auto& edge =
            edges[edgeIndex];

        const auto from =
            ids.find(edge.from);
        const auto to =
            ids.find(edge.to);

        if (
            from == ids.end() ||
            to == ids.end()
        ) {
            continue;
        }

        if (
            !dsu.unite(
                from->second,
                to->second)
        ) {
            continue;
        }

        result.selectedEdgeIndices.push_back(
            edgeIndex);
        result.totalCost +=
            edgeCost(edge);

        if (
            result.selectedEdgeIndices.size() ==
            cameraIds.size() - 1U
        ) {
            result.connected = true;
            break;
        }
    }

    return result;
}

CameraTreeResult
MinimumSpanningCameraTree::solve(
    std::size_t cameraCount,
    const std::vector<GraphEdge>& edges) const {
    if (
        cameraCount == 0 ||
        cameraCount >
            GlobalCameraGraphOptimizer::
                kMaximumCameras
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
            "C" +
            std::to_string(index));
    }

    return solve(
        cameraIds,
        edges);
}

} // namespace nox::native::calibration
