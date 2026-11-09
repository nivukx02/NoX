#include "scene/gaussian_deformation_graph.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>

namespace nox::native::scene {

std::size_t GaussianDeformationGraph::CellHash::operator()(
    const std::array<std::int64_t,3>& key) const noexcept {
    std::uint64_t x =
        static_cast<std::uint64_t>(key[0]);
    std::uint64_t y =
        static_cast<std::uint64_t>(key[1]);
    std::uint64_t z =
        static_cast<std::uint64_t>(key[2]);

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
        (y + 0x9e3779b97f4a7c15ULL +
            (x << 6U) + (x >> 2U)) ^
        (z + 0xc2b2ae3d27d4eb4fULL +
            (y << 7U) + (y >> 3U)));
}

std::array<std::int64_t,3>
GaussianDeformationGraph::cellFor(
    const Vec3& position) const noexcept {
    const double cell =
        std::max(
            options_.nodeSpacingMeters,
            1e-4);

    return {
        static_cast<std::int64_t>(
            std::floor(position.x / cell)),
        static_cast<std::int64_t>(
            std::floor(position.y / cell)),
        static_cast<std::int64_t>(
            std::floor(position.z / cell)),
    };
}

void GaussianDeformationGraph::indexNode(
    std::size_t index) {
    if (index >= nodes_.size()) return;

    nodeBuckets_[
        cellFor(nodes_[index].canonicalPosition)
    ].push_back(index);
}

GaussianDeformationGraph::GaussianDeformationGraph(
    GaussianDeformationOptions options)
    : options_(options) {
    options_.nodeSpacingMeters =
        std::max(
            options_.nodeSpacingMeters,
            0.005);
    options_.maximumNodes =
        std::max<std::size_t>(
            options_.maximumNodes,
            64U);
    options_.maximumInfluencesPerGaussian =
        std::clamp<std::size_t>(
            options_.maximumInfluencesPerGaussian,
            1U,
            8U);
    options_.maximumNodeTranslationMeters =
        std::max(
            options_.maximumNodeTranslationMeters,
            0.01);
    options_.maximumNodeRotationRadians =
        std::max(
            options_.maximumNodeRotationRadians,
            0.05);
    options_.maximumEdgeStrainRatio =
        std::max(
            options_.maximumEdgeStrainRatio,
            0.01);
    options_.minimumBindingConfidence =
        clamp01(
            options_.minimumBindingConfidence);
    options_.temporalSmoothing =
        std::clamp(
            options_.temporalSmoothing,
            0.0,
            0.99);
}

void GaussianDeformationGraph::reset() noexcept {
    std::unique_lock lock(mutex_);
    resetUnlocked();
}

void GaussianDeformationGraph::resetUnlocked() noexcept {
    nodes_.clear();
    bindings_.clear();
    nodeBuckets_.clear();
}

double GaussianDeformationGraph::clamp01(
    double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(
        value,
        0.0,
        1.0);
}

double GaussianDeformationGraph::distance(
    const Vec3& a,
    const Vec3& b) noexcept {
    return std::hypot(
        std::hypot(
            a.x - b.x,
            a.y - b.y),
        a.z - b.z);
}

Vec3 GaussianDeformationGraph::rotate(
    const math::SE3d& transform,
    const Vec3& value) noexcept {
    const auto rotated =
        transform.rotation.rotate({
            value.x,
            value.y,
            value.z,
        });

    return {
        rotated.x,
        rotated.y,
        rotated.z,
    };
}

std::size_t GaussianDeformationGraph::nearestNode(
    const Vec3& position) const noexcept {
    const auto base =
        cellFor(position);

    std::size_t best =
        nodes_.size();
    double bestDistance =
        std::numeric_limits<double>::infinity();

    for (
        std::int64_t dz = -1;
        dz <= 1;
        ++dz) {
        for (
            std::int64_t dy = -1;
            dy <= 1;
            ++dy) {
            for (
                std::int64_t dx = -1;
                dx <= 1;
                ++dx) {
                const std::array<std::int64_t,3>
                    cell{
                        base[0] + dx,
                        base[1] + dy,
                        base[2] + dz,
                    };

                const auto iterator =
                    nodeBuckets_.find(cell);

                if (iterator == nodeBuckets_.end()) {
                    continue;
                }

                for (const auto index :
                     iterator->second) {
                    const double current =
                        distance(
                            nodes_[index].canonicalPosition,
                            position);

                    if (current < bestDistance) {
                        bestDistance = current;
                        best = index;
                    }
                }
            }
        }
    }

    return best;
}

std::vector<std::size_t>
GaussianDeformationGraph::nearestNodes(
    const Vec3& position) const {
    constexpr std::size_t kMaxInfluences = 8U;
    const std::size_t limit =
        std::min(
            options_.maximumInfluencesPerGaussian,
            kMaxInfluences);

    std::array<std::size_t, kMaxInfluences> bestIndices{};
    std::array<double, kMaxInfluences> bestDistances{};
    std::size_t count = 0U;

    const auto base = cellFor(position);

    const auto insertCandidate =
        [&](std::size_t index, double currentDistance) {
            if (limit == 0U) return;

            std::size_t insertion = count;
            if (count < limit) {
                while (
                    insertion > 0U &&
                    (
                        currentDistance <
                            bestDistances[insertion - 1U] ||
                        (
                            currentDistance ==
                                bestDistances[insertion - 1U] &&
                            index <
                                bestIndices[insertion - 1U]
                        )
                    )
                ) {
                    bestDistances[insertion] =
                        bestDistances[insertion - 1U];
                    bestIndices[insertion] =
                        bestIndices[insertion - 1U];
                    --insertion;
                }
                bestDistances[insertion] =
                    currentDistance;
                bestIndices[insertion] =
                    index;
                ++count;
                return;
            }

            if (
                currentDistance >
                    bestDistances[limit - 1U] ||
                (
                    currentDistance ==
                        bestDistances[limit - 1U] &&
                    index >=
                        bestIndices[limit - 1U]
                )
            ) {
                return;
            }

            insertion = limit - 1U;
            while (
                insertion > 0U &&
                (
                    currentDistance <
                        bestDistances[insertion - 1U] ||
                    (
                        currentDistance ==
                            bestDistances[insertion - 1U] &&
                        index <
                            bestIndices[insertion - 1U]
                    )
                )
            ) {
                bestDistances[insertion] =
                    bestDistances[insertion - 1U];
                bestIndices[insertion] =
                    bestIndices[insertion - 1U];
                --insertion;
            }

            bestDistances[insertion] =
                currentDistance;
            bestIndices[insertion] =
                index;
        };

    for (
        std::int64_t dz = -1;
        dz <= 1;
        ++dz) {
        for (
            std::int64_t dy = -1;
            dy <= 1;
            ++dy) {
            for (
                std::int64_t dx = -1;
                dx <= 1;
                ++dx) {
                const std::array<std::int64_t,3>
                    cell{
                        base[0] + dx,
                        base[1] + dy,
                        base[2] + dz,
                    };

                const auto iterator =
                    nodeBuckets_.find(cell);

                if (iterator == nodeBuckets_.end()) {
                    continue;
                }

                for (const auto index :
                     iterator->second) {
                    if (index >= nodes_.size()) continue;
                    insertCandidate(
                        index,
                        distance(
                            nodes_[index].canonicalPosition,
                            position));
                }
            }
        }
    }

    std::vector<std::size_t> indices;
    indices.reserve(count);
    for (std::size_t i = 0U; i < count; ++i) {
        indices.push_back(bestIndices[i]);
    }
    return indices;
}

bool GaussianDeformationGraph::buildCanonicalGraph(
    const std::vector<GaussianSurfacePrimitive>& primitives) {
    std::unique_lock lock(mutex_);
    resetUnlocked();

    if (primitives.empty()) {
        return false;
    }

    nodes_.reserve(
        std::min(
            options_.maximumNodes,
            primitives.size()));
    nodeBuckets_.reserve(
        std::min(
            options_.maximumNodes,
            primitives.size()));

    for (
        const auto& primitive :
        primitives) {
        if (
            nodes_.size() >=
            options_.maximumNodes) {
            break;
        }

        if (
            primitive.confidence <
            options_.minimumBindingConfidence) {
            continue;
        }

        const std::size_t nearest =
            nearestNode(
                primitive.mean);

        if (
            nearest <
            nodes_.size() &&
            distance(
                nodes_[nearest].canonicalPosition,
                primitive.mean) <
                options_.nodeSpacingMeters) {
            nodes_[nearest].confidence =
                clamp01(
                    nodes_[nearest].confidence *
                        0.70 +
                    primitive.confidence *
                        0.30);
            nodes_[nearest].temporalStability =
                clamp01(
                    nodes_[nearest].temporalStability *
                        0.70 +
                    primitive.temporalStability *
                        0.30);
            ++nodes_[nearest].supportCount;
            continue;
        }

        GaussianDeformationNode node;
        node.id =
            static_cast<std::uint32_t>(
                nodes_.size());
        node.canonicalPosition =
            primitive.mean;
        node.currentTransform =
            math::SE3d{};
        node.velocity =
            primitive.velocity;
        node.confidence =
            clamp01(
                primitive.confidence);
        node.temporalStability =
            clamp01(
                primitive.temporalStability);
        node.supportCount =
            1U;

        nodes_.push_back(
            node);
        indexNode(
            nodes_.size() - 1U);
    }

    if (nodes_.empty()) {
        return false;
    }

    bindings_.clear();
    bindings_.resize(
        primitives.size());

    for (
        std::size_t primitiveIndex = 0;
        primitiveIndex < primitives.size();
        ++primitiveIndex) {
        const auto& primitive =
            primitives[primitiveIndex];

        const auto nearest =
            nearestNodes(
                primitive.mean);

        if (nearest.empty()) {
            continue;
        }

        GaussianDeformationBinding binding;
        binding.gaussianIndex =
            static_cast<std::uint32_t>(
                primitiveIndex);

        double sumWeight = 0.0;

        for (
            const auto nodeIndex :
            nearest) {
            const double d =
                distance(
                    nodes_[nodeIndex].canonicalPosition,
                    primitive.mean);

            const double distanceWeight =
                1.0 /
                std::max(
                    d * d,
                    1e-8);

            const double covarianceTrace =
                std::max(
                    0.0,
                    primitive.covariance[0]) +
                std::max(
                    0.0,
                    primitive.covariance[1]) +
                std::max(
                    0.0,
                    primitive.covariance[2]);

            const double uncertaintyRadius =
                std::sqrt(
                    covarianceTrace);

            const double uncertaintyScale =
                std::max(
                    options_.nodeSpacingMeters,
                    1e-3);

            const double uncertaintyWeight =
                1.0 /
                (
                    1.0 +
                    (
                        uncertaintyRadius /
                        uncertaintyScale
                    ) *
                    (
                        uncertaintyRadius /
                        uncertaintyScale
                    )
                );

            const double weight =
                distanceWeight *
                uncertaintyWeight;

            binding.nodeIds.push_back(
                nodes_[nodeIndex].id);
            binding.weights.push_back(
                weight);
            sumWeight +=
                weight;
        }

        if (
            sumWeight <=
            0.0) {
            continue;
        }

        for (double& weight :
             binding.weights) {
            weight /=
                sumWeight;
        }

        const auto anchor =
            nodes_[
                nearest.front()
            ].canonicalPosition;

        binding.canonicalOffset = {
            primitive.mean.x -
                anchor.x,
            primitive.mean.y -
                anchor.y,
            primitive.mean.z -
                anchor.z,
        };

        const double covarianceTrace =
            std::max(
                0.0,
                primitive.covariance[0]) +
            std::max(
                0.0,
                primitive.covariance[1]) +
            std::max(
                0.0,
                primitive.covariance[2]);
        const double uncertaintyRadius =
            std::sqrt(
                covarianceTrace);
        const double uncertaintyScale =
            std::max(
                options_.nodeSpacingMeters,
                1e-3);
        const double uncertaintyWeight =
            1.0 /
            (
                1.0 +
                (
                    uncertaintyRadius /
                    uncertaintyScale
                ) *
                (
                    uncertaintyRadius /
                    uncertaintyScale
                )
            );

        binding.confidence =
            clamp01(
                primitive.confidence *
                primitive.temporalStability *
                std::sqrt(
                    uncertaintyWeight));

        if (
            binding.confidence <
            options_.minimumBindingConfidence) {
            continue;
        }

        bindings_[primitiveIndex] =
            binding;
    }

    bool anyBinding = false;
    for (const auto& binding :
         bindings_) {
        if (!binding.nodeIds.empty()) {
            anyBinding = true;
            break;
        }
    }

    return anyBinding;
}

bool GaussianDeformationGraph::updateNodePose(
    std::uint32_t nodeId,
    const math::SE3d& transform,
    const Vec3& velocity,
    double confidence,
    double temporalStability) noexcept {
    std::unique_lock lock(mutex_);
    if (
        nodeId >=
        nodes_.size()) {
        return false;
    }

    auto& node =
        nodes_[nodeId];

    const auto delta =
        transform.translation;

    const double translationMagnitude =
        std::hypot(
            std::hypot(
                delta.x,
                delta.y),
            delta.z);

    if (
        translationMagnitude >
        options_.maximumNodeTranslationMeters) {
        return false;
    }

    const auto rotationVector =
        transform.rotation.toRotationVector();

    const double rotationMagnitude =
        std::hypot(
            std::hypot(
                rotationVector.x,
                rotationVector.y),
            rotationVector.z);

    if (
        rotationMagnitude >
        options_.maximumNodeRotationRadians) {
        return false;
    }

    node.currentTransform =
        transform;
    node.velocity =
        velocity;
    node.confidence =
        clamp01(
            node.confidence *
                options_.temporalSmoothing +
            clamp01(
                confidence) *
                (
                    1.0 -
                    options_.temporalSmoothing
                ));
    node.temporalStability =
        clamp01(
            node.temporalStability *
                options_.temporalSmoothing +
            clamp01(
                temporalStability) *
                (
                    1.0 -
                    options_.temporalSmoothing
                ));

    return true;
}

std::vector<GaussianSurfacePrimitive>
GaussianDeformationGraph::deform(
    const std::vector<GaussianSurfacePrimitive>& primitives) const {
    std::shared_lock lock(mutex_);
    std::vector<GaussianSurfacePrimitive> output =
        primitives;

    for (
        std::size_t primitiveIndex = 0;
        primitiveIndex < output.size();
        ++primitiveIndex) {
        auto& primitive =
            output[primitiveIndex];
        if (
            primitiveIndex >=
            bindings_.size()) {
            continue;
        }

        const auto& binding =
            bindings_[primitiveIndex];

        if (
            binding.nodeIds.empty() ||
            binding.gaussianIndex !=
                primitiveIndex) {
            continue;
        }

        Vec3 transformed{};
        Vec3 transformedVelocity{};
        double accumulatedConfidence = 0.0;

        for (
            std::size_t influence = 0;
            influence <
                binding.nodeIds.size();
            ++influence) {
            const auto nodeId =
                binding.nodeIds[
                    influence];

            if (
                nodeId >=
                nodes_.size()) {
                continue;
            }

            const double weight =
                binding.weights[
                    influence];

            const auto& node =
                nodes_[nodeId];

            const Vec3 rotatedOffset =
                rotate(
                    node.currentTransform,
                    binding.canonicalOffset);

            const Vec3 worldPoint = {
                node.canonicalPosition.x +
                    node.currentTransform.translation.x +
                    rotatedOffset.x,
                node.canonicalPosition.y +
                    node.currentTransform.translation.y +
                    rotatedOffset.y,
                node.canonicalPosition.z +
                    node.currentTransform.translation.z +
                    rotatedOffset.z,
            };

            transformed.x +=
                worldPoint.x *
                weight;
            transformed.y +=
                worldPoint.y *
                weight;
            transformed.z +=
                worldPoint.z *
                weight;

            transformedVelocity.x +=
                node.velocity.x *
                weight;
            transformedVelocity.y +=
                node.velocity.y *
                weight;
            transformedVelocity.z +=
                node.velocity.z *
                weight;

            accumulatedConfidence +=
                weight *
                node.confidence;
        }

        if (
            binding.weights.empty()) {
            continue;
        }

        primitive.mean =
            transformed;
        primitive.velocity =
            transformedVelocity;
        primitive.confidence =
            clamp01(
                primitive.confidence *
                    0.35 +
                accumulatedConfidence *
                    0.65);
        primitive.temporalStability =
            clamp01(
                primitive.temporalStability *
                    0.45 +
                binding.confidence *
                    0.25 +
                accumulatedConfidence *
                    0.30);
    }

    return output;
}

GaussianDeformationStats
GaussianDeformationGraph::stats() const noexcept {
    std::shared_lock lock(mutex_);
    GaussianDeformationStats result;
    result.nodeCount =
        nodes_.size();
    for (const auto& binding :
         bindings_) {
        if (!binding.nodeIds.empty()) {
            ++result.bindingCount;
        } else {
            ++result.rejectedBindings;
        }
    }

    if (
        nodes_.empty() &&
        bindings_.empty()) {
        return result;
    }

    double nodeConfidence = 0.0;
    double bindingConfidence = 0.0;
    double stability = 0.0;
    std::size_t trustedBindings = 0U;

    for (const auto& node :
         nodes_) {
        nodeConfidence +=
            clamp01(
                node.confidence);
        stability +=
            clamp01(
                node.temporalStability);
    }

    for (const auto& binding :
         bindings_) {
        if (binding.nodeIds.empty()) {
            continue;
        }

        bindingConfidence +=
            clamp01(
                binding.confidence);

        if (
            binding.confidence >=
                0.70) {
            ++trustedBindings;
        }
    }

    if (!nodes_.empty()) {
        result.meanNodeConfidence =
            nodeConfidence /
            static_cast<double>(
                nodes_.size());
        result.temporalStability +=
            (
                stability /
                static_cast<double>(
                    nodes_.size())
            ) *
            0.60;
    }

    if (result.bindingCount > 0U) {
        result.meanBindingConfidence =
            bindingConfidence /
            static_cast<double>(
                result.bindingCount);
        result.trustedBindingRatio =
            static_cast<double>(
                trustedBindings) /
            static_cast<double>(
                result.bindingCount);
        result.temporalStability +=
            result.meanBindingConfidence *
            0.40;
    }

    std::vector<double> strainValues;
    strainValues.reserve(
        result.bindingCount * 3U);

    const auto transformedNodePosition =
        [this](std::uint32_t nodeId) {
            if (nodeId >= nodes_.size()) {
                return Vec3{};
            }

            const auto& node =
                nodes_[nodeId];

            return Vec3{
                node.canonicalPosition.x +
                    node.currentTransform.translation.x,
                node.canonicalPosition.y +
                    node.currentTransform.translation.y,
                node.canonicalPosition.z +
                    node.currentTransform.translation.z,
            };
        };

    for (const auto& binding :
         bindings_) {
        if (
            binding.nodeIds.size() <
            2U) {
            continue;
        }

        for (
            std::size_t i = 0;
            i < binding.nodeIds.size();
            ++i) {
            for (
                std::size_t j = i + 1U;
                j < binding.nodeIds.size();
                ++j) {
                const auto a =
                    binding.nodeIds[i];
                const auto b =
                    binding.nodeIds[j];

                if (
                    a >= nodes_.size() ||
                    b >= nodes_.size()) {
                    continue;
                }

                const double canonicalDistance =
                    distance(
                        nodes_[a].canonicalPosition,
                        nodes_[b].canonicalPosition);

                if (
                    canonicalDistance <
                    1e-6) {
                    continue;
                }

                const double currentDistance =
                    distance(
                        transformedNodePosition(a),
                        transformedNodePosition(b));

                if (
                    !std::isfinite(
                        currentDistance)) {
                    continue;
                }

                strainValues.push_back(
                    std::abs(
                        currentDistance /
                            canonicalDistance -
                        1.0));
            }
        }
    }

    if (!strainValues.empty()) {
        std::sort(
            strainValues.begin(),
            strainValues.end());

        const double p95Position =
            0.95 *
            static_cast<double>(
                strainValues.size() - 1U);

        const auto lower =
            static_cast<std::size_t>(
                std::floor(
                    p95Position));
        const auto upper =
            std::min(
                strainValues.size() - 1U,
                lower + 1U);
        const double alpha =
            p95Position -
            static_cast<double>(
                lower);

        result.p95EdgeStrainRatio =
            strainValues[lower] +
            (
                strainValues[upper] -
                strainValues[lower]
            ) *
            alpha;
        result.maximumEdgeStrainRatio =
            strainValues.back();

        const double strainPenalty =
            std::exp(
                -result.p95EdgeStrainRatio /
                std::max(
                    options_.maximumEdgeStrainRatio,
                    1e-6));

        result.temporalStability =
            clamp01(
                result.temporalStability *
                    0.72 +
                strainPenalty *
                    0.28);
    }

    return result;
}

} // namespace nox::native::scene
