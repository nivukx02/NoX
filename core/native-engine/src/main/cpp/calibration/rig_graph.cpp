#include "calibration/rig_graph.h"

#include <cmath>
#include <queue>
#include <unordered_set>

namespace nox::native::calibration {

namespace {

double norm(const Quaternion& q) {
    return std::sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
}

} // namespace

Quaternion Quaternion::normalized() const noexcept {
    const double n = norm(*this);
    if (n < 1e-12) return {};
    return {w/n, x/n, y/n, z/n};
}

Quaternion Quaternion::conjugate() const noexcept {
    return {w, -x, -y, -z};
}

Quaternion Quaternion::multiply(const Quaternion& a, const Quaternion& b) noexcept {
    return {
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
    };
}

Vec3 Quaternion::rotate(Vec3 v) const noexcept {
    const Quaternion q = normalized();
    const Quaternion p{0.0, v.x, v.y, v.z};
    const Quaternion r = multiply(multiply(q, p), q.conjugate());
    return {r.x, r.y, r.z};
}

RigidPose RigidPose::compose(const RigidPose& next) const noexcept {
    return {
        Quaternion::multiply(rotation, next.rotation).normalized(),
        transform(next.translation),
    };
}

RigidPose RigidPose::inverse() const noexcept {
    const Quaternion inv = rotation.conjugate().normalized();
    const Vec3 t = inv.rotate({-translation.x, -translation.y, -translation.z});
    return {inv, t};
}

Vec3 RigidPose::transform(Vec3 point) const noexcept {
    const Vec3 r = rotation.rotate(point);
    return {r.x + translation.x, r.y + translation.y, r.z + translation.z};
}

void RigGraph::addEdge(RigEdge edge) {
    const auto inverse = edge.from_to.inverse();
    adjacency_[edge.from].push_back({edge.to, edge.from_to});
    adjacency_[edge.to].push_back({edge.from, inverse});
}

std::optional<RigidPose> RigGraph::resolve(
    const std::string& reference,
    const std::string& target) const {
    if (reference == target) return RigidPose{};
    struct State { std::string camera; RigidPose reference_to_camera; };
    std::queue<State> queue;
    std::unordered_set<std::string> visited;
    queue.push({reference, RigidPose{}});
    visited.insert(reference);

    while (!queue.empty()) {
        const auto current = queue.front();
        queue.pop();
        const auto found = adjacency_.find(current.camera);
        if (found == adjacency_.end()) continue;
        for (const auto& next : found->second) {
            if (visited.contains(next.camera)) continue;
            const auto composed = current.reference_to_camera.compose(next.pose);
            if (next.camera == target) return composed;
            visited.insert(next.camera);
            queue.push({next.camera, composed});
        }
    }
    return std::nullopt;
}

} // namespace nox::native::calibration
