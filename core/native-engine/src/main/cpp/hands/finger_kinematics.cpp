#include "hands/finger_kinematics.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace nox::native::hands {
namespace {
Vec3 subtract(Vec3 a,Vec3 b) noexcept {
    return {a.x-b.x,a.y-b.y,a.z-b.z};
}
double clamp01(double value) noexcept {
    if (!std::isfinite(value)) return 0.0;
    return std::clamp(value,0.0,1.0);
}
double dot(Vec3 a,Vec3 b) noexcept {
    return a.x*b.x+a.y*b.y+a.z*b.z;
}
double norm(Vec3 v) noexcept {
    return std::sqrt(dot(v,v));
}
double angle(Vec3 a,Vec3 b) noexcept {
    const double na=norm(a),nb=norm(b);
    if(na<1e-9||nb<1e-9)return 0.0;
    return std::acos(std::clamp(
        dot(a,b)/(na*nb),-1.0,1.0));
}
}

std::vector<FingerJointRotation> FingerKinematics::solve(
    const std::vector<Joint>& joints,
    const std::vector<double>& preferredLengths) const {
    std::vector<FingerJointRotation> output;
    output.reserve(joints.size());
    for(std::size_t i=0;i<joints.size();++i){
        double bend=0.0;
        double lengthResidual=0.0;
        double confidence=
            std::clamp(
                joints[i].confidence,
                0.0,
                1.0);

        if(i>=2U){
            const Vec3 incoming=subtract(
                joints[i-1].position,
                joints[i-2].position);
            const Vec3 outgoing=subtract(
                joints[i].position,
                joints[i-1].position);
            bend=angle(
                incoming,
                outgoing);

            const double expected=
                i-1U<preferredLengths.size()
                    ? preferredLengths[i-1U]
                    : 0.0;
            const double actual=
                norm(outgoing);

            if(expected>1e-6){
                lengthResidual=
                    std::abs(actual-expected)/
                    expected;
                confidence*=
                    std::exp(
                        -std::min(
                            4.0,
                            lengthResidual*3.0));
            }
        }

        const double uncertainty=
            std::clamp(
                1.0-confidence,
                0.0,
                1.0);

        output.push_back({
            joints[i].id,
            bend,
            uncertainty,
            lengthResidual,
            0.0,
            clamp01(
                1.0 -
                uncertainty),
            bend<=2.7 &&
                lengthResidual<=0.35,
        });
    }
    return output;
}

std::vector<FingerJointRotation> FingerKinematics::solveChains(
    const std::vector<Joint>& joints,
    const std::vector<FingerChain>& chains) const {
    std::unordered_map<std::uint32_t,std::size_t> index;
    index.reserve(joints.size());
    for(std::size_t i=0;i<joints.size();++i)index[joints[i].id]=i;

    std::vector<FingerJointRotation> output;
    for(const auto& chain:chains){
        if(chain.jointIndices.size()<3)continue;
        for(std::size_t i=1;i+1<chain.jointIndices.size();++i){
            const auto a=index.find(chain.jointIndices[i-1]);
            const auto b=index.find(chain.jointIndices[i]);
            const auto c=index.find(chain.jointIndices[i+1]);
            if(a==index.end()||b==index.end()||c==index.end())continue;

            const Vec3 incoming=subtract(
                joints[b->second].position,
                joints[a->second].position);
            const Vec3 outgoing=subtract(
                joints[c->second].position,
                joints[b->second].position);
            const double bend=angle(incoming,outgoing);
            const double uncertainty=std::clamp(
                1.0-std::min({
                    joints[a->second].confidence,
                    joints[b->second].confidence,
                    joints[c->second].confidence,
                }),
                0.0,
                1.0);
            output.push_back({
                joints[b->second].id,
                bend,
                uncertainty,
                0.0,
                0.0,
                clamp01(
                    1.0 -
                    uncertainty),
                bend>=chain.minBendRadians&&
                    bend<=chain.maxBendRadians,
            });
        }
    }
    return output;
}


std::vector<FingerJointRotation>
FingerKinematics::solveTemporal(
    const std::vector<Joint>& joints,
    const std::vector<FingerJointRotation>& previous,
    double dtSeconds,
    const std::vector<FingerChain>& chains) const {
    auto current =
        chains.empty()
            ? solve(
                joints,
                {})
            : solveChains(
                joints,
                chains);

    const double dt =
        std::clamp(
            std::isfinite(dtSeconds)
                ? dtSeconds
                : 0.0,
            1e-4,
            0.25);

    std::unordered_map<
        std::uint32_t,
        FingerJointRotation> previousById;
    previousById.reserve(
        previous.size());

    for (const auto& item :
         previous) {
        previousById[item.id] =
            item;
    }

    for (auto& item :
         current) {
        const auto iterator =
            previousById.find(
                item.id);

        if (
            iterator ==
            previousById.end()) {
            item.temporalConfidence =
                clamp01(
                    1.0 -
                    item.uncertainty);
            continue;
        }

        const auto& old =
            iterator->second;

        const double innovation =
            std::abs(
                item.bendRadians -
                old.bendRadians);

        item.angularVelocityRadiansPerSecond =
            innovation /
            dt;

        const double velocityGate =
            std::exp(
                -item.angularVelocityRadiansPerSecond /
                12.0);

        const double uncertaintyGate =
            clamp01(
                1.0 -
                item.uncertainty);

        item.temporalConfidence =
            clamp01(
                0.50 *
                    uncertaintyGate +
                0.30 *
                    clamp01(
                        old.temporalConfidence) +
                0.20 *
                    velocityGate);

        if (
            item.angularVelocityRadiansPerSecond >
            18.0) {
            item.withinLimits =
                false;
        }
    }

    return current;
}

} // namespace nox::native::hands
