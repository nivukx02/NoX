#include "tracking/joint_track.h"
#include <algorithm>
#include <cmath>
namespace nox::native::tracking{
static Vec3 add(Vec3 a,Vec3 b){return{a.x+b.x,a.y+b.y,a.z+b.z};}
static Vec3 sub(Vec3 a,Vec3 b){return{a.x-b.x,a.y-b.y,a.z-b.z};}
static Vec3 mul(Vec3 a,double s){return{a.x*s,a.y*s,a.z*s};}
static double mag(Vec3 a){return std::sqrt(a.x*a.x+a.y*a.y+a.z*a.z);}
JointState JointTrack::update(const JointObservation&o,std::uint64_t t,double trust) noexcept{
    if(!initialized_){initialized_=true;lastTimestampNanos_=t;state_ = JointState{
            id_,
            o.position,
            Vec3{},
            Vec3{},
            Vec3{},
            Vec3{},
            0.0,
            0.0,
            0.0,
            0.0,
            o.confidence * trust,
            1.0 - o.confidence * trust,
            0.0,
            false,
            false,
            false,
        };state_.visible=true;return state_;}
    const double dt=std::clamp((t-lastTimestampNanos_)*1e-9,1e-4,0.25);
    const Vec3 jerkTerm =
        mul(
            state_.jerk,
            (
                dt *
                dt *
                dt
            ) /
            6.0);
    const Vec3 predicted =
        add(
            add(
                add(
                    state_.position,
                    mul(
                        state_.velocity,
                        dt)),
                mul(
                    state_.acceleration,
                    0.5 * dt * dt)),
            jerkTerm);
    const double alpha=std::clamp(0.18+0.70*(o.confidence*trust),0.18,0.88);
    const Vec3 residual=sub(o.position,predicted);
    const double innovationMagnitude =
        mag(residual);
    const double predictedMotionScale =
        std::max(
            0.01,
            state_.velocityMagnitude * dt +
                0.5 *
                    state_.accelerationMagnitude *
                    dt * dt +
                (1.0 / 6.0) *
                    state_.jerkMagnitude *
                    dt * dt * dt);

    /*
     * Normalize innovation against the current uncertainty and predicted dynamic envelope.
     * A fixed meter-scale threshold silently fails for both small precise motion and close-range
     * high-speed motion. The scale below stays tied to the measured state and has a conservative
     * floor for ordinary camera landmark noise.
     */
    const double innovationScale =
        std::max(
            0.030,
            state_.uncertainty *
                0.08 +
                predictedMotionScale *
                0.35 +
                0.003,
            0.030 +
                0.18 *
                    predictedMotionScale);
    const double normalizedInnovation =
        innovationMagnitude /
        innovationScale;
    const double robustGateSigma =
        std::clamp(
            4.5 -
                1.2 *
                    std::clamp(
                        predictedMotionScale /
                            0.25,
                        0.0,
                        1.0) +
                0.35 *
                    std::clamp(
                        state_.confidence,
                        0.0,
                        1.0),
            3.0,
            5.0);
    const double cauchyWeight =
        1.0 /
            (
                1.0 +
                    (
                        normalizedInnovation /
                            robustGateSigma
                        ) *
                        (
                            normalizedInnovation /
                                robustGateSigma
                        )
            );
    if(
        normalizedInnovation >
            robustGateSigma &&
        cauchyWeight < 0.50){
        state_.innovation=residual;
        state_.innovationMagnitude=
            innovationMagnitude;
        state_.rejected=true;
        state_.predicted=true;
        state_.position=predicted;
        state_.velocityMagnitude=mag(state_.velocity);
        state_.accelerationMagnitude=mag(state_.acceleration);
        state_.jerkMagnitude=mag(state_.jerk);
        state_.predictionAgeSeconds += dt;
        state_.uncertainty=
            std::min(
                1.0,
                state_.uncertainty *
                    (
                        1.12 +
                        0.08 *
                            std::clamp(
                                state_.jerkMagnitude /
                                    10.0,
                                0.0,
                                1.0)
                    ) +
                    0.05);
        state_.confidence*=0.90;
        lastTimestampNanos_=t;
        return state_;
    }
    const double robustAlpha =
        std::clamp(
            alpha *
                (
                    0.35 +
                        0.65 *
                            cauchyWeight
                ),
            0.08,
            0.88);
    const Vec3 corrected=add(
        predicted,
        mul(
            residual,
            robustAlpha));
    const Vec3 newVelocity=mul(sub(corrected,state_.position),1.0/dt);
    const Vec3 newAcceleration = mul(sub(newVelocity,state_.velocity),1.0/dt);
    state_.jerk = mul(sub(newAcceleration,state_.acceleration),1.0/dt);
    state_.acceleration = newAcceleration;
    state_.velocity=newVelocity;
    state_.position=corrected;
    state_.innovation=residual;
    state_.innovationMagnitude=mag(residual);
    state_.velocityMagnitude=mag(newVelocity);
    state_.accelerationMagnitude=mag(newAcceleration);
    state_.jerkMagnitude=mag(state_.jerk);
    state_.predictionAgeSeconds=0.0;
    state_.confidence=std::clamp(
        0.55*state_.confidence+
            0.45*(o.confidence*trust)*
            cauchyWeight *
            std::exp(
                -state_.innovationMagnitude /
                0.025),
        0.0,
        1.0);
    state_.uncertainty=std::clamp(
        0.55*state_.uncertainty+
            0.45*(1.0-o.confidence*trust)+
            std::min(
                0.30,
                state_.innovationMagnitude / 0.10),
        0.0,
        1.0);
    state_.visible=o.confidence>=0.08;
    state_.predicted=false;
    state_.rejected=false;
    lastTimestampNanos_=t;
    return state_;
}
JointState JointTrack::predict(std::uint64_t t,double trust) noexcept{
    if(!initialized_)return state_;
    const double dt=std::clamp((t-lastTimestampNanos_)*1e-9,0.0,0.25);
    state_.position=add(
        add(
            add(
                state_.position,
                mul(
                    state_.velocity,
                    dt)),
            mul(
                state_.acceleration,
                0.5*dt*dt)),
        mul(
            state_.jerk,
            (
                dt *
                dt *
                dt
            ) /
            6.0));
    state_.innovation={};
    state_.innovationMagnitude=0.0;
    state_.velocityMagnitude=mag(state_.velocity);
    state_.accelerationMagnitude=mag(state_.acceleration);
    state_.jerk={};
    state_.jerkMagnitude=0.0;
    state_.predictionAgeSeconds += dt;
    state_.predicted=true;
    state_.visible=false;
    state_.confidence*=std::clamp(
        std::exp(-0.80*dt)*trust,
        0.0,
        1.0);
    state_.uncertainty=std::clamp(
        state_.uncertainty+
            0.10*dt+
            0.02*state_.predictionAgeSeconds,
        0.0,
        1.0);
    lastTimestampNanos_=t;
    return state_;
}
} // namespace nox::native::tracking
