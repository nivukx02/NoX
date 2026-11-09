#include "eyes/temporal_gaze_stabilizer.h"

#include <algorithm>
#include <cmath>

namespace nox::native::eyes {

TemporalGazeStabilizer::TemporalGazeStabilizer(
    TemporalGazeConfig config) noexcept
    : config_(config) {
    config_.baseSmoothing=
        std::clamp(
            config_.baseSmoothing,
            0.05,
            0.95);
    config_.fastMotionSmoothing=
        std::clamp(
            config_.fastMotionSmoothing,
            config_.baseSmoothing,
            1.0);
    config_.maximumPredictionSeconds=
        std::clamp(
            config_.maximumPredictionSeconds,
            0.0,
            0.25);
    config_.continuityAngleScaleDegrees=
        std::clamp(
            config_.continuityAngleScaleDegrees,
            0.5,
            45.0);
}

GazeState TemporalGazeStabilizer::update(
    std::uint64_t timestampNanos,
    const EyeGeometry& left,
    const EyeGeometry& right,
    const Vec3& headForward,
    const Vec3& worldUp) noexcept {
    const GazeState instantaneous=
        GazeFusion{}.update(
            left,
            right,
            headForward,
            worldUp);

    if(!initialized_) {
        state_=instantaneous;
        timestampNanos_=timestampNanos;
        initialized_=true;
        angularVelocityDegreesPerSecond_=0.0;
        state_.angular_velocity_degrees_per_second=0.0;
        state_.temporal_residual_degrees=0.0;
        return state_;
    }

    if(timestampNanos<=timestampNanos_) {
        state_.failure_flags|=
            GAZE_FAILURE_HIGH_UNCERTAINTY;
        state_.temporal_confidence=
            std::min(
                state_.temporal_confidence,
                0.25);
        state_.accepted=false;
        return state_;
    }

    const double dt=
        std::clamp(
            static_cast<double>(
                timestampNanos-timestampNanos_)/
                1e9,
            1e-3,
            0.5);

    const double angularStep=
        angleDegrees(
            state_.world_relative,
            instantaneous.world_relative);

    const Vec3 rotationAxis =
        normalize(
            cross(
                state_.world_relative,
                instantaneous.world_relative));

    const double axisMagnitude =
        std::sqrt(
            std::max(
                0.0,
                dot(
                    rotationAxis,
                    rotationAxis)));

    if (
        axisMagnitude >
        1e-6) {
        angularAxis_ =
            rotationAxis;
    }

    angularVelocityDegreesPerSecond_=
        angularStep/dt;

    const double fastMotionFactor=
        clamp01(
            angularVelocityDegreesPerSecond_/
            220.0);

    const double smoothing=
        config_.baseSmoothing+
        (
            config_.fastMotionSmoothing-
            config_.baseSmoothing
        ) *
        fastMotionFactor;

    const Vec3 filtered=
        blend(
            state_.world_relative,
            instantaneous.world_relative,
            smoothing);

    const double predictionSeconds =
        std::clamp(
            config_.maximumPredictionSeconds *
                fastMotionFactor,
            0.0,
            config_.maximumPredictionSeconds);

    const double predictionAngleRadians =
        angularVelocityDegreesPerSecond_ *
        predictionSeconds *
        3.14159265358979323846 /
        180.0;

    const Vec3 predictedDirection =
        predictionAngleRadians > 1e-8
            ? rotateAroundAxis(
                filtered,
                angularAxis_,
                predictionAngleRadians)
            : filtered;

    predictionConfidence_ =
        clamp01(
            std::exp(
                -angularStep /
                std::max(
                    config_.continuityAngleScaleDegrees,
                    1e-3)) *
            std::exp(
                -predictionSeconds *
                10.0));

    const double predictionBlend =
        clamp01(
            predictionConfidence_ *
            fastMotionFactor *
            0.35);

    const Vec3 predictedFiltered =
        blend(
            filtered,
            predictedDirection,
            predictionBlend);

    state_=instantaneous;
    state_.world_relative=predictedFiltered;
    state_.head_relative=
        normalize(
            blend(
                state_.head_relative,
                instantaneous.head_relative,
                smoothing));

    state_.angular_velocity_degrees_per_second=
        angularVelocityDegreesPerSecond_;
    state_.temporal_residual_degrees=
        angularStep;

    const double continuity=
        std::exp(
            -angularStep/
            config_.continuityAngleScaleDegrees);

    state_.temporal_confidence=
        clamp01(
            0.45*instantaneous.temporal_confidence+
            0.25*continuity+
            0.15*std::exp(
                -dt*8.0)+
            0.15*predictionConfidence_);

    state_.temporal_consistency=
        clamp01(
            0.45*instantaneous.temporal_consistency+
            0.55*continuity);

    timestampNanos_=
        timestampNanos;

    return state_;
}

void TemporalGazeStabilizer::reset() noexcept {
    state_={};
    timestampNanos_=0;
    angularVelocityDegreesPerSecond_=0.0;
    angularAxis_={0.0,0.0,1.0};
    predictionConfidence_=0.0;
    initialized_=false;
}

Vec3 TemporalGazeStabilizer::cross(
    const Vec3& a,
    const Vec3& b) noexcept {
    return {
        a.y*b.z-a.z*b.y,
        a.z*b.x-a.x*b.z,
        a.x*b.y-a.y*b.x,
    };
}

Vec3 TemporalGazeStabilizer::rotateAroundAxis(
    const Vec3& direction,
    const Vec3& axis,
    double angleRadians) noexcept {
    const Vec3 unitAxis =
        normalize(axis);
    const Vec3 unitDirection =
        normalize(direction);

    const double c =
        std::cos(angleRadians);
    const double s =
        std::sin(angleRadians);

    const Vec3 parallel =
        {
            unitAxis.x *
                dot(unitAxis,unitDirection),
            unitAxis.y *
                dot(unitAxis,unitDirection),
            unitAxis.z *
                dot(unitAxis,unitDirection),
        };

    const Vec3 perpendicular =
        {
            unitDirection.x-parallel.x,
            unitDirection.y-parallel.y,
            unitDirection.z-parallel.z,
        };

    const Vec3 crossTerm =
        cross(
            unitAxis,
            unitDirection);

    return normalize({
        parallel.x+
            perpendicular.x*c+
            crossTerm.x*s,
        parallel.y+
            perpendicular.y*c+
            crossTerm.y*s,
        parallel.z+
            perpendicular.z*c+
            crossTerm.z*s,
    });
}

double TemporalGazeStabilizer::clamp01(double value) noexcept {
    if(!std::isfinite(value)) return 0.0;
    return std::clamp(value,0.0,1.0);
}

Vec3 TemporalGazeStabilizer::normalize(
    const Vec3& value) noexcept {
    const double length=
        std::sqrt(
            std::max(
                0.0,
                dot(value,value)));
    if(length<=1e-12) {
        return {0.0,0.0,1.0};
    }
    return {
        value.x/length,
        value.y/length,
        value.z/length,
    };
}

double TemporalGazeStabilizer::dot(
    const Vec3& a,
    const Vec3& b) noexcept {
    return a.x*b.x+
        a.y*b.y+
        a.z*b.z;
}

double TemporalGazeStabilizer::angleDegrees(
    const Vec3& a,
    const Vec3& b) noexcept {
    constexpr double pi=
        3.14159265358979323846;
    const double cosine=
        std::clamp(
            dot(
                normalize(a),
                normalize(b)),
            -1.0,
            1.0);
    return std::acos(cosine)*
        180.0/pi;
}

Vec3 TemporalGazeStabilizer::blend(
    const Vec3& from,
    const Vec3& to,
    double amount) noexcept {
    const double alpha=
        clamp01(amount);
    return normalize({
        from.x*(1.0-alpha)+to.x*alpha,
        from.y*(1.0-alpha)+to.y*alpha,
        from.z*(1.0-alpha)+to.z*alpha,
    });
}

} // namespace nox::native::eyes
