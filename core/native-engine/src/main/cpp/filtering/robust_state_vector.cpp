#include "filtering/robust_state_vector.h"

#include <algorithm>
#include <cmath>

namespace nox::native::filtering {

MotionState3 RobustStateVector::predict(
    const MotionState3& state,
    double dt) noexcept {
    const double safeDt=std::clamp(dt,0.0,0.5);
    const double dt2=safeDt*safeDt;
    const double dt3=dt2*safeDt;
    const double dt4=dt2*dt2;
    const double dt5=dt4*safeDt;

    MotionState3 output=state;
    for(int axis=0;axis<3;++axis){
        output.position[axis]=
            state.position[axis]+
            state.velocity[axis]*safeDt+
            0.5*state.acceleration[axis]*dt2+
            (1.0/6.0)*state.jerk[axis]*dt3;

        output.velocity[axis]=
            state.velocity[axis]+
            state.acceleration[axis]*safeDt+
            0.5*state.jerk[axis]*dt2;

        output.acceleration[axis]=
            state.acceleration[axis]+
            state.jerk[axis]*safeDt;

        const double jerkNoise=
            0.004+
            0.006*std::abs(state.jerk[axis]);
        output.variance[axis]=std::min(
            100.0,
            state.variance[axis]+
            jerkNoise*(
                dt5/20.0+
                dt4/8.0+
                dt3/3.0
            )
        );
    }
    return output;
}

MotionState3 RobustStateVector::correct(
    const MotionState3& state,
    const std::array<double,3>& measurement,
    const std::array<double,3>& measurementVariance,
    double gain) noexcept {
    return correct(
        state,
        measurement,
        measurementVariance,
        gain,
        1.0 / 30.0);
}

MotionState3 RobustStateVector::correct(
    const MotionState3& state,
    const std::array<double,3>& measurement,
    const std::array<double,3>& measurementVariance,
    double gain,
    double dtSeconds) noexcept {
    MotionState3 output=state;

    const double externalGain=
        std::clamp(gain,0.0,1.0);
    const double dt=
        std::clamp(dtSeconds,1e-3,0.5);

    for(int axis=0;axis<3;++axis){
        const double sensorVariance=
            std::max(measurementVariance[axis],1e-9);
        const double priorVariance=
            std::max(state.variance[axis],1e-9);
        const double innovation=
            measurement[axis]-state.position[axis];
        const double kalmanGain=
            priorVariance/
            (priorVariance+sensorVariance);
        const double positionAlpha=
            std::clamp(
                kalmanGain*externalGain,
                0.0,
                1.0);

        output.position[axis]+=
            positionAlpha*innovation;

        const double measuredVelocity=
            innovation/dt;
        const double velocityResidual=
            measuredVelocity-
            state.velocity[axis];
        const double velocityAlpha=
            std::clamp(
                0.38*positionAlpha,
                0.02,
                0.68);
        output.velocity[axis]+=
            velocityAlpha*velocityResidual;

        const double measuredAcceleration=
            velocityResidual/dt;
        const double accelerationAlpha=
            std::clamp(
                0.12*positionAlpha,
                0.01,
                0.24);
        output.acceleration[axis]+=
            accelerationAlpha*(
                measuredAcceleration-
                state.acceleration[axis]);

        const double measuredJerk=
            (
                output.acceleration[axis]-
                state.acceleration[axis]
            )/dt;
        const double jerkAlpha=
            std::clamp(
                0.08*positionAlpha,
                0.01,
                0.12);
        output.jerk[axis]+=
            jerkAlpha*(
                measuredJerk-
                state.jerk[axis]);

        output.variance[axis]=std::max(
            1e-9,
            (1.0-positionAlpha)*priorVariance);
    }
    return output;
}

} // namespace nox::native::filtering
