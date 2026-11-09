#pragma once
namespace nox::native::imu{
struct Quaternion{
    double w{1};
    double x{0};
    double y{0};
    double z{0};
};

struct ImuState{
    Quaternion orientation{};
    double gravity_norm{9.80665};
    double confidence{0.0};
    double gyroResidual{0.0};
    double accelResidual{0.0};
    double biasConfidence{0.0};
    double stationaryConfidence{0.0};
    double inertialQuality{0.0};

    // Bounded world-frame translational evidence. These values are auxiliary
    // motion evidence, not authoritative metric position without external reference.
    double worldAccelerationX{0.0};
    double worldAccelerationY{0.0};
    double worldAccelerationZ{0.0};
    double velocityX{0.0};
    double velocityY{0.0};
    double velocityZ{0.0};
    double displacementX{0.0};
    double displacementY{0.0};
    double displacementZ{0.0};
    double velocityConfidence{0.0};
    double translationConfidence{0.0};
};

Quaternion normalize(Quaternion)noexcept;Quaternion multiply(Quaternion,Quaternion)noexcept;Quaternion fromAngularVelocity(double,double,double,double)noexcept;}