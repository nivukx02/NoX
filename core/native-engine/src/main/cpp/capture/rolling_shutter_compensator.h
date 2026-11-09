#pragma once
namespace nox::native::capture {
struct Point2 { double x{0},y{0}; };
class RollingShutterCompensator final { public: Point2 correct(Point2,double rowNormalized,double angularVelocity,double readoutSeconds) const noexcept; };
}