#pragma once
namespace nox::native::lighting {
class ExposureVignetteModel final { public: double correctedLuma(double luma,double exposure,double radius,double strength=.15) const noexcept; };
}