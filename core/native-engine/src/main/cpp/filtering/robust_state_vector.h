#pragma once
#include <array>
namespace nox::native::filtering{
struct MotionState3{std::array<double,3> position{0,0,0};std::array<double,3> velocity{0,0,0};std::array<double,3> acceleration{0,0,0};std::array<double,3> jerk{0,0,0};std::array<double,3> variance{1,1,1};};
class RobustStateVector final{
public:
    static MotionState3 predict(const MotionState3&,double)noexcept;
    static MotionState3 correct(const MotionState3&,const std::array<double,3>&,const std::array<double,3>&,double)noexcept;
    static MotionState3 correct(const MotionState3&,const std::array<double,3>&,const std::array<double,3>&,double,double)noexcept;
};
}