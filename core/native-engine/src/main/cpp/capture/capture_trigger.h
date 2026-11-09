#pragma once
#include <cstdint>
namespace nox::native::capture { struct TriggerState{bool armed{false};bool fired{false};std::uint64_t timestamp{0};}; class CaptureTrigger final{public:TriggerState evaluate(bool armed,double confidence,std::uint64_t timestamp,double threshold=.65)const noexcept;};}