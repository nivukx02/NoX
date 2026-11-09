#pragma once
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
namespace nox::native::tracking{class CameraVisibilityGraph final{public:void update(std::uint32_t camera,std::uint64_t subject,bool visible);bool visible(std::uint32_t camera,std::uint64_t subject)const noexcept;double coverage(std::uint64_t subject)const noexcept;private:std::unordered_map<std::uint32_t,std::unordered_set<std::uint64_t>> visible_;};}