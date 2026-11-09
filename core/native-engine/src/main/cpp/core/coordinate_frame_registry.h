#pragma once
#include <string>
#include <unordered_map>
namespace nox::native::core { struct Transform{double x{0},y{0},z{0};}; class CoordinateFrameRegistry final{public:void registerFrame(const std::string&,const std::string&);bool isCompatible(const std::string&,const std::string&)const;private:std::unordered_map<std::string,std::string>parents_;};}