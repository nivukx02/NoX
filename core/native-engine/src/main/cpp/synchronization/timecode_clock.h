#pragma once
#include <cstdint>
#include <string>
namespace nox::native::sync {
struct Timecode { std::uint32_t hours{0},minutes{0},seconds{0},frames{0}; };
class TimecodeClock final {
public:
 static bool parse(const std::string&,Timecode&) noexcept;
 static std::uint64_t toNanoseconds(const Timecode&,double fps) noexcept;
 static Timecode fromNanoseconds(std::uint64_t,double fps) noexcept;
};
}