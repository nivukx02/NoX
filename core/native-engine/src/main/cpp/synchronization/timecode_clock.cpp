#include "synchronization/timecode_clock.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace nox::native::sync {
bool TimecodeClock::parse(const std::string& value, Timecode& output) noexcept {
    unsigned hours = 0;
    unsigned minutes = 0;
    unsigned seconds = 0;
    unsigned frames = 0;
    const int parsed = std::sscanf(
        value.c_str(), "%u:%u:%u:%u",
        &hours, &minutes, &seconds, &frames);
    if (parsed != 4 || minutes >= 60 || seconds >= 60) return false;
    output = {
        hours,
        minutes,
        seconds,
        frames,
    };
    return true;
}

std::uint64_t TimecodeClock::toNanoseconds(
    const Timecode& value, double fps) noexcept {
    if (fps <= 0.0) return 0;
    const double frameNumber =
        ((value.hours * 3600ull +
          value.minutes * 60ull +
          value.seconds) * fps) +
        value.frames;
    return static_cast<std::uint64_t>(
        std::llround(frameNumber * 1e9 / fps));
}

Timecode TimecodeClock::fromNanoseconds(
    std::uint64_t nanoseconds, double fps) noexcept {
    if (fps <= 0.0) return {};
    const auto wholeFrames = static_cast<std::uint64_t>(
        std::llround(nanoseconds * fps / 1e9));
    const auto fpsInteger = std::max<std::uint64_t>(
        1,
        static_cast<std::uint64_t>(std::llround(fps)));
    const auto totalSeconds = wholeFrames / fpsInteger;
    return {
        static_cast<std::uint32_t>(totalSeconds / 3600),
        static_cast<std::uint32_t>((totalSeconds / 60) % 60),
        static_cast<std::uint32_t>(totalSeconds % 60),
        static_cast<std::uint32_t>(wholeFrames % fpsInteger),
    };
}
}
