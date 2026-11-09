#pragma once
#include <cstdint>
#include <string>
#include "replay/replay_reader.h"
namespace nox::native::replay {
struct ReplayResult { std::uint64_t outputDigest{0}; double synchronizationConfidence{0}; double calibrationConfidence{0}; double recoveryConfidence{0}; std::uint64_t droppedObservations{0}; bool valid{false}; };
class ReplayRunner final { public: ReplayResult run(const std::string&) const; };
}