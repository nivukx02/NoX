#pragma once
#include <vector>
namespace nox::native::sync {
struct AudioSyncResult { double offsetSamples{0}; double correlation{0}; double confidence{0}; bool valid{false}; };
class AudioEventSync final { public: AudioSyncResult estimate(const std::vector<double>&,const std::vector<double>&) const; };
}