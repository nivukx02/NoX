#pragma once
#include <cstdint>
#include <vector>
#include "tracking/joint_track.h"
namespace nox::native::tracking{
struct BodyTrackState{std::uint64_t timestamp_nanos{0};std::uint64_t subject_id{0};std::vector<JointState> joints;double mean_confidence{0};
    double mean_uncertainty{1};
    double occlusion_ratio{1};
    double mean_innovation{0};
    double p95_innovation{0};
    double mean_velocity{0};
    double mean_acceleration{0};
    double mean_jerk{0};
    double dynamic_stability{0};
    double micro_motion_retention{0};
    bool healthy{false};};
}
