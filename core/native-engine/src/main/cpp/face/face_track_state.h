#pragma once
#include <cstdint>
#include <vector>
namespace nox::native::face {
struct Vec2{double x{0},y{0};};
struct Vec3{double x{0},y{0},z{0};};
struct FaceLandmarkObservation{std::uint32_t id{0};std::uint32_t camera_id{0};Vec2 pixel{};Vec3 ray_origin{};Vec3 ray_direction{0,0,1};double confidence{0};};
struct FaceLandmarkState{
std::uint32_t id{0};
Vec3 position{};
double confidence{0};
double uncertainty{1};
double triangulation_conditioning{0};
std::uint32_t supporting_pairs{0};
double innovation_magnitude{0.0};
double temporal_confidence{0.0};
bool metric{false};
};
enum FaceFailureFlags : std::uint32_t {
 FACE_FAILURE_NONE = 0u,
 FACE_FAILURE_NO_MULTI_VIEW_SUPPORT = 1u << 0u,
 FACE_FAILURE_HIGH_OUTLIER_RATIO = 1u << 1u,
 FACE_FAILURE_LOW_CONDITIONING = 1u << 2u,
 FACE_FAILURE_HIGH_UNCERTAINTY = 1u << 3u,
 FACE_FAILURE_LOW_CONFIDENCE = 1u << 4u,
};

struct FaceGeometryState{
 std::uint64_t timestamp_nanos{0};
 Vec3 head_center{};
 Vec3 head_forward{0,0,1};
 std::vector<FaceLandmarkState> landmarks;
 double confidence{0};
 double reprojection_rms{0};
 double robust_support_ratio{0};
 double mean_landmark_uncertainty{0};
 double conditioning_score{0};
 double temporal_confidence{0};
 double temporal_consistency{0};
 double temporal_innovation_p95{0};
 double temporal_innovation_max{0};
 double temporal_velocity_meters_per_second{0};
 double mean_innovation{0};
 double p95_innovation{0};
 double mean_velocity{0};
 double mean_acceleration{0};
 double mean_jerk{0};
 double micro_expression_retention{1};
 double dynamic_stability{0};
 std::uint32_t rejected_pair_count{0};
 std::uint32_t failure_flags{FACE_FAILURE_NONE};
 bool metric{false};
};
}