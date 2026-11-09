#pragma once
#include <cstdint>
#include <vector>
namespace nox::native::tracking {
struct SubjectState { std::uint64_t id{0}; double x{0},y{0},z{0}; double confidence{0}; double vx{0},vy{0},vz{0}; };
struct Interaction {
    std::uint64_t a{0},b{0};
    double distance{0};
    double approachRate{0};
    double contactProbability{0};
    bool interacting{false};
    double contactAgeSeconds{0};
    double peakApproachRate{0};
    double impactScore{0};
    double interactionConfidence{0};
    bool staleEvidenceRejected{false};
};
class MultiPersonInteraction final { public: std::vector<Interaction> evaluate(const std::vector<SubjectState>&,double interactionRadius=1.0,double contactRadius=.15) const; };
}