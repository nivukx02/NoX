#pragma once
#include <cstdint>
#include <map>
#include <vector>
#include "tracking/multi_person_interaction.h"

namespace nox::native::tracking {

struct InteractionTemporalConfig {
    double timeConstantSeconds{0.12};
    double engageThreshold{0.45};
    double releaseThreshold{0.25};
    std::uint32_t maxMissingFrames{6};
    double velocitySmoothingAlpha{0.35};
    double contactAgeSmoothingAlpha{0.12};
    double impactApproachRate{0.90};
    double impactScoreScale{0.28};
    double maximumDeltaSeconds{0.25};
};

class InteractionTemporalFilter final {
public:
    explicit InteractionTemporalFilter(
        InteractionTemporalConfig config = {});

    void reset();

    std::vector<Interaction> update(
        const std::vector<Interaction>& observations,
        double deltaSeconds);

private:
    struct Track {
        Interaction value{};
        double confidence{0.0};
        double relativeVelocity{0.0};
        double peakApproachRate{0.0};
        double contactAgeSeconds{0.0};
        double impactScore{0.0};
        double distanceEma{0.0};
        std::uint32_t missingFrames{0};
        bool initialized{false};
    };

    using Key = std::pair<std::uint64_t,std::uint64_t>;
    static Key keyFor(const Interaction& interaction) noexcept;

    InteractionTemporalConfig config_;
    std::map<Key,Track> tracks_;
};

} // namespace nox::native::tracking
