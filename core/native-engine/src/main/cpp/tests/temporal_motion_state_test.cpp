#include <cassert>
#include <cmath>
#include "tracking/temporal_motion_state.h"

int main() {
    using namespace nox::native::tracking;
    TemporalMotionState state;
    state.reset(0, {0,0,0}, 1.0);
    state.correct(100000000, {1,0,0}, 1.0);
    const auto corrected = state.state();
    assert(corrected.position[0] > 0.1);

    const double before = state.confidence();
    state.correct(200000000, {100,0,0}, 1.0);
    assert(state.confidence() < before);

    const auto predicted = state.predict(300000000);
    assert(predicted.variance[0] >= corrected.variance[0]);
    assert(predicted.position[0] > corrected.position[0]);

    TemporalMotionState highRate;
    highRate.reset(
        0,
        {0,0,0},
        1.0);
    highRate.correct(
        8'333'333,
        {0.008333333,0,0},
        1.0);
    const auto highRateState=highRate.state();

    TemporalMotionState lowRate;
    lowRate.reset(
        0,
        {0,0,0},
        1.0);
    lowRate.correct(
        33'333'333,
        {0.033333333,0,0},
        1.0);
    const auto lowRateState=lowRate.state();

    assert(
        highRate.lastIntervalSeconds() <
        lowRate.lastIntervalSeconds());
    assert(std::abs(
        highRateState.velocity[0] -
        lowRateState.velocity[0]) < 1.0);

    TemporalMotionState backwards;
    backwards.reset(
        1'000'000'000ull,
        {1,0,0},
        1.0);
    const auto beforeBackwards=backwards.state();
    backwards.predict(900'000'000ull);
    assert(std::abs(
        backwards.state().position[0] -
        beforeBackwards.position[0]) < 1e-12);

    TemporalMotionState staleCorrection;
    staleCorrection.reset(
        1'000'000'000ull,
        {2, 0, 0},
        1.0);
    const auto beforeStale = staleCorrection.state();
    const auto staleResult = staleCorrection.correct(
        900'000'000ull,
        {99, 0, 0},
        1.0);
    assert(std::abs(
        staleResult.position[0] -
        beforeStale.position[0]) < 1e-12);
    assert(std::abs(
        staleCorrection.state().position[0] -
        beforeStale.position[0]) < 1e-12);

    return 0;
}
