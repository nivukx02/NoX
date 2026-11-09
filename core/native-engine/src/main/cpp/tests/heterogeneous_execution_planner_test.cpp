#include <cassert>

#include "runtime/heterogeneous_execution_planner.h"

using nox::native::runtime::ExecutionBackend;
using nox::native::runtime::ExecutionHardwareProfile;
using nox::native::runtime::ExecutionStageProfile;
using nox::native::runtime::HeterogeneousExecutionPlanner;

int main() {
    const HeterogeneousExecutionPlanner planner{};

    {
        const auto decision = planner.choose(
            ExecutionStageProfile{true, true, false, 1.0, 0.2},
            ExecutionHardwareProfile{true, true, true, true, 0.8, 0.9, 0.95, 0.8, 0.9});
        assert(decision.backend == ExecutionBackend::NPU);
        assert(!decision.shed_optional_work);
    }

    {
        const auto decision = planner.choose(
            ExecutionStageProfile{false, true, false, 1.0, 0.2},
            ExecutionHardwareProfile{true, true, true, false, 0.8, 0.9, 0.2, 0.7, 0.8});
        assert(decision.backend == ExecutionBackend::GPU);
    }

    {
        const auto decision = planner.choose(
            ExecutionStageProfile{false, true, true, 1.0, 0.2},
            ExecutionHardwareProfile{true, true, false, false, 0.1, 0.0, 0.0, 0.08, 0.09});
        assert(decision.shed_optional_work);
    }

    return 0;
}
