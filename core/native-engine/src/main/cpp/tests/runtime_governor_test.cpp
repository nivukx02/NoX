#include <cassert>
#include "runtime/runtime_budget.h"
#include "runtime/queue_pressure.h"
#include "runtime/thermal_policy.h"
#include "runtime/work_scheduler.h"
int main(){using namespace nox::native::runtime;RuntimeBudget b(100);b.beginFrame(1000);assert(b.remainingNanos(1050)==50);assert(b.endFrame(1101).missed);assert(QueuePressureModel{}.evaluate(90,100).shedding);auto th=ThermalPolicy{}.evaluate(4,.2);assert(th.constrained&&th.workloadScale<1);auto low=WorkScheduler{}.decide(Priority::Low,90,100,4,.9);assert(low.shed&&!low.accepted);auto critical=WorkScheduler{}.decide(Priority::Critical,90,100,4,.9);assert(critical.accepted);}