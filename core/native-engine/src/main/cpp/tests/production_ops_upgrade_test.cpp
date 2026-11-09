#include <cassert>
#include "capture/capture_trigger.h"
#include "runtime/network_health.h"
#include "synchronization/jitter_buffer.h"
#include "synchronization/device_clock_transport.h"

int main() {
    const auto trigger =
        nox::native::capture::CaptureTrigger{}.evaluate(true,0.9,100);
    assert(trigger.fired);

    const auto network =
        nox::native::runtime::NetworkHealthModel{}.evaluate(10.0,0.01,2.0);
    assert(network.healthy);

    nox::native::sync::JitterBuffer buffer(2);
    assert(buffer.push({100}));
    nox::native::sync::JitterSample sample{};
    assert(buffer.pop(sample));
    assert(sample.timestamp==100);

    const auto offset =
        nox::native::sync::DeviceClockTransport{}.estimateOffset({100,200});
    assert(offset==100);
    return 0;
}
