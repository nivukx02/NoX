#include "synchronization/device_clock_transport.h"

int main() {
    using namespace nox::native::sync;
    const ClockExchange exchange{1000,1600,1700,1100};
    const auto result=DeviceClockTransport{}.estimateExchange(exchange);
    return (
        result.valid &&
        result.offset==600 &&
        result.roundTripNanos==100 &&
        result.pathDelayNanos==0
    ) ? 0 : 1;
}
