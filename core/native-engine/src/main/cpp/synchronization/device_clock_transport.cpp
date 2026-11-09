#include "synchronization/device_clock_transport.h"
#include <algorithm>
#include <cmath>
namespace nox::native::sync { std::int64_t DeviceClockTransport::estimateOffset(const ClockPacket&p)const noexcept{return static_cast<std::int64_t>(p.local)-static_cast<std::int64_t>(p.remote);}
ClockExchangeEstimate DeviceClockTransport::estimateExchange(const ClockExchange&e)const noexcept{
ClockExchangeEstimate o;
if(e.localReceive<e.localSend||e.remoteSend<e.remoteReceive)return o;
const auto rtt=e.localReceive-e.localSend;
const auto remote=e.remoteSend-e.remoteReceive;
if(remote>rtt)return o;
const auto a=static_cast<std::int64_t>(e.remoteReceive)-static_cast<std::int64_t>(e.localSend);
const auto b=static_cast<std::int64_t>(e.remoteSend)-static_cast<std::int64_t>(e.localReceive);
o.offset=(a+b)/2;
o.roundTripNanos=rtt;
o.pathDelayNanos=rtt-remote;
o.confidence=std::clamp(std::exp(-static_cast<double>(o.pathDelayNanos)/2e6),0.0,1.0);
o.valid=true;
return o;
} }