#include <cassert>
#include "lighting/illumination_statistics.h"
#include "lighting/directional_evidence.h"
#include "lighting/light_event_detector.h"
int main(){
 using namespace nox::native::lighting;
 IlluminationStatistics stats;
 auto a=stats.update({0.4,0.1,0.5,0.4,10});
 auto b=stats.update({0.8,0.2,0.7,0.4,20});
 assert(b.samples==2);
 assert(b.meanLuma>a.meanLuma);
 auto cue=DirectionalEvidence{}.estimate({0.8,0.2,0.7,0.4,20},b);
 assert(cue.confidence>=0.0);
 auto ev=LightEventDetector{}.update({0.9,0.2,0.8,0.4,30},b);
 assert(ev.timestampNanos==30);
 assert(ev.magnitude>0);
}