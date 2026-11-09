#include "tracking/multi_person_interaction.h"
#include <algorithm>
#include <cmath>

namespace nox::native::tracking {

std::vector<Interaction> MultiPersonInteraction::evaluate(
    const std::vector<SubjectState>& subjects,
    double interactionRadius,
    double contactRadius) const {
    std::vector<Interaction> output;
    const double maxRadius=std::max(interactionRadius,contactRadius);
    const double safeContact=std::max(contactRadius,1e-6);

    for(std::size_t i=0;i<subjects.size();++i){
        for(std::size_t j=i+1;j<subjects.size();++j){
            const auto& a=subjects[i];
            const auto& b=subjects[j];
            const double dx=a.x-b.x;
            const double dy=a.y-b.y;
            const double dz=a.z-b.z;
            const double distance=std::sqrt(
                dx*dx+dy*dy+dz*dz);
            if(distance>maxRadius)continue;

            const double relativeVelocityX=a.vx-b.vx;
            const double relativeVelocityY=a.vy-b.vy;
            const double relativeVelocityZ=a.vz-b.vz;
            const double approachRate=distance>1e-6
                ? (dx*relativeVelocityX+
                   dy*relativeVelocityY+
                   dz*relativeVelocityZ)/distance
                : 0.0;

            const double geometricConfidence=
                std::exp(-distance/safeContact);
            const double approachingBoost=
                1.0+
                std::clamp(
                    -approachRate/0.5,
                    -0.4,
                    0.8);
            const double subjectConfidence=
                std::min(
                    std::clamp(a.confidence,0.0,1.0),
                    std::clamp(b.confidence,0.0,1.0));

            const double probability=std::clamp(
                subjectConfidence*
                geometricConfidence*
                std::clamp(approachingBoost,0.0,1.8),
                0.0,
                1.0);

            const bool contactState=
                distance<=safeContact&&probability>=0.25;
            const bool interactionState=
                probability>=0.35||
                (distance<=safeContact&&approachRate<0.0);

            output.push_back({
                a.id,
                b.id,
                distance,
                approachRate,
                probability,
                contactState||interactionState,
            });
        }
    }
    return output;
}

} // namespace nox::native::tracking
