#include "kinematics/skeleton_retargeter.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace nox::native::kinematics {
namespace {
double median(std::vector<double> values){
    if(values.empty())return 0.0;
    std::sort(values.begin(),values.end());
    const auto mid=values.size()/2;
    return values.size()%2
        ? values[mid]
        : 0.5*(values[mid-1]+values[mid]);
}
}

RetargetResult SkeletonRetargeter::solve(
    const std::vector<double>& sourceLengths,
    const std::vector<double>& targetLengths,
    const std::vector<JointPair>& mapping) const {
    RetargetResult output;
    if(mapping.empty())return output;

    std::vector<double> ratios;
    ratios.reserve(mapping.size());

    std::size_t validMappings=0;
    for(const auto& pair:mapping){
        if(pair.source>=sourceLengths.size()||
           pair.target>=targetLengths.size()||
           sourceLengths[pair.source]<=1e-9||
           targetLengths[pair.target]<=1e-9){
            continue;
        }
        ratios.push_back(
            targetLengths[pair.target]/
            sourceLengths[pair.source]);
        ++validMappings;
    }

    if(ratios.empty())return output;

    output.globalScale=median(ratios);
    const double coverage=
        static_cast<double>(validMappings)/
        static_cast<double>(mapping.size());
    output.mappingCoverage=coverage;

    std::vector<double> absoluteDeviations;
    absoluteDeviations.reserve(ratios.size());
    for(const double ratio:ratios){
        absoluteDeviations.push_back(
            std::abs(ratio-output.globalScale));
    }
    const double mad=median(absoluteDeviations);

    double weightedError=0.0;
    std::size_t produced=0;
    output.weights.reserve(validMappings);

    for(const auto& pair:mapping){
        if(pair.source>=sourceLengths.size()||
           pair.target>=targetLengths.size()||
           sourceLengths[pair.source]<=1e-9||
           targetLengths[pair.target]<=1e-9){
            continue;
        }
        const double ratio=
            targetLengths[pair.target]/
            sourceLengths[pair.source];
        const double deviation=
            std::abs(ratio-pair.lengthScale);
        const double robustWeight=
            1.0/(1.0+deviation/0.10);
        output.weights.push_back(ratio);
        weightedError+=robustWeight*deviation;
        ++produced;
    }

    output.meanError=
        produced==0
            ? 0.0
            : weightedError/static_cast<double>(produced);

    const double scaleConsistency=
        std::exp(-mad/std::max(output.globalScale*0.15,1e-6));
    const double mappingConfidence=
        std::clamp(coverage,0.0,1.0);
    const double errorConfidence=
        std::exp(-output.meanError/std::max(
            output.globalScale*0.20,1e-6));

    output.confidence=std::clamp(
        mappingConfidence*
        scaleConsistency*
        errorConfidence,
        0.0,
        1.0);
    output.valid=
        output.mappingCoverage>=0.70&&
        output.confidence>=0.70;
    return output;
}

} // namespace nox::native::kinematics
