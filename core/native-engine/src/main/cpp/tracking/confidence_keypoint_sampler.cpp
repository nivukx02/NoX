#include "tracking/confidence_keypoint_sampler.h"
#include <algorithm>
#include <map>

namespace nox::native::tracking {
std::vector<KeypointCandidate> ConfidenceKeypointSampler::select(
    const std::vector<KeypointCandidate>& input,
    std::size_t maxCount,
    std::size_t gridX,
    std::size_t gridY) const {
    if (maxCount == 0 || input.empty()) return {};

    std::map<std::pair<int, int>, KeypointCandidate> bins;
    for (const auto& candidate : input) {
        const int cellX = std::clamp(
            static_cast<int>(candidate.x * gridX),
            0,
            static_cast<int>(gridX) - 1);
        const int cellY = std::clamp(
            static_cast<int>(candidate.y * gridY),
            0,
            static_cast<int>(gridY) - 1);
        const auto key = std::make_pair(cellX, cellY);
        const auto iterator = bins.find(key);
        if (iterator == bins.end() ||
            iterator->second.confidence < candidate.confidence) {
            bins[key] = candidate;
        }
    }

    std::vector<KeypointCandidate> output;
    output.reserve(bins.size());
    for (const auto& entry : bins) output.push_back(entry.second);

    std::sort(output.begin(), output.end(),
              [](const auto& a, const auto& b) {
                  return a.confidence > b.confidence;
              });
    if (output.size() > maxCount) output.resize(maxCount);
    return output;
}
}


namespace nox::native::tracking {
std::vector<KeypointCandidate> ConfidenceKeypointSampler::selectSpatiotemporal(
    const std::vector<KeypointCandidate>& input,
    std::size_t maxCount,
    std::size_t gridX,
    std::size_t gridY,
    std::size_t temporalBins) const {
    if(maxCount==0||input.empty()||gridX==0||gridY==0||temporalBins==0)return{};

    using BinKey=std::tuple<int,int,std::uint32_t>;
    std::map<BinKey,KeypointCandidate> bins;
    for(const auto& candidate:input){
        const int cellX=std::clamp(
            static_cast<int>(candidate.x*gridX),0,static_cast<int>(gridX)-1);
        const int cellY=std::clamp(
            static_cast<int>(candidate.y*gridY),0,static_cast<int>(gridY)-1);
        const std::uint32_t t=static_cast<std::uint32_t>(
            candidate.temporalBin%static_cast<std::uint32_t>(temporalBins));
        const BinKey key{cellX,cellY,t};
        const auto it=bins.find(key);
        if(it==bins.end()||it->second.confidence<candidate.confidence)bins[key]=candidate;
    }

    std::vector<KeypointCandidate> output;
    output.reserve(std::min(maxCount,bins.size()));
    for(const auto& entry:bins)output.push_back(entry.second);
    std::sort(output.begin(),output.end(),[](const auto&a,const auto&b){
        if(a.confidence!=b.confidence)return a.confidence>b.confidence;
        if(a.temporalBin!=b.temporalBin)return a.temporalBin<b.temporalBin;
        if(a.y!=b.y)return a.y<b.y;
        if(a.x!=b.x)return a.x<b.x;
        return a.id<b.id;
    });
    if(output.size()>maxCount)output.resize(maxCount);
    return output;
}
}
