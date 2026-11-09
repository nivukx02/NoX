#include "tracking/adaptive_view_selector.h"

#include <cassert>
#include <vector>

int main() {
    using namespace nox::native::tracking;

    AdaptiveViewSelector selector;
    std::vector<ViewCandidate> basicCandidates;
    basicCandidates.reserve(4);
    basicCandidates.push_back(ViewCandidate{
        .cameraId = 1,
        .coverage = 0.9,
        .confidence = 0.9,
        .latency = 2.0,
        .observability = 0.9,
        .novelty = 0.9,
        .healthScore = 1.0,
    });
    basicCandidates.push_back(ViewCandidate{
        .cameraId = 2,
        .coverage = 0.95,
        .confidence = 0.95,
        .latency = 2.0,
        .observability = 0.2,
        .novelty = 0.1,
        .healthScore = 1.0,
    });
    basicCandidates.push_back(ViewCandidate{
        .cameraId = 3,
        .coverage = 0.7,
        .confidence = 0.8,
        .latency = 2.0,
        .observability = 0.9,
        .novelty = 0.9,
        .healthScore = 0.95,
    });
    basicCandidates.push_back(ViewCandidate{
        .cameraId = 4,
        .coverage = 0.99,
        .confidence = 0.99,
        .latency = 1.0,
        .observability = 0.99,
        .novelty = 0.99,
        .healthScore = 0.05,
    });
    const auto result =
        selector.select(basicCandidates,2);

    assert(result.selected.size()==2);
    assert(result.selected[0]==1);
    assert(result.selected[1]==3);

    std::vector<ViewCandidate> geometryCandidates;
    geometryCandidates.reserve(3);
    const auto geometryAware=
        selector.select(std::vector<ViewCandidate>{
            ViewCandidate{
                .cameraId=10,.coverage=0.98,.confidence=0.98,
                .latency=1.0,.observability=0.95,.novelty=0.95,
                .healthScore=1.0,.position={0.0,0.0,0.0},
                .viewDirection={0.0,0.0,1.0},
            },
            ViewCandidate{
                .cameraId=11,.coverage=0.97,.confidence=0.97,
                .latency=1.0,.observability=0.95,.novelty=0.95,
                .healthScore=1.0,.position={0.03,0.0,0.0},
                .viewDirection={0.0,0.0,0.98},
            },
            ViewCandidate{
                .cameraId=12,.coverage=0.93,.confidence=0.93,
                .latency=1.5,.observability=0.86,.novelty=0.88,
                .healthScore=0.98,.position={0.75,0.0,0.0},
                .viewDirection={0.75,0.0,0.66},
            },
        },2);

    assert(geometryAware.selected.size()==2);
    assert(geometryAware.selected.front()==10);
    assert(geometryAware.selected.back()==12);
    assert(geometryAware.geometricDiversity>0.40);


    const auto qualityAware=
        selector.select(std::vector<ViewCandidate>{
            ViewCandidate{
                .cameraId = 20,
                .coverage = 0.98,
                .confidence = 0.98,
                .latency = 1.0,
                .thermallyConstrained = false,
                .observability = 0.95,
                .novelty = 0.95,
                .healthScore = 1.0,
                .occlusionRisk = 0.05,
                .depthUncertainty = 0.05,
                .synchronizationConfidence = 0.98,
                .position = {0.0,0.0,0.0},
                .viewDirection = {0.0,0.0,1.0},
            },
            ViewCandidate{
                .cameraId = 21,
                .coverage = 0.99,
                .confidence = 0.99,
                .latency = 1.0,
                .thermallyConstrained = false,
                .observability = 0.95,
                .novelty = 0.95,
                .healthScore = 1.0,
                .occlusionRisk = 0.80,
                .depthUncertainty = 0.85,
                .synchronizationConfidence = 0.35,
                .position = {0.10,0.0,0.0},
                .viewDirection = {0.0,0.0,1.0},
            },
            ViewCandidate{
                .cameraId = 22,
                .coverage = 0.88,
                .confidence = 0.91,
                .latency = 1.5,
                .thermallyConstrained = false,
                .observability = 0.90,
                .novelty = 0.90,
                .healthScore = 0.96,
                .occlusionRisk = 0.08,
                .depthUncertainty = 0.12,
                .synchronizationConfidence = 0.94,
                .position = {0.85,0.0,0.0},
                .viewDirection = {0.70,0.0,0.71},
            },
        },2);

    assert(qualityAware.selected.size()==2);
    assert(qualityAware.selected.front()==20);
    assert(qualityAware.selected.back()==22);

    const auto oppositeViewSet =
        selector.select(
            {
                {
                    30,0.96,0.96,1.0,false,0.94,0.92,0.99,
                    0.03,0.05,0.99,
                    {0.0,0.0,0.0},
                    {0.0,0.0,1.0},
                },
                {
                    31,0.96,0.96,1.0,false,0.94,0.92,0.99,
                    0.03,0.05,0.99,
                    {0.8,0.0,0.0},
                    {0.0,0.0,-1.0},
                },
            },
            2);

    assert(oppositeViewSet.selected.size()==2);
    assert(oppositeViewSet.geometricDiversity>0.80);
    assert(oppositeViewSet.minimumPairwiseBaseline>0.70);
    assert(oppositeViewSet.minimumPairwiseParallaxRadians > 3.0);
    assert(oppositeViewSet.conditioningScore > 0.50);
    assert(oppositeViewSet.informationGainScore > 0.50);

    return 0;
}
