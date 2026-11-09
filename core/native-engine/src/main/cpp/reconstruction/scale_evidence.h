#pragma once
#include <string>
namespace nox::native::scale{
enum class Source{MeasuredRig,KnownSceneDimension,StableBaseline,SupportedDepth,BodyConstraint,RelativeDepthOnly};
struct ScaleEvidence {
    Source source{Source::RelativeDepthOnly};
    double scale{0};
    double uncertainty{1};
    std::string provenance;
    double confidence{1.0};
    double traceability{0.0};
    double temporalStability{1.0};
};
}