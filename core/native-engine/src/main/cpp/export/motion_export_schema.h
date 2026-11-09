#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace nox::native::exporting {
enum class ArtifactKind { MotionTrajectory, JointRotations, FaceReference, EyeGaze, HandTrajectory, CalibrationBundle, SynchronizationBundle, SceneReference, LightingTelemetry, CanonicalVideo, RawFrameArchive, DepthVideo };
struct MotionSession {
    std::uint32_t schemaVersion{1};
    std::uint32_t nativeAbiVersion{2};
    std::string provenanceHash;
    std::string scaleProvenance;
    double observedConfidence{0};
    double reportedConfidence{0};
    std::vector<ArtifactKind> artifacts;
};
}