#include <cassert>
#include <string>
#include "replay/replay_runner.h"
#include "export/export_validator.h"
int main(){
    const std::string fixture=R"({"cameraCount":2,"frameCount":240,"clockDriftPpm":12,"calibrationResidualPx":0.8,"occlusionRatio":0.1,"droppedFrames":3,"thermalStatus":0,"seed":42})";
    const auto a=nox::native::replay::ReplayRunner{}.run(fixture);
    const auto b=nox::native::replay::ReplayRunner{}.run(fixture);
    assert(a.valid);
    assert(a.outputDigest==b.outputDigest);
    nox::native::exporting::MotionSession session;
    session.provenanceHash="session-hash";
    session.scaleProvenance="measured-rig";
    session.observedConfidence=0.80;
    session.reportedConfidence=0.82;
    session.artifacts={nox::native::exporting::ArtifactKind::MotionTrajectory};
    assert(nox::native::exporting::ExportValidator::validate(session).valid);
    session.artifacts.push_back(nox::native::exporting::ArtifactKind::CanonicalVideo);
    assert(!nox::native::exporting::ExportValidator::validate(session).valid);
    session.artifacts.pop_back();
    session.reportedConfidence=0.95;
    assert(!nox::native::exporting::ExportValidator::validate(session).valid);
}