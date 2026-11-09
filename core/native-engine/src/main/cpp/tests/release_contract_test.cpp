#include <cassert>
#include <vector>
#include "export/schema_negotiator.h"
#include "replay/replay_matrix_validator.h"

int main() {
    using namespace nox::native;
    const exporting::SchemaCapabilities local{};
    const exporting::SchemaCapabilities remote{};
    const auto negotiated=exporting::SchemaNegotiator{}.negotiate(local,remote);
    assert(negotiated.compatible);
    assert(negotiated.version==1);

    const std::vector<replay::MatrixCase> cases={
        {"smoke",R"({"cameraCount":2,"frameCount":100,"clockDriftPpm":2,"calibrationResidualPx":0.5,"occlusionRatio":0.1,"droppedFrames":0,"thermalStatus":0,"seed":1})"},
        {"stress",R"({"cameraCount":6,"frameCount":600,"clockDriftPpm":20,"calibrationResidualPx":1.5,"occlusionRatio":0.2,"droppedFrames":5,"thermalStatus":1,"seed":2})"},
    };
    const auto result=replay::ReplayMatrixValidator{}.run(cases);
    assert(result.passed==2);
    assert(result.failed==0);
    return 0;
}
