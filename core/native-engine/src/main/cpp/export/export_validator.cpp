#include "export/export_validator.h"
#include <algorithm>
namespace nox::native::exporting {
ValidationResult ExportValidator::validate(const MotionSession& s,std::uint32_t abi,std::uint32_t schema) noexcept {
    if(s.schemaVersion!=schema)return{false,"schema-version-mismatch"};
    if(s.nativeAbiVersion!=abi)return{false,"native-abi-mismatch"};
    if(s.provenanceHash.empty())return{false,"missing-provenance"};
    if(s.reportedConfidence+1e-9<s.observedConfidence)return{false,"confidence-regression"};
    if(s.reportedConfidence>s.observedConfidence+0.05)return{false,"confidence-inflation"};
    if(s.scaleProvenance.empty())return{false,"missing-scale-provenance"};
    for(const auto kind:s.artifacts){
        if(kind==ArtifactKind::CanonicalVideo||kind==ArtifactKind::RawFrameArchive||kind==ArtifactKind::DepthVideo)
            return{false,"motion-only-contract-violation"};
    }
    return{true,"valid"};
}
}