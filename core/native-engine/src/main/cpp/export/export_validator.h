#pragma once
#include <string>
#include "export/motion_export_schema.h"
namespace nox::native::exporting {
struct ValidationResult { bool valid{false}; std::string reason; };
class ExportValidator final {
public: static ValidationResult validate(const MotionSession&,std::uint32_t expectedAbi=2,std::uint32_t expectedSchema=1) noexcept;
};
}