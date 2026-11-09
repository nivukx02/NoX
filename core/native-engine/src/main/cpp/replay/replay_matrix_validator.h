#pragma once
#include <string>
#include <vector>
#include "replay/replay_runner.h"
namespace nox::native::replay { struct MatrixCase{std::string name,fixture;double minSync{.5},minCalibration{.5},minRecovery{.4};};struct MatrixResult{std::size_t passed{0},failed{0};std::vector<std::string> failures;};class ReplayMatrixValidator final{public:MatrixResult run(const std::vector<MatrixCase>&)const;};}