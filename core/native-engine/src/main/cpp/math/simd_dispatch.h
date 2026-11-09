#pragma once
namespace nox::native::math { enum class SimdLevel{Scalar,SSE,AVX,NEON}; SimdLevel detectSimd()noexcept; }