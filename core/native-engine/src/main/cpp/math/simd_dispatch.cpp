#include "math/simd_dispatch.h"
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
namespace nox::native::math{SimdLevel detectSimd()noexcept{return SimdLevel::NEON;}}
#elif defined(__AVX2__)
namespace nox::native::math{SimdLevel detectSimd()noexcept{return SimdLevel::AVX;}}
#elif defined(__SSE4_2__)
namespace nox::native::math{SimdLevel detectSimd()noexcept{return SimdLevel::SSE;}}
#else
namespace nox::native::math{SimdLevel detectSimd()noexcept{return SimdLevel::Scalar;}}
#endif