#pragma once
namespace nox::native::lighting { struct RGB{double r{0},g{0},b{0};}; class CameraColorNormalizer final{public:RGB normalize(const RGB&,const RGB&referenceGray)const noexcept;};}