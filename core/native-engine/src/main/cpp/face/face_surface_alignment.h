#pragma once
#include "face/face_track_state.h"
namespace nox::native::face{
class FaceSurfaceAlignment final{
public:
 static double shapeConsistency(const FaceGeometryState&state) noexcept;
 static FaceGeometryState stabilize(const FaceGeometryState&previous,const FaceGeometryState&current,double alpha=0.35) noexcept;
};
}