#include "eyes/gaze_fusion.h"
#include "eyes/binocular_vergence.h"

#include <algorithm>
#include <cmath>

namespace nox::native::eyes {
namespace {
Vec3 add(Vec3 a,Vec3 b) noexcept { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
Vec3 sub(Vec3 a,Vec3 b) noexcept { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
Vec3 mul(Vec3 a,double s) noexcept { return {a.x*s,a.y*s,a.z*s}; }
double dot(Vec3 a,Vec3 b) noexcept { return a.x*b.x+a.y*b.y+a.z*b.z; }
double norm(Vec3 a) noexcept { return std::sqrt(dot(a,a)); }
bool finiteVec(Vec3 a) noexcept {
    return std::isfinite(a.x) &&
        std::isfinite(a.y) &&
        std::isfinite(a.z);
}
Vec3 normalize(Vec3 a) noexcept {
    if(!finiteVec(a)) return {0,0,1};
    const double n=norm(a);
    return !std::isfinite(n) || n<1e-12
        ? Vec3{0,0,1}
        : mul(a,1.0/n);
}
Vec3 cross(Vec3 a,Vec3 b) noexcept {
    return {
        a.y*b.z-a.z*b.y,
        a.z*b.x-a.x*b.z,
        a.x*b.y-a.y*b.x,
    };
}

double radiansToDegrees(double value) noexcept {
    constexpr double kPi =
        3.1415926535897932384626433832795;
    return value * 180.0 / kPi;
}

double angleDegrees(
    Vec3 a,
    Vec3 b) noexcept {
    const double cosine =
        std::clamp(
            dot(normalize(a),normalize(b)),
            -1.0,
            1.0);
    return radiansToDegrees(
        std::acos(cosine));
}

double rayParallaxDegrees(
    Vec3 gaze,
    Vec3 center,
    Vec3 fixation) noexcept {
    const Vec3 toward =
        sub(fixation,center);
    if(norm(toward)<1e-9)return 0.0;
    return angleDegrees(gaze,toward);
}

Vec3 headToWorld(
    Vec3 headRelative,
    Vec3 headForward,
    Vec3 worldUp) noexcept {
    const Vec3 forward=normalize(headForward);
    Vec3 up=sub(
        worldUp,
        mul(forward,dot(worldUp,forward)));
    up=normalize(up);
    const Vec3 right=normalize(cross(up,forward));
    const Vec3 correctedUp=normalize(cross(forward,right));
    return normalize(
        add(
            add(
                mul(right,headRelative.x),
                mul(correctedUp,headRelative.y)),
            mul(forward,headRelative.z)));
}
}

GazeState GazeFusion::update(
    const EyeGeometry& left,
    const EyeGeometry& right,
    const Vec3& head,
    const Vec3& up) const noexcept {
    GazeState output;
    if(
        !finiteVec(left.center) ||
        !finiteVec(right.center) ||
        !finiteVec(left.gaze) ||
        !finiteVec(right.gaze) ||
        !finiteVec(head) ||
        !finiteVec(up) ||
        !std::isfinite(left.confidence) ||
        !std::isfinite(right.confidence)) {
        output.failure_flags =
            GAZE_FAILURE_INVALID_EYE |
            GAZE_FAILURE_HIGH_UNCERTAINTY;
        output.accepted=false;
        output.consistent=false;
        return output;
    }

    const Vec3 mean=normalize(mul(add(left.gaze,right.gaze),0.5));
    const double agreement=
        std::clamp(dot(
            normalize(left.gaze),
            normalize(right.gaze)),
            0.0,
            1.0);
    output.angular_consistency=agreement;

    const double ipd =
        norm(sub(left.center,right.center));
    const double ipdQuality =
        ipd < 0.045
            ? 0.0
            : ipd > 0.085
                ? 0.0
                : ipd <= 0.062
                    ? (ipd - 0.045) / 0.017
                    : (0.085 - ipd) / 0.023;

    output.head_relative=mean;
    output.interpupillary_distance_meters=ipd;
    output.ipd_quality=std::clamp(ipdQuality,0.0,1.0);
    output.world_relative=headToWorld(mean,head,up);

    const auto vergence=BinocularVergenceSolver{}.solve(left,right);
    if(vergence.valid){
        output.fixationPoint=vergence.point;
        output.fixationDistance=vergence.distance;
        output.fixation_uncertainty_meters =
            std::max(0.0, vergence.uncertaintyMeters);
        output.vergence_confidence=
            std::clamp(
                vergence.confidence,
                0.0,
                1.0);
        output.parallax_quality=
            std::clamp(
                vergence.parallaxDegrees /
                    2.0,
                0.0,
                1.0);
        output.uncertainty_confidence=
            std::clamp(
                (
                    vergence.conditioning * 0.20 +
                    std::exp(
                        -vergence.uncertaintyMeters /
                        0.025) * 0.80
                ),
                0.0,
                1.0);
    }

    output.angular_disagreement_degrees =
        0.5 *
        (
            angleDegrees(
                left.gaze,
                mean) +
            angleDegrees(
                right.gaze,
                mean)
        );

    if(vergence.valid){
        const double leftParallax =
            rayParallaxDegrees(
                left.gaze,
                left.center,
                vergence.point);
        const double rightParallax =
            rayParallaxDegrees(
                right.gaze,
                right.center,
                vergence.point);
        output.parallax_degrees =
            0.5 *
            (leftParallax+rightParallax);
    }

    output.binocular_confidence=
        std::clamp(
            std::min(left.confidence,right.confidence),
            0.0,
            1.0)*
        agreement*
        (0.88 + 0.12*output.ipd_quality);

    const double headAlignment =
        std::max(
            0.0,
            dot(mean,normalize(head)));

    output.temporal_confidence=
        0.5*output.binocular_confidence+
        0.5*headAlignment;
    output.temporal_consistency=
        std::clamp(
            0.55*output.temporal_confidence+
                0.45*headAlignment,
            0.0,
            1.0);

    const double eyeConfidencePenalty =
        1.0 -
        std::min(
            left.confidence,
            right.confidence);
    const double angularPenalty =
        std::clamp(
            output.angular_disagreement_degrees /
                30.0,
            0.0,
            1.0);
    const double metricUncertaintyPenalty =
        vergence.valid
            ? std::clamp(
                vergence.uncertaintyMeters /
                    0.10,
                0.0,
                1.0)
            : 0.75;

    output.eye_uncertainty =
        std::clamp(
            eyeConfidencePenalty * 0.25 +
            angularPenalty * 0.20 +
            metricUncertaintyPenalty * 0.55,
            0.0,
            1.0);
    output.uncertainty_confidence=
        1.0-output.eye_uncertainty;
    output.parallax_quality=
        std::clamp(
            output.parallax_degrees/2.0,
            0.0,
            1.0);

    output.failure_flags=GAZE_FAILURE_NONE;
    if(!left.valid || !right.valid){
        output.failure_flags|=GAZE_FAILURE_INVALID_EYE;
    }
    if(output.binocular_confidence<=0.35){
        output.failure_flags|=GAZE_FAILURE_LOW_BINOCULAR_CONFIDENCE;
    }
    if(output.ipd_quality<=0.18){
        output.failure_flags|=GAZE_FAILURE_IPD_OUT_OF_RANGE;
    }
    if(output.angular_disagreement_degrees>=12.0){
        output.failure_flags|=GAZE_FAILURE_ANGULAR_DISAGREEMENT;
    }
    if(
        vergence.valid &&
        vergence.parallaxDegrees < 0.25 &&
        vergence.convergenceAngleDegrees < 8.0) {
        output.failure_flags|=GAZE_FAILURE_LOW_PARALLAX;
    }
    if(output.eye_uncertainty>=0.45){
        output.failure_flags|=GAZE_FAILURE_HIGH_UNCERTAINTY;
    }
    if(headAlignment<0.35){
        output.failure_flags|=GAZE_FAILURE_HEAD_MISALIGNMENT;
    }

    const auto fatalGazeFlags =
        GAZE_FAILURE_INVALID_EYE |
        GAZE_FAILURE_ANGULAR_DISAGREEMENT |
        GAZE_FAILURE_LOW_BINOCULAR_CONFIDENCE |
        GAZE_FAILURE_HIGH_UNCERTAINTY |
        GAZE_FAILURE_HEAD_MISALIGNMENT;

    output.accepted =
        (
            output.failure_flags &
            fatalGazeFlags
        ) == 0u &&
        output.binocular_confidence > 0.25 &&
        output.ipd_quality > 0.08 &&
        output.angular_disagreement_degrees < 15.0 &&
        output.eye_uncertainty < 0.55;

    output.consistent =
        agreement > 0.85 &&
        output.temporal_consistency > 0.70 &&
        (
            output.failure_flags &
                (
                    GAZE_FAILURE_INVALID_EYE |
                    GAZE_FAILURE_ANGULAR_DISAGREEMENT |
                    GAZE_FAILURE_LOW_BINOCULAR_CONFIDENCE |
                    GAZE_FAILURE_HIGH_UNCERTAINTY
                )
        ) == 0u;
    output.strict_submillimeter_qualified =
        output.accepted &&
        output.consistent &&
        output.fixation_uncertainty_meters <= 0.001 &&
        output.angular_disagreement_degrees <= 0.75 &&
        output.binocular_confidence >= 0.90 &&
        output.ipd_quality >= 0.90;
    return output;
}


GazeState GazeFusion::updateTemporal(
    const EyeGeometry& left,
    const EyeGeometry& right,
    const Vec3& head,
    const Vec3& up,
    const GazeState& previous,
    double dtSeconds,
    const GazeTemporalOptions& options) const noexcept {
    GazeState current =
        update(
            left,
            right,
            head,
            up);

    if (
        !std::isfinite(dtSeconds) ||
        dtSeconds <= 0.0 ||
        dtSeconds > options.maximumDtSeconds) {
        current.failure_flags |=
            GAZE_FAILURE_HIGH_UNCERTAINTY;
        current.temporal_confidence *= 0.50;
        current.temporal_consistency *= 0.50;
        return current;
    }

    if (
        !previous.accepted ||
        !finiteVec(previous.world_relative)) {
        return current;
    }

    const double innovationDegrees =
        angleDegrees(
            previous.world_relative,
            current.world_relative);

    const double angularVelocity =
        innovationDegrees /
        dtSeconds;

    const double velocityGate =
        options.maximumAngularVelocityDegreesPerSecond <= 0.0
            ? 0.0
            : std::clamp(
                1.0 -
                    std::max(
                        0.0,
                        angularVelocity -
                            options.maximumAngularVelocityDegreesPerSecond) /
                    options.maximumAngularVelocityDegreesPerSecond,
                0.0,
                1.0);

    const double innovationGate =
        options.maximumTemporalInnovationDegrees <= 0.0
            ? 0.0
            : std::clamp(
                1.0 -
                    std::max(
                        0.0,
                        innovationDegrees -
                            options.maximumTemporalInnovationDegrees) /
                    options.maximumTemporalInnovationDegrees,
                0.0,
                1.0);

    const double residualConfidence =
        options.innovationScaleDegrees <= 0.0
            ? innovationGate
            : std::clamp(
                std::exp(
                    -innovationDegrees /
                    options.innovationScaleDegrees),
                0.0,
                1.0);

    current.angular_velocity_degrees_per_second =
        angularVelocity;
    current.temporal_residual_degrees =
        innovationDegrees;

    current.temporal_consistency =
        std::clamp(
            current.temporal_consistency *
                0.45 +
            velocityGate *
                0.20 +
            innovationGate *
                0.15 +
            residualConfidence *
                0.20,
            0.0,
            1.0);

    current.temporal_confidence =
        std::clamp(
            current.temporal_confidence *
                0.55 +
            velocityGate *
                0.15 +
            innovationGate *
                0.15 +
            residualConfidence *
                0.15,
            0.0,
            1.0);

    if (
        current.temporal_confidence <
        options.minimumTemporalConfidence) {
        current.failure_flags |=
            GAZE_FAILURE_HIGH_UNCERTAINTY;
    }

    if (
        angularVelocity >
        options.maximumAngularVelocityDegreesPerSecond ||
        innovationDegrees >
        options.maximumTemporalInnovationDegrees) {
        current.accepted = false;
        current.consistent = false;
    }

    current.consistent =
        current.accepted &&
        current.temporal_consistency > 0.70 &&
        current.temporal_confidence >
            options.minimumTemporalConfidence;

    return current;
}

} // namespace nox::native::eyes
