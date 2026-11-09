#include "math/se3.h"

#include <algorithm>
#include <cmath>

namespace nox::native::math {

namespace {
double dot(const Vec3d& a, const Vec3d& b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
double norm(const Vec3d& value) noexcept {
    return std::sqrt(dot(value, value));
}
Vec3d matVec(
    const double matrix[3][3],
    const Vec3d& value) noexcept {
    return {
        matrix[0][0] * value.x + matrix[0][1] * value.y + matrix[0][2] * value.z,
        matrix[1][0] * value.x + matrix[1][1] * value.y + matrix[1][2] * value.z,
        matrix[2][0] * value.x + matrix[2][1] * value.y + matrix[2][2] * value.z,
    };
}
void skew(const Vec3d& w, double matrix[3][3]) noexcept {
    matrix[0][0] = 0.0; matrix[0][1] = -w.z; matrix[0][2] = w.y;
    matrix[1][0] = w.z; matrix[1][1] = 0.0; matrix[1][2] = -w.x;
    matrix[2][0] = -w.y; matrix[2][1] = w.x; matrix[2][2] = 0.0;
}
}

Quaterniond Quaterniond::normalized() const noexcept {
    const double magnitude =
        std::sqrt(w * w + x * x + y * y + z * z);
    if (magnitude < 1e-15) return {};
    return {w / magnitude, x / magnitude, y / magnitude, z / magnitude};
}

Quaterniond Quaterniond::conjugate() const noexcept {
    return {w, -x, -y, -z};
}

Quaterniond Quaterniond::multiply(
    const Quaterniond& a,
    const Quaterniond& b) noexcept {
    return {
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    };
}

Quaterniond Quaterniond::slerp(
    const Quaterniond& from,
    const Quaterniond& to,
    double alpha) noexcept {
    Quaterniond target=to.normalized();
    Quaterniond source=from.normalized();
    double cosine=
        source.w*target.w+
        source.x*target.x+
        source.y*target.y+
        source.z*target.z;
    if(cosine<0.0){
        target={-target.w,-target.x,-target.y,-target.z};
        cosine=-cosine;
    }
    const double t=std::clamp(alpha,0.0,1.0);
    if(cosine>0.9995){
        return Quaterniond{
            source.w+(target.w-source.w)*t,
            source.x+(target.x-source.x)*t,
            source.y+(target.y-source.y)*t,
            source.z+(target.z-source.z)*t,
        }.normalized();
    }
    const double theta=std::acos(std::clamp(cosine,-1.0,1.0));
    const double sinTheta=std::sin(theta);
    if(std::abs(sinTheta)<1e-12)return source;
    const double a=std::sin((1.0-t)*theta)/sinTheta;
    const double b=std::sin(t*theta)/sinTheta;
    return {
        source.w*a+target.w*b,
        source.x*a+target.x*b,
        source.y*a+target.y*b,
        source.z*a+target.z*b,
    };
}

Quaterniond Quaterniond::fromRotationVector(const Vec3d& omega) noexcept {
    const double theta = norm(omega);
    if (theta < 1e-9) {
        return Quaterniond{
            1.0,
            omega.x * 0.5,
            omega.y * 0.5,
            omega.z * 0.5,
        }.normalized();
    }
    const double half = 0.5 * theta;
    const double scale = std::sin(half) / theta;
    return {
        std::cos(half),
        omega.x * scale,
        omega.y * scale,
        omega.z * scale,
    };
}

Vec3d Quaterniond::rotate(const Vec3d& value) const noexcept {
    const Quaterniond q = normalized();
    const Quaterniond p{0.0, value.x, value.y, value.z};
    const Quaterniond result =
        multiply(multiply(q, p), q.conjugate());
    return {result.x, result.y, result.z};
}

Vec3d Quaterniond::toRotationVector() const noexcept {
    Quaterniond q = normalized();
    if (q.w < 0.0) q = {-q.w, -q.x, -q.y, -q.z};
    const double vectorMagnitude = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
    if (vectorMagnitude < 1e-9) {
        return {2.0 * q.x, 2.0 * q.y, 2.0 * q.z};
    }
    const double angle = 2.0 * std::atan2(vectorMagnitude, std::clamp(q.w, -1.0, 1.0));
    const double scale = angle / vectorMagnitude;
    return {q.x * scale, q.y * scale, q.z * scale};
}

SE3d SE3d::compose(const SE3d& other) const noexcept {
    return {
        Quaterniond::multiply(rotation, other.rotation).normalized(),
        transform(other.translation),
    };
}

SE3d SE3d::inverse() const noexcept {
    const Quaterniond inverseRotation = rotation.conjugate().normalized();
    return {
        inverseRotation,
        inverseRotation.rotate(translation * -1.0),
    };
}

Vec3d SE3d::transform(const Vec3d& value) const noexcept {
    return rotation.rotate(value) + translation;
}

Twist6d SE3d::log() const noexcept {
    const Vec3d omega = rotation.toRotationVector();
    const double theta = norm(omega);
    double omegaHat[3][3]{};
    skew(omega, omegaHat);

    double omegaHatSquared[3][3]{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            for (int k = 0; k < 3; ++k) {
                omegaHatSquared[r][c] += omegaHat[r][k] * omegaHat[k][c];
            }
        }
    }

    double coefficient = 1.0 / 12.0;
    if (theta >= 1e-5) {
        const double half = 0.5 * theta;
        const double cotHalf = std::cos(half) / std::max(std::sin(half), 1e-12);
        coefficient =
            (1.0 - half * cotHalf) / (theta * theta);
    }

    double inverseV[3][3]{
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    };
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            inverseV[r][c] -= 0.5 * omegaHat[r][c];
            inverseV[r][c] += coefficient * omegaHatSquared[r][c];
        }
    }

    return {omega, matVec(inverseV, translation)};
}

SE3d SE3d::exp(const Twist6d& twist) noexcept {
    const double theta = norm(twist.rotation);
    double omegaHat[3][3]{};
    skew(twist.rotation, omegaHat);

    double omegaHatSquared[3][3]{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            for (int k = 0; k < 3; ++k) {
                omegaHatSquared[r][c] += omegaHat[r][k] * omegaHat[k][c];
            }
        }
    }

    const double theta2 = theta * theta;
    const double theta3 = theta2 * theta;
    const double B = theta < 1e-7
        ? 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0
        : (1.0 - std::cos(theta)) / theta2;
    const double C = theta < 1e-7
        ? 1.0 / 6.0 - theta2 / 120.0 + theta2 * theta2 / 5040.0
        : (theta - std::sin(theta)) / theta3;

    double V[3][3]{
        {1.0,0.0,0.0},
        {0.0,1.0,0.0},
        {0.0,0.0,1.0},
    };
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            V[r][c] += B * omegaHat[r][c] + C * omegaHatSquared[r][c];
        }
    }

    return {
        Quaterniond::fromRotationVector(twist.rotation),
        matVec(V, twist.translation),
    };
}

} // namespace nox::native::math
