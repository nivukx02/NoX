package com.nivukx.nox.core.model

data class AvatarBonePose(
    val position: Vec3,
    val rotation: Quat,
)

data class AvatarPose(
    val timestampNanos: Long,
    val rootPositionMeters: Vec3,
    val scale: Double,
    val bones: Map<String, AvatarBonePose>,
    val confidence: Double,
)
