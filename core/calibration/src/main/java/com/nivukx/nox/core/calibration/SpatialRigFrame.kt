package com.nivukx.nox.core.calibration

import com.nivukx.nox.core.model.DeviceId
import com.nivukx.nox.core.model.DeviceRole
import com.nivukx.nox.core.model.Vec3

/**
 * Semantic world frame for distributed phones.
 *
 * +Y is gravity-up. +Z is the rig's north/look direction selected from the reference device.
 * +X is east. Camera roles such as LEFT/RIGHT/TOP/BOTTOM are labels over this common metric frame.
 */
data class SpatialRigFrame(
    val origin: Vec3,
    val north: Vec3,
    val east: Vec3,
    val up: Vec3,
    val deviceRoles: Map<DeviceId, DeviceRole>,
) {
    fun toWorld(local: Vec3): Vec3 =
        origin + east * local.x + up * local.y + north * local.z
}
