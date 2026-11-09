package com.nivukx.nox.core

/**
 * Canonical technology identity for the production motion-capture stack.
 *
 * NivukxNoX is the technology name. NoX remains the repository/application shorthand.
 * The identity is centralized so UI, telemetry, exported session metadata and diagnostics
 * do not drift into different product names.
 */
object NivukxNoXIdentity {
    const val TECHNOLOGY_NAME = "NivukxNoX"
    const val SHORT_NAME = "NoX"
    const val DOMAIN = "Production Motion Capture"
    const val MOTION_ONLY = true
    const val VERSION = "1.2.0"
}
