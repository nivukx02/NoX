package com.nivukx.nox.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class NivukxNoXIdentityTest {
    @Test
    fun canonicalIdentityIsStable() {
        assertEquals("NivukxNoX", NivukxNoXIdentity.TECHNOLOGY_NAME)
        assertEquals("NoX", NivukxNoXIdentity.SHORT_NAME)
        assertEquals("Production Motion Capture", NivukxNoXIdentity.DOMAIN)
        assertTrue(NivukxNoXIdentity.MOTION_ONLY)
    }
}
