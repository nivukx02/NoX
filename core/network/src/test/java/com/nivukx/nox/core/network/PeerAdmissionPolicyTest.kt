package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.DeviceId
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class PeerAdmissionPolicyTest {

    @Test
    fun allowlistRejectsUnknownEndpoint() {
        val policy =
            PeerAdmissionPolicy(
                PeerAdmissionPolicy.Config(
                    mode =
                        PeerAdmissionPolicy.Mode.TRUSTED_ALLOWLIST,
                ),
                trustedEndpoints =
                    setOf(DeviceId("trusted")),
            )

        assertFalse(
            policy.request(
                endpoint = DeviceId("intruder"),
                authenticationDigits = "123456",
                endpointName = "Unknown",
                nowNanos = 100L,
            ).acceptImmediately,
        )
    }

    @Test
    fun verificationModeRequiresMatchingAuthenticationDigits() {
        val policy =
            PeerAdmissionPolicy(
                PeerAdmissionPolicy.Config(
                    mode =
                        PeerAdmissionPolicy.Mode.VERIFICATION_REQUIRED,
                    requestTtlNanos = 1_000L,
                ),
            )

        val pending =
            policy.request(
                endpoint = DeviceId("relay-a"),
                authenticationDigits = "123456",
                endpointName = "Relay A",
                nowNanos = 100L,
            )

        assertFalse(pending.acceptImmediately)
        assertEquals(
            PeerAdmissionPolicy.Decision.PENDING_VERIFICATION,
            pending.decision,
        )

        assertFalse(
            policy.approve(
                endpoint = DeviceId("relay-a"),
                authenticationDigits = "654321",
                nowNanos = 200L,
            ),
        )

        assertTrue(
            policy.approve(
                endpoint = DeviceId("relay-a"),
                authenticationDigits = "123456",
                nowNanos = 200L,
            ),
        )
    }

    @Test
    fun expiredVerificationRequestCannotBeApproved() {
        val policy =
            PeerAdmissionPolicy(
                PeerAdmissionPolicy.Config(
                    mode =
                        PeerAdmissionPolicy.Mode.VERIFICATION_REQUIRED,
                    requestTtlNanos = 100L,
                ),
            )

        policy.request(
            endpoint = DeviceId("relay-b"),
            authenticationDigits = "222222",
            endpointName = "Relay B",
            nowNanos = 100L,
        )

        assertFalse(
            policy.approve(
                endpoint = DeviceId("relay-b"),
                authenticationDigits = "222222",
                nowNanos = 201L,
            ),
        )
    }

    @Test
    fun openModePreservesLegacyAdmission() {
        val policy = PeerAdmissionPolicy()

        assertTrue(
            policy.request(
                endpoint = DeviceId("any"),
                authenticationDigits = "000000",
                endpointName = "Any",
                nowNanos = 1L,
            ).acceptImmediately,
        )
    }
    @Test
    fun clearingPendingRequestsDoesNotEraseTrustedPeers() {
        val endpoint = DeviceId("trusted-relay")
        val policy =
            PeerAdmissionPolicy(
                PeerAdmissionPolicy.Config(
                    mode =
                        PeerAdmissionPolicy.Mode.VERIFICATION_REQUIRED,
                ),
            )

        policy.trust(endpoint)

        policy.request(
            endpoint = DeviceId("pending"),
            authenticationDigits = "111111",
            endpointName = "Pending",
            nowNanos = 10L,
        )

        policy.clearPendingRequests()

        assertTrue(policy.isTrusted(endpoint))
        assertEquals(0, policy.pendingCount(20L))
    }

}
