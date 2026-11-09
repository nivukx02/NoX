package com.nivukx.nox.core.network

import com.nivukx.nox.core.model.DeviceId

/**
 * Local peer-admission authority for distributed capture.
 *
 * This is intentionally separate from payload integrity. Nearby Connections already provides the
 * connection-level authentication token; this class decides whether that token, endpoint identity,
 * and operator policy are sufficient to admit a device into the capture graph.
 */
class PeerAdmissionPolicy(
    private val config: Config = Config(),
    trustedEndpoints: Set<DeviceId> = emptySet(),
) {

    enum class Mode {
        OPEN,
        TRUSTED_ALLOWLIST,
        VERIFICATION_REQUIRED,
    }

    enum class Decision {
        ACCEPT,
        REJECT,
        PENDING_VERIFICATION,
    }

    data class Config(
        val mode: Mode = Mode.OPEN,
        val maximumPendingRequests: Int = 64,
        val requestTtlNanos: Long = 30_000_000_000L,
    ) {
        init {
            require(maximumPendingRequests in 1..512)
            require(requestTtlNanos > 0L)
        }
    }

    data class Request(
        val endpoint: DeviceId,
        val authenticationDigits: String,
        val endpointName: String,
        val createdAtNanos: Long,
    ) {
        init {
            require(endpoint.value.isNotBlank())
            require(authenticationDigits.isNotBlank())
            require(endpointName.isNotBlank())
            require(createdAtNanos >= 0L)
        }
    }

    data class Result(
        val decision: Decision,
        val acceptImmediately: Boolean,
        val request: Request? = null,
        val reason: String,
    )

    private val trusted =
        LinkedHashSet<DeviceId>(
            trustedEndpoints,
        )

    private val pending =
        LinkedHashMap<DeviceId, Request>(
            16,
            0.75f,
            true,
        )

    @Synchronized
    fun request(
        endpoint: DeviceId,
        authenticationDigits: String,
        endpointName: String,
        nowNanos: Long,
    ): Result {
        require(nowNanos >= 0L)

        expire(nowNanos)

        val request =
            Request(
                endpoint = endpoint,
                authenticationDigits =
                    normalizeDigits(
                        authenticationDigits,
                    ),
                endpointName = endpointName.trim(),
                createdAtNanos = nowNanos,
            )

        return when (config.mode) {
            Mode.OPEN ->
                Result(
                    decision = Decision.ACCEPT,
                    acceptImmediately = true,
                    request = request,
                    reason = "open-admission-mode",
                )

            Mode.TRUSTED_ALLOWLIST ->
                if (trusted.contains(endpoint)) {
                    Result(
                        decision = Decision.ACCEPT,
                        acceptImmediately = true,
                        request = request,
                        reason = "trusted-endpoint",
                    )
                } else {
                    Result(
                        decision = Decision.REJECT,
                        acceptImmediately = false,
                        request = request,
                        reason = "endpoint-not-in-trusted-allowlist",
                    )
                }

            Mode.VERIFICATION_REQUIRED ->
                queuePending(request)
        }
    }

    @Synchronized
    fun approve(
        endpoint: DeviceId,
        authenticationDigits: String,
        nowNanos: Long,
    ): Boolean {
        require(nowNanos >= 0L)

        expire(nowNanos)

        val request =
            pending[endpoint]
                ?: return false

        val normalized =
            normalizeDigits(
                authenticationDigits,
            )

        if (
            normalized !=
                request.authenticationDigits
        ) {
            return false
        }

        pending.remove(endpoint)
        trusted += endpoint
        return true
    }

    @Synchronized
    fun reject(
        endpoint: DeviceId,
    ): Boolean =
        pending.remove(endpoint) != null

    @Synchronized
    fun revoke(
        endpoint: DeviceId,
    ) {
        trusted.remove(endpoint)
        pending.remove(endpoint)
    }

    @Synchronized
    fun trust(
        endpoint: DeviceId,
    ) {
        require(endpoint.value.isNotBlank())
        trusted += endpoint
    }

    @Synchronized
    fun untrust(
        endpoint: DeviceId,
    ) {
        trusted.remove(endpoint)
    }

    @Synchronized
    fun isTrusted(
        endpoint: DeviceId,
    ): Boolean =
        trusted.contains(endpoint)

    @Synchronized
    fun trustedEndpoints(): Set<DeviceId> =
        trusted.toSet()

    @Synchronized
    fun pendingRequests(
        nowNanos: Long,
    ): List<Request> {
        require(nowNanos >= 0L)
        expire(nowNanos)
        return pending.values.toList()
    }

    @Synchronized
    fun pendingCount(
        nowNanos: Long,
    ): Int {
        require(nowNanos >= 0L)
        expire(nowNanos)
        return pending.size
    }

    @Synchronized
    fun clearPendingRequests() {
        pending.clear()
    }

    @Synchronized
    fun clear() {
        pending.clear()
        trusted.clear()
    }

    private fun queuePending(
        request: Request,
    ): Result {
        if (
            pending.size >=
                config.maximumPendingRequests
        ) {
            pending.entries
                .firstOrNull()
                ?.key
                ?.let(pending::remove)
        }

        pending[request.endpoint] =
            request

        return Result(
            decision =
                Decision.PENDING_VERIFICATION,
            acceptImmediately = false,
            request = request,
            reason = "operator-verification-required",
        )
    }

    private fun expire(
        nowNanos: Long,
    ) {
        pending.entries.removeIf {
            nowNanos -
                it.value.createdAtNanos >=
                config.requestTtlNanos
        }
    }

    private fun normalizeDigits(
        digits: String,
    ): String =
        digits
            .filter(Char::isDigit)
            .take(32)
            .also {
                require(it.isNotEmpty()) {
                    "Authentication digits must contain at least one digit"
                }
            }
}
