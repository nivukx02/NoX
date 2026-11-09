package com.nivukx.nox.core.model

import java.security.MessageDigest

/**
 * Bounded immutable decision trace for production capture sessions.
 *
 * Stores why authority changed without retaining raw frames or unbounded payloads.
 */
data class RuntimeDecisionTrace(
    val sequence: Long,
    val timestampNanos: Long,
    val stage: String,
    val state: String,
    val decision: String,
    val severity: Severity,
    val score: Double,
    val causeCodes: List<String>,
    val evidenceDigest: String,
) {
    enum class Severity { INFO, WARNING, ERROR, CRITICAL }

    init {
        require(sequence > 0L)
        require(timestampNanos >= 0L)
        require(stage.isNotBlank())
        require(state.isNotBlank())
        require(decision.isNotBlank())
        require(score.isFinite() && score in 0.0..1.0)
        require(causeCodes.size <= 64)
        require(causeCodes.all { it.isNotBlank() })
        require(evidenceDigest.length == 64)
    }
}

/** Fixed-capacity trace buffer retaining only the newest decisions. */
class RuntimeDecisionTraceLedger(
    private val capacity: Int = 4096,
) : AutoCloseable {
    init { require(capacity in 1..16384) }

    private val entries = ArrayDeque<RuntimeDecisionTrace>()
    private var nextSequence = 1L
    private var closed = false

    @Synchronized
    fun append(
        timestampNanos: Long,
        stage: String,
        state: String,
        decision: String,
        severity: RuntimeDecisionTrace.Severity,
        score: Double,
        causeCodes: List<String> = emptyList(),
        evidenceMaterial: ByteArray = byteArrayOf(),
    ): RuntimeDecisionTrace {
        ensureOpen()
        val entry = RuntimeDecisionTrace(
            sequence = nextSequence++,
            timestampNanos = timestampNanos,
            stage = stage,
            state = state,
            decision = decision,
            severity = severity,
            score = score.coerceIn(0.0, 1.0),
            causeCodes = causeCodes.distinct().take(64),
            evidenceDigest = sha256(evidenceMaterial),
        )
        entries.addLast(entry)
        while (entries.size > capacity) entries.removeFirst()
        return entry
    }

    @Synchronized
    fun recent(limit: Int = 64): List<RuntimeDecisionTrace> {
        ensureOpen()
        require(limit >= 0)
        return entries.takeLast(limit.coerceAtMost(capacity)).toList()
    }

    @Synchronized
    fun size(): Int {
        ensureOpen()
        return entries.size
    }

    @Synchronized
    fun ledgerDigest(): String {
        ensureOpen()
        val material = buildString {
            entries.forEach { entry ->
                append(entry.sequence)
                append(":")
                append(entry.timestampNanos)
                append(":")
                append(entry.stage)
                append(":")
                append(entry.state)
                append(":")
                append(entry.decision)
                append(":")
                append(entry.severity.name)
                append(":")
                append(entry.score)
                append(":")
                append(entry.causeCodes.joinToString(","))
                append(":")
                append(entry.evidenceDigest)
                append("|")
            }
        }
        return sha256(material.toByteArray())
    }

    @Synchronized
    fun reset() {
        ensureOpen()
        entries.clear()
    }

    @Synchronized
    override fun close() {
        if (closed) return
        entries.clear()
        closed = true
    }

    private fun ensureOpen() { check(!closed) { "decision trace ledger is closed" } }

    private fun sha256(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256")
            .digest(bytes.copyOf())
            .joinToString("") { "%02x".format(it.toInt() and 0xff) }
}