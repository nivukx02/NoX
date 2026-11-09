package com.nivukx.nox.core.network

/**
 * Bounded per-source replay protection for secure remote evidence.
 *
 * The window accepts strictly newer sequences and also accepts unseen out-of-order packets inside
 * a bounded history. Older packets and duplicates are rejected. State is capped by source count.
 */
class SecureRemoteReplayWindow(
    private val config: Config = Config(),
) {

    data class Config(
        val windowSize: Int = 1_024,
        val maximumSources: Int = 256,
    ) {
        init {
            require(windowSize in 8..65_536)
            require(maximumSources in 1..4_096)
        }
    }

    private data class State(
        var highestSequence: Long,
        val seen:
            LinkedHashSet<Long> =
                LinkedHashSet(),
    )

    private val states =
        LinkedHashMap<String, State>(
            16,
            0.75f,
            true,
        )

    @Synchronized
    fun accept(
        source: String,
        sequence: Long,
    ): Boolean {
        require(source.isNotBlank())
        require(sequence >= 0L)

        val state =
            states[source]

        if (state == null) {
            while (
                states.size >=
                    config.maximumSources
            ) {
                states.entries
                    .firstOrNull()
                    ?.key
                    ?.let(states::remove)
            }

            states[source] =
                State(
                    highestSequence = sequence,
                ).also {
                    it.seen += sequence
                }

            return true
        }

        if (sequence > state.highestSequence) {
            state.highestSequence =
                sequence
            state.seen += sequence
            trim(state)
            return true
        }

        if (
            state.highestSequence -
                sequence >=
                config.windowSize
        ) {
            return false
        }

        if (!state.seen.add(sequence)) {
            return false
        }

        trim(state)
        return true
    }

    @Synchronized
    fun clearSource(
        source: String,
    ) {
        require(source.isNotBlank())
        states.remove(source)
    }

    @Synchronized
    fun clear() {
        states.clear()
    }

    @Synchronized
    fun sourceCount(): Int =
        states.size

    private fun trim(
        state: State,
    ) {
        val minimumSequence =
            state.highestSequence -
                config.windowSize.toLong() +
                1L

        val iterator =
            state.seen.iterator()

        while (iterator.hasNext()) {
            if (
                iterator.next() <
                    minimumSequence
            ) {
                iterator.remove()
            }
        }
    }
}
