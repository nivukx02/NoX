package com.nivukx.nox.core.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class RuntimeDecisionTraceLedgerTest {
    @Test
    fun ledgerIsBoundedAndDigestChangesWithDecisions() {
        val ledger = RuntimeDecisionTraceLedger(capacity = 2)
        val first = ledger.append(1L, "motion", "LIVE", "accepted", RuntimeDecisionTrace.Severity.INFO, 0.90, evidenceMaterial = "a".toByteArray())
        val digest1 = ledger.ledgerDigest()
        ledger.append(2L, "transport", "TRANSPORT_LIMITED", "hold", RuntimeDecisionTrace.Severity.WARNING, 0.50, evidenceMaterial = "b".toByteArray())
        ledger.append(3L, "recovery", "RECOVERING", "reset", RuntimeDecisionTrace.Severity.ERROR, 0.30, evidenceMaterial = "c".toByteArray())
        assertEquals(2, ledger.size())
        assertTrue(ledger.recent().none { it.sequence == first.sequence })
        assertTrue(digest1 != ledger.ledgerDigest())
    }

    @Test
    fun repeatedEvidenceProducesStableHash() {
        val ledger = RuntimeDecisionTraceLedger()
        val a = ledger.append(1L, "motion", "LIVE", "accepted", RuntimeDecisionTrace.Severity.INFO, 0.90, evidenceMaterial = "same".toByteArray())
        ledger.reset()
        val b = ledger.append(1L, "motion", "LIVE", "accepted", RuntimeDecisionTrace.Severity.INFO, 0.90, evidenceMaterial = "same".toByteArray())
        assertEquals(a.evidenceDigest, b.evidenceDigest)
    }
}