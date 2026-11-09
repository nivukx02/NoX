# Depth Fragment Integrity Contract

## Fragment identity

A logical depth frame is keyed by receiver endpoint, stream ID and frame sequence. Every fragment in that assembly must retain the same source device ID, declared chunk count and frame timestamp. Fragment payload bytes and CRC must agree when a duplicate index is received.

## Failure cleanup

If a fragment conflicts with an existing assembly, the reassembler discards that assembly before rejecting the fragment. If decoding, CRC validation, or the payload timestamp check fails after the final chunk arrives, all retained fragment bytes are released before the error propagates. Invalid traffic therefore cannot leave an incomplete or impossible assembly occupying the in-flight memory budget until ordinary timeout expiration.

## Existing protocol bounds

- Fragment payload size remains constrained by the existing NoX packet size limit.
- Depth assembly count, retained bytes, and age remain independently bounded.
- The fragment count stays at eight in this change because ACK/NACK masks and retransmission paths use 32-bit masks with an existing wire contract. Widening that limit requires a coordinated protocol/capability update and interoperability tests.
- This change does not alter the packet layout or wire version. It only strengthens consistency checks and failed-assembly cleanup.

## Verification boundary

Regression tests cover conflicting source IDs, mismatched frame timestamps, and corrupted completed payload timestamps, including retained-byte cleanup. Physical two-device packet-loss/retransmission validation remains separate.


The sender-side fragment codec validates the payload CRC and its embedded timestamp before constructing a wire packet. This rejects inconsistent in-process chunks at the sender rather than waiting for receiver rejection.