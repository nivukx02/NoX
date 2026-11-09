# NoX Fragment Reassembly Integrity Contract

## Shared invariants

A fragmented capture frame is one logical unit even though its payload arrives in multiple packets. Each reassembler must ensure:
- a stream/frame key retains one declared chunk count and timestamp;
- duplicate fragment indexes do not silently replace accepted bytes;
- conflicting fragments invalidate the affected partial assembly;
- successful assembly returns only after the underlying wire codec validates the complete frame;
- failed decoding releases every retained byte associated with the affected frame;
- memory accounting is decremented when a partial frame is removed by conflict, timeout, eviction, explicit endpoint clear, or decode failure.

## Stream-specific checks

Depth chunks carry a source-device ID and CRC at the fragment layer. Reassembly binds the origin to the first chunk and rejects origin, count, timestamp, duplicate-payload, and CRC disagreement.

Dense-landmark chunks validate fragment CRC and frame structure in the chunked codec. Multi-actor-pose and face-mesh wire paths retain their current packet layouts; their reassemblers reject changed metadata/duplicates and discard partial state if final decoding throws.

## Resource boundaries

The current maximum fragment counts, packet envelope, in-flight assembly counts, payload-byte budgets, and age expiration remain unchanged. This change does not widen fragment masks or renegotiate wire versions. Any future fragment-count increase must update sender sequence allocation, ACK/NACK masks, retransmission cache handling, packet-size checks, old-peer negotiation, and round-trip/interoperability tests together.

## Regression coverage

Tests now include timestamp-conflict cleanup in dense landmark, multi-actor pose, and face-mesh reassembly; depth tests verify mixed device identity, timestamp conflict, corrupt completed-frame cleanup, and sender-side CRC/timestamp rejection. Source-integrity tests preserve the shared failure-cleanup contract.

No Android build or physical-device transport validation was run as part of these edits.
