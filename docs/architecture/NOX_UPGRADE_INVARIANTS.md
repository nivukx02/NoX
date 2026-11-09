# NoX Upgrade Invariants

## Purpose

NoX is being evolved as a production-grade multi-camera motion-capture and 3D-reconstruction system. Future upgrades must increase capability without silently removing previously available repository assets or evidence paths.

## Repository preservation

- Work is performed on `main`.
- Existing source, test, native, build, documentation, and UI files are preserved.
- Feature work may modify an implementation when the old implementation is superseded, but no file is removed merely because a new implementation exists elsewhere.
- The repository integrity workflow rejects any source/test/documentation file deletion.
- Reusable subsystems must be retained when they expose distinct contracts, diagnostics, or testing value.

## Evidence integrity

- Timestamped camera evidence is never replaced with synthetic timestamps.
- Geometry-aware scores must be based on measured camera poses, calibrated intrinsics, visibility, timing, and uncertainty where those signals exist.
- Prediction is explicitly marked as prediction and loses confidence as the prediction horizon grows.
- A precision target is a qualification gate, not a guarantee of physical accuracy.
- Claims against active optical/infrared marker systems require matched external reference samples, reference uncertainty, confidence, and an explicit comparison protocol.

## Performance and memory

- Multi-camera state is bounded.
- Per-camera queues and temporal histories are bounded.
- Native and GPU resource leases remain generation-safe and observable.
- Runtime pressure may reduce optional fidelity stages, but core motion evidence is protected.
- Scaling to large federated rigs must avoid unbounded pairwise or job creation.

## Reconstruction quality

- Active-view selection favors useful triangulation conditioning rather than assuming larger ray angles are always better.
- Cross-view evidence is filtered by temporal compatibility and calibrated geometry when available.
- Dense reconstruction preserves uncertainty metadata into mesh and surface stages.
- Surface post-processing adapts denoising to measured uncertainty instead of treating all vertices as equally reliable.
- Topology, reprojection, temporal stability, and uncertainty remain explicit quality signals.

## Multi-person capture

- The operational target remains support for up to 50 actors with bounded global assignment.
- Actor identity continuity uses global assignment plus detector, geometry, appearance, skeleton, temporal, and body-space evidence.
- Severe overlap increases assignment conservatism so an ambiguous crossing becomes a prediction/hold rather than an arbitrary identity swap.
- Difficult actors may receive additional authority views within a bounded federation budget.

## Validation policy

- Prefer real Kotlin/JVM, native C++, Android/Gradle, and GitHub Actions execution.
- Do not substitute Python simulations for repository tests.
- When CI infrastructure fails before any job steps execute, report it as infrastructure/runner failure rather than inventing a code failure or code success.
- High line counts are not an engineering acceptance criterion. Source volume is increased through real features, diagnostics, invariants, tests, and production behavior, not repeated filler code.
