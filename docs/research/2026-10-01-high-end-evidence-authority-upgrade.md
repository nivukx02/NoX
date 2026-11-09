# NivukxNoX high-end evidence-authority continuation: 2026-10-01

This continuation adds an evidence-first precision authority, a bounded multi-actor compute scheduler, and a professional operator evidence surface.

## Engineering changes

- Precision authority separates modeled uncertainty, operational readiness and empirical matched-reference qualification.
- Bottleneck selection uses the weakest critical evidence dimensions instead of relying only on a weighted average.
- Empirical qualification accounts for reference-system uncertainty with root-sum-square aggregation.
- Crowd scheduling retains the existing 50-actor target and 64-actor hard ceiling.
- Optional face, eye, hand, dense-surface and global-4D work is explicitly budgeted after core body tracking.
- Aging and fairness prevent repeatedly deferred actors from permanent starvation.
- The operator surface presents geometry, timing, identity, surface and runtime evidence without turning a visual confidence score into a physical-accuracy claim.

## Research basis

The architecture remains aligned with current markerless multi-person research, including CVPR 2026 MAMMA, which emphasizes dense 2D evidence, uncertainty, visibility, contact and calibrated multi-view reconstruction. Recent temporal multi-person reconstruction research also supports motion-aware prediction and memory for occlusion recovery. Current Liquid Glass guidance emphasizes hierarchy, functional layering and legibility.

## Accuracy contract

The requested <1 mm requirement remains a qualification target. The implementation does not claim universal sub-millimetre accuracy or superiority to infrared/marker-based reference capture from architecture alone. A measured claim requires matched-condition physical benchmarking, documented reference uncertainty, repeated samples, P95/P99 statistics, identity continuity, dropout, temporal latency and calibration provenance.

## Validation workflow

No Python simulation and no Android application build were run in this continuation, per the requested workflow. Static source-level inspection was used for integration review.

## Continuation: geometric conditioning and native buffer ownership

- Dense reconstruction now has per-observation geometric conditioning that downweights weak-parallax or contradictory cameras before triangulation.
- The dense path keeps geometric conditioning telemetry and avoids repeating the same conditioning evaluation merely to populate diagnostics.
- Android capture now exposes an explicit AHardwareBuffer lease boundary with native reference counting, resident-byte telemetry, invalid-release accounting and teardown cleanup.
- The hardware-buffer bridge is deliberately a low-copy ownership primitive, not a fabricated promise that the entire camera-to-inference pipeline is zero-copy on every vendor device.
- Android's current native buffer documentation supports sharing AHardwareBuffer across hardware consumers and importing it through Vulkan external memory where supported.

## Continuation: semantic silhouette and capture-path hardening

The production chain now distinguishes semantic foreground evidence from instance identity and metric geometry:

1. **Bounded person segmentation**
   - MediaPipe multiclass segmentation is run asynchronously at an adaptive lower cadence.
   - GPU is preferred with an explicit CPU initialization fallback.
   - The live camera pipeline remains YUV; segmentation downsampling converts only requested frames to a bounded 256×256 ARGB input compatible with the documented segmenter input contract.
   - The retained evidence is a 64×64 confidence field rather than an unbounded mask image.

2. **Temporal segmentation stabilization**
   - A bounded previous mask is centroid-compensated before temporal blending.
   - The engine exposes temporal mask consistency, boundary stability and callback latency.
   - MPImage storage is explicitly released from the callback path.

3. **Per-actor silhouette evidence**
   - Actor boxes, centers and high-confidence joints are sampled against the semantic confidence field.
   - The result reports foreground support, strong support, joint support, boundary support, overlap ambiguity and reliability.
   - Because the underlying model is semantic rather than instance-segmented, this evidence is never treated as identity proof.

4. **Crowd-scale scheduling integration**
   - Per-actor silhouette reliability now influences expensive surface-detail allocation and scheduling modestly.
   - Identity scoring remains driven by the dedicated identity systems.
   - Aggregate actor support and overlap ambiguity are exposed to the operator UI.

5. **Metric reconstruction integration**
   - Dense observations are conditioned by camera-ray parallax, baseline diversity, peer geometry agreement and observation uncertainty before existing identity gating and triangulation.
   - Silhouette consistency is retained in the dense point contract and propagated from the live segmentation bridge.
   - Stale segmentation evidence is rejected at the post-capture reconstruction boundary.

6. **Native buffer ownership**
   - The native engine now owns explicit AHardwareBuffer leases with reference counting, resident-byte telemetry, invalid-release tracking and teardown cleanup.
   - This is a hardware-sharing primitive, not a universal claim of end-to-end zero-copy behavior across camera vendors.

### Qualification boundary

The architecture is now stronger at rejecting weak evidence and preserving dense surface support, but practical sub-millimeter accuracy remains an empirical requirement. The existing NoX metrology stack already contains matched-reference, uncertainty-adjusted P95/P99, regime, repeatability and qualification authorities. Those gates should be used to establish any real-world precision claim rather than deriving one from source-code complexity.

No Android build, device benchmark or Python simulation was run in this continuation, consistent with the requested workflow. Static source inspection and current platform documentation were used.

## Continuation: instance-aligned masks and distributed appearance evidence

- Pose Landmarker instance-aligned segmentation masks are enabled for the multi-actor front end. Each detected pose can now produce a bounded silhouette-quality result.
- The per-pose mask evidence is keyed by the stable local track ID and fused into the existing actor silhouette authority. It remains supporting evidence only and does not create a metric 3D point or an identity by itself.
- Dense landmark transport is now version 3. The packet preserves segmentation consistency, while versions 1 and 2 remain decodable.
- Multi-actor pose transport is now version 3. A fixed 16-value, quantized appearance continuity fingerprint plus quality is transported per actor. The full high-resolution appearance descriptor remains local to each device.
- Cross-camera identity fusion now compares the compact fingerprint rather than incorrectly mixing a full local descriptor with a compact remote representation.
- The 50-actor worst-case multi-actor payload remains bounded under the current NoX 32 KiB transport contract after the v3 extension.
- The appearance channel is explicitly continuity evidence, not facial recognition or a biometric identity claim.

## Protocol continuity and per-pose mask qualification

- Multi-actor wire format is now v3 with a fixed 17-byte appearance extension per actor, while v1 and v2 remain decodable.
- Dense-landmark wire format is now v3 with a 4-byte segmentation-consistency extension per point, while v1 and v2 remain decodable.
- The v3 payloads retain the current packet-size budget for the 50-actor target.
- Pose Landmarker per-pose segmentation masks are enabled as a bounded instance-aligned evidence channel. The mask contributes silhouette support and quality only; metric geometry remains governed by calibrated multi-view reconstruction.
- Appearance continuity is kept separate from identity authority. Its role is to reduce ambiguous assignment candidates, especially during crossings and partial occlusion.

## Continuation: metric geometry, identity, surface flow and operator control

- Per-observation geometric conditioning is now shared by body skeleton triangulation and dense facial mesh triangulation. Weak parallax, low baseline diversity and contradictory ray geometry are downweighted before robust refinement.
- Geometric conditioning exposes a minimum-observation score in addition to the global score so one pathological camera cannot disappear inside a healthy average.
- Robust triangulation now uses adaptive damping driven by accepted residual improvement, rather than one fixed damping value for every scene.
- Multi-actor identity association now fuses the existing appearance descriptor with its independent histogram fingerprint while retaining appearance-quality gating.
- Temporal native surface flow now uses reciprocal best-match consistency and ambiguity rejection instead of pure greedy nearest-neighbor acceptance. It reports mutual consistency and correspondence uniqueness.
- Scene-level segmentation is now bounded by capture policy. Operators can enable/disable it, tune refresh cadence and tune scheduling priority; the live pipeline keeps existing per-pose masks as the actor-specific evidence authority.
- Segmentation bitmap ownership is bounded by timestamp, reclaimed on callback/failure, and cleared during reset/teardown.
- The native Android hardware-buffer bridge remains an ownership/lifetime primitive. Android documents AHardwareBuffer as hardware-accessible shared memory with explicit native reference-counted lifetime management; NoX does not claim vendor-independent end-to-end zero-copy behavior.
- Current Compose guidance emphasizes 48dp minimum touch targets for custom interactive elements and accessible contrast; the new segmentation switch uses an explicit 48dp minimum row height.
- Recent CVPR 2026 work such as MAMMA is relevant as research precedent for markerless multi-person acquisition using dense visual evidence and calibrated multi-view reconstruction. NoX continues to distinguish research precedent, modeled precision and empirically qualified accuracy.

No Android build and no Python simulation were run in this continuation, per the requested workflow. Static source-level review was used to catch integration hazards such as Kotlin exhaustive-stage handling and C++ -Werror unused-state failures.

## Build verification checkpoint

- Triggered from `main` after the current precision, segmentation, geometric-conditioning and native-buffer changes; workflow results are the source of truth for compiler validation.
