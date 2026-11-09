# Eye Tracking Input Integrity

Eye tracking is sensitive to small spatial regions, timing and rolling-shutter/exposure artifacts. Validation now happens before temporal state changes or luma-plane access.

- Face-size and probability thresholds must be finite and in supported ranges. The tracking-map cap is bounded to prevent invalid configuration from creating unbounded state.
- Face bounds, eye centers, probabilities, pose angles and timestamps are validated before ROI calculations. Invalid detector output produces a zero-confidence empty result without updating an existing track.
- Timing metadata is evaluated before image traversal. Invalid timing returns an empty frame before track pruning or blink/gaze history updates.
- Luma-plane construction requires positive image dimensions and strides and validates the final byte index using 64-bit arithmetic. Every pixel index also uses Long arithmetic to prevent integer overflow; truncated planes are rejected.
- The stale-timestamp test was repaired into one complete regression test, and finite-config/timing and source-boundary tests were added.

These safeguards improve resilience and prevent corrupt input from poisoning temporal state; they are not a claim of measured gaze accuracy.
