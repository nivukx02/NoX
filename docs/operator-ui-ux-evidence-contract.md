# NoX Operator UI/UX Evidence Contract

## Design objective

NoX is an operator console for high-consequence capture feedback, not a decorative demo. The live viewport, identity continuity, timing/calibration warnings, and capture controls must remain visually prioritized above decorative material effects.

## Liquid Glass implementation rules

1. **Use glass selectively.** Reserve the strongest translucency/specular effects for navigation, focused action clusters, and a small number of high-level surfaces. Keep dense measurements, long tables, and detailed configuration on readable, higher-opacity material.
2. **Preserve content contrast.** Material opacity should respond to underlying luminance, content contrast, operator focus, capture priority, and accessibility options. Reduced-transparency mode must remain within the configured material opacity ceiling.
3. **Keep motion subordinate to capture.** Reduced-motion policy should damp transition effects without concealing state changes or degrading the reconstruction viewport.
4. **Retain native semantics.** Controls need descriptive labels, roles, state descriptions, predictable reading order, keyboard/focus behavior where supported, and adequate touch targets.
5. **Keep hierarchy obvious.** Separate navigation from scene content; group related settings; make the highest-priority action reachable and visually distinct; avoid stacking multiple translucent cards over dense imagery.

## Evidence and implementation links

- [Apple, Adopting Liquid Glass](https://developer.apple.com/documentation/TechnologyOverviews/adopting-liquid-glass) recommends using the material sparingly, preserving legibility, and validating custom effects with accessibility settings such as reduced transparency and reduced motion. NoX should apply those principles using Android-native components rather than attempting to reuse Apple's platform APIs.
- [Material Design 3, Accessibility designing](https://m3.material.io/foundations/designing) emphasizes structural hierarchy, color/contrast, clear labeling, and predictable focus/reading order.
- [Material Design 3, Structure](https://m3.material.io/foundations/designing/structure) recommends clear navigation and readable hierarchy, and documents a 48 dp target size for touch controls.
- [Android, Use multiple camera streams simultaneously](https://developer.android.com/media/camera/camera2/multiple-camera-streams-simultaneously) notes that supported stream combinations and performance vary by device/hardware level. The interface must present current verified capabilities, not just requested configuration.
- [Android CameraManager API](https://developer.android.com/reference/android/hardware/camera2/CameraManager) exposes the getConcurrentCameraIds() and isConcurrentSessionConfigurationSupported(...) APIs; runtime admission should be based on those device capabilities.

## Operator-facing recommendations

- Show requested versus effective camera/actor counts and explicitly disclose capacity shedding. Where a requested camera count is absent, label 128 as a capacity ceiling rather than an expected connected-camera count. Composite quality indices must not be labeled as physical precision; reserve qualification labels for the explicit authority contract.
- Give timing, calibration, identity, uncertainty, thermal, memory, and network degradation their own signal instead of collapsing them into one confidence number.
- Do not display a millimetre-grade label unless the currently selected tier has accepted independent reference evidence and the matching condition contract is satisfied. Composite quality scores and model-feasibility estimates must not be presented as measured error or as a qualified accuracy tier.
- Keep accessibility settings persistent and visible. Decorative reduction must never change measurement data or silently lower capture integrity.
- For dense settings, use section headings, sensible grouping, summaries for collapsed groups, and a clear path back to the live scene.

## Verification status

This document captures design constraints from public platform guidance. It does not claim that every screen has been manually audited on physical Android devices. Device layout, contrast, screen-reader order, and touch-target behavior still need instrumented UI and hardware validation.
