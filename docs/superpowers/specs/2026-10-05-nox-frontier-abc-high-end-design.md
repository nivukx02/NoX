# NivukxNoX Frontier A+B+C High-End Motion Reconstruction Design Specification
## 2026-10-05

### Status

Approved execution scope from the current conversation. This specification defines a production-oriented A+B+C upgrade that preserves the existing NoX feature surface and adds stronger learned world-space reasoning, reconstruction-heavy dynamic geometry, and a unified evidence/precision authority layer.

### 1. Objective

Upgrade NivukxNoX from its already substantial multi-camera markerless motion-capture architecture into a higher-end heterogeneous system combining:

A. Native-first deterministic geometric and runtime infrastructure;
B. learned world-space motion, association, and dense evidence proposals;
C. reconstruction-heavy dynamic human/scene refinement using surface, Gaussian, implicit, and temporal representations.

The upgrade must be feature-additive. Existing capture modes, body/hand/face/eye tracking, multi-camera synchronization, dense landmark fusion, reconstruction, 50-actor capacity targets, 128-camera federation, memory controls, diagnostics, Liquid Glass operator surfaces, and precision qualification contracts remain supported.

### 2. Non-negotiable engineering rules

- Work directly on the main branch because explicit authorization for main-branch work has been supplied.
- Do not remove an existing production feature solely to make a new subsystem easier.
- Do not inflate files with generated boilerplate or repetitive filler solely to reach a line-count target.
- New code must represent real functionality, real interfaces, real tests, or real diagnostics.
- Do not use Python simulation tests.
- Use Kotlin/JVM tests, Android tests where practical, and native C++ tests/replay fixtures for engine verification.
- Keep physical accuracy claims separate from modeled targets.
- A <1 mm setting is a qualification target, not a guaranteed measurement.
- “Better than infrared” is a hypothesis to be validated against a matched independent reference, never an architectural claim.
- 50 active actors is the runtime target; 64 remains the bounded engineering ceiling unless a later qualification proves otherwise.
- 128 cameras is the distributed federation ceiling; expensive reconstruction operates on a bounded authoritative subset.
- All queues, history stores, reassembly buffers, native allocations, and diagnostics remain bounded.
- Every authoritative result exposes provenance and uncertainty.

### 3. Architecture

The production graph is:

Camera/sensor acquisition
 -> exposure-midpoint and rolling-shutter timing authority
 -> per-camera conditioning and calibration state
 -> 2D perception and dense landmarks
 -> world-space lifting
 -> multi-camera/multi-actor hypothesis management
 -> hierarchical motion inference
 -> geometric robust solving
 -> dense surface reconstruction
 -> Gaussian/implicit dynamic refinement
 -> global 4D consistency
 -> precision authority
 -> operator/export evidence.

The system is deliberately hybrid. Learned models propose high-value hypotheses and dense features; calibrated geometry, physical constraints, robust optimization, and uncertainty determine authority.

### 4. Acquisition and timing upgrade

Introduce first-class capture timing evidence:
- exposure start, midpoint, and end timestamps;
- row-time/readout estimation;
- timing uncertainty;
- camera-local clock drift;
- timestamp lineage;
- duplicate/replay identity;
- frame-drop accounting;
- blur and exposure stability;
- focus-state provenance;
- lens-state fingerprint;
- sensor-mode fingerprint.

Timing uncertainty must flow into motion and geometry weighting rather than remain UI-only telemetry.

### 5. Motion engine upgrade

Extend MotionCaptureEngine into a hierarchical motion inference fabric:
- camera/system motion compensation;
- viewpoint-invariant motion representation;
- adaptive temporal receptive fields;
- motion-regime transitions;
- acceleration and jerk constraints;
- contact state;
- articulated constraints;
- world-space temporal reasoning;
- multi-hypothesis prediction;
- learned motion proposal plus deterministic robust solver;
- bounded occlusion recovery;
- uncertainty-aware weighting.

Motion states must retain confidence, uncertainty, provenance, temporal authority, and whether the state is observed, inferred, or predicted.

### 6. World-space learned reasoning

Introduce a learned world-space proposal layer informed by current 2025-2026 research directions:
- view-aware feature disentanglement;
- calibrated world-space lifted detections;
- physics-grounded temporal consistency;
- dense visibility-aware landmarks;
- learned cross-camera association proposals.

Learned outputs are bounded, finite-checked, versioned, and never trusted merely because a model returned a tensor.

### 7. Dense evidence

Extend body perception to dense surface evidence:
- visibility probability;
- contact probability;
- dense 2D surface landmarks;
- per-landmark uncertainty;
- local normals;
- local curvature;
- cross-view correspondence;
- persistent point-track IDs;
- semantic surface regions;
- occlusion state;
- photometric and silhouette consistency.

Dense evidence becomes a shared substrate for body, hands, face, clothing deformation proxies, and props.

### 8. Multi-actor identity

Upgrade identity handling to multi-stage hypothesis management:
- geometry;
- calibrated depth;
- skeleton topology;
- body-scale signature;
- motion history;
- appearance embedding;
- silhouette;
- dense landmarks;
- face geometry;
- temporal trajectory;
- interaction relationships;
- visibility/occlusion.

Maintain bounded identity hypotheses before hard assignment when ambiguity is material. Use one-to-one global assignment and deterministic tie-breaking.

### 9. Face engine

Upgrade to multi-scale dense facial geometry:
- high-resolution ROI refinement;
- dense mesh;
- visibility;
- temporal surface flow;
- expression dynamics;
- head pose;
- camera-conditioned uncertainty;
- cross-view geometric agreement;
- explicit freshness lineage.

### 10. Eye engine

Retain and strengthen:
- iris/pupil geometry;
- eyelid openness;
- blink state;
- ocular orientation;
- binocular vergence;
- head-relative gaze;
- world-space gaze rays;
- robust spherical consensus;
- saccade/fixation states;
- gaze velocity and acceleration;
- IPD calibration;
- temporal confidence;
- per-eye uncertainty.

### 11. 3D reconstruction

Use a multi-representation reconstruction fabric:
1. sparse metric landmarks;
2. dense surface evidence;
3. Gaussian surface state;
4. temporal deformation field;
5. SDF/surface-flow constraints;
6. mesh extraction;
7. global 4D refinement.

The reconstruction pipeline must separately report observed, geometrically constrained, temporally propagated, learned-prior, and completion-derived regions.

### 12. Sparse-view and low-overlap reconstruction

Add geometry-first sparse-view recovery:
- decoupled human/background reconstruction;
- geometric multi-view human association;
- body-model priors;
- adaptive view selection;
- temporal propagation;
- unseen-region provenance;
- generative completion only as a bounded refinement layer.

Observed pixels must receive greater authority than inferred pixels.

### 13. Global 4D consistency

Global refinement jointly reasons over:
- camera trajectories;
- human trajectories;
- surface deformation;
- contact constraints;
- scene geometry;
- temporal consistency.

The optimizer must have bounded iteration budgets and fail-closed behavior when conditioning or uncertainty becomes unacceptable.

### 14. Distributed camera federation

Preserve:
- 128 global camera identities;
- 32 authoritative views;
- 96 witness views.

Add:
- sector/shard topology;
- local ranking;
- global diversity-aware ranking;
- timing/calibration quarantine;
- evidence freshness;
- adaptive authority;
- actor-specific visibility;
- thermal/network pressure;
- selection hysteresis.

The global camera limit must never imply that a single Android process opens 128 saturated camera devices.

### 15. Runtime scheduler

Introduce a fairness-aware workload allocator per actor and stage.

Actor budgets:
- body;
- face;
- eye;
- hand;
- dense landmarks;
- surface;
- identity;
- temporal.

Priority function uses:
importance × uncertainty × occlusion × motion complexity × interaction risk × visibility × reconstruction value.

Preserve body-first behavior under pressure. Expensive modalities degrade before authoritative body state.

### 16. Heterogeneous execution

All expensive stages expose:
- preferred backend;
- minimum backend;
- estimated memory;
- latency class;
- thermal class;
- precision sensitivity;
- fallback path.

Backends include CPU, native SIMD, GPU, NPU/DSP/ISP-assisted paths when supported.

### 17. Native data model

Hot native paths prefer:
- structure-of-arrays;
- contiguous buffers;
- bounded containers;
- pooled temporary storage;
- fixed-capacity work queues;
- vectorized math;
- minimized lock contention;
- reusable linear-solver workspaces.

### 18. Memory safety

Each memory domain receives a hard byte ceiling:
camera buffers, image queues, geometry, actor state, face/eye history, dense evidence, Gaussian state, mesh state, network buffers, diagnostics, reconstruction history.

Memory pressure must be calculated from real allocator/buffer occupancy plus caller telemetry.

Every lease has owner, acquisition timestamp, byte size, state, and release path.

### 19. Precision authority

Expose:
- P50/P95/P99;
- bootstrap uncertainty;
- per-region tail error;
- temporal authority;
- timing residual;
- calibration residual;
- parallax;
- view diversity;
- reprojection tails;
- covariance;
- repeatability.

Qualification requires sufficient independent empirical evidence and matched reference runs.

### 20. Sub-millimetre target

The system may expose sub-millimetre feasibility estimates only as evidence-qualified feasibility.

A point can reach precision-qualified status only when:
- calibrated geometry is authoritative;
- sufficient independent views exist;
- view diversity passes;
- reprojection tail passes;
- covariance passes;
- temporal stability passes;
- sufficient empirical qualification evidence exists.

### 21. Reference comparison

Reference qualification must support independent optical/infrared reference systems.

Comparison requires:
- common coordinate frame;
- common time authority;
- synchronized populations;
- matched movement cohort;
- spatial-region breakdown;
- independent repeated runs;
- confidence intervals;
- tail statistics.

No UI copy or model name may imply measured superiority without this evidence.

### 22. Operator UX

Keep Liquid Glass but enforce functional material hierarchy:
scene -> glass shell -> readability plate -> content.

Operator console:
top system bar;
left operator rail;
center reconstruction viewport;
right evidence inspector;
bottom timeline and status deck.

Add:
- camera federation health;
- actor matrix;
- reconstruction provenance;
- uncertainty overlays;
- precision authority;
- thermal budget;
- memory budget;
- timing drift;
- calibration drift;
- evidence loss;
- recovery state.

Use explicit states LIVE, DEGRADED, RECOVERING, EVIDENCE-LIMITED, CALIBRATION-LIMITED, TIMING-LIMITED, THERMAL-LIMITED, PRECISION-QUALIFIED.

### 23. Customization

Advanced settings cover capture, motion, identity, face, eyes, hands, dense reconstruction, sparse-view policy, Gaussian detail, temporal refinement, crowd scheduling, precision qualification, network/federation policy, diagnostics, and accessibility.

### 24. Export and provenance

Each authoritative artifact retains:
- source camera set;
- calibration versions;
- focus/lens state;
- timing authority;
- detector/model versions;
- observation count;
- inlier count;
- uncertainty;
- reconstruction mode;
- observed/inferred ratio;
- qualification state.

### 25. Test strategy

No Python simulation is required.

Use:
- Kotlin unit tests;
- Android lifecycle and contract tests;
- native C++ unit tests;
- deterministic replay fixtures;
- memory-budget tests;
- scheduler fairness tests;
- timing corruption tests;
- duplicate/replay tests;
- identity crossing tests;
- occlusion/reacquisition tests;
- sparse-view geometry tests;
- dense surface tests;
- face freshness tests;
- gaze robustness tests;
- precision-qualification tests.

Every new feature requires a failing test before implementation and a passing targeted test after implementation.

### 26. Rollout order

1. provenance and shared evidence contracts;
2. timing and acquisition authority;
3. world-space learned proposals;
4. motion inference;
5. identity hypotheses;
6. dense point correspondence;
7. face/eye upgrade;
8. sparse-view reconstruction;
9. Gaussian/surface-flow refinement;
10. global 4D optimizer;
11. distributed rig authority;
12. heterogeneous runtime;
13. memory hardening;
14. operator UX;
15. qualification and replay suite.

### 27. Success criteria

The upgrade is successful when repository verification demonstrates:
- existing public interfaces remain available unless an explicit compatibility layer replaces them;
- new A+B+C capabilities are integrated into live architecture;
- memory and queue bounds remain enforced;
- invalid timing/geometry/evidence fails closed;
- the 50-actor scheduler remains fair;
- the 128-camera federation remains bounded;
- face and eye evidence retains freshness and uncertainty provenance;
- reconstruction distinguishes measured from inferred geometry;
- precision qualification cannot be asserted on insufficient evidence;
- no Python simulation tests are introduced;
- no source file is padded solely for line-count compliance.

### 28. Research grounding

The design is informed by 2026 research directions including markerless multi-person dense landmark capture, metric world-space multi-camera tracking, sparse-view 4D Gaussian reconstruction, dynamic human surface reconstruction, and joint temporal human/scene/camera reconstruction. The implementation adapts those ideas to NoX's bounded Android/native architecture rather than copying research prototypes verbatim.


### 30. Frontier implementation extensions

The production graph now includes the following concrete feature additions:

- **UltraFederatedMotionFusionEngine**: bounded 128-camera federation, 32-camera authority selection, 96-camera witness participation, robust ray fusion, timing uncertainty propagation and 50-actor scheduling integration.
- **UltraMultiCameraCalibrationAuthorityEngine**: bidirectional camera graph traversal, confidence-weighted graph costs, loop-consistency scoring, calibration uncertainty scoring, sector diversity and active calibration targets.
- **AdaptiveCameraPlacementPlanner**: deterministic recommendations for physical camera placement based on occlusion hotspots, angular sector coverage, baseline and depth conditioning. Recommendations remain advisory and do not assume arbitrary camera installation is possible.
- **WorldSpaceMotionProposalEngine robustification**: iterative Huber reweighting, conditioning gates and uncertainty-scaled learned priors.
- **ProbabilisticDynamicSurfaceFusionEngine**: bounded dynamic patches, temporal prediction, robust integration, topology stability, spatial indexing, deterministic eviction and measured/inferred provenance.
- **FrontierReconstructionCoordinator**: composition of camera federation, world-space motion, dynamic surface and the existing empirical precision authority.
- **FrontierCaptureCapacityCommandCenter**: operator visibility into camera capacity, actor capacity, precision evidence, uncertainty and resident memory through the existing Liquid Glass functional layer.
- **Native Gaussian invariant hardening**: same-cell eviction and stale spatial-index protections plus regression coverage.
- **CI runner modernization**: the build gate uses the current Ubuntu 26.04 hosted image after repeated pre-step runner allocation failures on the previous pool.

These extensions are feature-additive. They do not remove the existing sparse reconstruction, dynamic Gaussian, global 4D, face, eye, hand, identity, synchronization, transport, calibration, diagnostics or qualification layers.

### 31. Physical accuracy and comparison discipline

NoX may expose aggressive *targets* such as 0.75 mm P95 in qualification profiles, but these values remain configuration targets until empirical reference-aligned measurements satisfy the existing precision authority. Modeled covariance, solver residuals or AI confidence are not substitutes for an independent reference.

For any claim of better performance than optical/infrared marker systems, qualification must use matched capture conditions and report error distributions, identity continuity, temporal jitter, occlusion recovery, reconstruction coverage, latency, sustained frame rate, thermal behavior, dropped-frame ratio and reference uncertainty. A single favorable sequence is not sufficient evidence.

### 29. Known boundary

This design is intentionally more advanced than the current stack, but it does not claim technology “100 years ahead” in a scientifically meaningful sense. Research and engineering can target frontier capability; physical claims still require measurement.