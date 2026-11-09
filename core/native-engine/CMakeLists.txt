cmake_minimum_required(VERSION 3.22.1)
project(nox_native_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(NOX_NATIVE_HOST_TESTS "Build native host tests" OFF)

set(NOX_NATIVE_CORE_SOURCES
    src/main/cpp/ffi/native_engine_abi.cpp
    src/main/cpp/runtime/native_engine_runtime.cpp
    src/main/cpp/memory/native_buffer_pool.cpp
    src/main/cpp/memory/memory_lease_auditor.cpp
    src/main/cpp/diagnostics/native_engine_diagnostics.cpp
    src/main/cpp/core/observation/motion_observation.cpp
    src/main/cpp/capture/timestamp_normalizer.cpp
    src/main/cpp/capture/capture_gap_tracker.cpp
    src/main/cpp/capture/native_capture_ingestor.cpp
    src/main/cpp/calibration/brown_conrady_distortion.cpp
    src/main/cpp/calibration/rig_graph.cpp
    src/main/cpp/calibration/calibration_solver.cpp
    src/main/cpp/calibration/calibration_health.cpp
    src/main/cpp/synchronization/clock_model.cpp
    src/main/cpp/synchronization/event_correlation.cpp
    src/main/cpp/synchronization/motion_temporal_alignment.cpp
    src/main/cpp/synchronization/subframe_alignment.cpp
    src/main/cpp/synchronization/synchronization_estimator.cpp
    src/main/cpp/tracking/joint_track.cpp
    src/main/cpp/tracking/body_tracker.cpp
    src/main/cpp/tracking/camera_motion_blur_estimator.cpp
    src/main/cpp/tracking/camera_motion_compensator.cpp
    src/main/cpp/tracking/identity_track.cpp
    src/main/cpp/face/face_geometry_fusion.cpp
    src/main/cpp/face/face_surface_alignment.cpp
    src/main/cpp/eyes/eye_geometry.cpp
    src/main/cpp/eyes/gaze_fusion.cpp
    src/main/cpp/eyes/temporal_gaze_stabilizer.cpp
    src/main/cpp/eyes/binocular_vergence.cpp
    src/main/cpp/eyes/eyelid_state.cpp
    src/main/cpp/hands/hand_geometry_fusion.cpp
    src/main/cpp/hands/finger_kinematics.cpp
    src/main/cpp/filtering/robust_state_vector.cpp
    src/main/cpp/filtering/innovation_gate.cpp
    src/main/cpp/tracking/temporal_motion_state.cpp
    src/main/cpp/geometry3d/imu_state.cpp
    src/main/cpp/tracking/imu_bias_estimator.cpp
    src/main/cpp/tracking/imu_vision_fusion.cpp
    src/main/cpp/tracking/camera_visibility_graph.cpp
    src/main/cpp/tracking/multi_camera_assignment.cpp
    src/main/cpp/tracking/geometric_identity_track.cpp
    src/main/cpp/triangulation/weighted_triangulator.cpp
    src/main/cpp/triangulation/cheirality_gate.cpp
    src/main/cpp/triangulation/reprojection_evaluator.cpp
    src/main/cpp/triangulation/covariance_propagator.cpp
    src/main/cpp/reconstruction/metric_scale_solver.cpp
    src/main/cpp/scene/scene_point_map.cpp
    src/main/cpp/scene/depth_fusion.cpp
    src/main/cpp/scene/static_dynamic_separator.cpp
    src/main/cpp/scene/dynamic_motion_separator.cpp
    src/main/cpp/scene/normal_estimator.cpp
    src/main/cpp/scene/mesh_builder.cpp
    src/main/cpp/kinematics/skeleton_model.cpp
    src/main/cpp/kinematics/joint_limit_projector.cpp
    src/main/cpp/optimization/objective_terms.cpp
    src/main/cpp/optimization/robust_solver.cpp
    src/main/cpp/optimization/production_quality_gate.cpp
    src/main/cpp/kinematics/contact_constraints.cpp
    src/main/cpp/tracking/occlusion_state.cpp
    src/main/cpp/reconstruction/missing_data_recovery.cpp
    src/main/cpp/lighting/illumination_statistics.cpp
    src/main/cpp/lighting/directional_evidence.cpp
    src/main/cpp/lighting/light_event_detector.cpp
    src/main/cpp/runtime/runtime_budget.cpp
    src/main/cpp/runtime/queue_pressure.cpp
    src/main/cpp/runtime/thermal_policy.cpp
    src/main/cpp/runtime/work_scheduler.cpp
    src/main/cpp/runtime/session_integrity_journal.cpp
    src/main/cpp/diagnostics/evidence_trace.cpp
    src/main/cpp/export/motion_export_schema.cpp
    src/main/cpp/export/export_validator.cpp
    src/main/cpp/replay/replay_reader.cpp
    src/main/cpp/replay/replay_runner.cpp
    src/main/cpp/replay/frontier_replay_contract.cpp
    src/main/cpp/calibration/global_camera_graph_optimizer.cpp
    src/main/cpp/calibration/rig_observability_analyzer.cpp
    src/main/cpp/calibration/graph_cycle_validator.cpp
    src/main/cpp/calibration/calibration_readiness_gate.cpp
    src/main/cpp/calibration/rig_stability_monitor.cpp
    src/main/cpp/calibration/minimum_spanning_camera_tree.cpp
    src/main/cpp/triangulation/reprojection_consensus.cpp
    src/main/cpp/tracking/confidence_keypoint_sampler.cpp
    src/main/cpp/reconstruction/rigid_metric_constraint.cpp
    src/main/cpp/synchronization/audio_event_sync.cpp
    src/main/cpp/synchronization/cross_view_trajectory_aligner.cpp
    src/main/cpp/synchronization/timecode_clock.cpp
    src/main/cpp/tracking/camera_coverage_score.cpp
    src/main/cpp/tracking/motion_quality_fusion.cpp
    src/main/cpp/tracking/adaptive_view_selector.cpp
    src/main/cpp/tracking/multi_person_interaction.cpp
    src/main/cpp/face/surface_visibility.cpp
    src/main/cpp/kinematics/skeleton_retargeter.cpp
    src/main/cpp/kinematics/foot_lock_solver.cpp
    src/main/cpp/tracking/loop_closure_drift.cpp
    src/main/cpp/capture/rolling_shutter_compensator.cpp
    src/main/cpp/capture/rolling_shutter_pose_compensator.cpp
    src/main/cpp/lighting/exposure_vignette_model.cpp
    src/main/cpp/memory/allocation_telemetry.cpp
    src/main/cpp/export/schema_negotiator.cpp
    src/main/cpp/replay/replay_matrix_validator.cpp
    src/main/cpp/calibration/robust_bundle_adjustment.cpp
    src/main/cpp/calibration/lens_model.cpp
    src/main/cpp/geometry3d/imu_extrinsic_calibration.cpp
    src/main/cpp/lighting/camera_color_normalizer.cpp
    src/main/cpp/reconstruction/depth_uncertainty_fusion.cpp
    src/main/cpp/scene/mesh_temporal_coherence.cpp
    src/main/cpp/reconstruction/body_surface_deformation.cpp
    src/main/cpp/tracking/rigid_prop_tracker.cpp
    src/main/cpp/capture/capture_trigger.cpp
    src/main/cpp/runtime/network_health.cpp
    src/main/cpp/synchronization/jitter_buffer.cpp
    src/main/cpp/synchronization/device_clock_transport.cpp
    src/main/cpp/runtime/thermal_sensor_drift.cpp
    src/main/cpp/capture/capture_settings_optimizer.cpp
    src/main/cpp/runtime/gpu_execution_backend.cpp
    src/main/cpp/runtime/runtime_quality_controller.cpp
    src/main/cpp/runtime/heterogeneous_execution_planner.cpp
    src/main/cpp/math/simd_dispatch.cpp
    src/main/cpp/scene/volumetric_fusion.cpp
    src/main/cpp/geometry3d/world_pose_stabilizer.cpp
    src/main/cpp/scene/scene_flow_estimator.cpp
    src/main/cpp/core/coordinate_frame_registry.cpp
    src/main/cpp/math/se3.cpp
    src/main/cpp/math/robust_loss.cpp
    src/main/cpp/geometry3d/pose_timeline.cpp
    src/main/cpp/triangulation/multiview_bundle_adjuster.cpp
    src/main/cpp/scene/sparse_tsdf_volume.cpp
    src/main/cpp/scene/adaptive_tsdf_detail_controller.cpp
    src/main/cpp/scene/point_to_plane_icp.cpp
    src/main/cpp/scene/sparse_tsdf_native_session.cpp
    src/main/cpp/scene/temporal_gaussian_surface_state.cpp
    src/main/cpp/scene/gaussian_deformation_graph.cpp
    src/main/cpp/reconstruction/temporal_surface_flow.cpp
    src/main/cpp/tracking/observation_quality_model.cpp
    src/main/cpp/tracking/interaction_temporal_filter.cpp
    src/main/cpp/tracking/camera_health_monitor.cpp
    src/main/cpp/tracking/native_observation_processor.cpp
    src/main/cpp/tracking/adaptive_motion_regime.cpp
    src/main/cpp/tracking/dense_visibility_contact_fusion.cpp
    src/main/cpp/calibration/calibration_drift_monitor.cpp
    src/main/cpp/validation/motion_accuracy_evaluator.cpp
    src/main/cpp/validation/reference_system_comparison.cpp
    src/main/cpp/validation/paired_reference_statistics.cpp
    src/main/cpp/validation/metrology_precision_qualification.cpp
    src/main/cpp/validation/precision_authority_gate.cpp
    src/main/cpp/validation/precision_capture_admission_gate.cpp
    src/main/cpp/memory/frontier_memory_domain.cpp
    src/main/cpp/validation/dynamic_metrology_protocol.cpp
)

if(NOT ANDROID)
    find_package(Threads REQUIRED)
endif()

if(ANDROID)
    add_library(nox_native_engine SHARED
        ${NOX_NATIVE_CORE_SOURCES}
        src/main/cpp/ffi/native_motion_engine_jni.cpp
        src/main/cpp/ffi/native_calibration_jni.cpp
        src/main/cpp/ffi/native_sparse_tsdf_jni.cpp
        src/main/cpp/runtime/hardware_buffer_lease_registry.cpp
    )
    target_link_libraries(
        nox_native_engine
        PRIVATE
        log
        android
    )
else()
    add_library(nox_native_engine STATIC ${NOX_NATIVE_CORE_SOURCES})
    target_link_libraries(nox_native_engine PUBLIC Threads::Threads)
endif()

target_include_directories(nox_native_engine PUBLIC src/main/cpp)
target_compile_options(nox_native_engine PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_compile_definitions(nox_native_engine PRIVATE NOX_NATIVE_ENGINE_ABI_VERSION=12)

if(NOX_NATIVE_HOST_TESTS AND NOT ANDROID)
    enable_testing()
    add_compile_options(-UNDEBUG)
    add_executable(nox_native_accuracy_tests src/main/cpp/tests/motion_accuracy_evaluator_test.cpp)
    add_executable(nox_native_reference_comparison_tests src/main/cpp/tests/reference_system_comparison_test.cpp)
    target_link_libraries(nox_native_reference_comparison_tests PRIVATE nox_native_engine)
    target_compile_options(nox_native_reference_comparison_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME nox_native_reference_comparison_tests COMMAND nox_native_reference_comparison_tests)
    add_executable(nox_native_paired_reference_statistics_tests src/main/cpp/tests/paired_reference_statistics_test.cpp)
    target_link_libraries(nox_native_paired_reference_statistics_tests PRIVATE nox_native_engine)
    target_compile_options(nox_native_paired_reference_statistics_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME nox_native_paired_reference_statistics_tests COMMAND nox_native_paired_reference_statistics_tests)
    add_executable(nox_native_calibration_drift_tests src/main/cpp/tests/calibration_drift_monitor_test.cpp)
    add_executable(nox_native_allocation_telemetry_tests src/main/cpp/tests/allocation_telemetry_test.cpp)
    add_executable(nox_native_buffer_pool_tests src/main/cpp/tests/native_buffer_pool_test.cpp)
    target_link_libraries(nox_native_buffer_pool_tests PRIVATE nox_native_engine)
    target_compile_options(nox_native_buffer_pool_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_link_libraries(nox_native_allocation_telemetry_tests PRIVATE nox_native_engine)
    target_compile_options(nox_native_allocation_telemetry_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_link_libraries(nox_native_calibration_drift_tests PRIVATE nox_native_engine)
    target_compile_options(nox_native_calibration_drift_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_link_libraries(nox_native_accuracy_tests PRIVATE nox_native_engine)
    target_compile_options(nox_native_accuracy_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)

    add_executable(nox_native_runtime_tests src/main/cpp/tests/native_engine_runtime_test.cpp)
    target_link_libraries(nox_native_runtime_tests PRIVATE nox_native_engine)
    target_compile_options(nox_native_runtime_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_executable(nox_native_gpu_backend_tests src/main/cpp/tests/gpu_execution_backend_test.cpp)
    add_executable(nox_native_heterogeneous_execution_planner_tests src/main/cpp/tests/heterogeneous_execution_planner_test.cpp)
    target_link_libraries(nox_native_heterogeneous_execution_planner_tests PRIVATE nox_native_engine)
    target_compile_options(nox_native_heterogeneous_execution_planner_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME nox_native_heterogeneous_execution_planner_tests COMMAND nox_native_heterogeneous_execution_planner_tests)
    add_executable(nox_native_frontier_memory_domain_tests src/main/cpp/tests/frontier_memory_domain_test.cpp)
    target_link_libraries(nox_native_frontier_memory_domain_tests PRIVATE nox_native_engine)
    target_compile_options(nox_native_frontier_memory_domain_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME nox_native_frontier_memory_domain_tests COMMAND nox_native_frontier_memory_domain_tests)
    add_executable(nox_native_frontier_replay_contract_tests src/main/cpp/tests/frontier_replay_contract_test.cpp)
    target_link_libraries(nox_native_frontier_replay_contract_tests PRIVATE nox_native_engine)
    target_compile_options(nox_native_frontier_replay_contract_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME nox_native_frontier_replay_contract_tests COMMAND nox_native_frontier_replay_contract_tests)
    target_link_libraries(nox_native_gpu_backend_tests PRIVATE nox_native_engine)
    target_compile_options(nox_native_gpu_backend_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)

    add_executable(nox_native_evidence_trace_tests src/main/cpp/tests/evidence_trace_test.cpp)
    target_link_libraries(nox_native_evidence_trace_tests PRIVATE nox_native_engine)
    target_compile_options(nox_native_evidence_trace_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_executable(nox_native_diagnostics_tests src/main/cpp/tests/native_engine_diagnostics_test.cpp)
    target_link_libraries(nox_native_diagnostics_tests PRIVATE nox_native_engine)
    target_compile_options(nox_native_diagnostics_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)

    add_executable(nox_native_lease_tests src/main/cpp/tests/memory_lease_auditor_test.cpp)
    target_link_libraries(nox_native_lease_tests PRIVATE nox_native_engine)
    target_compile_options(nox_native_lease_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)

    add_executable(nox_native_dense_visibility_tests src/main/cpp/tests/dense_visibility_contact_fusion_test.cpp)
    target_link_libraries(nox_native_dense_visibility_tests PRIVATE nox_native_engine)
    target_compile_options(nox_native_dense_visibility_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)

    add_executable(nox_native_surface_flow_tests src/main/cpp/tests/temporal_surface_flow_test.cpp)
    target_link_libraries(nox_native_surface_flow_tests PRIVATE nox_native_engine)
    add_executable(nox_native_gaussian_covariance_tests src/main/cpp/tests/temporal_gaussian_surface_covariance_test.cpp)
    target_link_libraries(nox_native_gaussian_covariance_tests PRIVATE nox_native_engine)
    target_compile_options(nox_native_gaussian_covariance_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME nox_native_gaussian_covariance_tests COMMAND nox_native_gaussian_covariance_tests)
    target_compile_options(nox_native_surface_flow_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)

    add_executable(nox_native_metrology_tests src/main/cpp/tests/metrology_precision_qualification_test.cpp)
    target_link_libraries(nox_native_metrology_tests PRIVATE nox_native_engine)
    target_compile_options(nox_native_metrology_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)

    add_executable(nox_native_tests src/main/cpp/tests/capture_ingestion_test.cpp)
    target_link_libraries(nox_native_tests PRIVATE nox_native_engine)
    add_executable(nox_native_calibration_tests src/main/cpp/tests/calibration_solver_test.cpp)
    target_link_libraries(nox_native_calibration_tests PRIVATE nox_native_engine)
    add_executable(nox_native_sync_tests src/main/cpp/tests/synchronization_estimator_test.cpp)
    target_link_libraries(nox_native_sync_tests PRIVATE nox_native_engine)
    add_executable(nox_native_observation_processor_tests src/main/cpp/tests/native_observation_processor_test.cpp)
    target_link_libraries(nox_native_observation_processor_tests PRIVATE nox_native_engine)
    target_compile_options(nox_native_observation_processor_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_executable(nox_native_body_tests src/main/cpp/tests/body_tracker_test.cpp)
    target_link_libraries(nox_native_body_tests PRIVATE nox_native_engine)
    add_executable(nox_native_camera_motion_blur_tests src/main/cpp/tests/camera_motion_blur_estimator_test.cpp)
    target_link_libraries(nox_native_camera_motion_blur_tests PRIVATE nox_native_engine)
    target_compile_options(nox_native_camera_motion_blur_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME nox_native_camera_motion_blur_tests COMMAND nox_native_camera_motion_blur_tests)
    add_executable(nox_native_capture_settings_high_speed_tests src/main/cpp/tests/capture_settings_optimizer_high_speed_test.cpp)
    target_link_libraries(nox_native_capture_settings_high_speed_tests PRIVATE nox_native_engine)
    target_compile_options(nox_native_capture_settings_high_speed_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME nox_native_capture_settings_high_speed_tests COMMAND nox_native_capture_settings_high_speed_tests)
    add_executable(nox_native_camera_graph_capacity_tests src/main/cpp/tests/global_camera_graph_optimizer_capacity_test.cpp)
    target_link_libraries(nox_native_camera_graph_capacity_tests PRIVATE nox_native_engine)
    add_executable(nox_native_camera_graph_device_id_tests src/main/cpp/tests/global_camera_graph_optimizer_device_id_test.cpp)
    target_link_libraries(nox_native_camera_graph_device_id_tests PRIVATE nox_native_engine)
    target_compile_options(nox_native_camera_graph_device_id_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME nox_native_camera_graph_device_id_tests COMMAND nox_native_camera_graph_device_id_tests)
    add_executable(nox_native_brown_conrady_high_end_tests src/main/cpp/tests/brown_conrady_distortion_high_end_test.cpp)
    target_link_libraries(nox_native_brown_conrady_high_end_tests PRIVATE nox_native_engine)
    target_compile_options(nox_native_brown_conrady_high_end_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME nox_native_brown_conrady_high_end_tests COMMAND nox_native_brown_conrady_high_end_tests)
    target_compile_options(nox_native_camera_graph_capacity_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME nox_native_camera_graph_capacity_tests COMMAND nox_native_camera_graph_capacity_tests)
    add_executable(nox_native_camera_motion_tests src/main/cpp/tests/camera_motion_compensator_test.cpp)
    target_link_libraries(nox_native_camera_motion_tests PRIVATE nox_native_engine)
    add_executable(nox_native_face_tests src/main/cpp/tests/face_geometry_fusion_test.cpp)
    target_link_libraries(nox_native_face_tests PRIVATE nox_native_engine)
    add_executable(nox_native_eye_tests src/main/cpp/tests/gaze_fusion_test.cpp)
    target_link_libraries(nox_native_eye_tests PRIVATE nox_native_engine)
    add_executable(nox_native_temporal_eye_tests src/main/cpp/tests/temporal_gaze_stabilizer_test.cpp)
    target_link_libraries(nox_native_temporal_eye_tests PRIVATE nox_native_engine)
    target_compile_options(nox_native_temporal_eye_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_executable(nox_native_vergence_tests src/main/cpp/tests/binocular_vergence_test.cpp)
    target_link_libraries(nox_native_vergence_tests PRIVATE nox_native_engine)
    add_executable(nox_native_hand_tests src/main/cpp/tests/hand_geometry_fusion_test.cpp)
    target_link_libraries(nox_native_hand_tests PRIVATE nox_native_engine)
    add_executable(nox_native_temporal_tests src/main/cpp/tests/temporal_motion_state_test.cpp)
    target_link_libraries(nox_native_temporal_tests PRIVATE nox_native_engine)
    add_executable(nox_native_imu_tests src/main/cpp/tests/imu_vision_fusion_test.cpp)
    target_link_libraries(nox_native_imu_tests PRIVATE nox_native_engine)
    add_executable(nox_native_association_tests src/main/cpp/tests/multi_camera_assignment_test.cpp)
    target_link_libraries(nox_native_association_tests PRIVATE nox_native_engine)
    add_executable(nox_native_triangulation_tests src/main/cpp/tests/triangulation_test.cpp)
    target_link_libraries(nox_native_triangulation_tests PRIVATE nox_native_engine)
    add_executable(nox_native_scale_tests src/main/cpp/tests/metric_scale_solver_test.cpp)
    target_link_libraries(nox_native_scale_tests PRIVATE nox_native_engine)
    add_executable(nox_native_scene_tests src/main/cpp/tests/scene_reconstruction_test.cpp)
    target_link_libraries(nox_native_scene_tests PRIVATE nox_native_engine)
    add_executable(nox_native_reconstruction_surface_dynamics_tests src/main/cpp/tests/reconstruction_surface_dynamics_test.cpp)
    target_link_libraries(nox_native_reconstruction_surface_dynamics_tests PRIVATE nox_native_engine)
    add_executable(nox_native_dynamic_scene_tests src/main/cpp/tests/dynamic_motion_separator_test.cpp)
    target_link_libraries(nox_native_dynamic_scene_tests PRIVATE nox_native_engine)
    add_executable(nox_native_icp_tests src/main/cpp/tests/point_to_plane_icp_test.cpp)
    target_link_libraries(nox_native_icp_tests PRIVATE nox_native_engine)
    add_executable(nox_native_kinematic_tests src/main/cpp/tests/kinematic_optimizer_test.cpp)
    target_link_libraries(nox_native_kinematic_tests PRIVATE nox_native_engine)
    add_executable(nox_native_joint_limit_tests src/main/cpp/tests/joint_limit_projector_test.cpp)
    target_link_libraries(nox_native_joint_limit_tests PRIVATE nox_native_engine)
    add_executable(nox_native_occlusion_tests src/main/cpp/tests/missing_data_recovery_test.cpp)
    target_link_libraries(nox_native_occlusion_tests PRIVATE nox_native_engine)
    add_executable(nox_native_lighting_tests src/main/cpp/tests/lighting_telemetry_test.cpp)
    target_link_libraries(nox_native_lighting_tests PRIVATE nox_native_engine)
    add_executable(nox_native_runtime_governor_tests src/main/cpp/tests/runtime_governor_test.cpp)
    target_link_libraries(nox_native_runtime_governor_tests PRIVATE nox_native_engine)
    add_executable(nox_native_replay_tests src/main/cpp/tests/replay_validation_test.cpp)
    target_link_libraries(nox_native_replay_tests PRIVATE nox_native_engine)
    add_executable(nox_native_graph_tests src/main/cpp/tests/high_end_geometry_test.cpp)
    target_link_libraries(nox_native_graph_tests PRIVATE nox_native_engine)
    add_executable(nox_native_camera_tree_tests src/main/cpp/tests/minimum_spanning_camera_tree_test.cpp)
    target_link_libraries(nox_native_camera_tree_tests PRIVATE nox_native_engine)
    add_executable(nox_native_calibration_readiness_tests src/main/cpp/tests/calibration_readiness_gate_test.cpp)
    target_link_libraries(nox_native_calibration_readiness_tests PRIVATE nox_native_engine)
    add_executable(nox_native_graph_cycle_tests src/main/cpp/tests/graph_cycle_validator_test.cpp)
    target_link_libraries(nox_native_graph_cycle_tests PRIVATE nox_native_engine)
    add_executable(nox_native_rig_observability_tests src/main/cpp/tests/rig_observability_analyzer_test.cpp)
    target_link_libraries(nox_native_rig_observability_tests PRIVATE nox_native_engine)
    add_executable(nox_native_sync_selection_tests src/main/cpp/tests/sync_and_view_selection_test.cpp)
    target_link_libraries(nox_native_sync_selection_tests PRIVATE nox_native_engine)
    add_executable(nox_native_articulation_tests src/main/cpp/tests/articulation_upgrade_test.cpp)
    target_link_libraries(nox_native_articulation_tests PRIVATE nox_native_engine)
    add_executable(nox_native_identity_association_high_end_tests src/main/cpp/tests/identity_association_high_end_test.cpp)
    target_link_libraries(nox_native_identity_association_high_end_tests PRIVATE nox_native_engine)
    add_executable(nox_native_multiview_bundle_jacobian_tests src/main/cpp/tests/multiview_bundle_adjuster_analytic_point_jacobian_test.cpp)
    target_link_libraries(nox_native_multiview_bundle_jacobian_tests PRIVATE nox_native_engine)
    target_compile_options(nox_native_multiview_bundle_jacobian_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME nox_native_multiview_bundle_jacobian_tests COMMAND nox_native_multiview_bundle_jacobian_tests)
    add_executable(nox_native_stability_tests src/main/cpp/tests/stability_upgrade_test.cpp)
    target_link_libraries(nox_native_stability_tests PRIVATE nox_native_engine)
    add_executable(nox_native_rolling_shutter_pose_tests src/main/cpp/tests/rolling_shutter_pose_compensator_test.cpp)
    target_link_libraries(nox_native_rolling_shutter_pose_tests PRIVATE nox_native_engine)
    add_executable(nox_native_release_tests src/main/cpp/tests/release_contract_test.cpp)
    target_link_libraries(nox_native_release_tests PRIVATE nox_native_engine)
    add_executable(nox_native_calibration_depth_tests src/main/cpp/tests/calibration_depth_upgrade_test.cpp)
    target_link_libraries(nox_native_calibration_depth_tests PRIVATE nox_native_engine)
    add_executable(nox_native_precision_capture_admission_tests src/main/cpp/tests/precision_capture_admission_gate_test.cpp)
    target_link_libraries(nox_native_precision_capture_admission_tests PRIVATE nox_native_engine)
    add_executable(nox_native_surface_prop_tests src/main/cpp/tests/surface_prop_upgrade_test.cpp)
    target_link_libraries(nox_native_surface_prop_tests PRIVATE nox_native_engine)
    add_executable(nox_native_ops_tests src/main/cpp/tests/production_ops_upgrade_test.cpp)
    target_link_libraries(nox_native_ops_tests PRIVATE nox_native_engine)
    add_executable(nox_native_production_quality_gate_tests src/main/cpp/tests/production_quality_gate_test.cpp)
    target_link_libraries(nox_native_production_quality_gate_tests PRIVATE nox_native_engine)
    add_executable(nox_native_perf_tests src/main/cpp/tests/performance_upgrade_test.cpp)
    target_link_libraries(nox_native_perf_tests PRIVATE nox_native_engine)
    add_executable(nox_native_geometry_contract_tests src/main/cpp/tests/geometry_contract_test.cpp)
    target_link_libraries(nox_native_geometry_contract_tests PRIVATE nox_native_engine)
    add_executable(nox_native_missing_batch1_tests src/main/cpp/tests/high_end_missing_batch1_test.cpp)
    target_link_libraries(nox_native_missing_batch1_tests PRIVATE nox_native_engine)
    target_compile_options(nox_native_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_calibration_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_sync_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_body_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_camera_motion_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_face_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_eye_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_vergence_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_hand_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_temporal_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_imu_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_association_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_triangulation_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_scale_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_scene_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_dynamic_scene_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_executable(nox_native_missing_batch2_tests src/main/cpp/tests/high_end_missing_batch2_test.cpp)
    target_link_libraries(nox_native_missing_batch2_tests PRIVATE nox_native_engine)
    add_executable(nox_native_se3_graph_tests src/main/cpp/tests/se3_graph_optimizer_test.cpp)
    target_link_libraries(nox_native_se3_graph_tests PRIVATE nox_native_engine)
    add_executable(nox_native_sparse_tsdf_tests src/main/cpp/tests/sparse_tsdf_volume_test.cpp)
    target_link_libraries(nox_native_sparse_tsdf_tests PRIVATE nox_native_engine)
    add_executable(nox_native_adaptive_tsdf_detail_tests src/main/cpp/tests/adaptive_tsdf_detail_controller_test.cpp)
    target_link_libraries(nox_native_adaptive_tsdf_detail_tests PRIVATE nox_native_engine)
    add_executable(nox_native_bundle_adjustment_tests src/main/cpp/tests/multiview_bundle_adjustment_test.cpp)
    target_link_libraries(nox_native_bundle_adjustment_tests PRIVATE nox_native_engine)
    add_executable(nox_native_pose_timeline_tests src/main/cpp/tests/pose_timeline_test.cpp)
    target_link_libraries(nox_native_pose_timeline_tests PRIVATE nox_native_engine)
    add_executable(nox_native_motion_quality_tests src/main/cpp/tests/motion_quality_fusion_test.cpp)
    target_link_libraries(nox_native_motion_quality_tests PRIVATE nox_native_engine)
    add_executable(nox_native_tsdf_tests src/main/cpp/tests/sparse_tsdf_volume_test.cpp)
    target_link_libraries(nox_native_tsdf_tests PRIVATE nox_native_engine)
    add_executable(nox_native_quality_tests src/main/cpp/tests/observation_quality_model_test.cpp)
    target_link_libraries(nox_native_quality_tests PRIVATE nox_native_engine)
    add_executable(nox_native_interaction_temporal_tests src/main/cpp/tests/interaction_temporal_filter_test.cpp)
    target_link_libraries(nox_native_interaction_temporal_tests PRIVATE nox_native_engine)
    add_executable(nox_native_quality_uncertainty_tests src/main/cpp/tests/quality_uncertainty_upgrade_test.cpp)
    target_link_libraries(nox_native_quality_uncertainty_tests PRIVATE nox_native_engine)
    add_executable(nox_native_quality_bottleneck_tests src/main/cpp/tests/quality_bottleneck_test.cpp)
    target_link_libraries(nox_native_quality_bottleneck_tests PRIVATE nox_native_engine)
    add_executable(nox_native_jitter_order_tests src/main/cpp/tests/jitter_buffer_order_test.cpp)
    target_link_libraries(nox_native_jitter_order_tests PRIVATE nox_native_engine)
    add_executable(nox_native_clock_robust_tests src/main/cpp/tests/clock_model_robust_test.cpp)
    target_link_libraries(nox_native_clock_robust_tests PRIVATE nox_native_engine)
    add_executable(nox_native_clock_exchange_tests src/main/cpp/tests/device_clock_exchange_test.cpp)
    target_link_libraries(nox_native_clock_exchange_tests PRIVATE nox_native_engine)
    add_executable(nox_native_view_observability_tests src/main/cpp/tests/adaptive_view_observability_test.cpp)
    target_link_libraries(nox_native_view_observability_tests PRIVATE nox_native_engine)
    add_executable(nox_native_camera_health_tests src/main/cpp/tests/camera_health_monitor_test.cpp)
    target_link_libraries(nox_native_camera_health_tests PRIVATE nox_native_engine)
    add_executable(nox_native_dynamic_scene_separator_tests src/main/cpp/tests/static_dynamic_separator_test.cpp)
    target_link_libraries(nox_native_dynamic_scene_separator_tests PRIVATE nox_native_engine)
    add_executable(nox_native_scene_point_map_high_end_tests src/main/cpp/tests/scene_point_map_high_end_test.cpp)
    target_link_libraries(nox_native_scene_point_map_high_end_tests PRIVATE nox_native_engine)
    add_executable(nox_native_precision_authority_tests src/main/cpp/tests/precision_authority_gate_test.cpp)
    target_link_libraries(nox_native_precision_authority_tests PRIVATE nox_native_engine)
    add_executable(nox_native_motion_temporal_alignment_tests src/main/cpp/tests/motion_temporal_alignment_test.cpp)
    target_link_libraries(nox_native_motion_temporal_alignment_tests PRIVATE nox_native_engine)
    add_executable(nox_native_temporal_gaussian_surface_tests src/main/cpp/tests/temporal_gaussian_surface_state_test.cpp)
    target_link_libraries(nox_native_temporal_gaussian_surface_tests PRIVATE nox_native_engine)
    add_executable(nox_native_gaussian_deformation_graph_tests src/main/cpp/tests/gaussian_deformation_graph_test.cpp)
    target_link_libraries(nox_native_gaussian_deformation_graph_tests PRIVATE nox_native_engine)
    add_executable(nox_native_face_surface_alignment_high_end_tests src/main/cpp/tests/face_surface_alignment_high_end_test.cpp)
    target_link_libraries(nox_native_face_surface_alignment_high_end_tests PRIVATE nox_native_engine)
    add_executable(nox_native_dynamic_metrology_tests src/main/cpp/tests/dynamic_metrology_protocol_test.cpp)
    target_link_libraries(nox_native_dynamic_metrology_tests PRIVATE nox_native_engine)
    add_executable(nox_native_cross_view_trajectory_alignment_tests src/main/cpp/tests/cross_view_trajectory_aligner_test.cpp)
    target_link_libraries(nox_native_cross_view_trajectory_alignment_tests PRIVATE nox_native_engine)
    add_executable(nox_native_rig_stability_monitor_tests src/main/cpp/tests/rig_stability_monitor_test.cpp)
    target_link_libraries(nox_native_rig_stability_monitor_tests PRIVATE nox_native_engine)
    add_executable(nox_native_multi_camera_assignment_high_end_tests src/main/cpp/tests/multi_camera_assignment_high_end_test.cpp)
    target_link_libraries(nox_native_multi_camera_assignment_high_end_tests PRIVATE nox_native_engine)
    add_executable(nox_native_session_integrity_journal_tests src/main/cpp/tests/session_integrity_journal_test.cpp)
    target_link_libraries(nox_native_session_integrity_journal_tests PRIVATE nox_native_engine)
    add_executable(nox_native_runtime_quality_tests src/main/cpp/tests/runtime_quality_controller_test.cpp)
    target_link_libraries(nox_native_runtime_quality_tests PRIVATE nox_native_engine)
    target_compile_options(nox_native_missing_batch2_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_se3_graph_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_sparse_tsdf_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_adaptive_tsdf_detail_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_bundle_adjustment_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_pose_timeline_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_motion_quality_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_tsdf_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_quality_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_interaction_temporal_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_quality_uncertainty_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_quality_bottleneck_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_jitter_order_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_clock_robust_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_clock_exchange_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_view_observability_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_camera_health_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_dynamic_scene_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_runtime_quality_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_dynamic_scene_separator_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_identity_association_high_end_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(nox_native_reconstruction_surface_dynamics_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)

    add_test(NAME nox_native_calibration_drift_tests COMMAND nox_native_calibration_drift_tests)
    add_test(NAME nox_native_allocation_telemetry_tests COMMAND nox_native_allocation_telemetry_tests)
    add_test(NAME nox_native_buffer_pool_tests COMMAND nox_native_buffer_pool_tests)
    add_test(NAME nox_native_runtime_tests COMMAND nox_native_runtime_tests)
    add_test(NAME nox_native_gpu_backend_tests COMMAND nox_native_gpu_backend_tests)
    add_test(NAME nox_native_evidence_trace_tests COMMAND nox_native_evidence_trace_tests)
    add_test(NAME nox_native_diagnostics_tests COMMAND nox_native_diagnostics_tests)
    add_test(NAME nox_native_tests COMMAND nox_native_tests)
    add_test(NAME nox_native_calibration_tests COMMAND nox_native_calibration_tests)
    add_test(NAME nox_native_sync_tests COMMAND nox_native_sync_tests)
    add_test(NAME nox_native_body_tests COMMAND nox_native_body_tests)
    add_test(NAME nox_native_camera_motion_tests COMMAND nox_native_camera_motion_tests)
    add_test(NAME nox_native_face_tests COMMAND nox_native_face_tests)
    add_test(NAME nox_native_eye_tests COMMAND nox_native_eye_tests)
    add_test(NAME nox_native_vergence_tests COMMAND nox_native_vergence_tests)
    add_test(NAME nox_native_hand_tests COMMAND nox_native_hand_tests)
    add_test(NAME nox_native_temporal_tests COMMAND nox_native_temporal_tests)
    add_test(NAME nox_native_imu_tests COMMAND nox_native_imu_tests)
    add_test(NAME nox_native_association_tests COMMAND nox_native_association_tests)
    add_test(NAME nox_native_triangulation_tests COMMAND nox_native_triangulation_tests)
    add_test(NAME nox_native_scale_tests COMMAND nox_native_scale_tests)
    add_test(NAME nox_native_scene_tests COMMAND nox_native_scene_tests)
    add_test(NAME nox_native_reconstruction_surface_dynamics_tests COMMAND nox_native_reconstruction_surface_dynamics_tests)
    add_test(NAME nox_native_dynamic_scene_tests COMMAND nox_native_dynamic_scene_tests)
    add_test(NAME nox_native_dynamic_scene_separator_tests COMMAND nox_native_dynamic_scene_separator_tests)
    add_test(NAME nox_native_scene_point_map_high_end_tests COMMAND nox_native_scene_point_map_high_end_tests)
    add_test(NAME nox_native_precision_authority_tests COMMAND nox_native_precision_authority_tests)
    add_test(NAME nox_native_motion_temporal_alignment_tests COMMAND nox_native_motion_temporal_alignment_tests)
    add_test(NAME nox_native_temporal_gaussian_surface_tests COMMAND nox_native_temporal_gaussian_surface_tests)
    add_test(NAME nox_native_gaussian_deformation_graph_tests COMMAND nox_native_gaussian_deformation_graph_tests)
    add_test(NAME nox_native_face_surface_alignment_high_end_tests COMMAND nox_native_face_surface_alignment_high_end_tests)
    add_test(NAME nox_native_dynamic_metrology_tests COMMAND nox_native_dynamic_metrology_tests)
    add_test(NAME nox_native_cross_view_trajectory_alignment_tests COMMAND nox_native_cross_view_trajectory_alignment_tests)
    add_test(NAME nox_native_rig_stability_monitor_tests COMMAND nox_native_rig_stability_monitor_tests)
    add_test(NAME nox_native_multi_camera_assignment_high_end_tests COMMAND nox_native_multi_camera_assignment_high_end_tests)
    add_test(NAME nox_native_session_integrity_journal_tests COMMAND nox_native_session_integrity_journal_tests)
    add_test(NAME nox_native_icp_tests COMMAND nox_native_icp_tests)
    add_test(NAME nox_native_kinematic_tests COMMAND nox_native_kinematic_tests)
    add_test(NAME nox_native_joint_limit_tests COMMAND nox_native_joint_limit_tests)
    add_test(NAME nox_native_occlusion_tests COMMAND nox_native_occlusion_tests)
    add_test(NAME nox_native_lighting_tests COMMAND nox_native_lighting_tests)
    add_test(NAME nox_native_runtime_governor_tests COMMAND nox_native_runtime_governor_tests)
    add_test(NAME nox_native_replay_tests COMMAND nox_native_replay_tests)
    add_test(NAME nox_native_graph_tests COMMAND nox_native_graph_tests)
    add_test(NAME nox_native_camera_tree_tests COMMAND nox_native_camera_tree_tests)
    add_test(NAME nox_native_calibration_readiness_tests COMMAND nox_native_calibration_readiness_tests)
    add_test(NAME nox_native_graph_cycle_tests COMMAND nox_native_graph_cycle_tests)
    add_test(NAME nox_native_rig_observability_tests COMMAND nox_native_rig_observability_tests)
    add_test(NAME nox_native_sync_selection_tests COMMAND nox_native_sync_selection_tests)
    add_test(NAME nox_native_articulation_tests COMMAND nox_native_articulation_tests)
    add_test(NAME nox_native_identity_association_high_end_tests COMMAND nox_native_identity_association_high_end_tests)
    add_test(NAME nox_native_stability_tests COMMAND nox_native_stability_tests)
    add_test(NAME nox_native_rolling_shutter_pose_tests COMMAND nox_native_rolling_shutter_pose_tests)
    add_test(NAME nox_native_release_tests COMMAND nox_native_release_tests)
    add_test(NAME nox_native_calibration_depth_tests COMMAND nox_native_calibration_depth_tests)
    add_test(NAME nox_native_precision_capture_admission_tests COMMAND nox_native_precision_capture_admission_tests)
    add_test(NAME nox_native_surface_prop_tests COMMAND nox_native_surface_prop_tests)
    add_test(NAME nox_native_ops_tests COMMAND nox_native_ops_tests)
    add_test(NAME nox_native_perf_tests COMMAND nox_native_perf_tests)
    add_test(NAME nox_native_geometry_contract_tests COMMAND nox_native_geometry_contract_tests)
    add_test(NAME nox_native_missing_batch1_tests COMMAND nox_native_missing_batch1_tests)
    add_test(NAME nox_native_missing_batch2_tests COMMAND nox_native_missing_batch2_tests)
    add_test(NAME nox_native_se3_graph_tests COMMAND nox_native_se3_graph_tests)
    add_test(NAME nox_native_sparse_tsdf_tests COMMAND nox_native_sparse_tsdf_tests)
    add_test(NAME nox_native_adaptive_tsdf_detail_tests COMMAND nox_native_adaptive_tsdf_detail_tests)
    add_test(NAME nox_native_bundle_adjustment_tests COMMAND nox_native_bundle_adjustment_tests)
    add_test(NAME nox_native_pose_timeline_tests COMMAND nox_native_pose_timeline_tests)
    add_test(NAME nox_native_motion_quality_tests COMMAND nox_native_motion_quality_tests)
    add_test(NAME nox_native_tsdf_tests COMMAND nox_native_tsdf_tests)
    add_test(NAME nox_native_quality_tests COMMAND nox_native_quality_tests)
    add_test(NAME nox_native_interaction_temporal_tests COMMAND nox_native_interaction_temporal_tests)
    add_test(NAME nox_native_quality_uncertainty_tests COMMAND nox_native_quality_uncertainty_tests)
    add_test(NAME nox_native_quality_bottleneck_tests COMMAND nox_native_quality_bottleneck_tests)
    add_test(NAME nox_native_jitter_order_tests COMMAND nox_native_jitter_order_tests)
    add_test(NAME nox_native_clock_robust_tests COMMAND nox_native_clock_robust_tests)
    add_test(NAME nox_native_clock_exchange_tests COMMAND nox_native_clock_exchange_tests)
    add_test(NAME nox_native_view_observability_tests COMMAND nox_native_view_observability_tests)
    add_test(NAME nox_native_camera_health_tests COMMAND nox_native_camera_health_tests)
    add_test(NAME nox_native_runtime_quality_tests COMMAND nox_native_runtime_quality_tests)
endif()
