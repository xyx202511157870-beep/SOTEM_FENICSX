from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location("sotem_pipeline_for_model_consistency_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_model_consistency_reports_empymod_matching_depths_and_resistivity():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(sp.PipelineConfig())

    assert diagnostics["reference_mode"] == "noip"
    assert diagnostics["source_depth_start"] == pytest.approx(0.1)
    assert diagnostics["source_depth_end"] == pytest.approx(0.1)
    assert diagnostics["receiver_depth"] == pytest.approx(0.1)
    assert diagnostics["rho_air"] == pytest.approx(1.0e8)
    assert diagnostics["rho_earth"] == pytest.approx(100.0)
    assert diagnostics["layer_depths"] == []
    assert diagnostics["layer_resistivities"] == [pytest.approx(100.0)]


def test_layered_model_consistency_builds_shared_fem_and_empymod_layers():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        layer_depths=(2000.0, 2200.0),
        layer_resistivities=(200.0, 20.0, 200.0),
    )

    diagnostics = sp.validate_model_consistency(config)
    depth, res = sp._empymod_depth_res(config)

    assert diagnostics["layer_depths"] == [pytest.approx(2000.0), pytest.approx(2200.0)]
    assert diagnostics["layer_resistivities"] == [pytest.approx(200.0), pytest.approx(20.0), pytest.approx(200.0)]
    assert diagnostics["rho_earth"] == pytest.approx(200.0)
    assert diagnostics["sigma_earth"] == pytest.approx(1.0 / 200.0)
    assert depth == [pytest.approx(0.0), pytest.approx(2000.0), pytest.approx(2200.0)]
    assert res == [pytest.approx(config.rho_air), pytest.approx(200.0), pytest.approx(20.0), pytest.approx(200.0)]
    assert sp._earth_resistivity_at_depth(1999.9, config) == pytest.approx(200.0)
    assert sp._earth_resistivity_at_depth(2000.0, config) == pytest.approx(20.0)
    assert sp._earth_resistivity_at_depth(2199.9, config) == pytest.approx(20.0)
    assert sp._earth_resistivity_at_depth(2200.0, config) == pytest.approx(200.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"layer_depths": (2000.0,), "layer_resistivities": (200.0,)},
        {"layer_depths": (2200.0, 2000.0), "layer_resistivities": (200.0, 20.0, 200.0)},
        {"layer_depths": (0.0,), "layer_resistivities": (200.0, 20.0)},
        {"layer_depths": (2000.0,), "layer_resistivities": (200.0, -20.0)},
    ],
)
def test_model_consistency_rejects_invalid_layered_models(kwargs):
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="layer"):
        sp.validate_model_consistency(sp.PipelineConfig(**kwargs))


def test_model_consistency_rejects_cole_cole_window_spanning_multiple_resistivity_layers():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        polarization="cole-cole",
        layer_depths=(2000.0, 2200.0),
        layer_resistivities=(200.0, 20.0, 200.0),
    )

    with pytest.raises(ValueError, match="spans multiple resistivity layers"):
        sp.validate_model_consistency(config)


def test_model_consistency_rejects_analytic_halfspace_initial_for_nonuniform_layers():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        initial_dc_mode="analytic_halfspace",
        layer_depths=(2000.0, 2200.0),
        layer_resistivities=(200.0, 20.0, 200.0),
    )

    with pytest.raises(ValueError, match="analytic_halfspace"):
        sp.validate_model_consistency(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rho_air", 0.0),
        ("rho_earth", -100.0),
        ("mu_r_air", 0.0),
        ("mu_r_earth", -1.0),
    ],
)
def test_model_consistency_rejects_nonpositive_material_parameters(field, value):
    sp = _load_pipeline_module()
    kwargs = {field: value}

    with pytest.raises(ValueError, match=field):
        sp.validate_model_consistency(sp.PipelineConfig(**kwargs))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_start": (-50.0, 0.0, 0.0)},
        {"source_end": (50.0, 0.0, 0.1)},
        {"receiver": (500.0, 50.0, 0.0)},
    ],
)
def test_model_consistency_rejects_air_or_interface_electrodes(kwargs):
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="below the z=0 earth surface"):
        sp.validate_model_consistency(sp.PipelineConfig(**kwargs))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_current", 0.0),
        ("ramp_off_time", 0.0),
        ("t_min", 0.0),
        ("time_growth", 1.0),
        ("time_theta", 0.49),
        ("source_mesh_size", 0.0),
        ("receiver_mesh_size", -1.0),
        ("source_rhs_sign", 0.0),
        ("source_quadrature_points", -1),
        ("stop_after_outputs", -1),
        ("memory_limit_gb", -1.0),
        ("memory_safety_fraction", 0.0),
    ],
)
def test_model_consistency_rejects_nonpositive_runtime_parameters(field, value):
    sp = _load_pipeline_module()
    kwargs = {field: value}

    with pytest.raises(ValueError, match=field):
        sp.validate_model_consistency(sp.PipelineConfig(**kwargs))


def test_model_consistency_accepts_positive_source_rhs_sign_for_diagnostics():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(sp.PipelineConfig(source_rhs_sign=1.0))

    assert diagnostics["source_rhs_sign"] == pytest.approx(1.0)


def test_model_consistency_reports_manual_source_quadrature_points():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(sp.PipelineConfig(source_quadrature_points=201))

    assert diagnostics["source_quadrature_points"] == 201


def test_manual_line_auto_quadrature_resolves_source_mesh_scale():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        source_start=(-500.0, 200.0, -0.1),
        source_end=(500.0, 200.0, -0.1),
        source_mesh_size=40.0,
    )

    assert sp._manual_line_source_quadrature_count(1000.0, config) >= 5001
    assert sp._manual_line_source_quadrature_count(1000.0, sp.PipelineConfig(source_quadrature_points=2001)) == 2001


def test_source_line_segments_from_meshio_blocks_selects_and_sorts_physical_source_lines():
    sp = _load_pipeline_module()
    points = np.asarray(
        [
            [0.0, 1.0, -0.1],
            [1.0, 1.0, -0.1],
            [2.0, 1.0, -0.1],
            [3.0, 1.0, -0.1],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    line_cells = np.asarray([[2, 3], [0, 1], [1, 2], [0, 4]], dtype=np.int64)
    physical_tags = np.asarray([sp.PHYS_SOURCE_LINE, sp.PHYS_SOURCE_LINE, sp.PHYS_SOURCE_LINE, 999], dtype=np.int64)

    segments = sp._source_line_segments_from_meshio_blocks(
        points,
        [("line", line_cells)],
        [physical_tags],
        sp.PipelineConfig(source_start=(0.0, 1.0, -0.1), source_end=(3.0, 1.0, -0.1)),
    )

    assert segments is not None
    np.testing.assert_allclose(
        segments["segments"],
        np.asarray(
            [
                [[0.0, 1.0, -0.1], [1.0, 1.0, -0.1]],
                [[1.0, 1.0, -0.1], [2.0, 1.0, -0.1]],
                [[2.0, 1.0, -0.1], [3.0, 1.0, -0.1]],
            ]
        ),
    )
    assert segments["segment_count"] == 3
    assert segments["total_length"] == pytest.approx(3.0)
    assert segments["max_segment_length"] == pytest.approx(1.0)


def test_manual_line_integration_points_use_mesh_segments_when_available():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        source_start=(0.0, 1.0, -0.1),
        source_end=(2.0, 1.0, -0.1),
        source_mesh_size=1.0,
    )
    mesh_segments = {
        "segments": np.asarray(
            [
                [[0.0, 1.0, -0.1], [1.0, 1.0, -0.1]],
                [[1.0, 1.0, -0.1], [2.0, 1.0, -0.1]],
            ],
            dtype=float,
        ),
        "segment_count": 2,
        "total_length": 2.0,
        "min_segment_length": 1.0,
        "max_segment_length": 1.0,
        "mean_segment_length": 1.0,
    }

    points, weights, svals, diagnostics = sp._manual_line_source_integration_points(
        config,
        mesh_segments=mesh_segments,
    )

    assert diagnostics["integration_mode"] == "mesh_segments"
    assert diagnostics["segment_count"] == 2
    assert diagnostics["quadrature_points_per_segment_min"] >= 2
    assert diagnostics["quadrature_points_per_segment_max"] > 2
    assert points.shape[0] > 4
    assert svals.tolist() == pytest.approx(sorted(svals.tolist()))
    assert float(np.sum(weights)) == pytest.approx(2.0)


def test_transient_source_projection_uses_unit_current_shape():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(source_current=10.0)

    assert sp._endpoint_load_current(config, use_unit_current=False) == pytest.approx(10.0)
    assert sp._endpoint_load_current(config, use_unit_current=True) == pytest.approx(1.0)


def test_source_projection_mode_defaults_to_charge_conserving_and_accepts_raw_diagnostic_mode():
    sp = _load_pipeline_module()

    default_diagnostics = sp.validate_model_consistency(sp.PipelineConfig())
    raw_diagnostics = sp.validate_model_consistency(sp.PipelineConfig(source_projection_mode="raw"))

    assert sp.PipelineConfig().source_projection_mode == "charge_conserving"
    assert default_diagnostics["source_projection_mode"] == "charge_conserving"
    assert raw_diagnostics["source_projection_mode"] == "raw"


def test_model_consistency_rejects_unknown_source_projection_mode():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="source_projection_mode"):
        sp.validate_model_consistency(sp.PipelineConfig(source_projection_mode="bad"))


def test_empymod_receiver_mapping_supports_hz_and_dbdt():
    sp = _load_pipeline_module()

    hz_rec, hz_mrec, hz_signal, hz_factor = sp._empymod_rec_mapping((1.0, 2.0, -0.1), "Hz")
    dbdt_rec, dbdt_mrec, dbdt_signal, dbdt_factor = sp._empymod_rec_mapping((1.0, 2.0, -0.1), "dBzdt")

    assert hz_rec == [pytest.approx(1.0), pytest.approx(2.0), pytest.approx(0.1), pytest.approx(0.0), pytest.approx(90.0)]
    assert hz_mrec is True
    assert hz_signal == -1
    assert hz_factor == pytest.approx(1.0)
    assert dbdt_rec == hz_rec
    assert dbdt_mrec is True
    assert dbdt_signal == 0
    assert dbdt_factor < 0.0


def test_forward_components_include_hz_for_biot_magnetic_receiver():
    sp = _load_pipeline_module()

    assert sp._forward_components(sp.PipelineConfig()) == ["Ex", "Ey", "dBzdt"]
    assert sp._forward_components(sp.PipelineConfig(magnetic_receiver_mode="biot_current")) == [
        "Ex",
        "Ey",
        "Hz",
        "dBzdt",
    ]


def test_model_consistency_rejects_biot_rate_dbdt_without_biot_h_receiver():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="biot_rate"):
        sp.validate_model_consistency(sp.PipelineConfig(magnetic_receiver_mode="curl", magnetic_dbdt_mode="biot_rate"))

    diagnostics = sp.validate_model_consistency(
        sp.PipelineConfig(magnetic_receiver_mode="biot_current", magnetic_dbdt_mode="biot_rate")
    )
    assert diagnostics["magnetic_dbdt_mode"] == "biot_rate"


def test_model_consistency_reports_nedelec_order():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(sp.PipelineConfig(nedelec_order=2))

    assert diagnostics["nedelec_order"] == 2


@pytest.mark.parametrize("nedelec_order", [0, 3])
def test_model_consistency_rejects_unsupported_nedelec_order(nedelec_order):
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="nedelec_order"):
        sp.validate_model_consistency(sp.PipelineConfig(nedelec_order=nedelec_order))


def test_bdf2_step_coefficients_reduce_to_constant_step_formula():
    sp = _load_pipeline_module()

    coeffs = sp._bdf2_step_coefficients(dt=2.0, previous_dt=2.0)

    assert coeffs["lhs"] == pytest.approx(0.75)
    assert coeffs["old"] == pytest.approx(1.0)
    assert coeffs["older"] == pytest.approx(-0.25)


def test_model_consistency_reports_bdf2_time_method():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(sp.PipelineConfig(time_method="bdf2"))

    assert diagnostics["time_method"] == "bdf2"


def test_model_consistency_reports_conductivity_divergence_cleaning():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(sp.PipelineConfig(divergence_cleaning="conductivity"))

    assert diagnostics["divergence_cleaning"] == "conductivity"


def test_model_consistency_rejects_unknown_divergence_cleaning_mode():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="divergence_cleaning"):
        sp.validate_model_consistency(sp.PipelineConfig(divergence_cleaning="bad-mode"))


def test_model_consistency_accepts_ohmic_biot_magnetic_receiver_mode():
    sp = _load_pipeline_module()

    config = sp.PipelineConfig(magnetic_receiver_mode="biot_ohmic")
    diagnostics = sp.validate_model_consistency(config)

    assert diagnostics["magnetic_receiver_mode"] == "biot_ohmic"


def test_model_consistency_accepts_natural_outer_boundary_mode():
    sp = _load_pipeline_module()

    config = sp.PipelineConfig(outer_boundary_mode="natural")
    diagnostics = sp.validate_model_consistency(config)

    assert diagnostics["outer_boundary_mode"] == "natural"


def test_model_consistency_accepts_robin_outer_boundary_mode():
    sp = _load_pipeline_module()

    config = sp.PipelineConfig(outer_boundary_mode="robin", outer_boundary_robin_scale=0.5)
    diagnostics = sp.validate_model_consistency(config)

    assert diagnostics["outer_boundary_mode"] == "robin"
    assert diagnostics["outer_boundary_robin_scale"] == pytest.approx(0.5)


def test_model_consistency_rejects_unknown_outer_boundary_mode():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="outer_boundary_mode"):
        sp.validate_model_consistency(sp.PipelineConfig(outer_boundary_mode="bad"))


def test_model_consistency_rejects_negative_robin_boundary_scale():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="outer_boundary_robin_scale"):
        sp.validate_model_consistency(sp.PipelineConfig(outer_boundary_robin_scale=-1.0))


def test_model_consistency_accepts_mean_receiver_evaluation_mode():
    sp = _load_pipeline_module()

    config = sp.PipelineConfig(receiver_evaluation_mode="mean")
    diagnostics = sp.validate_model_consistency(config)

    assert diagnostics["receiver_evaluation_mode"] == "mean"


def test_model_consistency_reports_empymod_reference_settings():
    sp = _load_pipeline_module()

    config = sp.PipelineConfig(empymod_srcpts=33, empymod_ht="qwe", empymod_ft="qwe", reference_audit_srcpts=65)
    diagnostics = sp.validate_model_consistency(config)

    assert diagnostics["empymod_srcpts"] == 33
    assert diagnostics["empymod_ht"] == "qwe"
    assert diagnostics["empymod_ft"] == "qwe"
    assert diagnostics["reference_audit_srcpts"] == 65


@pytest.mark.parametrize(
    "kwargs",
    [
        {"empymod_srcpts": 0},
        {"empymod_ht": "bad"},
        {"empymod_ft": "bad"},
        {"reference_audit_srcpts": -1},
    ],
)
def test_model_consistency_rejects_invalid_empymod_reference_settings(kwargs):
    sp = _load_pipeline_module()

    with pytest.raises(ValueError):
        sp.validate_model_consistency(sp.PipelineConfig(**kwargs))


def test_receiver_evaluation_mode_defaults_to_median():
    sp = _load_pipeline_module()

    assert sp.PipelineConfig().receiver_evaluation_mode == "median"


def test_time_origin_defaults_to_after_ramp_and_is_reported():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(sp.PipelineConfig())

    assert sp.PipelineConfig().time_origin == "after_ramp"
    assert diagnostics["time_origin"] == "after_ramp"
    assert diagnostics["reference_ramp_window"] == "after_ramp"


def test_time_theta_defaults_to_backward_euler_and_is_reported():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(sp.PipelineConfig())

    assert sp.PipelineConfig().time_theta == pytest.approx(1.0)
    assert diagnostics["time_theta"] == pytest.approx(1.0)


def test_time_theta_coefficients_support_crank_nicolson():
    sp = _load_pipeline_module()

    coeffs = sp._theta_step_coefficients(dt=2.0, theta=0.5)

    assert coeffs["mass_lhs"] == pytest.approx(0.5)
    assert coeffs["stiffness_lhs"] == pytest.approx(0.5)
    assert coeffs["mass_rhs"] == pytest.approx(0.5)
    assert coeffs["stiffness_rhs"] == pytest.approx(-0.5)


def test_model_consistency_rejects_theta_polarized_model_until_debye_theta_is_supported():
    sp = _load_pipeline_module()

    config = sp.PipelineConfig(polarization="cole-cole", time_theta=0.5)
    with pytest.raises(ValueError, match="time_theta"):
        sp.validate_model_consistency(config)


def test_late_time_diffusion_audit_flags_default_box_for_uniform_200_late_window():
    sp = _load_pipeline_module()

    audit = sp._diffusion_refinement_audit(sp.PipelineConfig(rho_earth=200.0, t_max=1.0e-3))

    assert audit["diffusion_length"] == pytest.approx(564.1895, rel=1.0e-4)
    assert audit["recommended_radius"] == pytest.approx(1128.379, rel=1.0e-4)
    assert audit["box_radius"] == pytest.approx(1000.0)
    assert audit["box_depth"] == pytest.approx(500.0)
    assert audit["underresolved"] is True


def test_late_time_diffusion_audit_accepts_explicit_diffusion_refinement_factor():
    sp = _load_pipeline_module()

    config = sp.PipelineConfig(rho_earth=200.0, t_max=1.0e-3, diffusion_refinement_factor=2.0)
    audit = sp._diffusion_refinement_audit(config)

    assert audit["box_radius"] == pytest.approx(audit["recommended_radius"])
    assert audit["box_depth"] == pytest.approx(audit["recommended_radius"])
    assert audit["underresolved"] is False


def test_late_time_diffusion_audit_reports_finite_domain_separately_from_refinement_box():
    sp = _load_pipeline_module()

    config = sp.PipelineConfig(
        rho_earth=100.0,
        t_max=1.0,
        x_extent=30000.0,
        y_extent=30000.0,
        earth_depth=30000.0,
        diffusion_refinement_factor=0.0,
    )
    audit = sp._diffusion_refinement_audit(config)

    assert audit["box_radius"] == pytest.approx(1000.0)
    assert audit["box_depth"] == pytest.approx(500.0)
    assert audit["underresolved"] is True
    assert audit["domain_horizontal_radius"] == pytest.approx(30000.0)
    assert audit["domain_depth"] == pytest.approx(30000.0)
    assert audit["domain_underresolved"] is False


def test_ramp_start_time_origin_reports_ramp_start_reference_window():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(sp.PipelineConfig(time_origin="ramp_start"))

    assert diagnostics["time_origin"] == "ramp_start"
    assert diagnostics["reference_ramp_window"] == "ramp_start"


def test_model_consistency_rejects_invalid_time_origin():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="time_origin"):
        sp.validate_model_consistency(sp.PipelineConfig(time_origin="bad"))


def test_after_ramp_observation_schedule_solves_through_ramp_then_returns_observation_times():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        ramp_off_time=1.0e-5,
        time_origin="after_ramp",
        time_growth=5.0,
        min_steps_during_turnoff=10,
    )
    observation_times = [1.0e-6, 2.0e-6, 1.0e-5]

    schedule = sp._forward_observation_schedule(observation_times, config)

    assert schedule["return_times"].tolist() == pytest.approx(observation_times)
    assert schedule["output_internal_times"].tolist() == pytest.approx([1.1e-5, 1.2e-5, 2.0e-5])
    assert schedule["step_times"][0] == pytest.approx(1.0e-6)
    assert np.any(np.isclose(schedule["step_times"], 1.0e-5))
    assert schedule["step_times"][-1] == pytest.approx(2.0e-5)
    assert schedule["output_step_indices"] == [10, 11, 12]


def test_after_ramp_observation_schedule_keeps_ramp_grid_when_observations_end_before_ramp():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        ramp_off_time=1.0e-5,
        time_origin="after_ramp",
        time_growth=2.0,
        min_steps_during_turnoff=10,
    )
    observation_times = [1.0e-6, 2.0e-6]

    schedule = sp._forward_observation_schedule(observation_times, config)

    assert schedule["step_times"][:10].tolist() == pytest.approx(np.linspace(1.0e-6, 1.0e-5, 10))
    assert schedule["step_times"][-2:].tolist() == pytest.approx([1.1e-5, 1.2e-5])
    assert schedule["output_step_indices"] == [10, 11]


def test_after_ramp_observation_schedule_uses_ramp_solver_t_min_before_later_observation_start():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        ramp_off_time=1.0e-5,
        time_origin="after_ramp",
        time_growth=2.0,
        ramp_solver_t_min=1.0e-6,
        min_steps_during_turnoff=10,
    )
    observation_times = [2.0e-6, 4.0e-6]

    schedule = sp._forward_observation_schedule(observation_times, config)

    assert schedule["step_times"][:10].tolist() == pytest.approx(np.linspace(1.0e-6, 1.0e-5, 10))
    assert schedule["step_times"][-2:].tolist() == pytest.approx([1.2e-5, 1.4e-5])
    assert schedule["output_step_indices"] == [10, 11]


def test_ramp_start_observation_schedule_is_identity():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(ramp_off_time=1.0e-5, time_origin="ramp_start")
    observation_times = [1.0e-6, 2.0e-6, 1.0e-5]

    schedule = sp._forward_observation_schedule(observation_times, config)

    assert schedule["step_times"].tolist() == pytest.approx(observation_times)
    assert schedule["output_internal_times"].tolist() == pytest.approx(observation_times)
    assert schedule["return_times"].tolist() == pytest.approx(observation_times)
    assert schedule["output_step_indices"] == [0, 1, 2]


def test_model_consistency_rejects_invalid_time_window():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="t_max"):
        sp.validate_model_consistency(sp.PipelineConfig(t_min=1.0e-3, t_max=1.0e-3))


def test_model_consistency_requires_cole_rho0_to_match_fem_earth_resistivity():
    sp = _load_pipeline_module()

    config = sp.PipelineConfig(polarization="cole-cole", rho_earth=50.0, cole_rho0=100.0)
    with pytest.raises(ValueError, match="cole_rho0"):
        sp.validate_model_consistency(config)


def test_layered_cole_cole_model_accepts_explicit_polarizable_depth_window():
    sp = _load_pipeline_module()

    config = sp.PipelineConfig(
        polarization="cole-cole",
        layer_depths=(350.0, 650.0),
        layer_resistivities=(100.0, 100.0, 100.0),
        cole_rho0=100.0,
        cole_layer_top=350.0,
        cole_layer_bottom=650.0,
        earth_depth=1000.0,
    )

    diagnostics = sp.validate_model_consistency(config)

    assert diagnostics["reference_mode"] == "cole-cole"
    assert diagnostics["cole_layer_top"] == pytest.approx(350.0)
    assert diagnostics["cole_layer_bottom"] == pytest.approx(650.0)


def test_layered_cole_cole_model_requires_valid_polarizable_depth_window():
    sp = _load_pipeline_module()

    config = sp.PipelineConfig(
        polarization="cole-cole",
        layer_depths=(350.0, 650.0),
        layer_resistivities=(100.0, 100.0, 100.0),
        cole_rho0=100.0,
        cole_layer_top=650.0,
        cole_layer_bottom=350.0,
        earth_depth=1000.0,
    )

    with pytest.raises(ValueError, match="cole_layer"):
        sp.validate_model_consistency(config)


def test_empymod_polarizable_layer_indices_select_middle_layer_only():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(cole_layer_top=350.0, cole_layer_bottom=650.0)

    indices = sp._empymod_polarizable_layer_indices(
        [0.0, 350.0, 650.0],
        [1.0e6, 100.0, 100.0, 100.0],
        config,
    )

    assert indices == [2]
