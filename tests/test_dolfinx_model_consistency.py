from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

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
        ("t_min", 0.0),
        ("time_growth", 1.0),
        ("time_theta", 0.49),
        ("source_mesh_size", 0.0),
        ("receiver_mesh_size", -1.0),
        ("receiver_anchor_mesh_size", -1.0),
        ("source_rhs_sign", 0.0),
        ("source_quadrature_points", -1),
        ("stop_after_outputs", -1),
        ("memory_limit_gb", -1.0),
        ("memory_safety_fraction", 0.0),
        ("min_steps_before_first_observation", 0),
    ],
)
def test_model_consistency_rejects_nonpositive_runtime_parameters(field, value):
    sp = _load_pipeline_module()
    kwargs = {field: value}

    with pytest.raises(ValueError, match=field):
        sp.validate_model_consistency(sp.PipelineConfig(**kwargs))


def test_model_consistency_accepts_ideal_step_off():
    sp = _load_pipeline_module()

    result = sp.validate_model_consistency(sp.PipelineConfig(ramp_off_time=0.0))

    assert result["ramp_off_time"] == 0.0


def test_model_consistency_rejects_negative_ramp_off_time():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError) as exc_info:
        sp.validate_model_consistency(sp.PipelineConfig(ramp_off_time=-1.0e-5))

    assert str(exc_info.value) == "ramp_off_time must be finite and nonnegative"


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_model_consistency_rejects_invalid_divergence_cleaning_strength(value):
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="divergence_cleaning_strength"):
        sp.validate_model_consistency(sp.PipelineConfig(divergence_cleaning_strength=value))


def test_model_consistency_reports_divergence_cleaning_strength():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(
        sp.PipelineConfig(divergence_cleaning="conductivity", divergence_cleaning_strength=0.25)
    )

    assert diagnostics["divergence_cleaning"] == "conductivity"
    assert diagnostics["divergence_cleaning_strength"] == pytest.approx(0.25)


def test_model_consistency_rejects_negative_divergence_cleaning_t_obs_min():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="divergence_cleaning_t_obs_min"):
        sp.validate_model_consistency(sp.PipelineConfig(divergence_cleaning_t_obs_min=-1.0e-3))


def test_model_consistency_reports_divergence_cleaning_t_obs_min():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(
        sp.PipelineConfig(divergence_cleaning="conductivity", divergence_cleaning_t_obs_min=0.05)
    )

    assert diagnostics["divergence_cleaning_t_obs_min"] == pytest.approx(0.05)


def test_should_apply_divergence_cleaning_respects_observation_time_gate():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(ramp_off_time=1.0e-5, divergence_cleaning_t_obs_min=0.05)

    assert sp._should_apply_divergence_cleaning(0.04999 + 1.0e-5, config) is False
    assert sp._should_apply_divergence_cleaning(0.05 + 1.0e-5, config) is True


def test_model_consistency_rejects_negative_divergence_control_weight():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="divergence_control_weight"):
        sp.validate_model_consistency(sp.PipelineConfig(divergence_control_weight=-1.0))


def test_model_consistency_reports_divergence_control_settings():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(
        sp.PipelineConfig(
            divergence_control_weight=1.0e-8,
            divergence_control_t_obs_min=0.02,
            divergence_control_scale="lhs",
        )
    )

    assert diagnostics["divergence_control_weight"] == pytest.approx(1.0e-8)
    assert diagnostics["divergence_control_t_obs_min"] == pytest.approx(0.02)
    assert diagnostics["divergence_control_scale"] == "lhs"


def test_model_consistency_rejects_unknown_divergence_control_scale():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="divergence_control_scale"):
        sp.validate_model_consistency(sp.PipelineConfig(divergence_control_scale="bad-scale"))


def test_divergence_control_lhs_scale_uses_left_hand_operator_norm():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(divergence_control_weight=0.25, divergence_control_scale="lhs")

    stats = sp._divergence_control_step_stats(
        config,
        dt=2.0,
        lhs_mass=3.0,
        lhs_stiffness=4.0,
        mass_norm=5.0,
        stiffness_norm=7.0,
        control_norm=10.0,
    )

    assert stats["divergence_control_scale"] == "lhs"
    assert stats["divergence_control_weight"] == pytest.approx(0.25)
    assert stats["divergence_control_reference_norm"] == pytest.approx(43.0)
    assert stats["divergence_control_matrix_norm"] == pytest.approx(10.0)
    assert stats["divergence_control_applied_weight"] == pytest.approx(1.075)
    assert stats["divergence_control_relative_weight"] == pytest.approx(0.25)


def test_should_apply_divergence_control_respects_weight_and_observation_time_gate():
    sp = _load_pipeline_module()
    off = sp.PipelineConfig(ramp_off_time=1.0e-5, divergence_control_weight=0.0, divergence_control_t_obs_min=0.02)
    gated = sp.PipelineConfig(ramp_off_time=1.0e-5, divergence_control_weight=1.0e-8, divergence_control_t_obs_min=0.02)

    assert sp._should_apply_divergence_control(0.03, off) is False
    assert sp._should_apply_divergence_control(0.01999 + 1.0e-5, gated) is False
    assert sp._should_apply_divergence_control(0.02 + 1.0e-5, gated) is True


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


def test_manual_line_integration_points_report_line_orientation_diagnostics():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        source_start=(-500.0, 200.0, -0.1),
        source_end=(500.0, 200.0, -0.1),
        source_mesh_size=100.0,
    )

    _points, _weights, _svals, diagnostics = sp._manual_line_source_integration_points(config)

    orientation = diagnostics["line_orientation"]
    assert orientation["source_length_m"] == pytest.approx(1000.0)
    assert orientation["expected_displacement_m"] == pytest.approx([1000.0, 0.0, 0.0])
    assert orientation["integrated_displacement_m"] == pytest.approx([1000.0, 0.0, 0.0])
    assert orientation["quadrature_weight_sum_m"] == pytest.approx(1000.0)
    assert orientation["signed_parallel_projection_m"] == pytest.approx(1000.0)
    assert orientation["relative_parallel_length_error"] < 1.0e-12
    assert orientation["transverse_residual_m"] < 1.0e-12
    assert orientation["orientation_cosine"] == pytest.approx(1.0)
    assert orientation["s_parameter_min"] >= 0.0
    assert orientation["s_parameter_max"] <= 1.0
    assert orientation["s_parameter_monotonic"] is True
    assert orientation["reversed_orientation"] is False


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


def test_qwe_reference_identity_remains_available_for_function_level_diagnostics():
    sp = _load_pipeline_module()

    config = sp.PipelineConfig(
        empymod_srcpts=33,
        empymod_ht="qwe",
        empymod_ht_qwe_rtol=2.0e-11,
        empymod_ft="qwe",
        empymod_ft_pts_per_dec=13,
        empymod_ft_qwe_rtol=3.0e-9,
    )
    identity = sp._empymod_reference_identity(config)
    kwargs = sp._empymod_call_kwargs(config)

    assert identity["hankel_transform"]["parameters"]["rtol"] == 2.0e-11
    assert identity["fourier_transform"]["parameters"]["rtol"] == 3.0e-9
    assert kwargs["htarg"] == identity["hankel_transform"]["parameters"]
    assert kwargs["ftarg"] == identity["fourier_transform"]["parameters"]


def test_model_consistency_reports_resolved_quasistatic_reference_identity():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(sp.PipelineConfig())

    assert diagnostics["empymod_reference_identity"] == sp._empymod_reference_identity(
        sp.PipelineConfig()
    )
    assert diagnostics["empymod_reference_identity"]["equation"] == "quasistatic"
    assert diagnostics["empymod_reference_identity"]["fourier_transform"] == {
        "method": "dlf",
        "parameters": {
            "filter": "key_201_2012",
            "pts_per_dec": 0,
        },
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"empymod_ht": "qwe"},
        {"empymod_ht": "quad"},
        {"empymod_ft": "qwe", "empymod_ft_pts_per_dec": 20},
        {"empymod_ht_filter": "key_101_2009"},
        {"empymod_ft_filter": "key_101_2012"},
        {"empymod_ht_pts_per_dec": 1},
        {"empymod_ft_pts_per_dec": -1},
        {"empymod_eperm_h": 1.0},
        {"empymod_eperm_v": 1.0},
        {"empymod_mperm_h": 2.0},
        {"empymod_mperm_v": 2.0},
    ],
)
def test_formal_model_validation_rejects_diagnostic_only_reference_identity(changes):
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="diagnostic-only|approved"):
        sp.validate_model_consistency(sp.PipelineConfig(**changes))


def test_formal_model_validation_accepts_exact_approved_reference_identity():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(sp.PipelineConfig())

    assert diagnostics["empymod_reference_identity"] == sp._approved_empymod_reference_identity()


def test_acceptance_artifact_writer_rejects_diagnostic_identity_before_writing(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path / "acceptance", empymod_ht="qwe")
    times = np.asarray([1.0e-4])
    data = np.asarray([[1.0, 2.0, 3.0]])

    with pytest.raises(ValueError, match="diagnostic-only|approved"):
        sp.write_validation_artifacts(
            times,
            data,
            data.copy(),
            ("Ex", "Ey", "dBzdt"),
            config,
            case_type="noip",
            reference_type="empymod",
        )

    assert not config.workdir.exists()


def test_resolved_config_yaml_records_empymod_reference_identity_fields():
    sp = _load_pipeline_module()

    resolved = sp._resolved_config_yaml(sp.PipelineConfig())

    for line in (
        "empymod_equation: quasistatic\n",
        "empymod_eperm_h: 0\n",
        "empymod_eperm_v: 0\n",
        "empymod_mperm_h: 1\n",
        "empymod_mperm_v: 1\n",
        "empymod_ht: dlf\n",
        "empymod_ht_filter: key_201_2009\n",
        "empymod_ht_pts_per_dec: 0\n",
        "empymod_ft: dlf\n",
        "empymod_ft_filter: key_201_2012\n",
        "empymod_ft_pts_per_dec: 0\n",
    ):
        assert line in resolved


@pytest.mark.parametrize(
    "kwargs",
    [
        {"empymod_srcpts": 0},
        {"empymod_equation": "full-wave"},
        {"empymod_eperm_h": 1.0},
        {"empymod_eperm_v": 1.0},
        {"empymod_mperm_h": 2.0},
        {"empymod_mperm_v": 2.0},
        {"empymod_ht": "bad"},
        {"empymod_ht_filter": ""},
        {"empymod_ht_pts_per_dec": 0.5},
        {"empymod_ht_pts_per_dec": True},
        {"empymod_ft": "bad"},
        {"empymod_ft_filter": ""},
        {"empymod_ft_pts_per_dec": 0.5},
        {"empymod_ft_pts_per_dec": False},
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


def test_explicit_observation_times_define_effective_diffusion_window():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        rho_earth=100.0,
        t_max=1.0,
        observation_times=(1.0e-5, 1.0e-3, 10.0),
        diffusion_refinement_factor=2.0,
    )

    expected_length = sp._max_earth_diffusion_length(config, time=10.0)
    raw_t_max_length = sp._max_earth_diffusion_length(config, time=1.0)
    audit = sp._diffusion_refinement_audit(config)
    diagnostics = sp.validate_model_consistency(config)

    assert sp._effective_t_max(config) == pytest.approx(10.0)
    assert audit["diffusion_length"] == pytest.approx(expected_length)
    assert audit["diffusion_length"] != pytest.approx(raw_t_max_length)
    assert audit["box_radius"] == pytest.approx(2.0 * expected_length)
    assert audit["box_depth"] == pytest.approx(2.0 * expected_length)
    assert diagnostics["configured_t_max"] == pytest.approx(1.0)
    assert diagnostics["effective_t_max"] == pytest.approx(10.0)
    assert diagnostics["t_max"] == pytest.approx(10.0)


def test_h_static_initial_dt_ignores_explicit_observation_window():
    sp = _load_pipeline_module()
    short_window = sp.PipelineConfig(
        t_max=2.0,
        ramp_off_time=3.0,
        observation_times=(1.0e-5, 10.0),
    )
    long_window = sp.PipelineConfig(
        t_max=2.0,
        ramp_off_time=3.0,
        observation_times=(1.0e-5, 100.0),
    )
    expected = max(float(short_window.t_max), float(short_window.ramp_off_time), 1.0) * 1.0e9

    assert sp._h_static_initial_dt(short_window) == pytest.approx(expected)
    assert sp._h_static_initial_dt(long_window) == pytest.approx(expected)


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


def test_explicit_observation_times_override_growth_grid():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(observation_times=(1.0e-5, 1.0e-4, 1.0e-3))

    assert sp.generate_time_array(config).tolist() == pytest.approx(config.observation_times)


@pytest.mark.parametrize("observation_times", [(), [], np.asarray([], dtype=float)])
def test_empty_one_dimensional_observation_times_use_legacy_growth_grid(observation_times):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        t_min=1.0e-5,
        t_max=1.0e-3,
        time_growth=10.0,
        observation_times=observation_times,
    )

    assert sp.generate_time_array(config).tolist() == pytest.approx([1.0e-5, 1.0e-4, 1.0e-3])


def test_explicit_observation_times_return_independent_array_copy():
    sp = _load_pipeline_module()
    supplied = np.asarray([1.0e-5, 1.0e-4, 1.0e-3], dtype=float)

    generated = sp.generate_time_array(sp.PipelineConfig(observation_times=supplied))

    assert not np.shares_memory(generated, supplied)
    supplied[0] = 2.0e-5
    assert generated[0] == pytest.approx(1.0e-5)
    generated[1] = 2.0e-4
    assert supplied[1] == pytest.approx(1.0e-4)


@pytest.mark.parametrize(
    "observation_times",
    [
        (1.0e-5, float("nan"), 1.0e-3),
        (1.0e-5, float("inf"), 1.0e-3),
        (1.0e-5, float("-inf"), 1.0e-3),
        (0.0, 1.0e-4, 1.0e-3),
        (-1.0e-5, 1.0e-4, 1.0e-3),
        (1.0e-5, 1.0e-5, 1.0e-3),
        (1.0e-4, 1.0e-5, 1.0e-3),
    ],
)
def test_explicit_observation_times_reject_invalid_values(observation_times):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(observation_times=observation_times)

    with pytest.raises(ValueError) as exc_info:
        sp.generate_time_array(config)

    assert str(exc_info.value) == "observation_times must be finite, positive, and strictly increasing"


@pytest.mark.parametrize(
    "observation_times",
    [
        np.asarray(1.0e-5),
        np.empty((0, 1)),
    ],
)
def test_explicit_observation_times_reject_non_one_dimensional_inputs(observation_times):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(observation_times=observation_times)

    with pytest.raises(ValueError) as exc_info:
        sp.generate_time_array(config)

    assert str(exc_info.value) == "observation_times must be finite, positive, and strictly increasing"


def test_model_consistency_rejects_invalid_explicit_observation_times():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError) as exc_info:
        sp.validate_model_consistency(sp.PipelineConfig(observation_times=(1.0e-5, float("nan"))))

    assert str(exc_info.value) == "observation_times must be finite, positive, and strictly increasing"


def test_ideal_step_off_schedule_has_no_synthetic_ramp_steps():
    sp = _load_pipeline_module()
    times = np.array([1.0e-5, 1.0e-4, 1.0e-3])

    schedule = sp._forward_observation_schedule(
        times, sp.PipelineConfig(ramp_off_time=0.0, time_origin="after_ramp")
    )

    assert schedule["step_times"].tolist() == pytest.approx(times)
    assert schedule["output_step_indices"] == [0, 1, 2]


def test_ideal_step_off_schedule_subdivides_every_output_interval():
    sp = _load_pipeline_module()
    times = np.array([1.0e-5, 1.0e-4, 1.0e-3])

    schedule = sp._forward_observation_schedule(
        times,
        sp.PipelineConfig(
            ramp_off_time=0.0,
            time_origin="after_ramp",
            output_interval_substeps=4,
        ),
    )

    assert schedule["step_times"].tolist() == pytest.approx(
        [
            2.5e-6,
            5.0e-6,
            7.5e-6,
            1.0e-5,
            3.25e-5,
            5.5e-5,
            7.75e-5,
            1.0e-4,
            3.25e-4,
            5.5e-4,
            7.75e-4,
            1.0e-3,
        ]
    )
    assert schedule["output_internal_times"].tolist() == pytest.approx(times)
    assert schedule["return_times"].tolist() == pytest.approx(times)
    assert schedule["output_step_indices"] == [3, 7, 11]


@pytest.mark.parametrize("invalid_substeps", [True, 1.9, "2"])
def test_model_consistency_rejects_non_integer_output_interval_substeps(invalid_substeps):
    sp = _load_pipeline_module()

    with pytest.raises(
        ValueError,
        match="output_interval_substeps must be a non-bool positive integer",
    ):
        sp.validate_model_consistency(
            sp.PipelineConfig(output_interval_substeps=invalid_substeps)
        )


@pytest.mark.parametrize("invalid_substeps", [True, 1.9, "2", 0, -1])
def test_forward_schedule_rejects_invalid_output_interval_substeps(invalid_substeps):
    sp = _load_pipeline_module()

    with pytest.raises(
        ValueError,
        match="output_interval_substeps must be a non-bool positive integer",
    ):
        sp._forward_observation_schedule(
            [1.0e-5, 1.0e-4],
            sp.PipelineConfig(output_interval_substeps=invalid_substeps),
        )


def test_explicit_observation_times_round_trip_through_cli_and_resolved_yaml(monkeypatch, tmp_path):
    sp = _load_pipeline_module()
    captured = {}
    validate_model_consistency = sp.validate_model_consistency

    def capture_config(config):
        captured["config"] = config
        return validate_model_consistency(config)

    monkeypatch.setattr(sp, "validate_model_consistency", capture_config)
    monkeypatch.setattr(sp, "check_environment", lambda **_kwargs: {})

    result = sp.main(
        [
            "--workdir",
            str(tmp_path),
            "--observation-times",
            "1e-5,1e-4,1e-3",
            "--output-interval-substeps",
            "4",
            "--check-env-only",
            "--no-install",
        ]
    )

    assert result == 0
    assert captured["config"].observation_times == pytest.approx((1.0e-5, 1.0e-4, 1.0e-3))
    assert captured["config"].output_interval_substeps == 4
    resolved = sp._resolved_config_yaml(captured["config"])
    assert "observation_times: [1e-05, 0.0001, 0.001]\n" in resolved
    assert "configured_t_max: 1\n" in resolved
    assert "effective_t_max: 0.001\n" in resolved
    assert "t_max: 0.001\n" in resolved


def test_invalid_explicit_observation_times_fail_before_check_env_only(monkeypatch, tmp_path):
    sp = _load_pipeline_module()
    calls = {"environment": 0}

    def check_environment(**_kwargs):
        calls["environment"] += 1
        return {}

    monkeypatch.setattr(sp, "check_environment", check_environment)

    with pytest.raises(SystemExit) as exc_info:
        sp.main(
            [
                "--workdir",
                str(tmp_path),
                "--observation-times",
                "1e-5,nan",
                "--check-env-only",
                "--no-install",
            ]
        )

    assert str(exc_info.value) == "[model] observation_times must be finite, positive, and strictly increasing"
    assert calls["environment"] == 0


def test_invalid_explicit_observation_times_cannot_reach_mesh_only(monkeypatch, tmp_path):
    sp = _load_pipeline_module()
    calls = {"environment": 0, "mesh": 0}

    def check_environment(**_kwargs):
        calls["environment"] += 1
        return {}

    def generate_verification_mesh(_config):
        calls["mesh"] += 1
        return tmp_path / "unused.msh"

    monkeypatch.setattr(sp, "check_environment", check_environment)
    monkeypatch.setattr(sp, "generate_verification_mesh", generate_verification_mesh)
    monkeypatch.setitem(
        sys.modules,
        "mpi4py",
        SimpleNamespace(MPI=SimpleNamespace(COMM_WORLD=SimpleNamespace(size=1))),
    )

    with pytest.raises(SystemExit) as exc_info:
        sp.main(
            [
                "--workdir",
                str(tmp_path),
                "--observation-times",
                "1e-5,nan",
                "--mesh-only",
                "--no-install",
            ]
        )

    assert str(exc_info.value) == "[model] observation_times must be finite, positive, and strictly increasing"
    assert calls == {"environment": 0, "mesh": 0}


def test_malformed_observation_times_csv_is_reported_by_argparse(capsys):
    sp = _load_pipeline_module()

    with pytest.raises(SystemExit) as exc_info:
        sp.main(["--observation-times", "1e-5,,1e-3"])

    assert exc_info.value.code == 2
    assert "argument --observation-times" in capsys.readouterr().err


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


def test_after_ramp_observation_schedule_substeps_before_first_observation():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        ramp_off_time=1.0e-5,
        time_origin="after_ramp",
        ramp_solver_t_min=1.0e-6,
        min_steps_during_turnoff=10,
        min_steps_before_first_observation=4,
    )
    observation_times = [1.0e-5, 1.25e-5]

    schedule = sp._forward_observation_schedule(observation_times, config)

    assert np.any(np.isclose(schedule["step_times"], 1.25e-5))
    assert np.any(np.isclose(schedule["step_times"], 1.50e-5))
    assert np.any(np.isclose(schedule["step_times"], 1.75e-5))
    assert schedule["output_internal_times"].tolist() == pytest.approx([2.0e-5, 2.25e-5])
    assert schedule["return_times"].tolist() == pytest.approx(observation_times)
    assert schedule["output_step_indices"] == [13, 14]


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

    assert diagnostics["reference_mode"] == "cole-cole-exact"
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


@pytest.mark.parametrize(
    "builder_name",
    ["_exact_cole_cole_empymod_material", "_debye_cole_cole_empymod_material"],
)
def test_cole_cole_empymod_callbacks_flatten_standard_dlf_frequency_grid(
    builder_name, monkeypatch
):
    sp = _load_pipeline_module()
    if builder_name == "_debye_cole_cole_empymod_material":
        monkeypatch.setattr(
            sp,
            "fit_cole_cole_to_debye",
            lambda _config: SimpleNamespace(
                sigma_infinity=0.02,
                terms=(SimpleNamespace(delta_sigma=0.01, tau=0.1),),
            ),
        )
    _depth, material = getattr(sp, builder_name)(sp.PipelineConfig())

    eta_h, eta_v = material["func_eta"](
        material,
        {"freq": np.asarray([[1.0, 2.0], [3.0, 4.0]])},
    )

    assert eta_h.shape == (4, 2)
    assert eta_v.shape == (4, 2)
    assert np.isfinite(eta_h).all()
    np.testing.assert_allclose(eta_h, eta_v)


@pytest.mark.parametrize("value", [0.0, -0.01, float("nan"), float("inf"), -float("inf")])
def test_model_consistency_rejects_invalid_cole_fit_tolerance(value):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(cole_fit_tolerance=value)

    with pytest.raises(ValueError, match="cole_fit_tolerance"):
        sp.validate_model_consistency(config)


def test_cole_fit_tolerance_round_trips_through_cli_and_resolved_yaml(monkeypatch, tmp_path):
    sp = _load_pipeline_module()
    captured = {}
    validate_model_consistency = sp.validate_model_consistency

    def capture_config(config):
        captured["config"] = config
        return validate_model_consistency(config)

    monkeypatch.setattr(sp, "validate_model_consistency", capture_config)
    monkeypatch.setattr(sp, "check_environment", lambda **_kwargs: {})

    result = sp.main(
        [
            "--workdir",
            str(tmp_path),
            "--cole-fit-tolerance",
            "0.0075",
            "--check-env-only",
            "--no-install",
        ]
    )

    assert result == 0
    assert captured["config"].cole_fit_tolerance == pytest.approx(0.0075)
    resolved = sp._resolved_config_yaml(captured["config"])
    assert "cole_fit_tolerance: 0.0075\n" in resolved


def test_postprocess_cli_selects_exact_cole_cole_reference(monkeypatch, tmp_path):
    sp = _load_pipeline_module()
    captured = {}

    monkeypatch.setattr(sp, "check_environment", lambda **_kwargs: {})

    def postprocess_saved_forward(config, env, *, ref_mode, runtime):
        captured["config"] = config
        captured["ref_mode"] = ref_mode
        return {}

    monkeypatch.setattr(sp, "postprocess_saved_forward", postprocess_saved_forward)

    result = sp.main(
        [
            "--workdir",
            str(tmp_path),
            "--polarization",
            "cole-cole",
            "--postprocess-partial",
            "--no-install",
        ]
    )

    assert result == 0
    assert captured["config"].polarization == "cole-cole"
    assert captured["ref_mode"] == "cole-cole-exact"


@pytest.mark.parametrize("ref_mode", ["noip", "cole-cole-exact", "cole-cole-debye"])
def test_postprocess_records_actual_empymod_reference_mode(monkeypatch, tmp_path, ref_mode):
    sp = _load_pipeline_module()
    times = np.array([1.0e-4, 1.0e-3])
    components = ("Ex", "Ey", "dBzdt")
    data = np.ones((times.size, len(components)))
    fem_result = {"times": times, "data": data, "components": components, "solver_log": []}
    captured = {}

    monkeypatch.setattr(sp, "_load_forward_partial", lambda _config: fem_result)
    monkeypatch.setattr(
        sp,
        "get_empymod_reference",
        lambda requested_times, _config, *, mode, **_kwargs: {
            "times": requested_times,
            "data": data.copy(),
            "components": components,
            "reference_mode": mode,
        },
    )
    monkeypatch.setattr(sp, "compute_error", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(sp, "_save_npz", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sp, "plot_verification", lambda *_args, **_kwargs: None)

    def write_validation_artifacts(*_args, reference_mode=None, **_kwargs):
        captured["artifact_reference_mode"] = reference_mode
        return {}

    def write_report(_config, _env, _fem_result, ref_result, *_args, **_kwargs):
        captured["report_reference_mode"] = ref_result.get("reference_mode")

    monkeypatch.setattr(sp, "write_validation_artifacts", write_validation_artifacts)
    monkeypatch.setattr(sp, "write_report", write_report)
    config = sp.PipelineConfig(workdir=tmp_path, polarization="cole-cole" if ref_mode != "noip" else "none")

    sp.postprocess_saved_forward(config, {}, ref_mode=ref_mode)

    assert captured["artifact_reference_mode"] == ref_mode
    assert captured["report_reference_mode"] == ref_mode


def test_report_uses_reference_mode_from_reference_result(tmp_path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(workdir=tmp_path, polarization="cole-cole")
    times = np.array([1.0e-4, 1.0e-3])
    components = ("Ex", "Ey", "dBzdt")
    data = np.array([[1.0, 1.0e-4, 1.0e-6], [0.5, 5.0e-5, 5.0e-7]])
    fem_result = {
        "times": times,
        "data": data,
        "components": components,
        "solver_log": [],
    }
    ref_result = {
        "times": times,
        "data": data.copy(),
        "components": components,
        "reference_mode": "cole-cole-debye",
    }
    errors = sp.compute_error(data, data, components)

    sp.write_report(config, {}, fem_result, ref_result, errors, {"mode": "test"})

    report = config.output_report().read_text(encoding="utf-8")
    assert "reference_mode=cole-cole-debye" in report
    assert "reference_mode=cole-cole-exact" not in report


def test_validation_artifacts_record_actual_reference_mode(tmp_path):
    sp = _load_pipeline_module()
    times = np.array([1.0e-4, 1.0e-3])
    components = ("Ex", "Ey", "dBzdt")
    data = np.array([[1.0, 1.0e-4, 1.0e-6], [0.5, 5.0e-5, 5.0e-7]])
    config = sp.PipelineConfig(workdir=tmp_path)

    summary = sp.write_validation_artifacts(
        times,
        data,
        data.copy(),
        components,
        config,
        case_type="ip",
        reference_type="empymod",
        reference_mode="cole-cole-debye",
    )

    assert summary["reference_mode"] == "cole-cole-debye"
