from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location("sotem_pipeline_for_formulation_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_formulation_defaults_to_e_and_accepts_h():
    sp = _load_pipeline_module()

    assert sp.PipelineConfig().formulation == "e"
    assert sp.PipelineConfig(formulation="h").formulation == "h"


def test_formulation_validation_rejects_unknown_value():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="formulation"):
        sp.validate_formulation(sp.PipelineConfig(formulation="bad"))


def test_h_after_ramp_schedule_refines_before_first_observation_for_dbdt():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        formulation="h",
        time_origin="after_ramp",
        ramp_off_time=1.0e-5,
        t_min=1.0e-5,
        t_max=2.0e-4,
        time_growth=1.05,
        explicit_observation_times=(1.0e-5, 2.0e-4),
        time_method="bdf2",
        max_internal_dt=2.0e-6,
        min_steps_during_turnoff=10,
        min_steps_before_first_observation=1,
    )

    schedule = sp._forward_observation_schedule(sp.generate_time_array(config), config)
    first_output = int(schedule["output_step_indices"][0])
    step_times = schedule["step_times"]

    assert first_output >= 59
    assert step_times[first_output] == pytest.approx(2.0e-5)
    assert step_times[first_output] - step_times[first_output - 1] <= 2.0e-7 + 1.0e-15


def test_source_term_mode_defaults_to_impressed_current_and_accepts_primary_dc():
    sp = _load_pipeline_module()

    assert sp.PipelineConfig().source_term_mode == "impressed_current"
    diagnostics = sp.validate_model_consistency(sp.PipelineConfig(source_term_mode="primary_dc"))
    assert diagnostics["source_term_mode"] == "primary_dc"


def test_source_term_mode_validation_rejects_unknown_value():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="source_term_mode"):
        sp.validate_model_consistency(sp.PipelineConfig(source_term_mode="bad"))


def test_primary_secondary_current_correction_accepts_conductivity_mode():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(
        sp.PipelineConfig(
            source_term_mode="primary_secondary",
            primary_secondary_current_correction="conductivity",
        )
    )

    assert diagnostics["primary_secondary_current_correction"] == "conductivity"


def test_primary_secondary_current_correction_relaxation_accepts_fraction():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(
        sp.PipelineConfig(
            source_term_mode="primary_secondary",
            primary_secondary_current_correction="conductivity",
            primary_secondary_current_correction_relaxation=0.95,
        )
    )

    assert diagnostics["primary_secondary_current_correction_relaxation"] == pytest.approx(0.95)


def test_primary_secondary_current_correction_relaxation_schedule_interpolates_in_log_time():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        source_term_mode="primary_secondary",
        primary_secondary_current_correction="conductivity",
        primary_secondary_current_correction_relaxation=0.955,
        primary_secondary_current_correction_relaxation_late=0.965,
        primary_secondary_current_correction_relaxation_transition_time=0.1,
        primary_secondary_current_correction_relaxation_transition_width=1.0,
    )

    assert sp._primary_secondary_current_correction_relaxation_at_time(config, 0.01) == pytest.approx(0.955)
    assert sp._primary_secondary_current_correction_relaxation_at_time(config, 0.1) == pytest.approx(0.96)
    assert sp._primary_secondary_current_correction_relaxation_at_time(config, 1.0) == pytest.approx(0.965)


def test_primary_secondary_current_correction_relaxation_schedule_defaults_to_constant():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        source_term_mode="primary_secondary",
        primary_secondary_current_correction="conductivity",
        primary_secondary_current_correction_relaxation=0.955,
    )

    assert sp._primary_secondary_current_correction_relaxation_at_time(config, 0.01) == pytest.approx(0.955)
    assert sp._primary_secondary_current_correction_relaxation_at_time(config, 1.0) == pytest.approx(0.955)


def test_primary_secondary_current_correction_relaxation_rejects_invalid_values():
    sp = _load_pipeline_module()

    for value in (-0.1, float("inf")):
        with pytest.raises(ValueError, match="primary_secondary_current_correction_relaxation"):
            sp.validate_model_consistency(
                sp.PipelineConfig(
                    source_term_mode="primary_secondary",
                    primary_secondary_current_correction="conductivity",
                    primary_secondary_current_correction_relaxation=value,
                )
            )


def test_primary_secondary_current_correction_rejects_unknown_mode():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="primary_secondary_current_correction"):
        sp.validate_model_consistency(
            sp.PipelineConfig(
                source_term_mode="primary_secondary",
                primary_secondary_current_correction="bad",
            )
        )


def test_primary_secondary_rhs_scale_accepts_positive_values():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(
        sp.PipelineConfig(
            source_term_mode="primary_secondary",
            primary_secondary_rhs_scale=2.5,
        )
    )

    assert diagnostics["primary_secondary_rhs_scale"] == pytest.approx(2.5)


def test_primary_secondary_rhs_scale_rejects_nonpositive_values():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="primary_secondary_rhs_scale"):
        sp.validate_model_consistency(
            sp.PipelineConfig(
                source_term_mode="primary_secondary",
                primary_secondary_rhs_scale=0.0,
            )
        )


def test_primary_secondary_dc_rhs_sign_defaults_to_positive():
    sp = _load_pipeline_module()

    config = sp.PipelineConfig(source_term_mode="primary_secondary")
    diagnostics = sp.validate_model_consistency(config)

    assert config.primary_secondary_dc_rhs_sign == 1.0
    assert diagnostics["primary_secondary_dc_rhs_sign"] == pytest.approx(1.0)


def test_primary_secondary_dc_rhs_sign_accepts_negative_diagnostic():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(
        sp.PipelineConfig(
            source_term_mode="primary_secondary",
            primary_secondary_dc_rhs_sign=-1.0,
        )
    )

    assert diagnostics["primary_secondary_dc_rhs_sign"] == pytest.approx(-1.0)


def test_primary_secondary_dc_rhs_sign_rejects_invalid_values():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="primary_secondary_dc_rhs_sign"):
        sp.validate_model_consistency(
            sp.PipelineConfig(
                source_term_mode="primary_secondary",
                primary_secondary_dc_rhs_sign=0.0,
            )
        )


def test_primary_secondary_rhs_mass_mode_accepts_effective_mode():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(
        sp.PipelineConfig(
            source_term_mode="primary_secondary",
            primary_secondary_rhs_mass_mode="effective",
        )
    )

    assert diagnostics["source_term_mode"] == "primary_secondary"


def test_primary_secondary_rhs_mass_mode_defaults_to_unit_current_density_mass():
    sp = _load_pipeline_module()

    config = sp.PipelineConfig(source_term_mode="primary_secondary")
    diagnostics = sp.validate_model_consistency(config)

    assert config.primary_secondary_rhs_mass_mode == "unit"
    assert diagnostics["primary_secondary_rhs_mass_mode"] == "unit"


def test_primary_secondary_rhs_projection_accepts_conductivity_mode():
    sp = _load_pipeline_module()

    diagnostics = sp.validate_model_consistency(
        sp.PipelineConfig(
            source_term_mode="primary_secondary",
            primary_secondary_rhs_projection="conductivity",
        )
    )

    assert diagnostics["primary_secondary_rhs_projection"] == "conductivity"


def test_primary_secondary_rhs_projection_rejects_unknown_mode():
    sp = _load_pipeline_module()

    with pytest.raises(ValueError, match="primary_secondary_rhs_projection"):
        sp.validate_model_consistency(
            sp.PipelineConfig(
                source_term_mode="primary_secondary",
                primary_secondary_rhs_projection="bad",
            )
        )


def test_h_operator_assembly_accepts_pipeline_config_for_boundary_conditions():
    sp = _load_pipeline_module()

    signature = inspect.signature(sp.assemble_h_operators)

    assert "config" in signature.parameters


def test_h_source_inputs_prefers_preassembled_h_vector():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(formulation="h")
    sentinel_current_density = object()
    sentinel_h_vector = object()
    calls = []

    def build_regularized(*_args, **_kwargs):
        calls.append("regularized")
        return object()

    def assemble_regularized(*_args, **_kwargs):
        calls.append("assemble")
        return object()

    sp._build_regularized_current_density = build_regularized
    sp.assemble_h_source_vector = assemble_regularized

    current_density, source_vec, diagnostics = sp._h_source_inputs(
        msh=object(),
        spaces={},
        materials={},
        source={"current_density": sentinel_current_density, "h_vector": sentinel_h_vector},
        config=config,
        cell_tags=object(),
    )

    assert current_density is sentinel_current_density
    assert source_vec is sentinel_h_vector
    assert diagnostics["h_source_mode"] == "preassembled_line"
    assert calls == []


def test_physical_curl_from_reference_tabulation_uses_covariant_piola_curl_map():
    sp = _load_pipeline_module()
    tab = np.zeros((4, 1, 1, 3), dtype=float)
    # Reference curl = (dFz/dy - dFy/dz, dFx/dz - dFz/dx, dFy/dx - dFx/dy)
    #                = (1, 2, 3)
    tab[2, 0, 0, 2] = 4.0
    tab[3, 0, 0, 1] = 3.0
    tab[3, 0, 0, 0] = 7.0
    tab[1, 0, 0, 2] = 5.0
    tab[1, 0, 0, 1] = 11.0
    tab[2, 0, 0, 0] = 8.0
    J = np.diag([2.0, 3.0, 5.0])

    curl = sp._physical_curl_from_reference_tabulation(tab, J)

    np.testing.assert_allclose(curl, np.asarray([[2.0 / 30.0, 6.0 / 30.0, 15.0 / 30.0]]))


def test_regularized_h_source_cell_weight_samples_inside_cell_not_only_center():
    sp = _load_pipeline_module()
    p0 = np.asarray([-10.0, 0.0, -0.1])
    p1 = np.asarray([10.0, 0.0, -0.1])
    tangent = (p1 - p0) / np.linalg.norm(p1 - p0)
    vertices = np.asarray(
        [
            [-1.0, -4.0, -0.2],
            [1.0, -4.0, -0.2],
            [0.0, 0.0, -0.1],
            [0.0, -4.0, -4.0],
        ],
        dtype=float,
    )
    center = np.mean(vertices, axis=0)
    center_weight = sp._regularized_h_source_point_weight(center, p0, tangent, 20.0, 0.5)

    averaged_weight = sp._regularized_h_source_cell_weight(vertices, p0, tangent, 20.0, 0.5)

    assert center_weight < 1.0e-6
    assert averaged_weight > 1.0e-3
    assert averaged_weight > 1000.0 * center_weight


def test_h_source_inputs_builds_line_vector_for_manual_line_source(monkeypatch):
    sp = _load_pipeline_module()
    sentinel_current_density = object()
    sentinel_h_vector = object()
    calls = []

    def build_regularized(*_args, **_kwargs):
        calls.append("regularized")
        return sentinel_current_density

    def build_h_line(*_args, **_kwargs):
        calls.append("line")
        return sentinel_h_vector

    monkeypatch.setattr(sp, "_build_regularized_current_density", build_regularized)
    monkeypatch.setattr(sp, "_build_manual_line_h_source_vector", build_h_line)
    monkeypatch.setattr(sp, "assemble_h_source_vector", lambda *_args, **_kwargs: object())

    current_density, source_vec, diagnostics = sp._h_source_inputs(
        msh=object(),
        spaces={},
        materials={},
        source={"mode": "manual_line"},
        config=sp.PipelineConfig(formulation="h", source_mode="manual_line", h_source_mode="manual_line"),
        cell_tags=object(),
    )

    assert current_density is sentinel_current_density
    assert source_vec is sentinel_h_vector
    assert diagnostics["h_source_mode"] == "manual_line"
    assert calls == ["regularized", "line"]


def test_h_source_inputs_defaults_to_regularized_even_for_manual_line_source(monkeypatch):
    sp = _load_pipeline_module()
    sentinel_current_density = object()
    sentinel_regularized_vector = object()
    calls = []

    def build_regularized(*_args, **_kwargs):
        calls.append("regularized")
        return sentinel_current_density

    def assemble_regularized(*_args, **_kwargs):
        calls.append("assemble")
        return sentinel_regularized_vector

    monkeypatch.setattr(sp, "_build_regularized_current_density", build_regularized)
    monkeypatch.setattr(sp, "_build_manual_line_h_source_vector", lambda *_args, **_kwargs: calls.append("line"))
    monkeypatch.setattr(sp, "assemble_h_source_vector", assemble_regularized)

    current_density, source_vec, diagnostics = sp._h_source_inputs(
        msh=object(),
        spaces={},
        materials={},
        source={"mode": "manual_line"},
        config=sp.PipelineConfig(formulation="h", source_mode="manual_line"),
        cell_tags=object(),
    )

    assert current_density is sentinel_current_density
    assert source_vec is sentinel_regularized_vector
    assert diagnostics["h_source_mode"] == "regularized_volume"
    assert calls == ["regularized", "assemble"]
