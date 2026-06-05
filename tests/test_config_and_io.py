import h5py
import numpy as np

from atem3d.config import build_simulation, load_config
from atem3d.hj import HJMagneticSimulation
from atem3d.io import save_result_hdf5
from atem3d.simulation import ReceiverDataResult
from atem3d.source_history_runtime import (
    ChargeConservingInitialPolarizationSourceHistoryCorrection,
    DrivenRecoverySourceHistoryCorrection,
    InitialPolarizationSourceHistoryCorrection,
    SourceDiffusionKernelSourceHistoryCorrection,
    SourceHistoryCorrection,
    SourcePrimaryDelta6SourceHistoryCorrection,
)


def test_build_simulation_from_config_dict_runs_small_model(tmp_path):
    config = {
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {"sigma_infinity": 0.1},
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [
            {"location": [0.0, 0.5, 0.0], "component": "Ex"},
            {"location": [0.0, 0.5, 0.0], "component": "Hz"},
        ],
    }

    simulation = build_simulation(config)
    result = simulation.run()
    output = tmp_path / "result.h5"
    save_result_hdf5(output, result, config)

    with h5py.File(output, "r") as h5:
        np.testing.assert_allclose(h5["times"][:], result.times)
        assert h5["e"].shape == result.e.shape
        assert h5["b"].shape == result.b.shape
        assert h5["data"].shape == result.data.shape
        assert "config_yaml" in h5.attrs


def test_save_result_hdf5_accepts_receiver_data_only_result(tmp_path):
    result = ReceiverDataResult(
        times=np.array([0.0, 1.0e-3]),
        data=np.array([[0.0, 0.0], [1.0, 2.0]]),
        memories=[],
    )
    output = tmp_path / "data-only.h5"
    config = {"formulation": "eb"}

    save_result_hdf5(output, result, config)

    with h5py.File(output, "r") as h5:
        np.testing.assert_allclose(h5["times"][:], result.times)
        np.testing.assert_allclose(h5["data"][:], result.data)
        assert "e" not in h5
        assert "b" not in h5
        assert "h" not in h5
        assert bool(h5.attrs["receiver_data_only"]) is True
        assert h5.attrs["formulation"] == "eb"
        assert "config_yaml" in h5.attrs


def test_build_simulation_can_run_hj_formulation_and_save_h_fields(tmp_path):
    config = {
        "formulation": "hj",
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0, 1.0],
            "origin": "CCC",
        },
        "model": {
            "sigma_infinity": 0.2,
            "debye_terms": [{"delta_sigma": 0.02, "tau": 0.1}],
        },
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [
            {"location": [0.0, 0.5, 0.0], "component": "Ex"},
            {"location": [0.0, 0.5, 0.0], "component": "Hz"},
        ],
    }

    simulation = build_simulation(config)
    result = simulation.run()
    output = tmp_path / "hj-result.h5"
    save_result_hdf5(output, result, config)

    assert isinstance(simulation, HJMagneticSimulation)
    assert result.e.shape == (2, simulation.mesh.n_faces)
    assert result.h.shape == (2, simulation.mesh.n_edges)
    assert result.data.shape == (2, 2)
    with h5py.File(output, "r") as h5:
        assert h5.attrs["formulation"] == "hj"
        assert h5.attrs["electric_field_location"] == "faces"
        assert h5.attrs["magnetic_field_location"] == "edges"
        assert "h" in h5
        assert "b" not in h5
        assert h5["h"].shape == result.h.shape
        assert h5["e"].shape == result.e.shape
        assert h5["data"].shape == result.data.shape


def test_build_simulation_passes_hj_linear_solver_config():
    config = {
        "formulation": "hj",
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0, 1.0],
            "origin": "CCC",
        },
        "model": {"sigma_infinity": 0.1},
        "solver": {"type": "cg", "tolerance": 1.0e-10, "maxiter": 50, "preconditioner": "none"},
        "magnetic_receiver_mode": "current_biot",
        "magnetic_recovery_subdivisions": 2,
        "magnetic_recovery_polarization_scale": "low_frequency_ratio",
        "magnetic_recovery_initial_polarization_scale": 0.5,
        "magnetic_recovery_source_history": {
            "tau": 0.1,
            "max_order": 1,
            "source_moment_degrees": [0, 2],
            "coefficients": [0.1, -0.03, 0.04, 0.02],
        },
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [],
    }

    simulation = build_simulation(config)

    assert isinstance(simulation, HJMagneticSimulation)
    assert simulation.linear_solver == "cg"
    assert simulation.cg_tolerance == 1.0e-10
    assert simulation.cg_maxiter == 50
    assert simulation.cg_preconditioner == "none"
    assert simulation.magnetic_receiver_mode == "current_biot"
    assert simulation.magnetic_recovery_subdivisions == 2
    assert simulation.magnetic_recovery_polarization_scale == "low_frequency_ratio"
    assert simulation.magnetic_recovery_initial_polarization_scale == 0.5
    assert simulation.magnetic_recovery_source_history.tau == 0.1


def test_build_simulation_passes_source_face_projection_config():
    config = {
        "formulation": "hj",
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.0, -1.0],
        },
        "model": {"sigma_infinity": 0.1},
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "face_projection": "axis_aligned",
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [],
    }

    simulation = build_simulation(config)

    assert simulation.sources[0].face_projection == "axis_aligned"


def test_hj_sponge_can_leave_initial_model_unmodified():
    base_config = {
        "formulation": "hj",
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0, 1.0],
            "origin": [-1.5, -1.5, -1.5],
        },
        "model": {"sigma_infinity": 0.2},
        "source": {
            "start": [-1.0, 0.0, 0.0],
            "end": [1.0, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0, "on_value": 1.0},
        },
        "time_steps": [0.1],
    }
    no_boundary = build_simulation({**base_config, "boundary": {"kind": "none"}})
    sponge = build_simulation(
        {
            **base_config,
            "boundary": {
                "kind": "sponge",
                "thickness_cells": 1,
                "strength": 5.0,
                "apply_to_initial": False,
            },
        }
    )

    assert np.max(sponge.ip_model.sigma_infinity) > np.max(
        sponge.initial_ip_model.sigma_infinity
    )
    np.testing.assert_allclose(
        sponge.initial_ip_model.sigma_infinity,
        no_boundary.ip_model.sigma_infinity,
    )
    no_boundary_result = no_boundary.run()
    sponge_result = sponge.run()
    np.testing.assert_allclose(sponge_result.e[0], no_boundary_result.e[0])
    np.testing.assert_allclose(sponge_result.h[0], no_boundary_result.h[0])


def test_eb_sponge_can_leave_initial_model_unmodified():
    base_config = {
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0, 1.0],
            "origin": [-1.5, -1.5, -1.5],
        },
        "model": {"sigma_infinity": 0.2},
        "source": {
            "start": [-1.0, 0.0, 0.0],
            "end": [1.0, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0, "on_value": 1.0},
        },
        "time_steps": [0.1],
    }
    no_boundary = build_simulation({**base_config, "boundary": {"kind": "none"}})
    sponge = build_simulation(
        {
            **base_config,
            "boundary": {
                "kind": "sponge",
                "thickness_cells": 1,
                "strength": 5.0,
                "apply_to_initial": False,
            },
        }
    )

    assert np.max(sponge.ip_model.sigma_infinity) > np.max(
        sponge.initial_ip_model.sigma_infinity
    )
    np.testing.assert_allclose(
        sponge.initial_ip_model.sigma_infinity,
        no_boundary.ip_model.sigma_infinity,
    )
    no_boundary_result = no_boundary.run()
    sponge_result = sponge.run()
    np.testing.assert_allclose(sponge_result.e[0], no_boundary_result.e[0])
    np.testing.assert_allclose(sponge_result.b[0], no_boundary_result.b[0])


def test_build_simulation_rejects_hj_with_active_cpml():
    config = {
        "formulation": "hj",
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0, 1.0],
            "origin": "CCC",
        },
        "model": {"sigma_infinity": 0.1},
        "boundary": {"kind": "cpml", "thickness_cells": 1, "sigma_max": 10.0},
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [],
    }

    try:
        build_simulation(config)
    except ValueError as err:
        assert "H/J formulation does not support active CPML" in str(err)
    else:
        raise AssertionError("expected ValueError")


def test_build_simulation_accepts_boundary_config_alias():
    config = {
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0, 1.0],
            "origin": "CCC",
        },
        "model": {
            "sigma_infinity": 1.0,
            "debye_terms": [{"delta_sigma": 0.2, "tau": 0.1}],
        },
        "boundary": {
            "kind": "sponge",
            "thickness_cells": 1,
            "strength": 5.0,
            "power": 2.0,
            "disable_ip_in_shell": True,
            "sides": ["z_min"],
        },
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [],
    }

    simulation = build_simulation(config)

    assert simulation.ip_model.sigma_infinity.max() > 1.0
    changed = simulation.ip_model.sigma_infinity > 1.0
    assert np.all(
        simulation.mesh.cell_centers[changed, 2]
        == simulation.mesh.cell_centers[:, 2].min()
    )
    assert simulation.ip_model.terms[0].delta_sigma.min() == 0.0


def test_build_simulation_accepts_experimental_cpml_boundary_config():
    config = {
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {"sigma_infinity": 0.1},
        "boundary": {
            "kind": "cpml",
            "thickness_cells": 0,
            "sigma_max": 0.0,
            "alpha_max": 0.0,
            "kappa_max": 1.0,
            "power": 2.0,
        },
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [],
    }

    simulation = build_simulation(config)

    assert simulation.cpml is not None
    assert simulation.cpml.thickness_cells == 0


def test_build_simulation_accepts_initial_magnetic_mode():
    config = {
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {"sigma_infinity": 0.1},
        "initial_magnetic_field": "zero",
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [],
    }

    simulation = build_simulation(config)

    assert simulation.initial_magnetic_mode == "zero"


def test_build_simulation_accepts_biot_savart_wire_initial_magnetic_mode():
    config = {
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {"sigma_infinity": 0.1},
        "initial_magnetic_field": "biot_savart_wire",
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [],
    }

    simulation = build_simulation(config)

    assert simulation.initial_magnetic_mode == "biot_savart_wire"


def test_build_simulation_accepts_magnetic_receiver_mode():
    config = {
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {"sigma_infinity": 0.1},
        "magnetic_receiver_mode": "current_biot",
        "magnetic_recovery_subdivisions": 3,
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [{"location": [0.0, 0.5, 0.0], "component": "Hz"}],
    }

    simulation = build_simulation(config)

    assert simulation.magnetic_receiver_mode == "current_biot"
    assert simulation.magnetic_recovery_subdivisions == 3


def test_build_simulation_accepts_edge_basis_biot_magnetic_receiver_mode():
    config = {
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {"sigma_infinity": 0.1},
        "magnetic_receiver_mode": "edge_basis_biot",
        "magnetic_recovery_subdivisions": 2,
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [{"location": [0.0, 0.5, 0.0], "component": "Hz"}],
    }

    simulation = build_simulation(config)

    assert simulation.magnetic_receiver_mode == "edge_basis_biot"
    assert simulation.magnetic_recovery_subdivisions == 2


def test_build_simulation_accepts_edge_basis_cell_biot_magnetic_receiver_mode():
    config = {
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {"sigma_infinity": 0.1},
        "magnetic_receiver_mode": "edge_basis_cell_biot",
        "magnetic_recovery_subdivisions": 2,
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [{"location": [0.0, 0.5, 0.0], "component": "Hz"}],
    }

    simulation = build_simulation(config)

    assert simulation.magnetic_receiver_mode == "edge_basis_cell_biot"
    assert simulation.magnetic_recovery_subdivisions == 2


def test_build_simulation_accepts_magnetic_recovery_polarization_scale():
    config = {
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {
            "sigma_infinity": 0.2,
            "debye_terms": [{"delta_sigma": 0.05, "tau": 0.1}],
        },
        "magnetic_receiver_mode": "current_biot",
        "magnetic_recovery_polarization_scale": 1.25,
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [{"location": [0.0, 0.5, 0.0], "component": "Hz"}],
    }

    simulation = build_simulation(config)

    assert simulation.magnetic_recovery_polarization_scale == 1.25


def test_build_simulation_accepts_initial_polarization_scale_for_diagnostics():
    config = {
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {
            "sigma_infinity": 0.2,
            "debye_terms": [{"delta_sigma": 0.05, "tau": 0.1}],
        },
        "magnetic_receiver_mode": "current_biot",
        "magnetic_recovery_initial_polarization_scale": 0.25,
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [{"location": [0.0, 0.5, 0.0], "component": "Hz"}],
    }

    simulation = build_simulation(config)

    assert simulation.magnetic_recovery_initial_polarization_scale == 0.25


def test_build_simulation_accepts_source_primary_delta6_for_diagnostics():
    config = {
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {
            "sigma_infinity": 0.2,
            "debye_terms": [{"delta_sigma": 0.05, "tau": 0.1}],
        },
        "magnetic_receiver_mode": "current_biot",
        "magnetic_recovery_source_primary_delta6": True,
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [{"location": [0.0, 0.5, 0.0], "component": "Hz"}],
    }

    simulation = build_simulation(config)

    assert simulation.magnetic_recovery_source_primary_delta6 is True


def test_build_simulation_accepts_source_primary_delta6_edge_basis_for_diagnostics():
    config = {
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {
            "sigma_infinity": 0.2,
            "debye_terms": [{"delta_sigma": 0.05, "tau": 0.1}],
        },
        "magnetic_receiver_mode": "current_biot",
        "magnetic_recovery_source_primary_delta6": True,
        "magnetic_recovery_source_primary_delta6_basis": "edge_current",
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [{"location": [0.0, 0.5, 0.0], "component": "Hz"}],
    }

    simulation = build_simulation(config)

    assert simulation.magnetic_recovery_source_primary_delta6_basis == "edge_current"


def test_build_hj_simulation_accepts_source_primary_delta6_face_basis_for_diagnostics():
    config = {
        "formulation": "hj",
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {
            "sigma_infinity": 0.2,
            "debye_terms": [{"delta_sigma": 0.05, "tau": 0.1}],
        },
        "magnetic_receiver_mode": "current_biot",
        "magnetic_recovery_source_primary_delta6": True,
        "magnetic_recovery_source_primary_delta6_basis": "face_current",
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [{"location": [0.0, 0.5, 0.0], "component": "Hz"}],
    }

    simulation = build_simulation(config)

    assert isinstance(simulation, HJMagneticSimulation)
    assert simulation.magnetic_recovery_source_primary_delta6 is True
    assert simulation.magnetic_recovery_source_primary_delta6_basis == "face_current"


def test_build_simulation_accepts_prescribed_source_history_for_diagnostics():
    config = {
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {
            "sigma_infinity": 0.2,
            "debye_terms": [{"delta_sigma": 0.05, "tau": 0.1}],
        },
        "magnetic_receiver_mode": "current_biot",
        "magnetic_recovery_source_history": {
            "kind": "prescribed_source_moments",
            "tau": 0.1,
            "max_order": 1,
            "source_moment_degrees": [0, 2],
            "coefficients": [0.1, -0.03, 0.04, 0.02],
            "receiver_matrix": "current_biot",
        },
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [{"location": [0.0, 0.5, 0.0], "component": "Hz"}],
    }

    simulation = build_simulation(config)

    assert simulation.magnetic_recovery_source_history is not None
    assert simulation.magnetic_recovery_source_history.kind == "prescribed_source_moments"
    assert simulation.magnetic_recovery_source_history.source_moment_degrees == (0, 2)


def test_build_simulation_accepts_driven_recovery_source_history_for_diagnostics():
    config = {
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {
            "sigma_infinity": 0.2,
            "debye_terms": [{"delta_sigma": 0.05, "tau": 0.1}],
        },
        "magnetic_receiver_mode": "current_biot",
        "magnetic_recovery_source_history": {
            "kind": "driven_recovery_source_moments",
            "driver_tau": 0.1,
            "response_tau": 0.02,
            "source_moment_degrees": [0, 2],
            "coefficients": [0.1, -0.03],
            "receiver_matrix": "current_biot",
        },
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [{"location": [0.0, 0.5, 0.0], "component": "Hz"}],
    }

    simulation = build_simulation(config)

    correction = simulation.magnetic_recovery_source_history
    assert isinstance(correction, DrivenRecoverySourceHistoryCorrection)
    assert correction.kind == "driven_recovery_source_moments"
    assert correction.driver_tau == 0.1
    assert correction.response_tau == 0.02
    assert correction.source_moment_degrees == (0, 2)


def test_build_simulation_accepts_multimode_driven_recovery_source_history():
    config = {
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {
            "sigma_infinity": 0.2,
            "debye_terms": [{"delta_sigma": 0.05, "tau": 0.1}],
        },
        "magnetic_receiver_mode": "current_biot",
        "magnetic_recovery_source_history": {
            "kind": "driven_recovery_source_moments",
            "driver_tau": 0.1,
            "response_taus": [0.02, 0.04],
            "source_moment_degrees": [0, 2],
            "coefficients": [0.1, -0.03, 0.04, 0.02],
            "receiver_matrix": "current_biot",
        },
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [{"location": [0.0, 0.5, 0.0], "component": "Hz"}],
    }

    simulation = build_simulation(config)

    correction = simulation.magnetic_recovery_source_history
    assert isinstance(correction, DrivenRecoverySourceHistoryCorrection)
    assert correction.response_taus == (0.02, 0.04)
    assert correction.coefficients == (0.1, -0.03, 0.04, 0.02)


def test_build_simulation_accepts_initial_polarization_source_history_for_diagnostics():
    config = {
        "formulation": "hj",
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {
            "sigma_infinity": 0.2,
            "debye_terms": [{"delta_sigma": 0.05, "tau": 0.1}],
        },
        "magnetic_receiver_mode": "current_biot",
        "magnetic_recovery_source_history": {
            "kind": "initial_polarization_source_moments",
            "source_moment_degrees": [0, 2],
            "receiver_matrix": "face_current",
            "projection": "receiver_l2",
        },
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [{"location": [0.0, 0.5, 0.0], "component": "Hz"}],
    }

    simulation = build_simulation(config)

    correction = simulation.magnetic_recovery_source_history
    assert isinstance(correction, InitialPolarizationSourceHistoryCorrection)
    assert correction.source_moment_degrees == (0, 2)
    assert correction.receiver_matrix == "face_current"
    assert correction.projection == "receiver_l2"


def test_build_simulation_accepts_charge_conserving_source_history_for_diagnostics():
    config = {
        "formulation": "hj",
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {
            "sigma_infinity": 0.2,
            "debye_terms": [{"delta_sigma": 0.05, "tau": 0.1}],
        },
        "magnetic_receiver_mode": "current_biot",
        "magnetic_recovery_source_history": {
            "kind": "charge_conserving_initial_polarization_source_moments",
            "source_moment_degrees": [0, 2],
            "receiver_matrix": "face_current",
            "projection": "receiver_l2",
        },
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [{"location": [0.0, 0.5, 0.0], "component": "Hz"}],
    }

    simulation = build_simulation(config)

    correction = simulation.magnetic_recovery_source_history
    assert isinstance(correction, ChargeConservingInitialPolarizationSourceHistoryCorrection)
    assert correction.source_moment_degrees == (0, 2)
    assert correction.receiver_matrix == "face_current"
    assert correction.projection == "receiver_l2"


def test_build_simulation_accepts_source_primary_delta6_source_history():
    config = {
        "formulation": "hj",
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {
            "sigma_infinity": 0.2,
            "debye_terms": [{"delta_sigma": 0.05, "tau": 0.1}],
        },
        "magnetic_receiver_mode": "current_biot",
        "magnetic_recovery_source_history": {
            "kind": "source_primary_delta6_source_moments",
            "source_moment_degrees": [0, 2],
            "receiver_matrix": "auto",
        },
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [{"location": [0.0, 0.5, 0.0], "component": "Hz"}],
    }

    simulation = build_simulation(config)

    correction = simulation.magnetic_recovery_source_history
    assert isinstance(correction, SourcePrimaryDelta6SourceHistoryCorrection)
    assert correction.source_moment_degrees == (0, 2)
    assert correction.receiver_matrix == "auto"


def test_build_simulation_accepts_source_diffusion_kernel_source_history():
    config = {
        "formulation": "hj",
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {"sigma_infinity": 0.2},
        "magnetic_receiver_mode": "current_biot",
        "magnetic_recovery_source_history": {
            "kind": "source_diffusion_kernel_source_moments",
            "amplitude": -1.25e-4,
            "tau_multiplier": 1.5,
            "amplitude_time": 1.0e-4,
            "basis_kind": "be_decay",
            "source_moment_degrees": [0],
            "receiver_matrix": "auto",
        },
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "face_projection": "axis_aligned",
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [{"location": [0.0, 0.5, 0.0], "component": "Hz"}],
    }

    simulation = build_simulation(config)

    correction = simulation.magnetic_recovery_source_history
    assert isinstance(correction, SourceDiffusionKernelSourceHistoryCorrection)
    assert correction.kind == "source_diffusion_kernel_source_moments"
    assert correction.amplitude == -1.25e-4
    assert correction.coefficients == (-1.25e-4,)
    assert correction.tau_multiplier == 1.5
    assert correction.amplitude_time == 1.0e-4
    assert correction.basis_kind == "be_decay"
    assert correction.source_moment_degrees == (0,)


def test_build_simulation_accepts_source_diffusion_kernel_normalized_amplitude():
    config = {
        "formulation": "hj",
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {"sigma_infinity": 0.2},
        "magnetic_receiver_mode": "current_biot",
        "magnetic_recovery_source_history": {
            "kind": "source_diffusion_kernel_source_moments",
            "normalized_amplitude": -3.5,
            "tau_multiplier": 1.5,
            "amplitude_time": 1.0e-4,
            "source_moment_degrees": [0],
            "receiver_matrix": "auto",
        },
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "face_projection": "axis_aligned",
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [{"location": [0.0, 0.5, 0.0], "component": "Hz"}],
    }

    simulation = build_simulation(config)

    correction = simulation.magnetic_recovery_source_history
    assert isinstance(correction, SourceDiffusionKernelSourceHistoryCorrection)
    assert correction.normalized_amplitude == -3.5
    assert correction.coefficients is None
    assert correction.tau_multiplier == 1.5


def test_build_simulation_accepts_multiple_prescribed_source_history_terms():
    config = {
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {
            "sigma_infinity": 0.2,
            "debye_terms": [{"delta_sigma": 0.05, "tau": 0.1}],
        },
        "magnetic_receiver_mode": "current_biot",
        "magnetic_recovery_source_history": {
            "terms": [
                {
                    "tau": 0.1,
                    "max_order": 1,
                    "source_moment_degrees": [0],
                    "coefficients": [0.1, 0.04],
                },
                {
                    "tau": 0.5,
                    "max_order": 0,
                    "source_moment_degrees": [2],
                    "coefficients": [-0.03],
                    "receiver_matrix": "edge_current",
                },
            ],
        },
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [{"location": [0.0, 0.5, 0.0], "component": "Hz"}],
    }

    simulation = build_simulation(config)

    assert len(simulation.magnetic_recovery_source_history) == 2
    assert [term.tau for term in simulation.magnetic_recovery_source_history] == [0.1, 0.5]
    assert simulation.magnetic_recovery_source_history[1].receiver_matrix == "edge_current"


def test_build_simulation_accepts_prescribed_normalized_source_history_coefficients():
    config = {
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {
            "sigma_infinity": 0.2,
            "debye_terms": [{"delta_sigma": 0.05, "tau": 0.1}],
        },
        "magnetic_receiver_mode": "current_biot",
        "magnetic_recovery_source_history": {
            "tau": 0.1,
            "max_order": 1,
            "source_moment_degrees": [0],
            "normalized_coefficients": [6.0, 3.0],
        },
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [{"location": [0.0, 0.5, 0.0], "component": "Hz"}],
    }

    simulation = build_simulation(config)

    correction = simulation.magnetic_recovery_source_history
    assert isinstance(correction, SourceHistoryCorrection)
    assert correction.coefficients is None
    assert correction.normalized_coefficients == (6.0, 3.0)


def test_build_simulation_accepts_driven_normalized_source_history_coefficients():
    config = {
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {
            "sigma_infinity": 0.2,
            "debye_terms": [{"delta_sigma": 0.05, "tau": 0.1}],
        },
        "magnetic_receiver_mode": "current_biot",
        "magnetic_recovery_source_history": {
            "kind": "driven_recovery_source_moments",
            "driver_tau": 0.1,
            "response_tau": 0.05,
            "source_moment_degrees": [0, 2],
            "normalized_coefficients": [8.0, 2.0],
        },
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [{"location": [0.0, 0.5, 0.0], "component": "Hz"}],
    }

    simulation = build_simulation(config)

    correction = simulation.magnetic_recovery_source_history
    assert isinstance(correction, DrivenRecoverySourceHistoryCorrection)
    assert correction.coefficients is None
    assert correction.normalized_coefficients == (8.0, 2.0)


def test_build_simulation_accepts_low_frequency_ratio_polarization_scale():
    config = {
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {
            "sigma_infinity": 0.2,
            "debye_terms": [{"delta_sigma": 0.05, "tau": 0.1}],
        },
        "magnetic_receiver_mode": "current_biot",
        "magnetic_recovery_polarization_scale": "low_frequency_ratio",
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [{"location": [0.0, 0.5, 0.0], "component": "Hz"}],
    }

    simulation = build_simulation(config)

    assert simulation.magnetic_recovery_polarization_scale == "low_frequency_ratio"


def test_build_simulation_accepts_linear_solver_config():
    config = {
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {"sigma_infinity": 0.1},
        "solver": {"type": "cg", "tolerance": 1.0e-9, "maxiter": 123, "preconditioner": "jacobi"},
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [],
    }

    simulation = build_simulation(config)

    assert simulation.linear_solver == "cg"
    assert simulation.cg_tolerance == 1.0e-9
    assert simulation.cg_maxiter == 123
    assert simulation.cg_preconditioner == "jacobi"


def test_build_simulation_accepts_pardiso_solver_config():
    config = {
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {"sigma_infinity": 0.1},
        "solver": {"type": "pardiso"},
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [],
    }

    simulation = build_simulation(config)

    assert simulation.linear_solver == "pardiso"


def test_load_config_reads_yaml_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
mesh:
  hx: [1.0]
  hy: [1.0]
  hz: [1.0]
  origin: [0.0, 0.0, 0.0]
model:
  sigma_infinity: 1.0
source:
  start: [0.1, 0.1, 0.1]
  end: [0.9, 0.1, 0.1]
  current: 1.0
  waveform: {type: step_off, off_time: 0.0}
time_steps: [0.1]
receivers: []
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config["model"]["sigma_infinity"] == 1.0


def test_build_simulation_accepts_yaml_style_padding_widths():
    config = {
        "mesh": {
            "hx": [[1.0, 1, -1.2], [1.0, 2], [1.0, 1, 1.2]],
            "hy": [[1.0, 1, -1.2], [1.0, 2], [1.0, 1, 1.2]],
            "hz": [[1.0, 1], [1.0, 1]],
            "origin": "CCC",
        },
        "model": {"sigma_infinity": 0.1},
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [],
    }

    simulation = build_simulation(config)

    assert simulation.mesh.n_cells == 4 * 4 * 2


def test_build_simulation_accepts_repeated_time_step_tuples():
    config = {
        "mesh": {
            "hx": [1.0, 1.0, 1.0],
            "hy": [1.0, 1.0, 1.0],
            "hz": [1.0, 1.0],
            "origin": [-1.5, -1.5, -1.0],
        },
        "model": {"sigma_infinity": 0.1},
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [[0.001, 3], 0.005],
        "receivers": [],
    }

    simulation = build_simulation(config)

    np.testing.assert_allclose(simulation.time_steps, [0.001, 0.001, 0.001, 0.005])


def test_build_simulation_supports_layered_conductivity_and_receiver_line():
    config = {
        "mesh": {
            "hx": [1.0, 1.0],
            "hy": [1.0, 1.0],
            "hz": [1.0, 1.0, 1.0, 1.0],
            "origin": [-1.0, -1.0, -1.0],
        },
        "model": {
            "layers": [
                {"top": -1.0e9, "bottom": 0.0, "sigma_infinity": 1.0e-8},
                {"top": 0.0, "bottom": 1.0, "sigma_infinity": 0.1},
                {
                    "top": 1.0,
                    "bottom": 1.0e9,
                    "sigma_infinity": 0.2,
                    "debye_terms": [{"delta_sigma": 0.02, "tau": 0.1}],
                },
            ]
        },
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receiver_line": {
            "x": [-0.5, 0.0, 0.5],
            "y": 0.5,
            "z": 0.0,
            "components": ["Ex", "Ey", "Hz"],
        },
    }

    simulation = build_simulation(config)

    assert len(simulation.receivers) == 9
    assert simulation.ip_model.terms[0].delta_sigma.max() == 0.02
    assert simulation.ip_model.terms[0].delta_sigma.min() == 0.0


def test_build_simulation_assigns_z_up_layers_from_top_to_bottom():
    config = {
        "coordinate_system": "z_up",
        "mesh": {
            "hx": [1.0],
            "hy": [1.0],
            "hz": [1.0, 1.0, 1.0],
            "origin": [0.0, 0.0, -2.0],
        },
        "model": {
            "layers": [
                {"top": 1.0e9, "bottom": 0.0, "sigma_infinity": 1.0e-8},
                {"top": 0.0, "bottom": -1.0e9, "sigma_infinity": 0.1},
            ],
        },
        "source": {
            "start": [0.1, 0.1, -0.1],
            "end": [0.9, 0.1, -0.1],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [],
    }

    simulation = build_simulation(config)
    z = simulation.mesh.cell_centers[:, 2]

    np.testing.assert_allclose(simulation.ip_model.sigma_infinity[z > 0.0], 1.0e-8)
    np.testing.assert_allclose(simulation.ip_model.sigma_infinity[z < 0.0], 0.1)


def test_build_simulation_can_require_layer_boundaries_to_align_with_mesh_nodes():
    config = {
        "mesh": {
            "hx": [1.0],
            "hy": [1.0],
            "hz": [1.0, 1.0],
            "origin": [0.0, 0.0, -0.25],
        },
        "model": {
            "require_layer_boundary_alignment": True,
            "layers": [
                {"top": -1.0e9, "bottom": 0.0, "sigma_infinity": 1.0e-8},
                {"top": 0.0, "bottom": 1.0e9, "sigma_infinity": 0.1},
            ],
        },
        "source": {
            "start": [0.1, 0.1, 0.1],
            "end": [0.9, 0.1, 0.1],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [],
    }

    try:
        build_simulation(config)
    except ValueError as err:
        assert "layer boundary z=0.0 must align with a mesh node" in str(err)
    else:
        raise AssertionError("expected ValueError")


def test_build_simulation_expands_layer_pelton_model_to_debye_terms():
    config = {
        "mesh": {
            "hx": [1.0],
            "hy": [1.0],
            "hz": [1.0, 1.0],
            "origin": [0.0, 0.0, 0.0],
        },
        "model": {
            "fit_frequencies": [0.1, 1.0, 10.0],
            "layers": [
                {
                    "top": 0.0,
                    "bottom": 2.0,
                    "sigma_infinity": 0.01,
                    "ip_model": {
                        "type": "pelton",
                        "rho0": 100.0,
                        "chargeability": 0.2,
                        "tau": 0.1,
                        "c": 1.0,
                        "tau_grid": [0.1],
                    },
                }
            ],
        },
        "source": {
            "start": [0.1, 0.1, 0.1],
            "end": [0.9, 0.1, 0.1],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [],
    }

    simulation = build_simulation(config)

    assert len(simulation.ip_model.terms) == 1
    assert simulation.ip_model.terms[0].tau == 0.1
    assert np.all(simulation.ip_model.terms[0].delta_sigma > 0.0)


def test_build_simulation_uses_pelton_high_frequency_conductivity_for_layer_base():
    config = {
        "mesh": {
            "hx": [1.0],
            "hy": [1.0],
            "hz": [1.0],
            "origin": [0.0, 0.0, 0.0],
        },
        "model": {
            "fit_frequencies": [0.1, 1.0, 10.0],
            "layers": [
                {
                    "top": 0.0,
                    "bottom": 1.0,
                    "sigma_infinity": 999.0,
                    "ip_model": {
                        "type": "pelton",
                        "rho0": 100.0,
                        "chargeability": 0.2,
                        "tau": 0.1,
                        "c": 1.0,
                        "tau_grid": [0.1],
                    },
                }
            ],
        },
        "source": {
            "start": [0.1, 0.1, 0.1],
            "end": [0.9, 0.1, 0.1],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [],
    }

    simulation = build_simulation(config)

    np.testing.assert_allclose(
        simulation.ip_model.sigma_infinity,
        np.array([1.0 / (100.0 * (1.0 - 0.2))]),
    )


def test_build_simulation_expands_uniform_pelton_model_to_debye_terms():
    config = {
        "mesh": {
            "hx": [1.0, 1.0],
            "hy": [1.0],
            "hz": [1.0],
            "origin": [0.0, 0.0, 0.0],
        },
        "model": {
            "sigma_infinity": 999.0,
            "fit_frequencies": [0.1, 1.0, 10.0],
            "ip_model": {
                "type": "pelton",
                "rho0": 100.0,
                "chargeability": 0.2,
                "tau": 0.1,
                "c": 1.0,
                "tau_grid": [0.1],
            },
        },
        "source": {
            "start": [0.1, 0.1, 0.1],
            "end": [0.9, 0.1, 0.1],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01],
        "receivers": [],
    }

    simulation = build_simulation(config)

    np.testing.assert_allclose(
        simulation.ip_model.sigma_infinity,
        np.full(2, 1.0 / (100.0 * (1.0 - 0.2))),
    )
    assert len(simulation.ip_model.terms) == 1
    assert simulation.ip_model.terms[0].delta_sigma.shape == (2,)
