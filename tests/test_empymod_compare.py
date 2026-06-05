import numpy as np
import h5py
import yaml
from scipy.constants import mu_0

from atem3d.empymod_compare import (
    EmpymodSurvey,
    build_empymod_survey_from_config,
    build_empymod_survey_from_result,
    make_debye_resistivity_model,
    make_debye_resistivity_model_from_config,
    run_empymod_reference,
)
from atem3d.fit import fit_pelton_resistivity_debye


class FakeEmpymod:
    def __init__(self):
        self.calls = []

    def bipole(self, **kwargs):
        self.calls.append(kwargs)
        return np.full(len(kwargs["freqtime"]), len(self.calls), dtype=float)


def test_empymod_reference_maps_finite_wire_and_components_to_bipole_calls():
    backend = FakeEmpymod()
    survey = EmpymodSurvey(
        source_start=(-25.0, 0.0, 0.0),
        source_end=(25.0, 0.0, 0.0),
        receiver_locations=[(0.0, 10.0, 0.0)],
        components=["Ex", "Ey", "Hz"],
        times=np.array([1.0e-4, 1.0e-3]),
        depths=[0.0, 100.0],
        resistivities=[100.0, 20.0],
        strength=2.0,
        signal=-1,
    )

    data = run_empymod_reference(survey, backend=backend)

    assert data.shape == (2, 3)
    assert len(backend.calls) == 3
    assert backend.calls[0]["src"] == [-25.0, 25.0, 0.0, 0.0, 0.0, 0.0]
    assert backend.calls[0]["rec"] == [0.0, 10.0, 0.0, 0.0, 0.0]
    assert backend.calls[1]["rec"] == [0.0, 10.0, 0.0, 90.0, 0.0]
    assert backend.calls[2]["mrec"] is True
    assert backend.calls[2]["rec"] == [0.0, 10.0, 0.0, 0.0, 90.0]
    assert backend.calls[2]["strength"] == 2.0


def test_empymod_reference_maps_db_dt_to_mrec_b():
    backend = FakeEmpymod()
    survey = EmpymodSurvey(
        source_start=(-1.0, 0.0, 0.0),
        source_end=(1.0, 0.0, 0.0),
        receiver_locations=[(0.0, 2.0, 0.0)],
        components=["dBzdt"],
        times=np.array([1.0e-3]),
        depths=[0.0],
        resistivities=[1.0e8, 100.0],
    )

    run_empymod_reference(survey, backend=backend)

    assert backend.calls[0]["mrec"] == "b"


def test_empymod_reference_scales_magnetic_flux_density_from_h_field():
    class ConstantMagneticBackend:
        def bipole(self, **kwargs):
            assert kwargs["mrec"] is True
            return np.full(len(kwargs["freqtime"]), 2.0, dtype=float)

    survey = EmpymodSurvey(
        source_start=(-1.0, 0.0, 0.0),
        source_end=(1.0, 0.0, 0.0),
        receiver_locations=[(0.0, 2.0, 0.0)],
        components=["Bz"],
        times=np.array([1.0e-3, 2.0e-3]),
        depths=[0.0],
        resistivities=[1.0e8, 100.0],
    )

    data = run_empymod_reference(survey, backend=ConstantMagneticBackend())

    np.testing.assert_allclose(data[:, 0], 2.0 * mu_0)


def test_empymod_reference_forwards_source_and_receiver_integration_points():
    backend = FakeEmpymod()
    survey = EmpymodSurvey(
        source_start=(-1.0, 0.0, 0.0),
        source_end=(1.0, 0.0, 0.0),
        receiver_locations=[(0.0, 2.0, 0.0)],
        components=["Ex"],
        times=np.array([1.0e-3]),
        depths=[0.0],
        resistivities=[1.0e8, 100.0],
    )

    run_empymod_reference(survey, backend=backend, srcpts=7, recpts=5)

    assert backend.calls[0]["srcpts"] == 7
    assert backend.calls[0]["recpts"] == 5


def test_empymod_reference_maps_z_up_coordinates_to_empymod_depth_coordinates():
    backend = FakeEmpymod()
    survey = EmpymodSurvey(
        source_start=(-25.0, 0.0, -0.5),
        source_end=(25.0, 0.0, -0.5),
        receiver_locations=[(0.0, 10.0, -1.0)],
        components=["Ex", "Ez", "Hz", "dBzdt"],
        times=np.array([1.0e-3]),
        depths=[0.0, 40.0],
        resistivities=[1.0e8, 100.0, 33.3333333333],
        coordinate_system="z_up",
    )

    run_empymod_reference(survey, backend=backend)

    assert backend.calls[0]["src"] == [-25.0, 25.0, 0.0, 0.0, 0.5, 0.5]
    assert backend.calls[0]["rec"] == [0.0, 10.0, 1.0, 0.0, 0.0]
    assert backend.calls[1]["rec"] == [0.0, 10.0, 1.0, 0.0, -90.0]
    assert backend.calls[2]["rec"] == [0.0, 10.0, 1.0, 0.0, 90.0]
    assert backend.calls[3]["rec"] == [0.0, 10.0, 1.0, 0.0, 90.0]


def test_empymod_reference_applies_z_up_axial_vector_signs_to_magnetic_components():
    class ConstantBackend:
        def bipole(self, **kwargs):
            return np.ones(len(kwargs["freqtime"]), dtype=float)

    survey = EmpymodSurvey(
        source_start=(-25.0, 0.0, -0.5),
        source_end=(25.0, 0.0, -0.5),
        receiver_locations=[(0.0, 10.0, -1.0)],
        components=["Hx", "Hy", "Hz", "Bx", "By", "Bz", "dBxdt", "dBydt", "dBzdt"],
        times=np.array([1.0e-3]),
        depths=[0.0, 40.0],
        resistivities=[1.0e8, 100.0, 33.3333333333],
        coordinate_system="z_up",
    )

    data = run_empymod_reference(survey, backend=ConstantBackend())

    np.testing.assert_allclose(
        data[0],
        [-1.0, -1.0, 1.0, -mu_0, -mu_0, mu_0, -1.0, -1.0, 1.0],
    )


def test_make_debye_resistivity_model_builds_frequency_dependent_eta():
    model = make_debye_resistivity_model(
        sigma_infinity=[1.0e-8, 0.02],
        debye_terms=[
            {"delta_sigma": [0.0, 0.005], "tau": 0.1},
            {"delta_sigma": [0.0, 0.002], "tau": 1.0},
        ],
    )
    locals_dict = {
        "freq": np.array([1.0, 10.0]),
        "etaH": np.zeros((2, 2), dtype=complex),
        "etaV": np.zeros((2, 2), dtype=complex),
    }

    eta_h, eta_v = model["func_eta"](model, locals_dict)

    expected = np.array([1.0e-8, 0.02])[None, :] - (
        np.array([0.0, 0.005])[None, :] / (1.0 + 1j * 2.0 * np.pi * locals_dict["freq"][:, None] * 0.1)
        + np.array([0.0, 0.002])[None, :] / (1.0 + 1j * 2.0 * np.pi * locals_dict["freq"][:, None] * 1.0)
    )
    np.testing.assert_allclose(eta_h, expected)
    np.testing.assert_allclose(eta_v, expected)
    np.testing.assert_allclose(model["res"], [1.0e8, 50.0])


def test_run_empymod_reference_accepts_debye_resistivity_model():
    backend = FakeEmpymod()
    res_model = make_debye_resistivity_model(
        sigma_infinity=[1.0e-8, 0.02],
        debye_terms=[{"delta_sigma": [0.0, 0.005], "tau": 0.1}],
    )
    survey = EmpymodSurvey(
        source_start=(-1.0, 0.0, 0.0),
        source_end=(1.0, 0.0, 0.0),
        receiver_locations=[(0.0, 2.0, 0.0)],
        components=["Ex"],
        times=np.array([1.0e-3]),
        depths=[0.0],
        resistivities=res_model,
    )

    run_empymod_reference(survey, backend=backend)

    assert backend.calls[0]["res"] is res_model


def test_make_debye_resistivity_model_from_layer_config_uses_depth_intervals():
    config = {
        "coordinate_system": "depth_down",
        "model": {
            "layers": [
                {"top": -1.0e9, "bottom": 0.0, "sigma_infinity": 1.0e-8},
                {
                    "top": 0.0,
                    "bottom": 40.0,
                    "sigma_infinity": 0.02,
                    "debye_terms": [{"delta_sigma": 0.005, "tau": 0.1}],
                },
                {
                    "top": 40.0,
                    "bottom": 1.0e9,
                    "sigma_infinity": 0.03,
                    "debye_terms": [{"delta_sigma": 0.001, "tau": 0.1}],
                },
            ]
        }
    }

    model = make_debye_resistivity_model_from_config(config, depths=[0.0, 40.0])
    eta_h, _ = model["func_eta"](model, {"freq": np.array([1.0])})

    expected = np.array([[1.0e-8, 0.02, 0.03]], dtype=complex)
    expected[:, 1] -= 0.005 / (1.0 + 1j * 2.0 * np.pi * 1.0 * 0.1)
    expected[:, 2] -= 0.001 / (1.0 + 1j * 2.0 * np.pi * 1.0 * 0.1)
    np.testing.assert_allclose(eta_h, expected)
    np.testing.assert_allclose(model["res"], [1.0e8, 50.0, 33.333333333333336])


def test_make_debye_resistivity_model_from_z_up_layer_config_uses_depth_intervals():
    config = {
        "coordinate_system": "z_up",
        "model": {
            "layers": [
                {"top": 1.0e9, "bottom": 0.0, "sigma_infinity": 1.0e-8},
                {
                    "top": 0.0,
                    "bottom": -40.0,
                    "sigma_infinity": 0.02,
                    "debye_terms": [{"delta_sigma": 0.005, "tau": 0.1}],
                },
                {
                    "top": -40.0,
                    "bottom": -1.0e9,
                    "sigma_infinity": 0.03,
                    "debye_terms": [{"delta_sigma": 0.001, "tau": 0.1}],
                },
            ]
        },
    }

    model = make_debye_resistivity_model_from_config(config, depths=[0.0, 40.0])
    eta_h, _ = model["func_eta"](model, {"freq": np.array([1.0])})

    expected = np.array([[1.0e-8, 0.02, 0.03]], dtype=complex)
    expected[:, 1] -= 0.005 / (1.0 + 1j * 2.0 * np.pi * 1.0 * 0.1)
    expected[:, 2] -= 0.001 / (1.0 + 1j * 2.0 * np.pi * 1.0 * 0.1)
    np.testing.assert_allclose(eta_h, expected)
    np.testing.assert_allclose(model["res"], [1.0e8, 50.0, 33.333333333333336])


def test_make_debye_resistivity_model_from_layer_config_includes_pelton_ip_model_fit():
    frequencies = np.array([0.5, 2.0, 8.0])
    tau_grid = np.array([0.01, 0.1])
    config = {
        "coordinate_system": "depth_down",
        "model": {
            "fit_frequencies": frequencies.tolist(),
            "layers": [
                {"top": -1.0e9, "bottom": 0.0, "sigma_infinity": 1.0e-8},
                {"top": 0.0, "bottom": 40.0, "sigma_infinity": 0.02},
                {
                    "top": 40.0,
                    "bottom": 1.0e9,
                    "sigma_infinity": 999.0,
                    "ip_model": {
                        "type": "pelton",
                        "rho0": 40.0,
                        "chargeability": 0.25,
                        "tau": 0.02,
                        "c": 0.7,
                        "tau_grid": tau_grid.tolist(),
                    },
                },
            ],
        }
    }

    model = make_debye_resistivity_model_from_config(config, depths=[0.0, 40.0])
    eta_h, _ = model["func_eta"](model, {"freq": np.array([1.0, 10.0])})

    fit = fit_pelton_resistivity_debye(
        rho0=40.0,
        chargeability=0.25,
        tau=0.02,
        c=0.7,
        frequencies=frequencies,
        tau_grid=tau_grid,
    )
    expected = np.repeat(np.array([[1.0e-8, 0.02, fit.sigma_infinity]], dtype=complex), 2, axis=0)
    for term in fit.terms:
        expected[:, 2] -= float(term.delta_sigma[0]) / (
            1.0 + 1j * 2.0 * np.pi * np.array([1.0, 10.0]) * term.tau
        )

    np.testing.assert_allclose(eta_h, expected)
    np.testing.assert_allclose(model["res"], [1.0e8, 50.0, 1.0 / fit.sigma_infinity])


def test_make_debye_resistivity_model_from_uniform_config_expands_to_empymod_layers():
    config = {
        "model": {
            "sigma_infinity": 0.02,
            "debye_terms": [{"delta_sigma": 0.005, "tau": 0.1}],
        }
    }

    model = make_debye_resistivity_model_from_config(config, depths=[0.0])
    eta_h, _ = model["func_eta"](model, {"freq": np.array([1.0])})

    expected_layer = 0.02 - 0.005 / (1.0 + 1j * 2.0 * np.pi * 1.0 * 0.1)
    np.testing.assert_allclose(eta_h, [[expected_layer, expected_layer]])
    np.testing.assert_allclose(model["res"], [50.0, 50.0])


def test_make_debye_resistivity_model_from_uniform_config_includes_pelton_ip_model_fit():
    frequencies = np.array([0.5, 2.0, 8.0])
    config = {
        "model": {
            "sigma_infinity": 999.0,
            "fit_frequencies": frequencies.tolist(),
            "ip_model": {
                "type": "pelton",
                "rho0": 40.0,
                "chargeability": 0.25,
                "tau": 0.02,
                "c": 0.7,
                "tau_grid": [0.01, 0.1],
            },
        }
    }

    model = make_debye_resistivity_model_from_config(config, depths=[0.0])
    eta_h, _ = model["func_eta"](model, {"freq": np.array([1.0])})

    fit = fit_pelton_resistivity_debye(
        rho0=40.0,
        chargeability=0.25,
        tau=0.02,
        c=0.7,
        frequencies=frequencies,
        tau_grid=np.array([0.01, 0.1]),
    )
    expected_layer = fit.sigma_infinity
    for term in fit.terms:
        expected_layer -= float(term.delta_sigma[0]) / (1.0 + 1j * 2.0 * np.pi * 1.0 * term.tau)

    np.testing.assert_allclose(eta_h, [[expected_layer, expected_layer]])
    np.testing.assert_allclose(model["res"], [1.0 / fit.sigma_infinity, 1.0 / fit.sigma_infinity])


def test_build_empymod_survey_from_result_hdf5_uses_receiver_line(tmp_path):
    config = {
        "source": {
            "start": [-25.0, 0.0, 0.0],
            "end": [25.0, 0.0, 0.0],
            "current": 3.0,
        },
        "receiver_line": {
            "x": [-10.0, 0.0],
            "y": 10.0,
            "z": 0.0,
            "components": ["Ex", "Hz"],
        },
    }
    path = tmp_path / "result.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("times", data=np.array([0.0, 1.0e-3]))
        h5.create_dataset("data", data=np.zeros((2, 4)))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    survey, names = build_empymod_survey_from_result(
        path,
        depths=[0.0, 40.0],
        resistivities=[1.0e8, 100.0, 50.0],
        signal=-1,
    )

    assert survey.source_start == (-25.0, 0.0, 0.0)
    assert survey.receiver_locations == [(-10.0, 10.0, 0.0), (0.0, 10.0, 0.0)]
    assert survey.components == ["Ex", "Hz"]
    assert survey.strength == 3.0
    assert names == ["Ex@x=-10", "Hz@x=-10", "Ex@x=0", "Hz@x=0"]


def test_build_empymod_survey_from_config_uses_supplied_times_and_receiver_line():
    config = {
        "coordinate_system": "z_up",
        "source": {
            "start": [-25.0, 0.0, -0.5],
            "end": [25.0, 0.0, -0.5],
            "current": 2.0,
        },
        "receiver_line": {
            "x": [-20.0, 0.0],
            "y": 10.0,
            "z": -0.5,
            "components": ["Ex", "Hz"],
        },
    }

    survey, names = build_empymod_survey_from_config(
        config,
        times=np.array([1.0e-4, 2.0e-4]),
        depths=[0.0, 40.0],
        resistivities=[1.0e8, 100.0, 33.0],
        signal=-1,
    )

    assert names == ["Ex@x=-20", "Hz@x=-20", "Ex@x=0", "Hz@x=0"]
    np.testing.assert_allclose(survey.times, [1.0e-4, 2.0e-4])
    assert survey.strength == 2.0
    assert survey.coordinate_system == "z_up"
    assert survey.receiver_components == [
        ((-20.0, 10.0, -0.5), "Ex"),
        ((-20.0, 10.0, -0.5), "Hz"),
        ((0.0, 10.0, -0.5), "Ex"),
        ((0.0, 10.0, -0.5), "Hz"),
    ]


def test_build_empymod_survey_rejects_wrong_layer_count(tmp_path):
    config = {
        "source": {
            "start": [-25.0, 0.0, 0.0],
            "end": [25.0, 0.0, 0.0],
            "current": 1.0,
        },
        "receivers": [{"location": [0.0, 10.0, 0.0], "component": "Ex"}],
    }
    path = tmp_path / "result.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("times", data=np.array([1.0e-3]))
        h5.create_dataset("data", data=np.zeros((1, 1)))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    try:
        build_empymod_survey_from_result(
            path,
            depths=[0.0, 40.0],
            resistivities=[100.0, 50.0],
        )
    except ValueError as err:
        assert "len(resistivities) must equal len(depths) + 1" in str(err)
    else:
        raise AssertionError("expected ValueError")


def test_build_empymod_survey_from_explicit_receivers_preserves_flat_columns(tmp_path):
    config = {
        "source": {
            "start": [-1.0, 0.0, 0.0],
            "end": [1.0, 0.0, 0.0],
            "current": 1.0,
        },
        "receivers": [
            {"location": [0.0, 2.0, 0.0], "component": "Ex"},
            {"location": [0.0, 2.0, 0.0], "component": "Hz"},
        ],
    }
    path = tmp_path / "result.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("times", data=np.array([1.0e-3]))
        h5.create_dataset("data", data=np.zeros((1, 2)))
        h5.attrs["config_yaml"] = yaml.safe_dump(config)

    survey, names = build_empymod_survey_from_result(
        path,
        depths=[0.0],
        resistivities=[1.0e8, 100.0],
        signal=-1,
    )
    data = run_empymod_reference(survey, backend=FakeEmpymod())

    assert names == ["Ex@0", "Hz@1"]
    assert data.shape == (1, 2)
