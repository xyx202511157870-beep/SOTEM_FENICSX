import numpy as np
from discretize import TensorMesh
from simpeg import maps
from simpeg.electromagnetics import time_domain as tdem

from atem3d.config import build_simulation
from atem3d.hj import HJMagneticSimulation
from atem3d.ip import DebyeIPModel
from atem3d.sources import GroundedWireSource, StepOffWaveform


def test_no_ip_solver_matches_simpeg_e_form_for_grounded_line_source():
    mesh = TensorMesh([np.ones(3), np.ones(3), np.ones(3)], origin="CCC")
    config = {
        "mesh": {"hx": [1.0, 1.0, 1.0], "hy": [1.0, 1.0, 1.0], "hz": [1.0, 1.0, 1.0], "origin": "CCC"},
        "model": {"sigma_infinity": 0.1},
        "source": {
            "start": [-0.5, 0.0, 0.0],
            "end": [0.5, 0.0, 0.0],
            "current": 1.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        },
        "time_steps": [0.01, 0.01],
        "receivers": [],
    }
    ours = build_simulation(config).run()

    src = tdem.sources.LineCurrent(
        [],
        location=np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        current=1.0,
        waveform=tdem.sources.StepOffWaveform(off_time=0.0),
    )
    survey = tdem.Survey([src])
    simpeg_sim = tdem.Simulation3DElectricField(
        mesh,
        survey=survey,
        sigmaMap=maps.IdentityMap(nP=mesh.n_cells),
        time_steps=[0.01, 0.01],
    )
    fields = simpeg_sim.fields(np.full(mesh.n_cells, 0.1))
    simpeg_e = np.column_stack([fields[src, "e", i] for i in range(3)]).T

    np.testing.assert_allclose(ours.e, simpeg_e, rtol=1e-10, atol=1e-12)


def test_no_ip_hj_solver_matches_simpeg_h_form_with_empymod_current_orientation():
    mesh = TensorMesh([np.ones(3), np.ones(3), np.ones(3)], origin="CCC")
    source = GroundedWireSource(
        start=(-0.5, 0.0, 0.0),
        end=(0.5, 0.0, 0.0),
        current=1.0,
        waveform=StepOffWaveform(off_time=0.0),
    )
    ours = HJMagneticSimulation(
        mesh=mesh,
        ip_model=DebyeIPModel.no_ip(np.full(mesh.n_cells, 0.1)),
        time_steps=[0.01, 0.01],
        sources=[source],
    ).run()

    simpeg_source = tdem.sources.LineCurrent(
        [],
        location=source.locations,
        current=1.0,
        waveform=tdem.sources.StepOffWaveform(off_time=0.0),
    )
    survey = tdem.Survey([simpeg_source])
    simpeg_sim = tdem.Simulation3DMagneticField(
        mesh,
        survey=survey,
        sigmaMap=maps.IdentityMap(nP=mesh.n_cells),
        time_steps=[0.01, 0.01],
    )
    fields = simpeg_sim.fields(np.full(mesh.n_cells, 0.1))
    simpeg_h = np.column_stack([fields[simpeg_source, "h", i] for i in range(3)]).T
    simpeg_e = np.column_stack([fields[simpeg_source, "e", i] for i in range(3)]).T

    np.testing.assert_allclose(ours.h, -simpeg_h, rtol=1.0e-9, atol=1.0e-11)
    np.testing.assert_allclose(ours.e, -simpeg_e, rtol=1.0e-9, atol=1.0e-11)
