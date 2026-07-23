import csv
import json
from pathlib import Path

import numpy as np
import pytest

from atem3d.zhou2020_reference import (
    build_zhou_empymod_survey,
    run_reference_sweep,
)


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "benchmarks/sotem/zhou2020_grounded_wire.yaml"
PROVENANCE = ROOT / "benchmarks/sotem/zhou2020_parameter_provenance.json"


class SrcptsSensitiveEmpymod:
    __version__ = "test-double"

    def __init__(self):
        self.calls = []

    def bipole(self, **kwargs):
        self.calls.append(kwargs)
        srcpts = int(kwargs["srcpts"])
        component_scale = len(self.calls) % 3 + 1
        times = np.asarray(kwargs["freqtime"], dtype=float)
        convergence_term = 1.0 + 0.005 / srcpts**2
        return component_scale * convergence_term / (1.0 + times)


def test_build_zhou_empymod_survey_uses_canonical_layers_and_components():
    noip = build_zhou_empymod_survey(CASE, variant="noip")
    ip = build_zhou_empymod_survey(CASE, variant="ip")

    assert noip.source_start == (-500.0, 0.0, 0.0)
    assert noip.source_end == (500.0, 0.0, 0.0)
    assert noip.receiver_locations == [(0.0, 1000.0, 0.0)]
    assert noip.components == ("Ex", "Hz", "dBzdt")
    assert noip.depths == [0.0, 500.0, 520.0]
    assert noip.resistivities == [1.0e8, 100.0, 10.0, 200.0]
    assert noip.strength == 10.0
    assert noip.signal == -1

    assert isinstance(ip.resistivities, dict)
    eta_dc, _ = ip.resistivities["func_eta"](
        ip.resistivities, {"freq": np.array([0.0])}
    )
    np.testing.assert_allclose(eta_dc, [[1.0e-8, 0.01, 0.1, 0.005]])


def test_build_zhou_empymod_survey_applies_explicit_surface_offset():
    survey = build_zhou_empymod_survey(
        CASE,
        variant="noip",
        surface_offset_m=0.1,
    )

    assert survey.source_start == (-500.0, 0.0, 0.1)
    assert survey.source_end == (500.0, 0.0, 0.1)
    assert survey.receiver_locations == [(0.0, 1000.0, 0.1)]


@pytest.mark.parametrize("variant", ["bad", "", "IP"])
def test_build_zhou_empymod_survey_rejects_unknown_variant(variant):
    with pytest.raises(ValueError, match="variant"):
        build_zhou_empymod_survey(CASE, variant=variant)


def test_reference_sweep_writes_signed_canonical_evidence(tmp_path):
    backend = SrcptsSensitiveEmpymod()

    result = run_reference_sweep(
        case_path=CASE,
        provenance_path=PROVENANCE,
        output_dir=tmp_path,
        srcpts_values=(3, 5, 9, 17),
        surface_offsets_m=(0.0, 0.1),
        backend=backend,
    )

    assert result["status"] == "reference_verified"
    assert result["source_convergence"]["passed"] is True
    assert result["source_convergence"]["gate"] == pytest.approx(0.005)
    assert result["surface_offset_sensitivity"]["offsets_m"] == [0.0, 0.1]

    required = {
        "empymod_noip.csv",
        "empymod_ip.csv",
        "empymod_srcpts_convergence.json",
        "surface_offset_sensitivity.json",
        "empymod_metadata.json",
        "reference_manifest.json",
    }
    assert required <= {path.name for path in tmp_path.iterdir()}

    with (tmp_path / "empymod_ip.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert tuple(rows[0]) == (
        "time_s",
        "Ex_V_per_m",
        "Hz_A_per_m",
        "dBzdt_T_per_s",
    )
    assert len(rows) == 101
    assert all(np.isfinite(float(value)) for row in rows for value in row.values())

    manifest = json.loads(
        (tmp_path / "reference_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema"] == "atem3d.zhou2020.reference-manifest/v1"
    assert manifest["case_id"] == "zhou2020_grounded_wire"
    assert manifest["status"] == "reference_verified"
    assert set(manifest["file_sha256"]) == required - {"reference_manifest.json"}
    assert all(len(value) == 64 for value in manifest["file_sha256"].values())
    metadata = json.loads(
        (tmp_path / "empymod_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["schema"] == "atem3d.zhou2020.empymod-metadata/v2"
    assert metadata["component_conventions"]["dBzdt"] == {
        "empymod_receiver": "H",
        "empymod_signal": 0,
        "scale": "-mu0",
        "source_waveform": "ideal_step_off",
    }


def test_reference_sweep_fails_closed_when_source_convergence_exceeds_gate(
    tmp_path,
):
    class NonConvergentEmpymod(SrcptsSensitiveEmpymod):
        def bipole(self, **kwargs):
            self.calls.append(kwargs)
            times = np.asarray(kwargs["freqtime"], dtype=float)
            return float(kwargs["srcpts"]) / (1.0 + times)

    result = run_reference_sweep(
        case_path=CASE,
        provenance_path=PROVENANCE,
        output_dir=tmp_path,
        srcpts_values=(3, 5, 9, 17),
        surface_offsets_m=(0.0,),
        backend=NonConvergentEmpymod(),
    )

    assert result["status"] == "failed_with_reproducible_evidence"
    manifest = json.loads(
        (tmp_path / "reference_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed_with_reproducible_evidence"
