# COMSOL Uniform Halfspace Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a COMSOL CSV reference path that validates ATEM3D/SimPEG predictions against `Ex`, `Ey`, and `dBzdt` from `COMSOL/均匀半空间.mph` under the `z_down` coordinate convention.

**Architecture:** Keep COMSOL as an external diagnostic reference for the first version. Add a small `atem3d.comsol_validation` adapter for CSV/HDF5 table loading and a `validate-comsol-3comp` CLI command that delegates artifact generation to the existing three-component validation writer.

**Tech Stack:** Python 3.11, NumPy, h5py, PyYAML, pytest, existing `atem3d.validation_3comp` artifact writer.

---

## File Structure

- Modify `src/atem3d/validation_3comp.py`: register `comsol_uniform_halfspace` as a diagnostic reference type and allow callers to choose the reference CSV artifact filename while preserving the old default.
- Create `src/atem3d/comsol_validation.py`: focused readers and validators for COMSOL reference CSVs and ATEM3D prediction CSV/HDF5 files.
- Modify `src/atem3d/cli.py`: add `validate-comsol-3comp` command and route it to the adapter.
- Create `tests/test_comsol_validation.py`: unit tests for readers, validation failures, CLI output, and `z_down` metadata.
- Optional manual artifact after tests pass: run the new CLI on a synthetic CSV pair to inspect generated files. Do not run or modify the `.mph` model in this plan.

### Task 1: Register COMSOL As A Diagnostic Reference

**Files:**
- Modify: `src/atem3d/validation_3comp.py`
- Modify: `tests/test_validation_3comp_cli.py`

- [ ] **Step 1: Write the failing diagnostic reference test**

Append this test to `tests/test_validation_3comp_cli.py`:

```python
def test_comsol_uniform_halfspace_reference_is_diagnostic_not_final(tmp_path):
    times = np.array([1.0e-5, 1.0e-3, 1.0])
    reference = np.array(
        [
            [1.0, 0.1, 1.0e-9],
            [0.5, 0.05, 5.0e-10],
            [0.1, 0.01, 1.0e-10],
        ]
    )

    summary = write_three_component_validation_artifacts(
        ThreeComponentValidationInput(
            output_dir=tmp_path / "comsol",
            times=times,
            predictions=1.01 * reference,
            reference=reference,
            component_names=["Ex", "Ey", "dBzdt"],
            case_type="noip",
            reference_type="comsol_uniform_halfspace",
            magnetic_quantity="dBzdt",
            validation_scope="diagnostic_comsol_reference",
        )
    )

    assert summary["reference_type"] == "comsol_uniform_halfspace"
    assert summary["acceptance_status"]["reference_type_supported"] is True
    assert summary["acceptance_status"]["reference_type_ok"] is False
    assert "comsol_uniform_halfspace" in summary["acceptance_status"]["diagnostic_reference_types"]
    assert "reference_type_not_final_acceptance" in summary["acceptance_status"]["blocking_reasons"]
    assert summary["final_acceptance_passed"] is False
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```powershell
python -m pytest tests/test_validation_3comp_cli.py::test_comsol_uniform_halfspace_reference_is_diagnostic_not_final -q
```

Expected: FAIL with `ValueError: reference_type must be one of:` because `comsol_uniform_halfspace` is not yet registered.

- [ ] **Step 3: Register the diagnostic reference type**

Edit `src/atem3d/validation_3comp.py` near the existing `DIAGNOSTIC_REFERENCE_TYPES` definition:

```python
DIAGNOSTIC_REFERENCE_TYPES = {
    "dolfinx_refined",
    "self_convergence",
    "manufactured",
    "published_response_curve",
    "comsol_uniform_halfspace",
}
```

- [ ] **Step 4: Run the focused test and verify it passes**

Run:

```powershell
python -m pytest tests/test_validation_3comp_cli.py::test_comsol_uniform_halfspace_reference_is_diagnostic_not_final -q
```

Expected: PASS.

- [ ] **Step 5: Add configurable reference artifact filename test**

Append this test to `tests/test_validation_3comp_cli.py`:

```python
def test_three_component_writer_accepts_reference_output_name(tmp_path):
    times = np.array([1.0e-5, 1.0e-3, 1.0])
    reference = np.array(
        [
            [1.0, 0.1, 1.0e-9],
            [0.5, 0.05, 5.0e-10],
            [0.1, 0.01, 1.0e-10],
        ]
    )

    write_three_component_validation_artifacts(
        ThreeComponentValidationInput(
            output_dir=tmp_path / "comsol_named",
            times=times,
            predictions=reference.copy(),
            reference=reference,
            component_names=["Ex", "Ey", "dBzdt"],
            case_type="noip",
            reference_type="comsol_uniform_halfspace",
            magnetic_quantity="dBzdt",
            validation_scope="diagnostic_comsol_reference",
            reference_output_name="reference_comsol.csv",
        )
    )

    assert (tmp_path / "comsol_named" / "reference_comsol.csv").is_file()
    assert not (tmp_path / "comsol_named" / "reference_empymod_or_1d.csv").exists()
```

- [ ] **Step 6: Run the new test and verify it fails**

Run:

```powershell
python -m pytest tests/test_validation_3comp_cli.py::test_three_component_writer_accepts_reference_output_name -q
```

Expected: FAIL with `TypeError` because `ThreeComponentValidationInput` does not yet accept `reference_output_name`.

- [ ] **Step 7: Add the dataclass field and use it**

In `src/atem3d/validation_3comp.py`, add this field to `ThreeComponentValidationInput` after `validation_scope`:

```python
    reference_output_name: str = "reference_empymod_or_1d.csv"
```

Then replace the fixed reference output line in `write_three_component_validation_artifacts`:

```python
    _write_response_csv(output_dir / case.reference_output_name, times, reference, component_names)
```

Keep the existing `predictions.csv`, `errors.csv`, JSON, YAML, and PNG outputs unchanged.

- [ ] **Step 8: Run validation writer tests**

Run:

```powershell
python -m pytest tests/test_validation_3comp_cli.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit Task 1**

Run:

```powershell
git add src/atem3d/validation_3comp.py tests/test_validation_3comp_cli.py
git commit -m "feat: register COMSOL diagnostic validation reference"
```

Expected: commit succeeds and only these two files are staged.

### Task 2: Add COMSOL Validation Readers

**Files:**
- Create: `src/atem3d/comsol_validation.py`
- Create: `tests/test_comsol_validation.py`

- [ ] **Step 1: Write reader tests**

Create `tests/test_comsol_validation.py` with:

```python
from __future__ import annotations

import h5py
import numpy as np
import pytest
import yaml

from atem3d.comsol_validation import (
    COMSOL_COMPONENT_NAMES,
    read_comsol_reference_csv,
    read_prediction_table,
)


def _write_response_csv(path, *, times=None, scale=1.0, header=("time_obs", "Ex", "Ey", "dBzdt")):
    if times is None:
        times = np.array([1.0e-5, 1.0e-3, 1.0])
    values = np.array(
        [
            [1.0, 0.1, 1.0e-9],
            [0.5, 0.05, 5.0e-10],
            [0.1, 0.01, 1.0e-10],
        ],
        dtype=float,
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(header) + "\n")
        for time, row in zip(times, values * scale):
            payload = {"time_obs": time, "Ex": row[0], "Ey": row[1], "dBzdt": row[2]}
            handle.write(",".join(f"{float(payload[name]):.12g}" for name in header) + "\n")


def test_read_comsol_reference_csv_returns_ordered_components(tmp_path):
    path = tmp_path / "reference.csv"
    _write_response_csv(path, header=("Ey", "time_obs", "dBzdt", "Ex"))

    table = read_comsol_reference_csv(path)

    assert table.component_names == COMSOL_COMPONENT_NAMES
    assert table.times.tolist() == pytest.approx([1.0e-5, 1.0e-3, 1.0])
    assert table.values[:, 0].tolist() == pytest.approx([1.0, 0.5, 0.1])
    assert table.metadata["coordinate_system"] == "z_down"


def test_read_comsol_reference_csv_rejects_missing_component(tmp_path):
    path = tmp_path / "bad.csv"
    _write_response_csv(path, header=("time_obs", "Ex", "Ey"))

    with pytest.raises(ValueError, match="bad.csv.*missing columns.*dBzdt"):
        read_comsol_reference_csv(path)


def test_read_comsol_reference_csv_rejects_unsorted_times(tmp_path):
    path = tmp_path / "bad_times.csv"
    _write_response_csv(path, times=np.array([1.0e-5, 1.0e-5, 1.0]))

    with pytest.raises(ValueError, match="bad_times.csv.*strictly increasing"):
        read_comsol_reference_csv(path)


def test_read_prediction_table_reads_csv(tmp_path):
    path = tmp_path / "prediction.csv"
    _write_response_csv(path, scale=1.02)

    table = read_prediction_table(path, component_names=COMSOL_COMPONENT_NAMES)

    assert table.values.shape == (3, 3)
    assert table.component_names == COMSOL_COMPONENT_NAMES


def test_read_prediction_table_reads_hdf5_and_rejects_conflicting_coordinate_system(tmp_path):
    path = tmp_path / "prediction.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("times", data=np.array([1.0e-5, 1.0e-3, 1.0]))
        h5.create_dataset(
            "data",
            data=np.array(
                [
                    [1.0, 0.1, 1.0e-9],
                    [0.5, 0.05, 5.0e-10],
                    [0.1, 0.01, 1.0e-10],
                ]
            ),
        )
        h5.attrs["config_yaml"] = yaml.safe_dump({"coordinate_system": "z_up"})

    with pytest.raises(ValueError, match="prediction.h5.*coordinate_system.*z_up"):
        read_prediction_table(path, component_names=COMSOL_COMPONENT_NAMES)
```

- [ ] **Step 2: Run reader tests and verify import failure**

Run:

```powershell
python -m pytest tests/test_comsol_validation.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'atem3d.comsol_validation'`.

- [ ] **Step 3: Create the COMSOL validation adapter**

Create `src/atem3d/comsol_validation.py`:

```python
"""COMSOL reference adapters for three-component validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import yaml

COMSOL_COMPONENT_NAMES = ["Ex", "Ey", "dBzdt"]
SUPPORTED_COORDINATE_SYSTEM = "z_down"


@dataclass(frozen=True)
class ResponseTable:
    times: np.ndarray
    values: np.ndarray
    component_names: list[str]
    metadata: dict = field(default_factory=dict)


def read_comsol_reference_csv(path: str | Path) -> ResponseTable:
    """Read a COMSOL reference CSV in time_obs, Ex, Ey, dBzdt format."""

    return _read_csv_response(
        path,
        component_names=COMSOL_COMPONENT_NAMES,
        metadata={
            "source": "comsol",
            "reference_type": "comsol_uniform_halfspace",
            "coordinate_system": SUPPORTED_COORDINATE_SYSTEM,
        },
    )


def read_prediction_table(path: str | Path, *, component_names: Sequence[str]) -> ResponseTable:
    """Read an ATEM3D prediction table from CSV or HDF5."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv_response(
            path,
            component_names=component_names,
            metadata={"source": "prediction_csv", "coordinate_system": SUPPORTED_COORDINATE_SYSTEM},
        )
    if suffix in {".h5", ".hdf5"}:
        return _read_hdf5_response(path, component_names=component_names)
    raise ValueError(f"{path}: unsupported prediction format {path.suffix!r}; expected .csv, .h5, or .hdf5")


def assert_matching_times(prediction: ResponseTable, reference: ResponseTable) -> None:
    """Require exact time-node alignment between prediction and reference."""

    if prediction.times.shape != reference.times.shape or not np.allclose(
        prediction.times,
        reference.times,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("prediction and COMSOL reference time_obs columns must match exactly")


def _read_csv_response(
    path: str | Path,
    *,
    component_names: Sequence[str],
    metadata: dict,
) -> ResponseTable:
    path = Path(path)
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    if table.ndim == 0:
        table = np.asarray([table], dtype=table.dtype)
    names = list(table.dtype.names or [])
    required = ["time_obs", *[str(name) for name in component_names]]
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    times = np.asarray(table["time_obs"], dtype=float)
    values = np.column_stack([np.asarray(table[str(name)], dtype=float) for name in component_names])
    _validate_times(path, times)
    _validate_values(path, values)
    return ResponseTable(
        times=times,
        values=values,
        component_names=[str(name) for name in component_names],
        metadata=dict(metadata),
    )


def _read_hdf5_response(path: Path, *, component_names: Sequence[str]) -> ResponseTable:
    with h5py.File(path, "r") as h5:
        if "times" not in h5 or "data" not in h5:
            raise ValueError(f"{path}: HDF5 prediction must contain 'times' and 'data' datasets")
        times = np.asarray(h5["times"][:], dtype=float)
        values = np.asarray(h5["data"][:], dtype=float)
        metadata = _metadata_from_hdf5(h5)
    if values.ndim != 2 or values.shape != (times.size, len(component_names)):
        raise ValueError(
            f"{path}: HDF5 data shape {values.shape} does not match "
            f"(n_times={times.size}, n_components={len(component_names)})"
        )
    _validate_times(path, times)
    _validate_values(path, values)
    coordinate_system = str(metadata.get("coordinate_system", SUPPORTED_COORDINATE_SYSTEM))
    if coordinate_system != SUPPORTED_COORDINATE_SYSTEM:
        raise ValueError(
            f"{path}: coordinate_system {coordinate_system!r} conflicts with required "
            f"{SUPPORTED_COORDINATE_SYSTEM!r}"
        )
    metadata["coordinate_system"] = SUPPORTED_COORDINATE_SYSTEM
    metadata["source"] = "prediction_hdf5"
    return ResponseTable(
        times=times,
        values=values,
        component_names=[str(name) for name in component_names],
        metadata=metadata,
    )


def _metadata_from_hdf5(h5) -> dict:
    metadata: dict = {}
    raw = h5.attrs.get("config_yaml")
    if raw is not None:
        config = yaml.safe_load(raw)
        if isinstance(config, dict):
            metadata.update(config)
    return metadata


def _validate_times(path: Path, times: np.ndarray) -> None:
    if times.ndim != 1 or times.size == 0:
        raise ValueError(f"{path}: time_obs must be a non-empty 1D column")
    if not np.all(np.isfinite(times)):
        raise ValueError(f"{path}: time_obs contains NaN or infinite values")
    if np.any(times <= 0.0):
        raise ValueError(f"{path}: time_obs values must be positive")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError(f"{path}: time_obs values must be strictly increasing")


def _validate_values(path: Path, values: np.ndarray) -> None:
    if values.ndim != 2:
        raise ValueError(f"{path}: response values must be a 2D array")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: response values contain NaN or infinite values")
```

- [ ] **Step 4: Run reader tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_comsol_validation.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add src/atem3d/comsol_validation.py tests/test_comsol_validation.py
git commit -m "feat: add COMSOL validation table readers"
```

Expected: commit succeeds.

### Task 3: Add validate-comsol-3comp CLI

**Files:**
- Modify: `src/atem3d/cli.py`
- Modify: `tests/test_comsol_validation.py`

- [ ] **Step 1: Add CLI test**

Append this test to `tests/test_comsol_validation.py`:

```python
from atem3d import cli


def test_validate_comsol_3comp_cli_writes_artifacts(tmp_path):
    prediction = tmp_path / "prediction.csv"
    reference = tmp_path / "reference.csv"
    _write_response_csv(prediction, scale=1.01)
    _write_response_csv(reference, scale=1.0)
    output_dir = tmp_path / "artifacts"

    exit_code = cli.main(
        [
            "validate-comsol-3comp",
            "--prediction",
            str(prediction),
            "--reference",
            str(reference),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "predictions.csv").is_file()
    assert (output_dir / "reference_comsol.csv").is_file()
    assert (output_dir / "errors.csv").is_file()
    assert (output_dir / "error_summary.json").is_file()
    assert (output_dir / "diagnostics.json").is_file()
    assert (output_dir / "comparison_3comp.png").is_file()
    summary = yaml.safe_load((output_dir / "error_summary.json").read_text(encoding="utf-8"))
    assert summary["reference_type"] == "comsol_uniform_halfspace"
    assert summary["magnetic_quantity"] == "dBzdt"
    assert summary["validation_scope"] == "diagnostic_comsol_reference"
    resolved = yaml.safe_load((output_dir / "run_config_resolved.yaml").read_text(encoding="utf-8"))
    assert resolved["coordinate_system"] == "z_down"
    assert resolved["component_names"] == ["Ex", "Ey", "dBzdt"]
```

- [ ] **Step 2: Run CLI test and verify it fails**

Run:

```powershell
python -m pytest tests/test_comsol_validation.py::test_validate_comsol_3comp_cli_writes_artifacts -q
```

Expected: FAIL because `validate-comsol-3comp` is not routed and the CLI tries to treat it as a config path.

- [ ] **Step 3: Add command to help list and main router**

In `src/atem3d/cli.py`, add this item to `_TOP_LEVEL_COMMANDS`:

```python
    ("validate-comsol-3comp", "Compare Ex/Ey/dBzdt predictions against a COMSOL CSV reference."),
```

Add this branch in `main` before the `validate-noip-3comp` branch:

```python
    if argv and argv[0] == "validate-comsol-3comp":
        return _main_validate_comsol_3comp(argv[1:])
```

- [ ] **Step 4: Add the CLI implementation**

In `src/atem3d/cli.py`, add this function near `_main_validate`:

```python
def _main_validate_comsol_3comp(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare Ex/Ey/dBzdt predictions against a COMSOL uniform-halfspace CSV reference."
    )
    parser.add_argument("--prediction", type=Path, required=True, help="Prediction CSV/HDF5 file")
    parser.add_argument("--reference", type=Path, required=True, help="COMSOL reference CSV file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Validation artifact directory")
    parser.add_argument("--threshold", type=float, default=0.05, help="Pointwise relative-error threshold")
    parser.add_argument("--coordinate-system", default="z_down", choices=["z_down"])
    args = parser.parse_args(argv)

    from .comsol_validation import (
        COMSOL_COMPONENT_NAMES,
        SUPPORTED_COORDINATE_SYSTEM,
        assert_matching_times,
        read_comsol_reference_csv,
        read_prediction_table,
    )

    if args.coordinate_system != SUPPORTED_COORDINATE_SYSTEM:
        parser.error(f"--coordinate-system must be {SUPPORTED_COORDINATE_SYSTEM!r}")
    prediction = read_prediction_table(args.prediction, component_names=COMSOL_COMPONENT_NAMES)
    reference = read_comsol_reference_csv(args.reference)
    assert_matching_times(prediction, reference)
    diagnostics = {
        "reference_source": "comsol",
        "reference_model": "COMSOL/均匀半空间.mph",
        "reference_csv": str(args.reference),
        "prediction_path": str(args.prediction),
        "coordinate_system": SUPPORTED_COORDINATE_SYSTEM,
        "prediction_metadata": prediction.metadata,
        "reference_metadata": reference.metadata,
    }
    resolved_config = {
        "prediction": str(args.prediction),
        "reference": str(args.reference),
        "coordinate_system": SUPPORTED_COORDINATE_SYSTEM,
        "component_names": COMSOL_COMPONENT_NAMES,
    }
    summary = write_three_component_validation_artifacts(
        ThreeComponentValidationInput(
            output_dir=args.output_dir,
            times=prediction.times,
            predictions=prediction.values,
            reference=reference.values,
            component_names=COMSOL_COMPONENT_NAMES,
            case_type="noip",
            reference_type="comsol_uniform_halfspace",
            magnetic_quantity="dBzdt",
            threshold=float(args.threshold),
            diagnostics=diagnostics,
            resolved_config=resolved_config,
            validation_scope="diagnostic_comsol_reference",
            reference_output_name="reference_comsol.csv",
        )
    )
    print(f"wrote {args.output_dir}")
    print(f"reference_type: {summary['reference_type']}; pass_all_components: {summary['pass_all_components']}")
    return 0
```

- [ ] **Step 5: Run CLI test and verify it passes**

Run:

```powershell
python -m pytest tests/test_comsol_validation.py::test_validate_comsol_3comp_cli_writes_artifacts -q
```

Expected: PASS.

- [ ] **Step 6: Run related tests**

Run:

```powershell
python -m pytest tests/test_comsol_validation.py tests/test_validation_3comp_cli.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 3**

Run:

```powershell
git add src/atem3d/cli.py tests/test_comsol_validation.py
git commit -m "feat: add COMSOL three-component validation CLI"
```

Expected: commit succeeds.

### Task 4: Manual Smoke And Final Verification

**Files:**
- No new source files unless a small manual fixture is needed under `.codex_tmp`.

- [ ] **Step 1: Run full focused verification**

Run:

```powershell
python -m pytest tests/test_comsol_validation.py tests/test_validation_3comp_cli.py -q
```

Expected: PASS.

- [ ] **Step 2: Run CLI help**

Run:

```powershell
python -m atem3d.cli --help
```

Expected: command list includes `validate-comsol-3comp`.

- [ ] **Step 3: Create synthetic smoke CSVs outside tracked source**

Run:

```powershell
New-Item -ItemType Directory -Force -Path '.codex_tmp\comsol_smoke' | Out-Null
@'
time_obs,Ex,Ey,dBzdt
1e-5,1.01,0.101,1.01e-9
1e-3,0.505,0.0505,5.05e-10
1,0.101,0.0101,1.01e-10
'@ | Set-Content -LiteralPath '.codex_tmp\comsol_smoke\prediction.csv' -Encoding UTF8
@'
time_obs,Ex,Ey,dBzdt
1e-5,1,0.1,1e-9
1e-3,0.5,0.05,5e-10
1,0.1,0.01,1e-10
'@ | Set-Content -LiteralPath '.codex_tmp\comsol_smoke\reference.csv' -Encoding UTF8
```

Expected: two CSV files exist under `.codex_tmp\comsol_smoke`.

- [ ] **Step 4: Run synthetic CLI smoke**

Run:

```powershell
python -m atem3d.cli validate-comsol-3comp `
  --prediction '.codex_tmp\comsol_smoke\prediction.csv' `
  --reference '.codex_tmp\comsol_smoke\reference.csv' `
  --output-dir '.codex_tmp\comsol_smoke\artifacts'
```

Expected: output says `reference_type: comsol_uniform_halfspace; pass_all_components: True`, and `.codex_tmp\comsol_smoke\artifacts\reference_comsol.csv` exists.

- [ ] **Step 5: Inspect git status**

Run:

```powershell
git status --short
```

Expected: only intentional source/test changes are committed. `.codex_tmp` may remain untracked and should not be staged.

- [ ] **Step 6: Final commit if any verification-only adjustments were needed**

If Task 4 required small source or test fixes, commit them:

```powershell
git add src/atem3d tests
git commit -m "test: verify COMSOL validation workflow"
```

Expected: commit succeeds only if there are actual tracked changes.

## Self-Review

Spec coverage:

- COMSOL CSV reference path: Task 2 and Task 3.
- `Ex`, `Ey`, `dBzdt` component contract: Task 2 tests and adapter constants.
- `z_down` coordinate convention: Task 2 HDF5 rejection test and Task 3 resolved metadata.
- Diagnostic `comsol_uniform_halfspace` reference type: Task 1.
- `reference_comsol.csv` artifact: Task 1 configurable output name and Task 3 CLI test.
- No LiveLink or `.mph` mutation: Task 4 manual smoke uses synthetic CSVs only.

Placeholder scan: no banned placeholder markers are present.

Type consistency:

- `ResponseTable.times`, `ResponseTable.values`, and `ResponseTable.component_names` are used consistently by the CLI.
- `reference_output_name` is added to `ThreeComponentValidationInput` with a default, so existing callers keep working.
- CLI uses `relative_error_threshold` semantics through the existing `threshold` field.
