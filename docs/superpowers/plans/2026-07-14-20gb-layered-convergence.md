# 20 GB Layered Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the paper-baseline FEniCSx convergence study reproducibly executable on a 20 GB workstation by using a fixed 6/12/18 km domain axis and an auditable 20/6/14 GB memory contract.

**Architecture:** Add one immutable resource-contract value object to the existing convergence module, thread it through the stage-two command builder, preflight, evaluator, and report, and enforce a live fail-closed resource gate in the Windows runner. Preserve the general solver, legacy stage-one behavior, physical model, estimator, checkpoints, and scientific thresholds.

**Tech Stack:** Python 3.12, dataclasses, NumPy, meshio, pytest, Windows process/memory APIs, FEniCSx through WSL, JSON evidence artifacts.

---

## File Map

- Modify `src/atem3d/layered_convergence.py`: define and validate the memory contract, replace the stage-two 24 km level with 18 km, allow the pipeline memory limit to be supplied, and carry the contract into reports.
- Modify `dolfinx/run_layered_convergence_study.py`: expose CLI memory values, write/require matching preflight evidence, collect the live host snapshot, and reject concurrent COMSOL or insufficient available memory.
- Modify `dolfinx/audit_layered_convergence.py`: independently verify the stored resource contract and the generated baseline/large-run preflights.
- Modify `tests/test_layered_convergence.py`: cover level isolation, contract validation and propagation, stale-preflight rejection, live resource gates, report output, and independent audit.
- Preserve `output/publication_validation/convergence/layered_resistive_offset100_stage2/runs/domain_large_24km_dt005_mesh8_6`: it is pre-solve evidence explaining why the original design was rejected, not an input to the revised evaluator.

### Task 1: Define The Resource Contract And Revised Domain

**Files:**
- Modify: `src/atem3d/layered_convergence.py:22-45`
- Modify: `src/atem3d/layered_convergence.py:133-234`
- Test: `tests/test_layered_convergence.py:14-115`

- [ ] **Step 1: Write failing tests for the 18 km level and memory contract**

Add `PublicationMemoryContract` to the imports in `tests/test_layered_convergence.py`, change the expected large domain and run id, and add:

```python
def test_publication_memory_contract_defaults_to_20_6_14():
    contract = layered_convergence.PublicationMemoryContract()

    assert contract.total_memory_gb == 20.0
    assert contract.reserve_memory_gb == 6.0
    assert contract.solver_memory_limit_gb == 14.0
    assert contract.as_dict() == {
        "total_memory_gb": 20.0,
        "reserve_memory_gb": 6.0,
        "solver_memory_limit_gb": 14.0,
    }


@pytest.mark.parametrize(
    ("total", "reserve"),
    [
        (float("nan"), 6.0),
        (20.0, float("inf")),
        (0.0, 0.0),
        (20.0, -1.0),
        (20.0, 20.0),
        (20.0, 21.0),
    ],
)
def test_publication_memory_contract_rejects_invalid_values(total, reserve):
    with pytest.raises(ValueError, match="memory contract"):
        layered_convergence.PublicationMemoryContract(total, reserve)
```

Change the domain expectation to:

```python
    assert [
        (level.level_id, level.x_extent, level.earth_depth, level.air_height)
        for level in levels["domain"]
    ] == [
        ("small", 6000.0, 6000.0, 600.0),
        ("standard", 12000.0, 12000.0, 1200.0),
        ("large", 18000.0, 18000.0, 1800.0),
    ]
```

Replace every stage-two fixture occurrence of
`domain_large_24km_dt005_mesh8_6` with
`domain_large_18km_dt005_mesh8_6`.

- [ ] **Step 2: Run the focused tests and verify the intended failure**

Run:

```powershell
python -m pytest tests/test_layered_convergence.py -q
```

Expected: failures report the missing `PublicationMemoryContract` and the
still-configured 24 km level.

- [ ] **Step 3: Implement the immutable contract and 18 km level**

Add below `ConvergenceResponse` in `src/atem3d/layered_convergence.py`:

```python
@dataclass(frozen=True)
class PublicationMemoryContract:
    total_memory_gb: float = 20.0
    reserve_memory_gb: float = 6.0

    def __post_init__(self) -> None:
        total = float(self.total_memory_gb)
        reserve = float(self.reserve_memory_gb)
        if (
            not math.isfinite(total)
            or total <= 0.0
            or not math.isfinite(reserve)
            or reserve < 0.0
            or reserve >= total
        ):
            raise ValueError(
                "memory contract requires finite total > reserve >= 0"
            )

    @property
    def solver_memory_limit_gb(self) -> float:
        return float(self.total_memory_gb) - float(self.reserve_memory_gb)

    def as_dict(self) -> dict[str, float]:
        return {
            "total_memory_gb": float(self.total_memory_gb),
            "reserve_memory_gb": float(self.reserve_memory_gb),
            "solver_memory_limit_gb": self.solver_memory_limit_gb,
        }
```

Replace the paper-baseline large level with:

```python
            level(
                "domain",
                "large",
                "domain_large_18km_dt005_mesh8_6",
                x_extent=18000.0,
                y_extent=18000.0,
                earth_depth=18000.0,
                air_height=1800.0,
            ),
```

- [ ] **Step 4: Run the focused tests**

Run:

```powershell
python -m pytest tests/test_layered_convergence.py -q
```

Expected: the new contract and level tests pass; later tests may still fail
because they expect a 24 GB command and preflight.

- [ ] **Step 5: Commit the level and contract**

```powershell
git add src/atem3d/layered_convergence.py tests/test_layered_convergence.py
git commit -m "feat: define 20gb layered convergence contract"
```

### Task 2: Propagate The Contract Through Commands And Preflight

**Files:**
- Modify: `src/atem3d/layered_convergence.py:249-320`
- Modify: `dolfinx/run_layered_convergence_study.py:21-169`
- Modify: `dolfinx/run_layered_convergence_study.py:210-421`
- Test: `tests/test_layered_convergence.py:100-214`
- Test: `tests/test_layered_convergence.py:897-1025`

- [ ] **Step 1: Write failing propagation and stale-evidence tests**

Change the direct command test to construct the default contract and call:

```python
    contract = layered_convergence.PublicationMemoryContract()
    arguments = build_pipeline_command_arguments(
        baseline,
        memory_limit_gb=contract.solver_memory_limit_gb,
    )
    assert _option_value(arguments, "--memory-limit-gb") == "14"
```

Extend `test_stage_two_mesh_mode_writes_strict_preflight_from_source_diagnostics`
with:

```python
    assert preflight["total_memory_gb"] == 20.0
    assert preflight["reserve_memory_gb"] == 6.0
    assert preflight["solver_memory_limit_gb"] == 14.0
    assert preflight["memory_limit_gb"] == 14.0
```

Change the over-budget parameterized case to
`estimated_memory_gb=14.1`, expected message `14 GB`, and
`memory_limit_gb=14.0`. Change the complete-evidence acceptance case to use an
estimate exactly equal to the limit and assert equality is accepted:

```python
    result = layered_convergence.validate_publication_preflight(
        mesh_path=mesh,
        diagnostics={
            "estimated_memory_gb": 14.0,
            "source_coverage_passed": True,
            "receiver_found": True,
            "source_divergence_passed": True,
        },
        memory_limit_gb=14.0,
    )
    assert result["estimated_memory_gb"] == 14.0
    assert result["memory_limit_gb"] == 14.0
```

Add a custom CLI propagation test:

```python
def test_stage_two_custom_memory_contract_reaches_command(tmp_path):
    output_root = tmp_path / "stage2"
    _run_study_cli(
        "--study", "paper-baseline",
        "--output-root", str(output_root),
        "--layered-root", str(tmp_path / "layered"),
        "--prior-convergence-root", str(tmp_path / "stage1"),
        "--axis", "mesh",
        "--level", "coarse",
        "--mode", "full",
        "--total-memory-gb", "19",
        "--memory-reserve-gb", "5",
        "--dry-run",
    )

    command = (
        output_root
        / "runs"
        / "mesh_coarse_12km_dt005_mesh12_9"
        / "command.txt"
    ).read_text(encoding="utf-8")
    assert "--memory-limit-gb 14" in command
```

Add this helper-level stale contract test. Add `import importlib.util` near the
test imports.

```python
def _load_convergence_runner_module():
    path = Path("dolfinx/run_layered_convergence_study.py").resolve()
    spec = importlib.util.spec_from_file_location("convergence_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_rejects_preflight_from_a_different_memory_contract(tmp_path):
    runner = _load_convergence_runner_module()
    level = layered_convergence.build_paper_baseline_convergence_levels(
        tmp_path / "layered",
        tmp_path / "stage2",
        tmp_path / "stage1",
    )["mesh"][0]
    level.workdir.mkdir(parents=True)
    mesh = level.workdir / "verification_mesh.msh"
    mesh.write_bytes(b"mesh")
    (level.workdir / "preflight.json").write_text(
        json.dumps(
            {
                "passed": True,
                "mesh_sha256": hashlib.sha256(b"mesh").hexdigest(),
                "estimated_memory_gb": 10.0,
                "memory_limit_gb": 18.0,
                "total_memory_gb": 24.0,
                "reserve_memory_gb": 6.0,
                "solver_memory_limit_gb": 18.0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="memory contract changed"):
        runner._require_publication_preflight(
            level,
            layered_convergence.PublicationMemoryContract(),
        )
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_layered_convergence.py -q
```

Expected: failures show that `build_pipeline_command_arguments` has no
`memory_limit_gb` parameter, the CLI options are unknown, and preflight lacks
the three contract fields.

- [ ] **Step 3: Add optional command memory-limit replacement**

Change the command builder signature and body in
`src/atem3d/layered_convergence.py`:

```python
def build_pipeline_command_arguments(
    level: ConvergenceLevel,
    *,
    memory_limit_gb: float | None = None,
) -> list[str]:
    case = build_layered_cases(
        offsets=(100.0,),
        basement_resistivities=(1000.0,),
    )[0]
    profile = LayeredRunProfile(
        profile_id=f"convergence_{level.axis}_{level.level_id}",
        x_extent=level.x_extent,
        y_extent=level.y_extent,
        air_height=level.air_height,
        earth_depth=level.earth_depth,
        far_field_mesh_size=level.far_field_mesh_size,
        max_internal_dt=level.max_internal_dt,
    )
    arguments = build_pipeline_arguments(case, profile, level.workdir)
    _replace_option(arguments, "--source-mesh-size", _number(level.source_mesh_size))
    _replace_option(
        arguments,
        "--receiver-mesh-size",
        _number(level.receiver_mesh_size),
    )
    _replace_option(
        arguments,
        "--max-internal-dt-fraction",
        _number(level.max_internal_dt_fraction),
    )
    if memory_limit_gb is not None:
        value = float(memory_limit_gb)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("memory limit must be finite and positive")
        _replace_option(arguments, "--memory-limit-gb", _number(value))
    arguments.extend(("--stop-after-outputs", "25"))
    if level.reuse_mesh_path is not None:
        arguments.extend(("--reuse-mesh", str(level.reuse_mesh_path)))
    return arguments
```

- [ ] **Step 4: Thread the contract through the runner**

Import `math` and `PublicationMemoryContract`. Add parser options:

```python
    parser.add_argument("--total-memory-gb", type=float, default=20.0)
    parser.add_argument("--memory-reserve-gb", type=float, default=6.0)
```

After parsing, resolve a contract only for `paper-baseline`:

```python
    memory_contract = (
        PublicationMemoryContract(
            args.total_memory_gb,
            args.memory_reserve_gb,
        )
        if args.study == "paper-baseline"
        else None
    )
```

Replace the preflight helpers with contract-aware versions. Keep the existing
locked-mesh block inside `_write_publication_preflight` between validation and
payload construction.

```python
def _write_publication_preflight(
    level,
    memory_contract: PublicationMemoryContract,
) -> dict:
    diagnostics = _publication_preflight_diagnostics(level.workdir)
    evidence = validate_publication_preflight(
        mesh_path=level.workdir / "verification_mesh.msh",
        diagnostics=diagnostics,
        memory_limit_gb=memory_contract.solver_memory_limit_gb,
    )
    if level.reuse_mesh_path is not None:
        locked_mesh_path = Path(level.reuse_mesh_path)
        if not locked_mesh_path.is_file():
            raise FileNotFoundError(f"locked mesh is missing: {locked_mesh_path}")
        locked_sha256 = sha256_file(locked_mesh_path)
        if evidence["mesh_sha256"] != locked_sha256:
            raise ValueError(
                "locked_mesh_hash_mismatch: "
                f"run={evidence['mesh_sha256']}, locked={locked_sha256}"
            )
        evidence["locked_mesh_path"] = str(locked_mesh_path)
        evidence["locked_mesh_sha256"] = locked_sha256
    payload = {
        "run_id": level.run_id,
        **diagnostics,
        **evidence,
        **memory_contract.as_dict(),
    }
    _write_json(level.workdir / "preflight.json", payload)
    return payload


def _require_publication_preflight(
    level,
    memory_contract: PublicationMemoryContract,
) -> dict:
    preflight_path = level.workdir / "preflight.json"
    preflight = _read_json(preflight_path)
    if not isinstance(preflight, dict) or not preflight.get("passed", False):
        raise ValueError(f"passing publication preflight is required: {preflight_path}")
    expected = memory_contract.as_dict()
    for key, expected_value in expected.items():
        try:
            actual_value = float(preflight[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("publication preflight memory contract changed") from exc
        if not math.isclose(actual_value, expected_value, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("publication preflight memory contract changed")
    if not math.isclose(
        float(preflight.get("memory_limit_gb", math.nan)),
        memory_contract.solver_memory_limit_gb,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("publication preflight memory contract changed")
    mesh_path = level.workdir / "verification_mesh.msh"
    if sha256_file(mesh_path) != preflight.get("mesh_sha256"):
        raise ValueError(f"publication preflight mesh hash changed: {mesh_path}")
    if level.reuse_mesh_path is not None:
        locked_sha256 = sha256_file(Path(level.reuse_mesh_path))
        if locked_sha256 != preflight.get("locked_mesh_sha256"):
            raise ValueError("locked_mesh_hash_mismatch")
    return preflight
```

Build paper-baseline commands with:

```python
            pipeline_arguments = build_pipeline_command_arguments(
                level,
                memory_limit_gb=(
                    memory_contract.solver_memory_limit_gb
                    if memory_contract is not None
                    else None
                ),
            )
```

Pass the same contract to preflight writing and requiring. Leave stage-one
commands at their existing builder default.

- [ ] **Step 5: Run the focused suite**

Run:

```powershell
python -m pytest tests/test_layered_convergence.py -q
```

Expected: all command, preflight, resume, and level tests pass.

- [ ] **Step 6: Commit contract propagation**

```powershell
git add src/atem3d/layered_convergence.py dolfinx/run_layered_convergence_study.py tests/test_layered_convergence.py
git commit -m "feat: enforce convergence memory preflight"
```

### Task 3: Add The Live Fail-Closed Resource Gate

**Files:**
- Modify: `src/atem3d/layered_convergence.py:249-286`
- Modify: `dolfinx/run_layered_convergence_study.py:1-32`
- Modify: `dolfinx/run_layered_convergence_study.py:399-413`
- Test: `tests/test_layered_convergence.py`

- [ ] **Step 1: Write failing pure-gate tests**

Add:

```python
def test_live_resource_gate_passes_exact_available_memory():
    result = layered_convergence.evaluate_publication_live_resources(
        estimated_memory_gb=13.25,
        available_memory_gb=13.25,
        comsol_processes=[],
    )
    assert result["passed"] is True
    assert result["blocking_reasons"] == []


@pytest.mark.parametrize(
    ("available", "processes", "reason"),
    [
        (13.24, [], "insufficient_available_memory"),
        (20.0, ["comsolmphserver.exe"], "comsol_process_running"),
        (float("nan"), [], "invalid_available_memory"),
    ],
)
def test_live_resource_gate_rejects_unsafe_launch(available, processes, reason):
    result = layered_convergence.evaluate_publication_live_resources(
        estimated_memory_gb=13.25,
        available_memory_gb=available,
        comsol_processes=processes,
    )
    assert result["passed"] is False
    assert reason in result["blocking_reasons"]
```

- [ ] **Step 2: Run the gate tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_layered_convergence.py -q
```

Expected: failure reports missing
`evaluate_publication_live_resources`.

- [ ] **Step 3: Implement the pure gate**

Add to `src/atem3d/layered_convergence.py`:

```python
def evaluate_publication_live_resources(
    *,
    estimated_memory_gb: float,
    available_memory_gb: float,
    comsol_processes: list[str] | tuple[str, ...],
) -> dict:
    estimate = float(estimated_memory_gb)
    available = float(available_memory_gb)
    processes = sorted({str(value) for value in comsol_processes})
    reasons: list[str] = []
    if not math.isfinite(estimate) or estimate <= 0.0:
        reasons.append("invalid_estimated_memory")
    if not math.isfinite(available) or available < 0.0:
        reasons.append("invalid_available_memory")
    elif math.isfinite(estimate) and available < estimate:
        reasons.append("insufficient_available_memory")
    if processes:
        reasons.append("comsol_process_running")
    return {
        "passed": not reasons,
        "estimated_memory_gb": estimate,
        "available_memory_gb": available,
        "comsol_processes": processes,
        "blocking_reasons": reasons,
    }
```

- [ ] **Step 4: Collect and persist the host snapshot before a real solve**

In `dolfinx/run_layered_convergence_study.py`, add Windows/Linux helpers that:

```python
def _available_physical_memory_gb() -> float:
    if os.name == "nt":
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return float(status.ullAvailPhys) / (1024.0**3)
    pages = os.sysconf("SC_AVPHYS_PAGES")
    page_size = os.sysconf("SC_PAGE_SIZE")
    return float(pages * page_size) / (1024.0**3)


def _comsol_process_names() -> list[str]:
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/fo", "csv", "/nh"],
            check=True,
            capture_output=True,
            text=True,
            encoding="mbcs",
        )
        import csv
        import io

        names = [row[0] for row in csv.reader(io.StringIO(result.stdout)) if row]
    else:
        result = subprocess.run(
            ["ps", "-eo", "comm="],
            check=True,
            capture_output=True,
            text=True,
        )
        names = result.stdout.splitlines()
    return sorted({name.strip() for name in names if "comsol" in name.lower()})
```

Store the return value from `_require_publication_preflight` in `preflight`, set
`preflight = None` before the mode branch, and add the following immediately
before `subprocess.run(command, ...)`. Import `datetime` and `timezone` from
`datetime`, and import `evaluate_publication_live_resources` from the convergence
module.

```python
            starts_forward_solve = (
                args.study == "paper-baseline"
                and args.mode == "full"
                and not postprocess_partial
            )
            if not args.dry_run and starts_forward_solve:
                if preflight is None:
                    raise AssertionError("paper-baseline solve requires preflight")
                live_check = evaluate_publication_live_resources(
                    estimated_memory_gb=float(preflight["estimated_memory_gb"]),
                    available_memory_gb=_available_physical_memory_gb(),
                    comsol_processes=_comsol_process_names(),
                )
                live_check["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
                _write_json(
                    level.workdir / "live_resource_check.json",
                    live_check,
                )
                if not live_check["passed"]:
                    raise ValueError(
                        "publication live resource gate failed: "
                        + ", ".join(live_check["blocking_reasons"])
                    )
```

Skip this gate for `--dry-run`, legacy stage one, existing levels, and
postprocessing-only commands that do not start FEniCSx time stepping.

- [ ] **Step 5: Run focused tests and a dry-run smoke command**

Run:

```powershell
python -m pytest tests/test_layered_convergence.py -q
python dolfinx/run_layered_convergence_study.py --study paper-baseline --output-root output/publication_validation/convergence/layered_resistive_offset100_stage2 --axis mesh --level coarse --mode full --dry-run
```

Expected: tests pass; the dry-run command contains
`--memory-limit-gb 14` and does not require a live snapshot.

- [ ] **Step 6: Commit the live gate**

```powershell
git add src/atem3d/layered_convergence.py dolfinx/run_layered_convergence_study.py tests/test_layered_convergence.py
git commit -m "feat: guard convergence solver resources"
```

### Task 4: Put The Contract In Reports And Independent Audit

**Files:**
- Modify: `src/atem3d/layered_convergence.py:689-767`
- Modify: `src/atem3d/layered_convergence.py:840-994`
- Modify: `dolfinx/run_layered_convergence_study.py:274-287`
- Modify: `dolfinx/audit_layered_convergence.py:1-170`
- Test: `tests/test_layered_convergence.py:677-800`

- [ ] **Step 1: Write failing report and audit assertions**

Pass `PublicationMemoryContract()` to synthetic stage-two evaluations through a
new `resource_contract` keyword. Assert:

```python
    assert summary["resource_contract"] == {
        "total_memory_gb": 20.0,
        "reserve_memory_gb": 6.0,
        "solver_memory_limit_gb": 14.0,
    }
```

Extend the baseline acceptance test with:

```python
    assert acceptance["resource_contract"]["solver_memory_limit_gb"] == 14.0
```

Have `_write_complete_convergence_run` create deterministic resource evidence:

```python
    (run_dir / "preflight.json").write_text(
        json.dumps(
            {
                "passed": True,
                "estimated_memory_gb": 1.0,
                "memory_limit_gb": 14.0,
                "total_memory_gb": 20.0,
                "reserve_memory_gb": 6.0,
                "solver_memory_limit_gb": 14.0,
            }
        ),
        encoding="utf-8",
    )
```

Extend the audit test with:

```python
    assert audit["resource_contract_verified"] is True
    assert audit["resource_preflight_count"] == 2
```

- [ ] **Step 2: Run the report/audit tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_layered_convergence.py -q
```

Expected: failures report the unknown evaluator keyword and missing audit
resource fields.

- [ ] **Step 3: Add the optional contract to evaluator and reports**

Extend `evaluate_convergence_study` with:

```python
    resource_contract: PublicationMemoryContract | None = None,
```

After constructing `result`, add:

```python
    if resource_contract is not None:
        result["resource_contract"] = resource_contract.as_dict()
```

When constructing `acceptance`, add:

```python
            "resource_contract": summary.get("resource_contract"),
```

Before the report table, append:

```python
    resource_contract = summary.get("resource_contract")
    if isinstance(resource_contract, dict):
        markdown[4:4] = [
            f"- Total memory contract: {resource_contract['total_memory_gb']:g} GB",
            f"- Operating-system reserve: {resource_contract['reserve_memory_gb']:g} GB",
            f"- Solver memory limit: {resource_contract['solver_memory_limit_gb']:g} GB",
        ]
```

In runner evaluate mode, call:

```python
        summary = evaluate_convergence_study(
            levels,
            selected_axes=selected_axes,
            study_id=(
                "layered_resistive_offset100_stage2"
                if args.study == "paper-baseline"
                else "layered_resistive_offset100"
            ),
            resource_contract=memory_contract,
        )
```

- [ ] **Step 4: Independently audit the two generated preflights**

Add this audit helper:

```python
def _audit_resource_contract(summary: dict, axes: dict[str, dict]) -> dict:
    contract = summary.get("resource_contract")
    if not isinstance(contract, dict):
        raise AssertionError("summary resource contract missing")
    keys = (
        "total_memory_gb",
        "reserve_memory_gb",
        "solver_memory_limit_gb",
    )
    expected = {key: float(contract[key]) for key in keys}
    if not all(math.isfinite(value) for value in expected.values()):
        raise AssertionError("summary resource contract is nonfinite")
    baseline_dir = Path(summary["candidate_baseline"]["run_dir"])
    large_dir = next(
        Path(level["run_dir"])
        for level in axes["domain"]["levels"]
        if level["level_id"] == "large"
    )
    checked = []
    for run_dir in (baseline_dir, large_dir):
        path = run_dir / "preflight.json"
        preflight = json.loads(path.read_text(encoding="utf-8"))
        if preflight.get("passed") is not True:
            raise AssertionError(f"resource preflight did not pass: {path}")
        for key, expected_value in expected.items():
            if float(preflight.get(key, math.nan)) != expected_value:
                raise AssertionError(f"resource contract mismatch: {path} {key}")
        estimate = float(preflight.get("estimated_memory_gb", math.nan))
        if not math.isfinite(estimate) or estimate > expected["solver_memory_limit_gb"]:
            raise AssertionError(f"resource estimate exceeds contract: {path}")
        checked.append(str(path))
    return {
        "resource_contract_verified": True,
        "resource_preflight_count": len(checked),
        "resource_preflights": checked,
    }
```

Call it after constructing `axes` and merge it into the returned audit object:

```python
        "resource_contract_verified": True,
        "resource_preflight_count": 2,
```

Missing, stale, nonfinite, or over-budget evidence raises `AssertionError` via
the code above.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
python -m pytest tests/test_layered_convergence.py -q
```

Expected: all convergence, report, and audit tests pass.

- [ ] **Step 6: Commit reporting and audit**

```powershell
git add src/atem3d/layered_convergence.py dolfinx/run_layered_convergence_study.py dolfinx/audit_layered_convergence.py tests/test_layered_convergence.py
git commit -m "feat: audit convergence memory evidence"
```

### Task 5: Verify Code And Migrate Existing Stage-Two Preflights

**Files:**
- Verify: `src/atem3d/layered_convergence.py`
- Verify: `dolfinx/run_layered_convergence_study.py`
- Verify: `dolfinx/audit_layered_convergence.py`
- Verify: `tests/test_layered_convergence.py`
- Regenerate: `output/publication_validation/convergence/layered_resistive_offset100_stage2/runs/*/preflight.json`

- [ ] **Step 1: Run syntax, focused, and related regression suites**

```powershell
python -m py_compile src/atem3d/layered_convergence.py dolfinx/run_layered_convergence_study.py dolfinx/audit_layered_convergence.py
python -m pytest tests/test_layered_convergence.py tests/test_layered_publication_validation.py tests/test_dolfinx_partial_forward.py -q
```

Expected: compilation succeeds and all selected tests pass.

- [ ] **Step 2: Confirm only intended source files changed**

```powershell
git status --short
git diff --check
git diff -- src/atem3d/layered_convergence.py dolfinx/run_layered_convergence_study.py dolfinx/audit_layered_convergence.py tests/test_layered_convergence.py
```

Expected: no whitespace errors; unrelated dirty-worktree files remain untouched.

- [ ] **Step 3: Rewrite preflight evidence for completed generated meshes**

After all COMSOL processes exit, run:

```powershell
python dolfinx/run_layered_convergence_study.py --study paper-baseline --output-root output/publication_validation/convergence/layered_resistive_offset100_stage2 --axis time --level standard --level fine --mode mesh --dry-run
python dolfinx/run_layered_convergence_study.py --study paper-baseline --output-root output/publication_validation/convergence/layered_resistive_offset100_stage2 --axis mesh --level coarse --level fine --mode mesh --dry-run
```

Expected: each existing generated mesh receives a passing 20/6/14 GB
`preflight.json`; no mesh or forward field is regenerated.

- [ ] **Step 4: Verify migrated evidence from disk**

```powershell
python -c "import json,pathlib; root=pathlib.Path(r'output/publication_validation/convergence/layered_resistive_offset100_stage2/runs'); ids=['baseline_12km_dt005_mesh8_6','time_fine_12km_dt0025_mesh8_6','mesh_coarse_12km_dt005_mesh12_9','mesh_fine_12km_dt005_mesh6_4p5']; rows=[json.loads((root/i/'preflight.json').read_text()) for i in ids]; assert all(r['passed'] and r['total_memory_gb']==20 and r['reserve_memory_gb']==6 and r['solver_memory_limit_gb']==14 and r['estimated_memory_gb']<=14 for r in rows); print('PREFLIGHT_MIGRATION_OK')"
```

Expected: `PREFLIGHT_MIGRATION_OK`.

### Task 6: Generate And Decide The 18 km Preflight

**Files:**
- Generate: `output/publication_validation/convergence/layered_resistive_offset100_stage2/runs/domain_large_18km_dt005_mesh8_6/verification_mesh.msh`
- Generate: `output/publication_validation/convergence/layered_resistive_offset100_stage2/runs/domain_large_18km_dt005_mesh8_6/preflight.json`

- [ ] **Step 1: Confirm the host is idle enough for mesh generation**

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'comsol' } | Select-Object ProcessId,Name,CommandLine
Get-CimInstance Win32_OperatingSystem | Select-Object @{Name='AvailableGB';Expression={[math]::Round($_.FreePhysicalMemory/1MB,2)}}
```

Expected: no COMSOL process and available memory comfortably exceeds the
predicted 13.26 GB mesh/solve estimate.

- [ ] **Step 2: Generate only the 18 km mesh and strict preflight**

```powershell
python dolfinx/run_layered_convergence_study.py --study paper-baseline --output-root output/publication_validation/convergence/layered_resistive_offset100_stage2 --axis domain --level large --mode mesh
```

Expected: source-only generation completes and writes `preflight.json`; no
`verification_data.npz` is created.

- [ ] **Step 3: Enforce the predeclared decision rule**

```powershell
python -c "import json,pathlib; p=pathlib.Path(r'output/publication_validation/convergence/layered_resistive_offset100_stage2/runs/domain_large_18km_dt005_mesh8_6/preflight.json'); d=json.loads(p.read_text()); print(json.dumps(d,indent=2)); assert d['passed'] is True; assert d['estimated_memory_gb']<=14.0; assert d['total_memory_gb']==20.0; assert d['reserve_memory_gb']==6.0; assert d['solver_memory_limit_gb']==14.0"
```

Expected: assertions pass. If they fail, stop before forward modeling, preserve
the mesh, and begin a new resource-only design amendment without examining an
18 km field response.

### Task 7: Resume Formal Solves Serially And Audit The Study

**Files:**
- Resume: `output/publication_validation/convergence/layered_resistive_offset100_stage2/runs/time_fine_12km_dt0025_mesh8_6`
- Generate: `output/publication_validation/convergence/layered_resistive_offset100_stage2/runs/mesh_coarse_12km_dt005_mesh12_9`
- Generate: `output/publication_validation/convergence/layered_resistive_offset100_stage2/runs/mesh_fine_12km_dt005_mesh6_4p5`
- Generate: `output/publication_validation/convergence/layered_resistive_offset100_stage2/runs/domain_large_18km_dt005_mesh8_6`
- Generate: `output/publication_validation/convergence/layered_resistive_offset100_stage2/convergence_summary.json`
- Generate: `output/publication_validation/convergence/layered_resistive_offset100_stage2/independent_audit.json`

- [ ] **Step 1: Resume and verify the time-fine run**

```powershell
python dolfinx/run_layered_convergence_study.py --study paper-baseline --output-root output/publication_validation/convergence/layered_resistive_offset100_stage2 --axis time --level fine --mode full
```

Expected: the command contains `--resume-forward`, reaches exactly 25 outputs,
and all KSP reasons are positive.

- [ ] **Step 2: Run coarse and fine local meshes one at a time**

```powershell
python dolfinx/run_layered_convergence_study.py --study paper-baseline --output-root output/publication_validation/convergence/layered_resistive_offset100_stage2 --axis mesh --level coarse --mode full
python dolfinx/run_layered_convergence_study.py --study paper-baseline --output-root output/publication_validation/convergence/layered_resistive_offset100_stage2 --axis mesh --level fine --mode full
```

Expected: both runs complete 25 outputs and pass their empymod postprocessing
gate; no COMSOL process is allowed to overlap either command.

- [ ] **Step 3: Run the 18 km large-domain solve**

```powershell
python dolfinx/run_layered_convergence_study.py --study paper-baseline --output-root output/publication_validation/convergence/layered_resistive_offset100_stage2 --axis domain --level large --mode full
```

Expected: the live resource check passes, the solve completes 25 outputs, and
the run retains the exact preflight mesh hash.

- [ ] **Step 4: Evaluate all axes and run the independent audit**

```powershell
python dolfinx/run_layered_convergence_study.py --study paper-baseline --output-root output/publication_validation/convergence/layered_resistive_offset100_stage2 --mode evaluate
python dolfinx/audit_layered_convergence.py --summary output/publication_validation/convergence/layered_resistive_offset100_stage2/convergence_summary.json --output output/publication_validation/convergence/layered_resistive_offset100_stage2/independent_audit.json
```

Expected: `CONVERGENCE_COMPLETE=3`, `CONVERGENCE_PASSED=3`, and
`INDEPENDENT_RECOMPUTE_OK`.

- [ ] **Step 5: Perform final artifact and image verification**

Check every generated NPZ for 25 finite rows and positive KSP reasons, compare
reported metrics with the audit, and inspect
`convergence_curves.png` and `convergence_differences.png` at original
resolution. Run:

```powershell
python -m pytest tests/test_layered_convergence.py tests/test_layered_publication_validation.py tests/test_dolfinx_partial_forward.py -q
git status --short
```

Expected: tests pass; generated outputs remain untracked or ignored according
to project policy; no unrelated user change is reverted.

## Completion Boundary

This plan is complete only when the 18 km preflight is within 14 GB and all
three FEniCSx axes pass independent audit. It does not establish whole-project
20 GB compatibility. The next separately designed subproject must make the
COMSOL 3D and nonzero-IP validation pathway remain below the same physical
memory ceiling before a final publication-readiness claim is allowed.
