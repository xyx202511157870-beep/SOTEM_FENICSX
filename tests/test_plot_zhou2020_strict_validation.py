from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/plot_zhou2020_strict_validation.py"
HARDENED_AUDIT = (
    ROOT
    / "output/zhou2020_strict_validation/reference_audit_hardened_v2"
)
FORMAL_RUN = (
    ROOT
    / "generated/validation/zhou2020_grounded_wire/runs"
    / "20260723T062004Z_zhou_strict_v2"
)


def _load_plotter():
    spec = importlib.util.spec_from_file_location("zhou_plotter_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sample_data():
    times = np.geomspace(1.0e-4, 1.0e-2, 8)
    base = np.geomspace(1.0e-7, 1.0e-9, 8)
    return {
        "time_s": times,
        "Ex_V_per_m": np.array([-base[0], 0.0, *(-base[2:])]),
        "Hz_A_per_m": 2.0 * base,
        "dBzdt_T_per_s": -3.0 * base,
    }


def _write_verification_npz(path: Path, scale: float) -> None:
    times = np.geomspace(1.0e-4, 1.0e-2, 8)
    values = np.column_stack(
        [scale * times, 2.0 * scale * times, -3.0 * scale * times]
    )
    np.savez_compressed(
        path,
        times=times,
        components=np.array(["Ex", "Hz", "dBzdt"]),
        fem=values,
    )


def _metrics() -> dict:
    total = {}
    for variant in ("noip", "ip"):
        total[variant] = {}
        for index, component in enumerate(("Ex", "Hz", "dBzdt"), start=1):
            total[variant][component] = {
                "relative_l2": 0.01 * index,
                "gate": 0.05,
                "passed": True,
            }
    increment = {
        "Ex": {"relative_l2": 0.1632057472, "gate": 0.10, "passed": False},
        "Hz": {"relative_l2": 0.6125712432, "gate": 0.10, "passed": False},
        "dBzdt": {"relative_l2": 0.1140375542, "gate": 0.10, "passed": False},
    }
    return {
        "total_field": total,
        "ip_increment": increment,
        "zero_crossings": {
            "noip": {
                "Hz": {
                    "prediction": [0.0222724651],
                    "reference": [],
                    "count_match": False,
                    "passed": False,
                }
            }
        },
    }


def _audit(times: np.ndarray) -> dict:
    return {
        "status": "inconclusive",
        "qwe": {
            "converged": False,
            "direct_difference_qwe_converged": False,
            "separate_ip_qwe_converged": True,
            "separate_noip_qwe_converged": True,
        },
        "default_dlf": {"sign_changes_first20": 2},
        "stable_window": {"start_s": float(times[3])},
        "all_samples_retained": True,
    }


def test_positive_magnitude_masks_only_exact_zeros():
    plotter = _load_plotter()
    values = np.array([-2.0, 0.0, 3.0])
    plotted = plotter._positive_magnitude(values)
    np.testing.assert_allclose(plotted.compressed(), [2.0, 3.0])
    np.testing.assert_array_equal(np.ma.getmaskarray(plotted), [False, True, False])
    np.testing.assert_allclose(values, [-2.0, 0.0, 3.0])


def test_total_field_axes_are_log_log_and_never_symlog():
    plotter = _load_plotter()
    noip_fem = _sample_data()
    noip_ref = _sample_data()
    ip_fem = {name: values.copy() for name, values in noip_fem.items()}
    ip_ref = {name: values.copy() for name, values in noip_ref.items()}

    fig = plotter.plot_total_fields(noip_fem, ip_fem, noip_ref, ip_ref, None)

    assert all(ax.get_xscale() == "log" for ax in fig.axes)
    assert all(ax.get_yscale() == "log" for ax in fig.axes)
    title = fig._suptitle.get_text().lower()
    assert "absolute magnitude" in title
    assert "literature-style log-log" in title
    assert "sign" in title
    for ax in fig.axes:
        for line in ax.lines:
            values = np.ma.asarray(line.get_ydata())
            assert np.all(values.compressed() > 0.0)
    plt.close(fig)


def test_total_field_rejects_nonpositive_log_time():
    plotter = _load_plotter()
    noip_fem = _sample_data()
    noip_fem["time_s"][0] = 0.0
    noip_ref = _sample_data()
    ip_fem = _sample_data()
    ip_ref = _sample_data()

    with pytest.raises(ValueError, match="strictly positive"):
        plotter.plot_total_fields(noip_fem, ip_fem, noip_ref, ip_ref, None)


def test_total_field_rejects_shifted_equal_length_time_grid():
    plotter = _load_plotter()
    noip_fem = _sample_data()
    noip_ref = _sample_data()
    ip_fem = _sample_data()
    ip_ref = _sample_data()
    ip_ref["time_s"] = ip_ref["time_s"] + 1.0e-12

    with pytest.raises(ValueError, match="exactly equal"):
        plotter.plot_total_fields(noip_fem, ip_fem, noip_ref, ip_ref, None)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_total_field_rejects_nonfinite_component(bad_value):
    plotter = _load_plotter()
    noip_fem = _sample_data()
    noip_ref = _sample_data()
    ip_fem = _sample_data()
    ip_ref = _sample_data()
    ip_ref["Ex_V_per_m"][2] = bad_value

    with pytest.raises(ValueError, match="finite 1-D"):
        plotter.plot_total_fields(noip_fem, ip_fem, noip_ref, ip_ref, None)


def test_model_contract_callouts_do_not_overlap_adjacent_labels():
    plotter = _load_plotter()

    fig = plotter.plot_model(None)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    labels = {
        text.get_text(): text.get_window_extent(renderer)
        for ax in fig.axes
        for text in ax.texts
    }
    assert not labels["IP: 10 ohm m, 20 m\nm=0.1, tau=1 s, c=0.3"].overlaps(
        labels["Half-space\n200 ohm m"]
    )
    assert not labels["Rx (0, 1000 m)"].overlaps(
        labels["1000 m perpendicular offset"]
    )
    plt.close(fig)


def test_signed_reference_panel_uses_linear_y_and_shades_unstable_window():
    plotter = _load_plotter()
    times = np.geomspace(1.0e-4, 1.0e-1, 25)
    arrays = {
        "time_s": times,
        "default_dlf": np.resize(
            np.array([1.0, -1.0, 1.0, -1.0, 2.0]) * 1.0e-10,
            times.size,
        ),
        "separate_total_qwe": np.geomspace(0.8e-13, 0.8e-9, times.size),
        "direct_frequency_qwe": np.geomspace(1.0e-13, 1.0e-9, times.size),
        "fenicsx_increment": np.geomspace(1.1e-13, 1.1e-9, times.size),
    }

    fig = plotter.plot_reference_stability(arrays, _audit(times), None)

    assert fig.axes[0].get_xscale() == "log"
    assert fig.axes[0].get_yscale() == "log"
    assert fig.axes[1].get_yscale() == "linear"
    assert fig.axes[1].get_xscale() == "log"
    np.testing.assert_allclose(fig.axes[1].get_xlim(), (times[0], times[19]))
    assert fig.axes[0].get_xlim()[1] >= times[-1]
    early_values = np.concatenate(
        [
            np.asarray(arrays[name])[:20]
            for name in (
                "default_dlf",
                "separate_total_qwe",
                "direct_frequency_qwe",
                "fenicsx_increment",
            )
        ]
    )
    signed_ylim = fig.axes[1].get_ylim()
    assert signed_ylim[0] < min(0.0, float(np.min(early_values)))
    assert signed_ylim[1] > max(0.0, float(np.max(early_values)))
    assert signed_ylim[1] < float(np.max(arrays["fenicsx_increment"]))
    assert all(len(ax.patches) >= 1 for ax in fig.axes)
    assert "signed diagnostic" in fig.axes[1].get_title().lower()
    assert any(np.allclose(line.get_ydata(), 0.0) for line in fig.axes[1].lines)
    sampled_lines = [
        line
        for ax in fig.axes
        for line in ax.lines
        if len(line.get_xdata()) > 2
    ]
    assert all(len(line.get_xdata()) == times.size for line in sampled_lines)
    text = "\n".join(item.get_text() for ax in fig.axes for item in ax.texts)
    for phrase in (
        "reference-transform unstable",
        "sign changes",
        "QWE converged=False",
        "status=inconclusive",
        "all samples retained",
    ):
        assert phrase in text
    plt.close(fig)


def test_reference_stability_rejects_false_all_samples_retained_evidence():
    plotter = _load_plotter()
    times = np.geomspace(1.0e-4, 1.0e-2, 8)
    arrays = {
        "time_s": times,
        "default_dlf": np.ones(times.size),
        "separate_total_qwe": np.ones(times.size),
        "direct_frequency_qwe": np.ones(times.size),
        "fenicsx_increment": np.ones(times.size),
    }
    audit = _audit(times)
    audit["all_samples_retained"] = False

    with pytest.raises(ValueError, match="all_samples_retained"):
        plotter.plot_reference_stability(arrays, audit, None)


def test_reference_evidence_box_does_not_overlap_signed_data_region():
    plotter = _load_plotter()
    times = np.geomspace(1.0e-4, 1.0e-2, 25)
    arrays = {
        "time_s": times,
        "default_dlf": np.sin(np.arange(times.size)) * 1.0e-10,
        "separate_total_qwe": np.geomspace(1.0e-13, 1.0e-9, times.size),
        "direct_frequency_qwe": np.geomspace(1.2e-13, 1.2e-9, times.size),
        "fenicsx_increment": np.geomspace(1.1e-13, 1.1e-9, times.size),
    }

    fig = plotter.plot_reference_stability(arrays, _audit(times), None)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    signed_ax = fig.axes[1]
    evidence = next(
        text for text in signed_ax.texts if "QWE converged" in text.get_text()
    )

    assert not evidence.get_window_extent(renderer).overlaps(
        signed_ax.get_window_extent(renderer)
    )
    plt.close(fig)


def test_reference_legend_is_outside_absolute_data_region_and_inside_figure():
    plotter = _load_plotter()
    times = np.geomspace(1.0e-4, 3.0, 101)
    arrays = {
        "time_s": times,
        "default_dlf": np.geomspace(2.0e-10, 2.0e-14, times.size),
        "separate_total_qwe": np.geomspace(1.0e-13, 1.0e-9, times.size),
        "direct_frequency_qwe": np.geomspace(1.2e-13, 1.2e-9, times.size),
        "fenicsx_increment": np.geomspace(1.1e-13, 1.1e-9, times.size),
    }

    fig = plotter.plot_reference_stability(arrays, _audit(times), None)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    absolute_ax = fig.axes[0]
    legend = (
        fig.legends[0]
        if fig.legends
        else absolute_ax.get_legend()
    )
    legend_bbox = legend.get_window_extent(renderer)
    axes_bbox = absolute_ax.get_window_extent(renderer)
    figure_bbox = fig.get_window_extent(renderer)

    assert not legend_bbox.overlaps(axes_bbox)
    assert not legend_bbox.overlaps(
        absolute_ax.xaxis.label.get_window_extent(renderer)
    )
    assert legend_bbox.x0 >= figure_bbox.x0
    assert legend_bbox.y0 >= figure_bbox.y0
    assert legend_bbox.x1 <= figure_bbox.x1
    assert legend_bbox.y1 <= figure_bbox.y1
    plt.close(fig)


def test_gate_summary_uses_only_total_fields_as_formal_bars():
    plotter = _load_plotter()
    audit = _audit(np.geomspace(1.0e-4, 1.0e-2, 8))

    fig = plotter.plot_gate_summary(_metrics(), audit, None)

    ax = fig.axes[0]
    assert len(ax.patches) == 6
    text = "\n".join(item.get_text() for item in ax.texts)
    assert "IP increments: sensitivity only" in text
    assert "Ex 16.32%, Hz 61.26%, dBz/dt 11.40%" in text
    assert "reference stability not established" in text
    assert "inconclusive" in text
    assert "Hz false reversal" in text
    assert "0.022 s" in text
    plt.close(fig)


def test_gate_summary_rejects_empty_or_nonfailed_hz_crossing_record():
    plotter = _load_plotter()
    audit = _audit(np.geomspace(1.0e-4, 1.0e-2, 8))
    metrics = _metrics()
    metrics["zero_crossings"]["noip"]["Hz"]["prediction"] = []

    with pytest.raises(ValueError, match="false-reversal"):
        plotter.plot_gate_summary(metrics, audit, None)

    metrics = _metrics()
    metrics["zero_crossings"]["noip"]["Hz"]["passed"] = True
    with pytest.raises(ValueError, match="false-reversal"):
        plotter.plot_gate_summary(metrics, audit, None)


def test_debye_metric_is_four_relative_to_sixteen_not_empymod(tmp_path):
    plotter = _load_plotter()
    noip_path = tmp_path / "noip.npz"
    ip4_path = tmp_path / "ip4.npz"
    ip16_path = tmp_path / "ip16.npz"
    _write_verification_npz(noip_path, 1.0)
    _write_verification_npz(ip4_path, 1.9)
    _write_verification_npz(ip16_path, 2.0)

    result, fig = plotter.plot_debye_order_diagnostic(
        noip_path, ip4_path, ip16_path, None
    )

    for component in ("Ex", "Hz", "dBzdt"):
        item = result["comparison"][component]
        assert item["debye_4_vs_16_relative_l2"] == pytest.approx(0.1)
        assert "debye_4_relative_l2" not in item
    assert all(ax.get_xscale() == "log" for ax in fig.axes)
    assert all(ax.get_yscale() == "log" for ax in fig.axes)
    assert all(
        "empymod" not in line.get_label().lower()
        for ax in fig.axes
        for line in ax.lines
    )
    assert "pole-count sensitivity" in fig._suptitle.get_text().lower()
    plt.close(fig)


def test_save_all_does_not_create_pdf(tmp_path):
    plotter = _load_plotter()
    fig = plt.figure()
    plotter._save_all(fig, tmp_path / "figure")
    assert (tmp_path / "figure.svg").exists()
    assert (tmp_path / "figure.png").exists()
    assert (tmp_path / "figure.tiff").exists()
    assert not (tmp_path / "figure.pdf").exists()
    plt.close(fig)


def test_save_failure_leaves_existing_output_set_untouched(tmp_path, monkeypatch):
    plotter = _load_plotter()
    stem = tmp_path / "figure"
    expected = {}
    for suffix in plotter.EXPORT_FORMATS:
        path = stem.with_suffix(suffix)
        expected[path] = f"old-{suffix}".encode()
        path.write_bytes(expected[path])
    fig = plt.figure()
    original = fig.savefig
    calls = 0

    def fail_second_save(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected save failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(fig, "savefig", fail_second_save)

    with pytest.raises(RuntimeError, match="injected save failure"):
        plotter._save_all(fig, stem)

    for path, content in expected.items():
        assert path.read_bytes() == content
    assert not list(tmp_path.glob(".figure.*"))
    plt.close(fig)


def test_finish_or_save_closes_figure_when_export_fails(tmp_path, monkeypatch):
    plotter = _load_plotter()
    fig = plt.figure()
    figure_number = fig.number

    def fail_export(*_args, **_kwargs):
        raise RuntimeError("injected export failure")

    monkeypatch.setattr(plotter, "_save_all", fail_export)
    with pytest.raises(RuntimeError, match="injected export failure"):
        plotter._finish_or_save(fig, tmp_path / "figure")

    assert not plt.fignum_exists(figure_number)


def test_hardened_reference_audit_is_loaded_through_manifest_validator():
    plotter = _load_plotter()

    validated = plotter.load_reference_audit(HARDENED_AUDIT)

    assert validated["audit"]["status"] == "inconclusive"
    assert validated["audit"]["qwe"]["converged"] is False
    assert validated["arrays"]["time_s"].shape == (101,)
    assert validated["manifest"]["artifacts"]["reference_stability.json"][
        "sha256"
    ]
    plotter._cross_bind_run_to_audit(FORMAL_RUN, validated)


def test_reference_audit_identity_binds_exact_generation():
    plotter = _load_plotter()
    identity = plotter._validated_reference_audit_identity(HARDENED_AUDIT)
    manifest = json.loads(
        (HARDENED_AUDIT / "manifest.json").read_text(encoding="utf-8")
    )
    audit = json.loads(
        (HARDENED_AUDIT / "reference_stability.json").read_text(
            encoding="utf-8"
        )
    )

    assert identity["manifest_sha256"] == plotter._sha256_file(
        HARDENED_AUDIT / "manifest.json"
    )
    assert identity["artifacts"] == {
        name: manifest["artifacts"][name]["sha256"]
        for name in plotter.REFERENCE_AUDIT_ARTIFACT_NAMES
    }
    assert identity["manifest_schema"] == manifest["schema"]
    assert identity["audit_schema"] == audit["schema"]
    for key in ("status", "input_sha256", "code_sha256", "methods", "qwe"):
        assert identity[key] == manifest[key] == audit[key]


def test_reference_audit_loader_rejects_unmanifested_files(tmp_path):
    plotter = _load_plotter()
    (tmp_path / "reference_stability.json").write_text("{}", encoding="utf-8")
    np.savez_compressed(tmp_path / "reference_stability.npz", time_s=[1.0])

    with pytest.raises((FileNotFoundError, ValueError)):
        plotter.load_reference_audit(tmp_path)


def test_cross_binding_rejects_tampered_formal_run(tmp_path):
    plotter = _load_plotter()
    run = tmp_path / "run"
    paths = plotter._audit_bound_inputs(run)
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"input:{name}".encode())
    input_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }
    validated = {
        "manifest": {"input_sha256": input_hashes},
        "audit": {
            "input_sha256": input_hashes,
            "methods": {"fenicsx_increment": {"spatial_case": "S1T1B1"}},
        },
    }
    plotter._cross_bind_run_to_audit(run, validated)

    paths["strict_comparison.json"].write_bytes(b"tampered")

    with pytest.raises(ValueError, match="hash mismatch"):
        plotter._cross_bind_run_to_audit(run, validated)


def test_cli_requires_reference_audit(monkeypatch):
    plotter = _load_plotter()
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "--run",
            "run",
            "--compute-root",
            "compute",
            "--output",
            "output",
        ],
    )

    with pytest.raises(SystemExit):
        plotter._parse_args()


def test_cli_accepts_explicit_reference_audit(monkeypatch):
    plotter = _load_plotter()
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "--run",
            "run",
            "--compute-root",
            "compute",
            "--output",
            "output",
            "--reference-audit",
            "audit",
        ],
    )

    args = plotter._parse_args()

    assert args.reference_audit == Path("audit")


def test_no_pdf_preflight_waiver_is_explicit_and_does_not_enable_pdf():
    plotter = _load_plotter()

    assert plotter.EXPORT_POLICY["pdf"] == "forbidden_by_user"
    assert "preflight" in plotter.EXPORT_POLICY["known_waiver"].lower()
    assert "pdf" in plotter.EXPORT_POLICY["known_waiver"].lower()
    assert ".pdf" not in plotter.EXPORT_FORMATS


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_debye_npz_rejects_nonfinite_field(tmp_path, bad_value):
    plotter = _load_plotter()
    path = tmp_path / "bad.npz"
    times = np.geomspace(1.0e-4, 1.0e-2, 8)
    values = np.ones((times.size, 3))
    values[2, 1] = bad_value
    np.savez_compressed(
        path,
        times=times,
        components=np.array(["Ex", "Hz", "dBzdt"]),
        fem=values,
    )

    with pytest.raises(ValueError, match="finite"):
        plotter._load_verification(path)


def test_debye_npz_rejects_incomplete_shape(tmp_path):
    plotter = _load_plotter()
    path = tmp_path / "bad-shape.npz"
    times = np.geomspace(1.0e-4, 1.0e-2, 8)
    np.savez_compressed(
        path,
        times=times,
        components=np.array(["Ex", "Hz", "dBzdt"]),
        fem=np.ones((times.size - 1, 3)),
    )

    with pytest.raises(ValueError, match="shape"):
        plotter._load_verification(path)


def test_scale_safe_relative_l2_avoids_overflow():
    plotter = _load_plotter()
    denominator = np.array([1.0e308, -1.0e308])
    numerator = 0.1 * denominator

    assert plotter._relative_l2(numerator, denominator) == pytest.approx(0.1)


def _write_minimal_bundle_artifacts(
    plotter,
    staging: Path,
    marker: str = "default",
) -> None:
    for stem in plotter.FIGURE_STEMS:
        for suffix in plotter.EXPORT_FORMATS:
            (staging / f"{stem}{suffix}").write_bytes(
                f"{marker}:{stem}{suffix}".encode()
            )
    plotter._write_json(
        staging / plotter.DEBYE_JSON_NAME,
        {
            "schema": "atem3d.zhou2020.debye-order-diagnostic/v2",
            "comparison": {},
        },
    )


def _bundle_metadata() -> dict:
    audit_identity = {
        "root_relative": "test/reference_audit",
        "manifest_sha256": "1" * 64,
        "manifest_schema": "atem3d.zhou2020.reference-stability-manifest/v1",
        "audit_schema": "atem3d.zhou2020.reference-stability/v1",
        "status": "inconclusive",
        "artifacts": {
            "reference_stability.json": "2" * 64,
            "reference_stability.npz": "3" * 64,
        },
        "input_sha256": {"case.yaml": "0" * 64},
        "code_sha256": {"audit.py": "4" * 64},
        "methods": {"qwe": {"nquad": 51}},
        "qwe": {"converged": False},
    }
    canonical = json.dumps(
        audit_identity,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    audit_identity["canonical_sha256"] = hashlib.sha256(canonical).hexdigest()
    return {
        "run_id": "test-run",
        "spatial_case": "S1T1B1",
        "input_sha256": {"case.yaml": "0" * 64},
        "reference_audit_status": "inconclusive",
        "qwe_converged": False,
        "reference_audit": audit_identity,
    }


def _assert_no_transaction_debris(target: Path) -> None:
    assert not list(target.parent.glob(f".{target.name}.staging-*"))
    assert not list(target.parent.glob(f".{target.name}.backup-*"))


def test_bundle_success_writes_manifest_last_and_loader_verifies_hashes(
    tmp_path,
    monkeypatch,
):
    plotter = _load_plotter()
    target = tmp_path / "publication_bundle"
    writes = []
    original_write_json = plotter._write_json

    def record_json_write(path, payload):
        writes.append(Path(path).name)
        return original_write_json(path, payload)

    monkeypatch.setattr(plotter, "_write_json", record_json_write)

    validated = plotter._publish_bundle(
        target,
        lambda staging: _write_minimal_bundle_artifacts(plotter, staging),
        _bundle_metadata(),
    )

    assert writes[-1] == plotter.COMPLETION_MANIFEST_NAME
    assert validated["manifest"]["status"] == "complete"
    assert set(validated["manifest"]["artifacts"]) == set(
        plotter.BUNDLE_ARTIFACT_NAMES
    )
    for name, item in validated["manifest"]["artifacts"].items():
        assert item["sha256"] == hashlib.sha256(
            (target / name).read_bytes()
        ).hexdigest()
    assert not list(target.rglob("*.pdf"))
    plotter.load_validated_bundle(target)
    _assert_no_transaction_debris(target)


def test_bundle_late_failure_after_fig03_preserves_existing_target(tmp_path):
    plotter = _load_plotter()
    target = tmp_path / "publication_bundle"
    target.mkdir()
    (target / "old-generation.txt").write_text("old", encoding="utf-8")

    def fail_after_fig03(staging):
        for index in range(1, 4):
            (staging / f"fig0{index}.png").write_bytes(b"new")
        raise RuntimeError("late failure after fig03")

    with pytest.raises(RuntimeError, match="after fig03"):
        plotter._publish_bundle(target, fail_after_fig03, _bundle_metadata())

    assert (target / "old-generation.txt").read_text(encoding="utf-8") == "old"
    assert list(target.iterdir()) == [target / "old-generation.txt"]
    _assert_no_transaction_debris(target)


def test_bundle_debye_json_failure_preserves_existing_target(
    tmp_path,
    monkeypatch,
):
    plotter = _load_plotter()
    target = tmp_path / "publication_bundle"
    target.mkdir()
    (target / "old-generation.txt").write_text("old", encoding="utf-8")

    def fail_at_json(*_args, **_kwargs):
        raise RuntimeError("Debye JSON failure")

    monkeypatch.setattr(plotter, "_write_json", fail_at_json)

    with pytest.raises(RuntimeError, match="Debye JSON"):
        plotter._publish_bundle(
            target,
            lambda staging: _write_minimal_bundle_artifacts(plotter, staging),
            _bundle_metadata(),
        )

    assert (target / "old-generation.txt").read_text(encoding="utf-8") == "old"
    _assert_no_transaction_debris(target)


def test_bundle_invalid_debye_json_rolls_back_before_exposure(tmp_path):
    plotter = _load_plotter()
    target = tmp_path / "publication_bundle"
    target.mkdir()
    (target / "old-generation.txt").write_text("old", encoding="utf-8")

    def write_invalid_json(staging):
        for stem in plotter.FIGURE_STEMS:
            for suffix in plotter.EXPORT_FORMATS:
                (staging / f"{stem}{suffix}").write_bytes(b"new")
        (staging / plotter.DEBYE_JSON_NAME).write_text(
            "{not-json",
            encoding="utf-8",
        )

    with pytest.raises(json.JSONDecodeError):
        plotter._publish_bundle(
            target,
            write_invalid_json,
            _bundle_metadata(),
        )

    assert (target / "old-generation.txt").read_text(encoding="utf-8") == "old"
    _assert_no_transaction_debris(target)


def test_bundle_manifest_failure_preserves_existing_target(
    tmp_path,
    monkeypatch,
):
    plotter = _load_plotter()
    target = tmp_path / "publication_bundle"
    target.mkdir()
    (target / "old-generation.txt").write_text("old", encoding="utf-8")

    def fail_manifest(*_args, **_kwargs):
        raise RuntimeError("manifest failure")

    monkeypatch.setattr(plotter, "_write_completion_manifest", fail_manifest)

    with pytest.raises(RuntimeError, match="manifest failure"):
        plotter._publish_bundle(
            target,
            lambda staging: _write_minimal_bundle_artifacts(plotter, staging),
            _bundle_metadata(),
        )

    assert (target / "old-generation.txt").read_text(encoding="utf-8") == "old"
    _assert_no_transaction_debris(target)


def test_bundle_final_replace_failure_rolls_back_existing_target(
    tmp_path,
    monkeypatch,
):
    plotter = _load_plotter()
    target = tmp_path / "publication_bundle"
    target.mkdir()
    (target / "old-generation.txt").write_text("old", encoding="utf-8")
    original_replace = os.replace

    def fail_staging_publish(source, destination):
        source = Path(source)
        destination = Path(destination)
        if ".staging-" in source.name and destination == target:
            raise OSError("injected final os.replace failure")
        return original_replace(source, destination)

    monkeypatch.setattr(plotter.os, "replace", fail_staging_publish)

    with pytest.raises(OSError, match="final os.replace"):
        plotter._publish_bundle(
            target,
            lambda staging: _write_minimal_bundle_artifacts(plotter, staging),
            _bundle_metadata(),
        )

    assert (target / "old-generation.txt").read_text(encoding="utf-8") == "old"
    assert list(target.iterdir()) == [target / "old-generation.txt"]
    _assert_no_transaction_debris(target)


def test_bundle_keyboard_interrupt_during_replace_rolls_back_and_unlocks(
    tmp_path,
    monkeypatch,
):
    plotter = _load_plotter()
    target = tmp_path / "publication_bundle"
    target.mkdir()
    (target / "old-generation.txt").write_text("old", encoding="utf-8")
    original_replace = os.replace

    def interrupt_staging_publish(source, destination):
        source = Path(source)
        destination = Path(destination)
        if ".staging-" in source.name and destination == target:
            raise KeyboardInterrupt("injected publish interrupt")
        return original_replace(source, destination)

    monkeypatch.setattr(plotter.os, "replace", interrupt_staging_publish)

    with pytest.raises(KeyboardInterrupt, match="publish interrupt"):
        plotter._publish_bundle(
            target,
            lambda staging: _write_minimal_bundle_artifacts(plotter, staging),
            _bundle_metadata(),
        )

    assert (target / "old-generation.txt").read_text(encoding="utf-8") == "old"
    assert not plotter._bundle_lock_path(target).exists()
    _assert_no_transaction_debris(target)


def test_bundle_lock_collision_fails_closed_without_touching_target(tmp_path):
    plotter = _load_plotter()
    target = tmp_path / "publication_bundle"
    target.mkdir()
    (target / "old-generation.txt").write_text("old", encoding="utf-8")
    lock_path = plotter._bundle_lock_path(target)
    other_payload = b'{"owner_token":"other","pid":999}\n'
    lock_path.write_bytes(other_payload)

    with pytest.raises(FileExistsError, match="publication lock"):
        plotter._publish_bundle(
            target,
            lambda staging: _write_minimal_bundle_artifacts(plotter, staging),
            _bundle_metadata(),
        )

    assert (target / "old-generation.txt").read_text(encoding="utf-8") == "old"
    assert lock_path.read_bytes() == other_payload
    _assert_no_transaction_debris(target)


def test_two_publishers_are_serialized_by_exclusive_lock(tmp_path):
    plotter = _load_plotter()
    target = tmp_path / "publication_bundle"
    first_owner = plotter._acquire_bundle_lock(target)
    lock_record = json.loads(first_owner.payload.decode("utf-8"))

    assert lock_record["owner_token"] == first_owner.owner_token
    assert lock_record["pid"] == os.getpid()
    with pytest.raises(FileExistsError, match="publication lock"):
        plotter._publish_bundle(
            target,
            lambda staging: _write_minimal_bundle_artifacts(plotter, staging),
            _bundle_metadata(),
        )

    plotter._release_bundle_lock(first_owner)
    plotter._publish_bundle(
        target,
        lambda staging: _write_minimal_bundle_artifacts(plotter, staging),
        _bundle_metadata(),
    )

    assert not plotter._bundle_lock_path(target).exists()
    plotter.load_validated_bundle(target)
    _assert_no_transaction_debris(target)


def test_owned_lock_is_not_deleted_after_other_owner_replaces_contents(tmp_path):
    plotter = _load_plotter()
    target = tmp_path / "publication_bundle"
    lock_path = plotter._bundle_lock_path(target)
    other_payload = b'{"owner_token":"replacement","pid":123}\n'

    def replace_lock_owner(staging):
        _write_minimal_bundle_artifacts(plotter, staging)
        lock_path.write_bytes(other_payload)

    plotter._publish_bundle(target, replace_lock_owner, _bundle_metadata())

    assert lock_path.read_bytes() == other_payload
    plotter.load_validated_bundle(target)
    lock_path.unlink()
    _assert_no_transaction_debris(target)


def test_bundle_fsyncs_artifacts_manifest_lock_and_rename_directories(
    tmp_path,
    monkeypatch,
):
    plotter = _load_plotter()
    target = tmp_path / "publication_bundle"
    file_calls = []
    directory_calls = []
    original_file_fsync = plotter._fsync_file
    original_directory_fsync = plotter._fsync_directory

    def record_file(path):
        file_calls.append(Path(path))
        return original_file_fsync(path)

    def record_directory(path):
        directory_calls.append(Path(path))
        return original_directory_fsync(path)

    monkeypatch.setattr(plotter, "_fsync_file", record_file)
    monkeypatch.setattr(plotter, "_fsync_directory", record_directory)

    plotter._publish_bundle(
        target,
        lambda staging: _write_minimal_bundle_artifacts(plotter, staging),
        _bundle_metadata(),
    )

    assert {path.name for path in file_calls} >= set(
        plotter.BUNDLE_ARTIFACT_NAMES
    ) | {plotter.COMPLETION_MANIFEST_NAME, plotter._bundle_lock_path(target).name}
    assert directory_calls.count(target.parent) >= 3
    assert any(".staging-" in path.name for path in directory_calls)
    assert not plotter._bundle_lock_path(target).exists()
    _assert_no_transaction_debris(target)


def test_bundle_keyboard_interrupt_after_fig03_cleans_stage_and_owned_lock(
    tmp_path,
):
    plotter = _load_plotter()
    target = tmp_path / "publication_bundle"
    target.mkdir()
    (target / "old-generation.txt").write_text("old", encoding="utf-8")

    def interrupt_after_fig03(staging):
        for index in range(1, 4):
            (staging / f"fig0{index}.png").write_bytes(b"new")
        raise KeyboardInterrupt("interrupt after fig03")

    with pytest.raises(KeyboardInterrupt, match="after fig03"):
        plotter._publish_bundle(
            target,
            interrupt_after_fig03,
            _bundle_metadata(),
        )

    assert (target / "old-generation.txt").read_text(encoding="utf-8") == "old"
    assert not plotter._bundle_lock_path(target).exists()
    _assert_no_transaction_debris(target)


def test_bundle_stage_cleanup_retries_one_keyboard_interrupt(
    tmp_path,
    monkeypatch,
):
    plotter = _load_plotter()
    target = tmp_path / "publication_bundle"
    target.mkdir()
    (target / "old-generation.txt").write_text("old", encoding="utf-8")
    original_rmtree = plotter.shutil.rmtree
    interrupted = False

    def interrupt_first_stage_cleanup(path, *args, **kwargs):
        nonlocal interrupted
        path = Path(path)
        if ".staging-" in path.name and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("cleanup interrupt")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(plotter.shutil, "rmtree", interrupt_first_stage_cleanup)

    with pytest.raises(RuntimeError, match="builder failed"):
        plotter._publish_bundle(
            target,
            lambda _staging: (_ for _ in ()).throw(
                RuntimeError("builder failed")
            ),
            _bundle_metadata(),
        )

    assert interrupted is True
    assert (target / "old-generation.txt").read_text(encoding="utf-8") == "old"
    assert not plotter._bundle_lock_path(target).exists()
    _assert_no_transaction_debris(target)


def test_bundle_owned_lock_cleanup_retries_one_keyboard_interrupt(
    tmp_path,
    monkeypatch,
):
    plotter = _load_plotter()
    target = tmp_path / "publication_bundle"
    lock_path = plotter._bundle_lock_path(target)
    original_unlink = Path.unlink
    interrupted = False

    def interrupt_first_lock_unlink(path, *args, **kwargs):
        nonlocal interrupted
        if Path(path) == lock_path and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("lock cleanup interrupt")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", interrupt_first_lock_unlink)

    plotter._publish_bundle(
        target,
        lambda staging: _write_minimal_bundle_artifacts(plotter, staging),
        _bundle_metadata(),
    )

    assert interrupted is True
    assert not lock_path.exists()
    plotter.load_validated_bundle(target)
    _assert_no_transaction_debris(target)


def test_bundle_artifact_fsync_interrupt_preserves_old_target_and_unlocks(
    tmp_path,
    monkeypatch,
):
    plotter = _load_plotter()
    target = tmp_path / "publication_bundle"
    target.mkdir()
    (target / "old-generation.txt").write_text("old", encoding="utf-8")
    original_fsync = plotter._fsync_file

    def interrupt_staged_artifact(path):
        path = Path(path)
        if (
            ".staging-" in path.parent.name
            and path.name == plotter.BUNDLE_ARTIFACT_NAMES[0]
        ):
            raise KeyboardInterrupt("artifact fsync interrupt")
        return original_fsync(path)

    monkeypatch.setattr(plotter, "_fsync_file", interrupt_staged_artifact)

    with pytest.raises(KeyboardInterrupt, match="artifact fsync"):
        plotter._publish_bundle(
            target,
            lambda staging: _write_minimal_bundle_artifacts(plotter, staging),
            _bundle_metadata(),
        )

    assert (target / "old-generation.txt").read_text(encoding="utf-8") == "old"
    assert not plotter._bundle_lock_path(target).exists()
    _assert_no_transaction_debris(target)


def test_bundle_publish_never_deletes_legacy_parent_files(tmp_path):
    plotter = _load_plotter()
    target = tmp_path / "publication_bundle"
    legacy = tmp_path / "fig01_model_contract.png"
    legacy.write_bytes(b"legacy-parent-file")

    plotter._publish_bundle(
        target,
        lambda staging: _write_minimal_bundle_artifacts(plotter, staging),
        _bundle_metadata(),
    )

    assert legacy.read_bytes() == b"legacy-parent-file"
    plotter.load_validated_bundle(target)


def test_post_commit_partial_backup_cleanup_never_rolls_back_new_target(
    tmp_path,
    monkeypatch,
):
    plotter = _load_plotter()
    target = tmp_path / "publication_bundle"
    plotter._publish_bundle(
        target,
        lambda staging: _write_minimal_bundle_artifacts(
            plotter,
            staging,
            "old",
        ),
        _bundle_metadata(),
    )
    original_rmtree = plotter.shutil.rmtree

    def partially_delete_backup_then_interrupt(path, *args, **kwargs):
        path = Path(path)
        if ".backup-" in path.name:
            victim = next(item for item in path.iterdir() if item.is_file())
            victim.unlink()
            raise KeyboardInterrupt("partial backup cleanup interrupt")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(
        plotter.shutil,
        "rmtree",
        partially_delete_backup_then_interrupt,
    )

    with pytest.raises(plotter.PublicationCleanupError) as caught:
        plotter._publish_bundle(
            target,
            lambda staging: _write_minimal_bundle_artifacts(
                plotter,
                staging,
                "new",
            ),
            _bundle_metadata(),
        )

    error = caught.value
    assert error.publication_committed is True
    assert error.backup_path.exists()
    assert ".backup-" in error.backup_path.name
    validated = plotter.load_validated_bundle(target)
    assert validated["manifest"]["status"] == "complete"
    first_artifact = target / plotter.BUNDLE_ARTIFACT_NAMES[0]
    assert first_artifact.read_bytes().startswith(b"new:")
    committed_manifest = (
        target / plotter.COMPLETION_MANIFEST_NAME
    ).read_bytes()
    residual_names = {
        path.name
        for path in target.parent.glob(f".{target.name}.backup-*")
    }
    assert residual_names == {error.backup_path.name}
    assert not plotter._bundle_lock_path(target).exists()
    assert not list(target.parent.glob(f".{target.name}.staging-*"))

    with pytest.raises(plotter.PublicationDebrisError) as debris_error:
        plotter._publish_bundle(
            target,
            lambda staging: _write_minimal_bundle_artifacts(
                plotter,
                staging,
                "third",
            ),
            _bundle_metadata(),
        )

    assert error.backup_path in debris_error.value.backup_paths
    assert (
        target / plotter.COMPLETION_MANIFEST_NAME
    ).read_bytes() == committed_manifest
    assert first_artifact.read_bytes().startswith(b"new:")
    assert error.backup_path.exists()
    assert not plotter._bundle_lock_path(target).exists()
    assert not list(target.parent.glob(f".{target.name}.staging-*"))


def test_bundle_loader_rejects_tampered_artifact(tmp_path):
    plotter = _load_plotter()
    target = tmp_path / "publication_bundle"
    plotter._publish_bundle(
        target,
        lambda staging: _write_minimal_bundle_artifacts(plotter, staging),
        _bundle_metadata(),
    )
    artifact = target / plotter.BUNDLE_ARTIFACT_NAMES[0]
    original = artifact.read_bytes()
    artifact.write_bytes(bytes([original[0] ^ 0xFF]) + original[1:])

    with pytest.raises(ValueError, match="hash mismatch"):
        plotter.load_validated_bundle(target)


def test_bundle_loader_rejects_unmanifested_directory(tmp_path):
    plotter = _load_plotter()
    target = tmp_path / "publication_bundle"
    plotter._publish_bundle(
        target,
        lambda staging: _write_minimal_bundle_artifacts(plotter, staging),
        _bundle_metadata(),
    )
    (target / "unmanifested").mkdir()

    with pytest.raises(ValueError, match="file set"):
        plotter.load_validated_bundle(target)


def test_main_real_run_publishes_self_validating_bundle(tmp_path):
    plotter = _load_plotter()
    compute_root = (
        ROOT
        / "generated/validation/zhou2020_grounded_wire/compute"
    )

    plotter.main(
        [
            "--run",
            str(FORMAL_RUN),
            "--compute-root",
            str(compute_root),
            "--reference-audit",
            str(HARDENED_AUDIT),
            "--output",
            str(tmp_path),
        ]
    )

    bundle = tmp_path / plotter.PUBLISHED_BUNDLE_NAME
    validated = plotter.load_validated_bundle(bundle, HARDENED_AUDIT)
    assert validated["manifest"]["metadata"]["run_id"] == FORMAL_RUN.name
    assert (
        validated["manifest"]["metadata"]["reference_audit"]
        == plotter._validated_reference_audit_identity(HARDENED_AUDIT)
    )
    diagnostic = json.loads(
        (bundle / plotter.DEBYE_JSON_NAME).read_text(encoding="utf-8")
    )
    assert diagnostic["schema"] == "atem3d.zhou2020.debye-order-diagnostic/v2"
    assert not list(bundle.rglob("*.pdf"))
    _assert_no_transaction_debris(bundle)


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("manifest", lambda identity: identity.__setitem__(
            "manifest_sha256", "a" * 64
        )),
        ("artifact", lambda identity: identity["artifacts"].__setitem__(
            "reference_stability.json", "b" * 64
        )),
        ("code", lambda identity: identity["code_sha256"].__setitem__(
            next(iter(identity["code_sha256"])), "c" * 64
        )),
        ("method", lambda identity: identity["methods"][
            "direct_frequency_qwe"
        ]["ftarg"].__setitem__("nquad", 53)),
        ("qwe", lambda identity: identity["qwe"].__setitem__(
            "direct_difference_qwe_converged", True
        )),
    ],
)
def test_loader_rejects_same_inputs_but_different_audit_generation(
    tmp_path,
    monkeypatch,
    field,
    mutate,
):
    plotter = _load_plotter()
    target = tmp_path / "publication_bundle"
    audit_identity = plotter._validated_reference_audit_identity(
        HARDENED_AUDIT
    )
    metadata = _bundle_metadata()
    metadata["input_sha256"] = dict(audit_identity["input_sha256"])
    metadata["reference_audit"] = audit_identity
    plotter._publish_bundle(
        target,
        lambda staging: _write_minimal_bundle_artifacts(plotter, staging),
        metadata,
        reference_audit=HARDENED_AUDIT,
    )
    stored_manifest = (
        target / plotter.COMPLETION_MANIFEST_NAME
    ).read_bytes()
    different_identity = json.loads(json.dumps(audit_identity))
    mutate(different_identity)
    different_identity["canonical_sha256"] = plotter._canonical_identity_digest(
        {
            key: value
            for key, value in different_identity.items()
            if key != "canonical_sha256"
        }
    )
    assert different_identity["input_sha256"] == audit_identity["input_sha256"]

    monkeypatch.setattr(
        plotter,
        "_validated_reference_audit_identity",
        lambda _path: different_identity,
    )

    with pytest.raises(ValueError, match="audit generation identity mismatch"):
        plotter.load_validated_bundle(target, tmp_path / f"different-{field}")

    assert (
        target / plotter.COMPLETION_MANIFEST_NAME
    ).read_bytes() == stored_manifest
    plotter.load_validated_bundle(target)
