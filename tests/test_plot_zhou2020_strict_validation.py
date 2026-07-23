from __future__ import annotations

import importlib.util
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
    times = np.geomspace(1.0e-4, 1.0e-2, 8)
    arrays = {
        "time_s": times,
        "default_dlf": np.array([1, -1, 1, 2, 3, 4, 5, 6]) * 1.0e-10,
        "separate_total_qwe": np.geomspace(0.8e-13, 0.8e-9, 8),
        "direct_frequency_qwe": np.geomspace(1.0e-13, 1.0e-9, 8),
        "fenicsx_increment": np.geomspace(1.1e-13, 1.1e-9, 8),
    }

    fig = plotter.plot_reference_stability(arrays, _audit(times), None)

    assert fig.axes[0].get_xscale() == "log"
    assert fig.axes[0].get_yscale() == "log"
    assert fig.axes[1].get_yscale() == "linear"
    assert fig.axes[1].get_xscale() == "log"
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


def test_hardened_reference_audit_is_loaded_through_manifest_validator():
    plotter = _load_plotter()

    validated = plotter.load_reference_audit(HARDENED_AUDIT)

    assert validated["audit"]["status"] == "inconclusive"
    assert validated["audit"]["qwe"]["converged"] is False
    assert validated["arrays"]["time_s"].shape == (101,)
    assert validated["manifest"]["artifacts"]["reference_stability.json"][
        "sha256"
    ]


def test_reference_audit_loader_rejects_unmanifested_files(tmp_path):
    plotter = _load_plotter()
    (tmp_path / "reference_stability.json").write_text("{}", encoding="utf-8")
    np.savez_compressed(tmp_path / "reference_stability.npz", time_s=[1.0])

    with pytest.raises((FileNotFoundError, ValueError)):
        plotter.load_reference_audit(tmp_path)


def test_cli_default_reference_audit_targets_hardened_v2(monkeypatch):
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

    args = plotter._parse_args()

    assert args.reference_audit.name == "reference_audit_hardened_v2"
