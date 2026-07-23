from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from xml.etree import ElementTree

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_zhou2020_validation_report.py"
FIGURE_NAMES = (
    "fig01_model_contract.png",
    "fig02_total_fields.png",
    "fig03_reference_stability.png",
    "fig04_gate_summary.png",
    "fig05_debye_order_diagnostic.png",
)


def _load_report_script():
    spec = importlib.util.spec_from_file_location(
        "zhou_report_under_test",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metric(value: float, gate: float, passed: bool) -> dict[str, object]:
    return {"relative_l2": value, "gate": gate, "passed": passed}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_report_fixture(tmp_path: Path):
    run_id = "synthetic-run"
    run = (
        tmp_path
        / "generated/validation/zhou2020_grounded_wire/runs"
        / run_id
    )
    comparison = run / "comparisons/S1T1B1"
    reference = run / "reference"
    bundle = tmp_path / "publication_bundle"
    audit_dir = tmp_path / "reference_audit_hardened_v2"
    comparison.mkdir(parents=True)
    reference.mkdir(parents=True)
    bundle.mkdir()
    audit_dir.mkdir()

    total = {
        variant: {
            component: _metric(0.03, 0.05, True)
            for component in ("Ex", "Hz", "dBzdt")
        }
        for variant in ("noip", "ip")
    }
    metrics = {
        "schema": "atem3d.zhou2020.strict-comparison/v1",
        "status": "failed_with_reproducible_evidence",
        "total_field": total,
        "ip_increment": {
            "Ex": _metric(0.1632, 0.10, False),
            "Hz": _metric(0.6126, 0.10, False),
            "dBzdt": _metric(0.1140, 0.10, False),
        },
        "zero_crossings": {
            "noip": {
                "Hz": {
                    "prediction": [0.022],
                    "reference": [],
                    "passed": False,
                }
            },
            "ip": {
                "Hz": {
                    "prediction": [0.022],
                    "reference": [],
                    "passed": False,
                },
                "Ex": {
                    "prediction": [0.0576942],
                    "reference": [0.0582826],
                    "max_relative_time_error": 0.010096,
                    "passed": True,
                },
            },
        },
    }
    (comparison / "strict_comparison.json").write_text(
        json.dumps(metrics),
        encoding="utf-8",
    )
    strict_hash = _sha256(comparison / "strict_comparison.json")
    run_manifest = {
        "schema": "atem3d.zhou2020.validation-run/v1",
        "run_id": run_id,
        "comparisons": {
            "full/S1T1B1": {
                "path": "comparisons/S1T1B1/strict_comparison.json",
                "sha256": strict_hash,
                "status": "failed_with_reproducible_evidence",
            }
        },
    }
    (run / "run_manifest.json").write_text(
        json.dumps(run_manifest),
        encoding="utf-8",
    )
    input_hashes = {
        "run_manifest.json": _sha256(run / "run_manifest.json"),
        "strict_comparison.json": strict_hash,
    }
    (reference / "empymod_srcpts_convergence.json").write_text(
        json.dumps({"max_relative_difference": 0.001}),
        encoding="utf-8",
    )

    audit = {
        "status": "inconclusive",
        "qwe": {
            "converged": False,
            "all_converged": False,
            "noip_total_converged": False,
            "ip_total_converged": False,
            "direct_frequency_converged": False,
        },
        "default_dlf": {"sign_changes_first20": 10},
        "stable_window": {"start_s": 5.77e-4},
        "transform_difference": {
            "default_dlf_vs_direct_qwe_relative_l2_full": 0.0644,
            "default_dlf_vs_direct_qwe_relative_l2_first20": 0.9365,
        },
        "fenicsx_vs_direct_qwe": {
            "relative_l2_full": 0.0944,
            "relative_l2_stable_window": 0.0939,
        },
    }
    diagnostic = {
        "schema": "atem3d.zhou2020.debye-order-diagnostic/v2",
        "comparison": {
            "Ex": {"debye_4_vs_16_relative_l2": 0.0920},
            "Hz": {"debye_4_vs_16_relative_l2": 0.0984},
            "dBzdt": {"debye_4_vs_16_relative_l2": 0.0640},
        },
    }
    manifest = {
        "schema": "atem3d.zhou2020.figure-bundle/v1",
        "status": "complete",
        "metadata": {
            "run_id": run_id,
            "spatial_case": "S1T1B1",
            "reference_audit_status": "inconclusive",
            "qwe_converged": False,
            "input_sha256": input_hashes,
        },
    }

    for index, name in enumerate(FIGURE_NAMES, start=1):
        fig, ax = plt.subplots(figsize=(2.0, 1.0))
        ax.plot([0.0, 1.0], [0.0, float(index)])
        ax.set_title(name)
        fig.savefig(bundle / name, dpi=80)
        plt.close(fig)

    fixture = {
        "root": tmp_path,
        "run": run,
        "bundle": bundle,
        "audit_dir": audit_dir,
        "validated_bundle": {
            "manifest": manifest,
            "diagnostic": diagnostic,
        },
        "validated_audit": {
            "manifest": {
                "status": "inconclusive",
                "input_sha256": input_hashes,
            },
            "audit": audit,
            "arrays": {},
        },
    }
    fixture["validated_audit"]["audit"]["input_sha256"] = input_hashes
    return fixture


def _document_text(path: Path) -> str:
    doc = Document(path)
    paragraphs = [paragraph.text for paragraph in doc.paragraphs]
    cells = [
        cell.text
        for table in doc.tables
        for row in table.rows
        for cell in row.cells
    ]
    return "\n".join([*paragraphs, *cells])


def _patch_validators(report, fixture, monkeypatch) -> None:
    monkeypatch.setattr(
        report,
        "_load_validated_bundle",
        lambda path: fixture["validated_bundle"],
    )
    monkeypatch.setattr(
        report,
        "_load_validated_reference_audit",
        lambda path: fixture["validated_audit"],
    )
    monkeypatch.setattr(report, "_git_state", lambda root: ("test-commit", False))


def test_report_consumes_validated_publication_bundle_and_audit(
    tmp_path,
    monkeypatch,
):
    report = _load_report_script()
    fixture = _make_report_fixture(tmp_path)
    calls: list[tuple[str, Path]] = []

    def load_bundle(path: Path):
        calls.append(("bundle", Path(path)))
        return fixture["validated_bundle"]

    def load_audit(path: Path):
        calls.append(("audit", Path(path)))
        return fixture["validated_audit"]

    monkeypatch.setattr(report, "_load_validated_bundle", load_bundle)
    monkeypatch.setattr(report, "_load_validated_reference_audit", load_audit)
    monkeypatch.setattr(report, "_git_state", lambda root: ("test-commit", False))
    output = tmp_path / "report.docx"

    report.build_report(
        fixture["root"],
        fixture["run"],
        fixture["bundle"],
        fixture["audit_dir"],
        output,
    )

    assert calls == [
        ("bundle", fixture["bundle"]),
        ("audit", fixture["audit_dir"]),
    ]


def test_report_language_and_package_are_scientifically_fail_closed(
    tmp_path,
    monkeypatch,
):
    report = _load_report_script()
    fixture = _make_report_fixture(tmp_path)
    monkeypatch.setattr(
        report,
        "_load_validated_bundle",
        lambda path: fixture["validated_bundle"],
    )
    monkeypatch.setattr(
        report,
        "_load_validated_reference_audit",
        lambda path: fixture["validated_audit"],
    )
    monkeypatch.setattr(report, "_git_state", lambda root: ("test-commit", False))
    output = tmp_path / "report.docx"

    report.build_report(
        fixture["root"],
        fixture["run"],
        fixture["bundle"],
        fixture["audit_dir"],
        output,
    )

    text = _document_text(output)
    for required in (
        "绝对值双对数",
        "绝对值不提供符号信息",
        "signed diagnostic",
        "reference-transform unstable",
        "inconclusive",
        "QWE direct 未收敛",
        "早期 DLF 震荡不是物理响应",
        "0.022 s",
        "false reversal",
        "11.40%",
        "仅为敏感性",
        "9.44%",
        "不是通过",
        "4 项相对 16 项",
        "内部敏感性",
        "failed_with_reproducible_evidence",
        "未完全通过",
        "layout rendering not performed",
        "LibreOffice unavailable",
    ):
        assert required in text, required
    for forbidden in (
        "dBz/dt 极化模块严格通过",
        "Ex、Hz 与 dBz/dt 均未同时满足 10% 严格门槛",
        "11.40% 因此正式失败",
        "9.44% 因此通过",
    ):
        assert forbidden not in text

    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
        media = sorted(
            name for name in names if name.startswith("word/media/")
        )
        assert len(media) == 5
        relationships = ElementTree.fromstring(
            archive.read("word/_rels/document.xml.rels")
        )
        image_relationships = [
            relation
            for relation in relationships
            if relation.attrib.get("Type", "").endswith("/image")
        ]
        assert len(image_relationships) == 5
        assert all(
            relation.attrib.get("TargetMode") != "External"
            and relation.attrib["Target"].startswith("media/")
            for relation in image_relationships
        )
        assert not any(
            relation.attrib.get("TargetMode") == "External"
            for relation in relationships
        )
        document_xml = archive.read("word/document.xml")
        document_root = ElementTree.fromstring(document_xml)
        drawing_nodes = document_root.findall(
            ".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}drawing"
        )
        assert len(drawing_nodes) == 5


def test_report_refuses_changed_reference_audit_state(tmp_path, monkeypatch):
    report = _load_report_script()
    fixture = _make_report_fixture(tmp_path)
    changed = fixture["validated_audit"]
    changed["audit"]["status"] = "passed"
    changed["audit"]["qwe"]["converged"] = True
    monkeypatch.setattr(
        report,
        "_load_validated_bundle",
        lambda path: fixture["validated_bundle"],
    )
    monkeypatch.setattr(
        report,
        "_load_validated_reference_audit",
        lambda path: changed,
    )
    monkeypatch.setattr(report, "_git_state", lambda root: ("test-commit", False))

    try:
        report.build_report(
            fixture["root"],
            fixture["run"],
            fixture["bundle"],
            fixture["audit_dir"],
            tmp_path / "report.docx",
        )
    except ValueError as error:
        assert "re-review" in str(error)
    else:
        raise AssertionError("changed audit state must stop stale report prose")


def test_report_refuses_strict_metric_tamper_without_publishing_docx(
    tmp_path,
    monkeypatch,
):
    report = _load_report_script()
    fixture = _make_report_fixture(tmp_path)
    _patch_validators(report, fixture, monkeypatch)
    strict = fixture["run"] / "comparisons/S1T1B1/strict_comparison.json"
    payload = json.loads(strict.read_text(encoding="utf-8"))
    payload["ip_increment"]["dBzdt"]["relative_l2"] = 0.01
    strict.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "tampered.docx"

    with pytest.raises(ValueError, match="hash|identity"):
        report.build_report(
            fixture["root"],
            fixture["run"],
            fixture["bundle"],
            fixture["audit_dir"],
            output,
        )

    assert not output.exists()


def test_report_refuses_equal_looking_substituted_run_directory(
    tmp_path,
    monkeypatch,
):
    report = _load_report_script()
    fixture = _make_report_fixture(tmp_path)
    _patch_validators(report, fixture, monkeypatch)
    substituted = tmp_path / "elsewhere" / "equal-looking-other-run"
    shutil.copytree(fixture["run"], substituted)
    output = tmp_path / "substituted.docx"

    with pytest.raises(ValueError, match="run_id|identity|path"):
        report.build_report(
            fixture["root"],
            substituted,
            fixture["bundle"],
            fixture["audit_dir"],
            output,
        )

    assert not output.exists()


def test_report_refuses_same_run_id_copied_outside_bound_run_root(
    tmp_path,
    monkeypatch,
):
    report = _load_report_script()
    fixture = _make_report_fixture(tmp_path)
    _patch_validators(report, fixture, monkeypatch)
    substituted = tmp_path / "elsewhere" / fixture["run"].name
    shutil.copytree(fixture["run"], substituted)
    output = tmp_path / "same-id-substituted.docx"

    with pytest.raises(ValueError, match="path|identity|run root"):
        report.build_report(
            fixture["root"],
            substituted,
            fixture["bundle"],
            fixture["audit_dir"],
            output,
        )

    assert not output.exists()


def test_unbound_source_convergence_file_is_not_read(tmp_path, monkeypatch):
    report = _load_report_script()
    fixture = _make_report_fixture(tmp_path)
    _patch_validators(report, fixture, monkeypatch)
    unbound = (
        fixture["run"] / "reference/empymod_srcpts_convergence.json"
    )
    unbound.write_text("{deliberately invalid and unbound", encoding="utf-8")
    output = tmp_path / "report.docx"

    report.build_report(
        fixture["root"],
        fixture["run"],
        fixture["bundle"],
        fixture["audit_dir"],
        output,
    )

    assert output.is_file()
    assert "source quadrature difference" not in _document_text(output)


def test_cli_requires_bundle_audit_and_docx_output_only():
    report = _load_report_script()
    parser = report.build_parser()

    args = parser.parse_args(
        [
            "--root",
            ".",
            "--run",
            "run",
            "--publication-bundle",
            "bundle",
            "--reference-audit",
            "audit",
            "--output",
            "report.docx",
        ]
    )
    assert args.publication_bundle == Path("bundle")
    assert args.reference_audit == Path("audit")
    assert args.output.suffix == ".docx"

    try:
        parser.parse_args(
            [
                "--root",
                ".",
                "--run",
                "run",
                "--publication-bundle",
                "bundle",
                "--reference-audit",
                "audit",
                "--output",
                "report.pdf",
            ]
        )
    except SystemExit:
        pass
    else:
        raise AssertionError("CLI must reject non-DOCX output")
