from __future__ import annotations

from pathlib import Path

from docx import Document


def test_report_profile_uses_model_audit_instead_of_baseline_constants() -> None:
    from tools.build_seepage_channel_word_report import _report_profile

    profile = _report_profile(
        {
            "channel": {
                "size_m": [60.0, 1.0, 1.0],
                "bounds_m": [[-30.0, 30.0], [-0.5, 0.5], [19.5, 20.5]],
                "theoretical_volume_m3": 60.0,
            },
            "variant": "thin_60x1x1",
        }
    )

    assert profile["size_text"] == "60 x 1 x 1 m"
    assert profile["depth_text"] == "19.5-20.5 m"
    assert profile["theoretical_volume_m3"] == 60.0
    assert profile["variant"] == "thin_60x1x1"


def test_summary_conclusion_reports_failed_targets_honestly() -> None:
    from tools.build_seepage_channel_word_report import _summary_conclusion

    summary = {
        "background": {
            "SimPEG": {name: {"pass": True} for name in ("Ex", "dBzdt", "Hz")},
            "FEniCSx": {name: {"pass": True} for name in ("Ex", "dBzdt", "Hz")},
        },
        "channel_delta": {
            "Ex": {"pass": True},
            "dBzdt": {"pass": False},
            "Hz": {"pass": True},
        },
    }

    text = _summary_conclusion(summary)

    assert "dBzdt" in text
    assert "未通过" in text


def test_build_seepage_channel_report_contains_required_sections(tmp_path: Path) -> None:
    from tools.build_seepage_channel_word_report import build_report

    result_dir = Path("output/seepage_channel_100m_5rx")
    report_path = tmp_path / "seepage_channel_report.docx"

    built = build_report(result_dir=result_dir, output_path=report_path)

    assert built == report_path
    assert report_path.is_file()
    document = Document(report_path)
    paragraph_text = [paragraph.text for paragraph in document.paragraphs]
    table_text = [cell.text for table in document.tables for row in table.rows for cell in row.cells]
    text = "\n".join([*paragraph_text, *table_text])
    assert "100 m 长导线" in text
    assert "三算法正演对比报告" in text
    assert "SimPEG" in text
    assert "empymod" in text
    assert "FEniCSx" in text
    assert "explicit_full_domain" in text
    assert "不使用单侧求解后对称镜像" in text
    assert "60 x 10 x 10 m" in text
    assert len(document.tables) >= 5
    assert len(document.inline_shapes) >= 5
