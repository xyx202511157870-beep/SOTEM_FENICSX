from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Mm
from PIL import Image
import pytest

from tools.fenicsx_progress_report.docx_writer import (
    UnknownFigureError,
    build_docx,
)


def _figure(image_path: Path, *, figure_id: str = "test_figure") -> dict[str, str]:
    return {
        "id": figure_id,
        "path": str(image_path),
        "phase": "uniform_three_way",
        "status": "已验证",
        "placement": "body",
        "caption": "测试图。",
        "problem": "测试问题。",
        "change": "测试修改。",
        "result": "测试结果。",
        "source_run": "fixture",
    }


def test_build_docx_uses_compact_a4_and_embeds_figure(tmp_path: Path) -> None:
    image_path = tmp_path / "figure.png"
    Image.new("RGB", (800, 500), "white").save(image_path)
    source = "\n".join(
        [
            "# 测试报告",
            "**报告性质：** 测试技术档案",
            "## 1. 摘要与当前状态",
            "正文包含 `dBz/dt`。",
            "[[FIGURE:test_figure]]",
        ]
    )
    output = tmp_path / "report.docx"

    build_docx(
        source,
        {"figures": [_figure(image_path)]},
        project_root=tmp_path,
        output_path=output,
    )

    document = Document(output)
    section = document.sections[0]
    assert abs(section.page_width - Mm(210)) < 1000
    assert abs(section.page_height - Mm(297)) < 1000
    assert abs(section.left_margin - Mm(17)) < 1000
    assert abs(section.right_margin - Mm(17)) < 1000
    assert len(document.inline_shapes) == 1
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    assert "测试报告" in text
    assert "测试图" in text
    assert "FEniCSx" in section.header.paragraphs[0].text


def test_unknown_figure_directive_raises(tmp_path: Path) -> None:
    with pytest.raises(UnknownFigureError, match="missing"):
        build_docx(
            "# 报告\n## 1. 摘要与当前状态\n[[FIGURE:missing]]",
            {"figures": []},
            project_root=tmp_path,
            output_path=tmp_path / "report.docx",
        )


def test_figure_requires_caption_status_and_source(tmp_path: Path) -> None:
    image_path = tmp_path / "figure.png"
    Image.new("RGB", (300, 200), "white").save(image_path)
    record = _figure(image_path)
    record["caption"] = ""

    with pytest.raises(ValueError, match="caption"):
        build_docx(
            "# 报告\n## 1. 摘要与当前状态\n[[FIGURE:test_figure]]",
            {"figures": [record]},
            project_root=tmp_path,
            output_path=tmp_path / "report.docx",
        )


def test_markdown_table_has_explicit_geometry_and_repeating_header(
    tmp_path: Path,
) -> None:
    source = "\n".join(
        [
            "# 报告",
            "## 1. 摘要与当前状态",
            "| 指标 | 数值 | 状态 |",
            "| --- | ---: | --- |",
            "| 中位误差 | 2.75% | 已验证 |",
        ]
    )
    output = tmp_path / "table.docx"

    build_docx(
        source,
        {"figures": []},
        project_root=tmp_path,
        output_path=output,
    )

    document = Document(output)
    table = document.tables[0]
    table_width = table._tbl.tblPr.find(qn("w:tblW"))
    assert table_width is not None
    assert int(table_width.get(qn("w:w"))) > 9000
    grid_cols = table._tbl.tblGrid.findall(qn("w:gridCol"))
    assert len(grid_cols) == 3
    assert sum(int(col.get(qn("w:w"))) for col in grid_cols) == int(
        table_width.get(qn("w:w"))
    )
    assert table.rows[0]._tr.find(qn("w:trPr")).find(qn("w:tblHeader")) is not None


def test_figure_atlas_embeds_appendix_images_four_up(tmp_path: Path) -> None:
    figures = []
    for index in range(4):
        image_path = tmp_path / f"figure-{index}.png"
        Image.new("RGB", (600 + index, 400), "white").save(image_path)
        record = _figure(image_path, figure_id=f"appendix_{index}")
        record["placement"] = "appendix"
        figures.append(record)
    output = tmp_path / "atlas.docx"

    build_docx(
        "# 报告\n## 附录 B：调试效果图谱\n[[FIGURE_ATLAS]]",
        {"figures": figures},
        project_root=tmp_path,
        output_path=output,
    )

    document = Document(output)
    assert len(document.inline_shapes) == 4
    assert any(len(table.rows) == 2 and len(table.columns) == 2 for table in document.tables)
