import json
from pathlib import Path
import subprocess
import sys

from PIL import Image

from tools.fenicsx_progress_report.inventory import (
    deduplicate_images,
    discover_images,
    main,
    parse_git_timeline,
    write_inventory,
)


def test_discover_and_deduplicate_images(tmp_path: Path) -> None:
    Image.new("RGB", (40, 20), "white").save(tmp_path / "a.png")
    (tmp_path / "b.png").write_bytes((tmp_path / "a.png").read_bytes())
    Image.new("RGB", (20, 40), "black").save(tmp_path / "c.jpg")
    excluded = tmp_path / ".git"
    excluded.mkdir()
    Image.new("RGB", (10, 10), "red").save(excluded / "ignored.png")

    records = discover_images(tmp_path)
    deduplicated = deduplicate_images(records)

    assert [record.path for record in records] == ["a.png", "b.png", "c.jpg"]
    assert records[0].width == 40
    assert records[0].height == 20
    assert records[0].sha256 == records[1].sha256
    assert sum(record.duplicate_of is not None for record in deduplicated) == 1
    assert deduplicated[1].duplicate_of == "a.png"


def test_write_inventory_uses_stable_utf8_json_and_csv(tmp_path: Path) -> None:
    Image.new("RGB", (32, 16), "white").save(tmp_path / "效果图.png")
    records = deduplicate_images(discover_images(tmp_path))
    json_path = tmp_path / "out" / "inventory.json"
    csv_path = tmp_path / "out" / "inventory.csv"

    write_inventory(records, json_path, csv_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["images"][0]["path"] == "效果图.png"
    assert csv_path.read_text(encoding="utf-8-sig").splitlines()[1].startswith("效果图.png,")


def test_parse_git_timeline_preserves_subject_tabs() -> None:
    text = "abc123\t2026-07-13T12:00:00+08:00\tfix: keep\ttab\n"

    events = parse_git_timeline(text)

    assert events == [
        {
            "commit": "abc123",
            "date": "2026-07-13T12:00:00+08:00",
            "subject": "fix: keep\ttab",
        }
    ]


def test_inventory_cli_writes_images_and_git_timeline(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "report@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Report Test"], cwd=tmp_path, check=True
    )
    Image.new("RGB", (64, 32), "white").save(tmp_path / "result.png")
    subprocess.run(["git", "add", "result.png"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "test: add result"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    output = tmp_path / "out"

    exit_code = main(
        [
            "--root",
            str(tmp_path),
            "--json",
            str(output / "inventory.json"),
            "--csv",
            str(output / "inventory.csv"),
            "--timeline",
            str(output / "timeline.json"),
        ]
    )

    assert exit_code == 0
    assert json.loads((output / "inventory.json").read_text(encoding="utf-8"))[
        "images"
    ][0]["path"] == "result.png"
    timeline = json.loads((output / "timeline.json").read_text(encoding="utf-8"))
    assert timeline[0]["subject"] == "test: add result"


def test_module_entrypoint_has_no_runpy_preload_warning() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "tools.fenicsx_progress_report.inventory", "--help"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert "RuntimeWarning" not in completed.stderr
