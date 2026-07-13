import json
from pathlib import Path

from PIL import Image

from tools.fenicsx_progress_report.selection import (
    ALLOWED_STATUSES,
    audit_selection,
    bootstrap_selection,
    classify_phase,
    load_selection,
    main,
    write_contact_sheets,
)


def test_classify_known_evidence_paths() -> None:
    assert (
        classify_phase(
            "COMSOL/exports/three_way_final_ideal_stepoff_1e-5_1e-2_dBzdt_decay.png"
        )
        == "uniform_three_way"
    )
    assert (
        classify_phase(
            "output/publication_validation/convergence/layered_resistive_offset100/error.png"
        )
        == "layered_convergence"
    )
    assert (
        classify_phase("dolfinx/dam_leakage_model/model.png") == "complex_models"
    )
    assert (
        classify_phase("simpeg3D有限体积/final_artifacts/noip_response.png")
        == "legacy_simpeg"
    )
    assert classify_phase("unrelated/image.png") == "unclassified"


def test_audit_rejects_duplicates_invalid_status_and_uncovered_records() -> None:
    inventory = [
        {"path": "a.png", "sha256": "same", "duplicate_of": None},
        {"path": "b.png", "sha256": "same", "duplicate_of": "a.png"},
        {"path": "c.png", "sha256": "other", "duplicate_of": None},
    ]
    manifest = {
        "figures": [
            {
                "id": "f1",
                "path": "a.png",
                "phase": "source_electrodes",
                "status": "诊断用途",
            },
            {"id": "f2", "path": "b.png", "phase": "", "status": "未知"},
        ],
        "excluded": [],
    }

    report = audit_selection(inventory, manifest)

    assert report.duplicate_selections == ["b.png"]
    assert report.unclassified == ["f2"]
    assert report.invalid_statuses == ["f2:未知"]
    assert report.uncovered == ["c.png"]


def test_audit_accepts_explicit_exclusion_rationale() -> None:
    inventory = [{"path": "a.png", "sha256": "one", "duplicate_of": None}]
    manifest = {
        "figures": [],
        "excluded": [{"path": "a.png", "reason": "自动测试夹具，无算法证据。"}],
    }

    report = audit_selection(inventory, manifest)

    assert report.ok
    assert set(ALLOWED_STATUSES) == {
        "已验证",
        "通过",
        "未通过",
        "待完成",
        "诊断用途",
        "历史实现",
    }


def test_load_selection_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps(
            {
                "figures": [
                    {"id": "same", "path": "a.png"},
                    {"id": "same", "path": "b.png"},
                ],
                "excluded": [],
            }
        ),
        encoding="utf-8",
    )

    try:
        load_selection(path)
    except ValueError as exc:
        assert "duplicate figure id" in str(exc)
    else:
        raise AssertionError("duplicate IDs must fail")


def test_write_contact_sheets_preserves_aspect_and_labels_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    Image.new("RGB", (400, 100), "white").save(root / "wide.png")
    Image.new("RGB", (100, 400), "black").save(root / "tall.png")
    records = [
        {
            "path": "wide.png",
            "width": 400,
            "height": 100,
            "phase": "formulation",
        },
        {
            "path": "tall.png",
            "width": 100,
            "height": 400,
            "phase": "formulation",
        },
    ]

    outputs = write_contact_sheets(
        records,
        root=root,
        output_dir=tmp_path / "sheets",
        columns=2,
        thumb_size=(200, 120),
    )

    assert len(outputs) == 1
    assert outputs[0].name == "formulation-001.png"
    with Image.open(outputs[0]) as sheet:
        assert sheet.width == 460
        assert sheet.height >= 180


def test_selection_cli_builds_sheets_and_passes_complete_audit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    image_dir = root / "COMSOL" / "exports"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "three_way_decay.png"
    Image.new("RGB", (300, 200), "white").save(image_path)
    relative = "COMSOL/exports/three_way_decay.png"
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "path": relative,
                        "sha256": "one",
                        "duplicate_of": None,
                        "width": 300,
                        "height": 200,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "figures": [
                    {
                        "id": "decay",
                        "path": relative,
                        "phase": "uniform_three_way",
                        "status": "已验证",
                    }
                ],
                "excluded": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--root",
                str(root),
                "--inventory",
                str(inventory_path),
                "--contact-sheet-dir",
                str(tmp_path / "sheets"),
            ]
        )
        == 0
    )
    assert (tmp_path / "sheets" / "uniform_three_way-001.png").exists()
    generated_selection = tmp_path / "generated-selection.json"
    assert (
        main(
            [
                "--root",
                str(root),
                "--inventory",
                str(inventory_path),
                "--bootstrap-selection",
                str(generated_selection),
            ]
        )
        == 0
    )
    assert json.loads(generated_selection.read_text(encoding="utf-8"))["figures"]
    assert (
        main(
            [
                "--root",
                str(root),
                "--inventory",
                str(inventory_path),
                "--selection",
                str(selection_path),
                "--audit-only",
            ]
        )
        == 0
    )


def test_bootstrap_selection_covers_canonical_images_with_explicit_policy() -> None:
    inventory = [
        {
            "path": "COMSOL/exports/three_way_final_dBzdt_decay.png",
            "sha256": "one",
            "duplicate_of": None,
            "width": 1200,
            "height": 800,
        },
        {
            "path": "dolfinx/paper_reproduction/song2025/pdf_pages/song2025_page-05.png",
            "sha256": "two",
            "duplicate_of": None,
            "width": 1600,
            "height": 2200,
        },
        {
            "path": "simpeg3D有限体积/final_artifacts/result_display.png",
            "sha256": "three",
            "duplicate_of": None,
            "width": 1000,
            "height": 600,
        },
    ]

    manifest = bootstrap_selection(inventory)
    report = audit_selection(inventory, manifest)

    assert report.ok
    assert [item["path"] for item in manifest["figures"]] == [
        "COMSOL/exports/three_way_final_dBzdt_decay.png"
    ]
    assert len(manifest["excluded"]) == 2
    assert manifest["figures"][0]["placement"] == "body"
    assert manifest["figures"][0]["status"] == "已验证"
    assert manifest["figures"][0]["caption"]
    assert manifest["figures"][0]["problem"]
    assert manifest["figures"][0]["change"]
    assert manifest["figures"][0]["result"]
