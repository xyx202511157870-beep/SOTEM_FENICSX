import importlib.util
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


def load_full_domain_module():
    path = Path("dolfinx/seepage_channel_full_domain.py")
    spec = importlib.util.spec_from_file_location("seepage_receiver_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_pipeline_module():
    path = Path("dolfinx/sotem_pipeline.py")
    spec = importlib.util.spec_from_file_location("seepage_checkpoint_pipeline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_five_receiver_set_calls_evaluator_five_times() -> None:
    module = load_full_domain_module()
    calls = []

    def evaluator(_electric_field, _dbdt, _mesh, config):
        calls.append(config.receiver)
        return {
            "Ex": config.receiver[1],
            "dBzdt": 2 * config.receiver[1],
            "Hz": 3 * config.receiver[1],
        }

    config = SimpleNamespace(
        receiver_locations=tuple(
            (0.0, y, 0.1) for y in (-20, -10, 0, 10, 20)
        ),
        receiver=(0.0, 0.0, 0.1),
    )
    records = module.evaluate_receiver_set(
        None,
        None,
        None,
        config,
        evaluator=evaluator,
    )
    assert calls == list(config.receiver_locations)
    assert [row["provenance"] for row in records] == [
        "explicit_full_domain"
    ] * 5
    assert module.records_to_array(
        records,
        ("Ex", "dBzdt", "Hz"),
    ).shape == (5, 3)


def test_five_receiver_set_evaluates_each_magnetic_diagnostic_independently() -> None:
    module = load_full_domain_module()
    magnetic_calls = []

    def electric_evaluator(_electric_field, _dbdt, _mesh, config):
        return {"Ex": config.receiver[1]}

    def magnetic_evaluator(_electric_field, _dbdt, _mesh, config):
        magnetic_calls.append(config.receiver)
        y = float(config.receiver[1])
        return {
            "dBzdt_curl": y,
            "dBzdt_biot_rate": 2.0 * y,
            "dBzdt_faraday_loop": 3.0 * y,
            "Hz_biot_center": 4.0 * y,
            "Hz_biot_tetra4": 5.0 * y,
        }

    config = SimpleNamespace(
        receiver_locations=tuple((0.0, y, 0.1) for y in (-20, -10, 0, 10, 20)),
        receiver=(0.0, 0.0, 0.1),
    )
    records = module.evaluate_receiver_set(
        None,
        None,
        None,
        config,
        evaluator=electric_evaluator,
        magnetic_evaluator=magnetic_evaluator,
    )

    assert magnetic_calls == list(config.receiver_locations)
    expected = {
        "dBzdt_curl",
        "dBzdt_biot_rate",
        "dBzdt_faraday_loop",
        "Hz_biot_center",
        "Hz_biot_tetra4",
        "provenance",
    }
    assert all(expected <= set(record) for record in records)


def test_full_domain_module_has_no_mirror_expansion() -> None:
    source = Path("dolfinx/seepage_channel_full_domain.py").read_text(
        encoding="utf-8"
    )
    assert "mirror_crossline_values" not in source
    assert "output[0] = output[4]" not in source
    assert "explicit_full_domain" in source


def test_write_predictions_5rx_uses_long_form_rows(tmp_path) -> None:
    module = load_full_domain_module()
    path = tmp_path / "predictions_5rx.csv"
    times = np.array([1.0e-5, 2.0e-5])
    locations = np.array([[0.0, -20.0, 0.1], [0.0, 20.0, 0.1]])
    data = np.arange(12, dtype=float).reshape(2, 2, 3)
    module.write_predictions_5rx(
        path,
        times=times,
        receiver_locations=locations,
        data=data,
        components=("Ex", "dBzdt", "Hz"),
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == (
        "receiver_id,receiver_x_m,receiver_y_m,receiver_z_m,time_obs,"
        "Ex,dBzdt,Hz,provenance"
    )
    assert len(lines) == 5
    assert all(line.endswith("explicit_full_domain") for line in lines[1:])


def test_write_magnetic_receiver_diagnostics_csv_records_methods_and_audits(tmp_path) -> None:
    module = load_full_domain_module()
    path = tmp_path / "magnetic_receiver_diagnostics.csv"
    records = [
        {
            "receiver_id": "Rx1",
            "receiver_x_m": 0.0,
            "receiver_y_m": -20.0,
            "receiver_z_m": 0.1,
            "time_obs": 1.0e-5,
            "dBzdt_curl": 1.0,
            "dBzdt_biot_rate": 2.0,
            "dBzdt_faraday_loop": 3.0,
            "Hz_biot_center": 4.0,
            "Hz_biot_tetra4": 5.0,
            "faraday_audit": {"point_count": 16},
            "biot_tetra4_audit": {"sample_count": 40},
            "provenance": "explicit_full_domain",
        }
    ]

    module.write_magnetic_receiver_diagnostics_csv(path, records)

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["dBzdt_faraday_loop"] == "3.0"
    assert json.loads(rows[0]["faraday_audit_json"])["point_count"] == 16
    assert json.loads(rows[0]["biot_tetra4_audit_json"])["sample_count"] == 40


def test_multi_receiver_checkpoint_rows_preserve_output_axis() -> None:
    module = load_pipeline_module()
    rows = np.arange(30, dtype=float).reshape(2, 5, 3)
    stored = module._checkpoint_rows_array(rows, component_count=3)
    assert stored.shape == (2, 5, 3)
    assert module._checkpoint_output_count(stored) == 2


def test_receiver_set_npz_preserves_three_dimensional_data(tmp_path) -> None:
    module = load_full_domain_module()
    path = tmp_path / "fenicsx_result_5rx.npz"
    values = np.ones((5, 2, 3), dtype=float)
    module.write_receiver_set_npz(
        path,
        times=np.array([1.0e-5, 2.0e-5]),
        receiver_locations=np.asarray(
            [(0.0, y, 0.1) for y in (-20, -10, 0, 10, 20)]
        ),
        components=np.asarray(("Ex", "dBzdt", "Hz")),
        data=values,
        receiver_provenance=np.asarray(["explicit_full_domain"] * 5),
        material_audit={"global_cell_count": 65},
    )
    with np.load(path, allow_pickle=False) as stored:
        assert stored["data"].shape == (5, 2, 3)
        assert stored["receiver_provenance"].tolist() == [
            "explicit_full_domain"
        ] * 5


def test_biot_h_is_only_recomputed_at_outputs_when_previous_field_is_available() -> None:
    module = load_pipeline_module()
    assert not module._biot_h_required_for_step("curl", is_output=False)
    assert module._biot_h_required_for_step("curl", is_output=True)
    assert module._biot_h_required_for_step("biot_rate", is_output=False)
    assert not module._biot_h_required_for_step(
        "biot_rate",
        is_output=False,
        can_recompute_previous=True,
    )
    assert module._biot_h_required_for_step(
        "biot_rate",
        is_output=True,
        can_recompute_previous=True,
    )


def test_biot_rate_diagnostic_skips_internal_history_when_previous_field_is_available() -> None:
    module = load_pipeline_module()
    assert module._biot_h_required_for_step(
        "faraday_loop",
        is_output=False,
        diagnostic_methods=("biot_rate",),
    )
    assert not module._biot_h_required_for_step(
        "faraday_loop",
        is_output=False,
        diagnostic_methods=("biot_rate",),
        can_recompute_previous=True,
    )
    assert not module._biot_h_required_for_step(
        "faraday_loop",
        is_output=False,
        diagnostic_methods=("curl", "faraday_loop"),
    )
