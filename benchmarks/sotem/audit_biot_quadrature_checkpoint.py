#!/usr/bin/env python3
"""Audit Biot-Savart tetrahedron quadrature on a saved FEniCSx checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import yaml


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _load_pipeline(repo_root: Path):
    path = repo_root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location("sotem_pipeline_for_quadrature_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pipeline from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference_hz(run_dir: Path) -> float | None:
    path = run_dir / "verification_data.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        names_key = "components" if "components" in data.files else "component_names"
        if names_key not in data.files or "empymod" not in data.files:
            return None
        names = [str(value) for value in data[names_key].tolist()]
        if "Hz" not in names:
            return None
        reference = np.asarray(data["empymod"], dtype=float)
        return float(reference[-1, names.index("Hz")])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--degrees", default="2,4,6,8,10")
    parser.add_argument("--nedelec-order", type=int, default=2)
    parser.add_argument("--rho-air", type=float, default=1.0e6)
    parser.add_argument("--layer-depths", default="300")
    parser.add_argument("--layer-resistivities", default="100,100")
    parser.add_argument("--source-current-at-checkpoint", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    repo_root = args.repo_root.resolve()
    checkpoint_path = run_dir / "forward_checkpoint.npz"
    resolved_path = run_dir / "run_config_resolved.yaml"
    if not checkpoint_path.exists() or not resolved_path.exists():
        raise FileNotFoundError("run directory must contain forward_checkpoint.npz and run_config_resolved.yaml")

    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    layer_depths = _parse_floats(args.layer_depths)
    layer_resistivities = _parse_floats(args.layer_resistivities)
    degrees = _parse_ints(args.degrees)
    if not degrees or any(degree <= 0 for degree in degrees):
        raise ValueError("degrees must contain positive integers")

    pipeline = _load_pipeline(repo_root)
    config = pipeline.PipelineConfig(
        workdir=run_dir,
        source_start=tuple(resolved["source_start"]),
        source_end=tuple(resolved["source_end"]),
        source_current=float(resolved["source_current"]),
        receiver=tuple(resolved["receiver"]),
        receiver_type="point",
        receiver_mesh_size=float(resolved["receiver_mesh_size"]),
        receiver_refinement_radius=float(resolved["receiver_refinement_radius"]),
        nedelec_order=int(args.nedelec_order),
        rho_air=float(args.rho_air),
        rho_earth=float(layer_resistivities[0]),
        layer_depths=layer_depths,
        layer_resistivities=layer_resistivities,
    )

    msh, cell_tags, _facet_tags = pipeline.load_mesh(config)
    spaces = pipeline.build_function_spaces(msh, config)
    materials = pipeline.assign_materials(msh, cell_tags, spaces, config)

    from dolfinx import fem

    field = fem.Function(spaces["V"], name="checkpoint_E")
    with np.load(checkpoint_path, allow_pickle=True) as checkpoint:
        values = np.asarray(checkpoint["e_old"])
    if values.shape != field.x.array.shape:
        raise ValueError(f"checkpoint E shape {values.shape} != function shape {field.x.array.shape}")
    field.x.array[:] = values
    field.x.scatter_forward()

    reference_hz = _reference_hz(run_dir)
    records = []
    for degree in degrees:
        parts = pipeline._biot_savart_total_h_components_at_receiver(
            field,
            msh,
            materials,
            config,
            float(args.source_current_at_checkpoint),
            quadrature_degree=degree,
        )
        hz = float(parts["total_h"][2])
        record = {
            "degree": degree,
            "conductive_h": np.asarray(parts["conductive_h"], dtype=float).tolist(),
            "wire_h": np.asarray(parts["wire_h"], dtype=float).tolist(),
            "total_h": np.asarray(parts["total_h"], dtype=float).tolist(),
            "hz": hz,
        }
        if reference_hz is not None:
            record["reference_hz"] = reference_hz
            record["relative_error_percent"] = 100.0 * abs(hz - reference_hz) / abs(reference_hz)
        records.append(record)

    result = {
        "schema": "atem3d.biot_quadrature_checkpoint_audit.v1",
        "run_dir": str(run_dir),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "source_current_at_checkpoint": float(args.source_current_at_checkpoint),
        "receiver": list(config.receiver),
        "degrees": list(degrees),
        "records": records,
    }
    output = args.output or (run_dir / "biot_quadrature_checkpoint_audit.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"[audit] saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
