#!/usr/bin/env python3
"""Apply and verify the DOLFINx magnetic-six production-output contract.

The production contract is opt-in through
``--magnetic-output-contract=magnetic6`` and fixes the output order to

    Hx, Hy, Hz, dBxdt, dBydt, dBzdt.

For the E formulation, absolute H is recovered from total conductive plus
impressed-wire current (``magnetic_receiver_mode=biot_current``), while dB/dt
is evaluated independently from ``-curl(E)``.  The H formulation uses its
native H field.  Legacy Ex/Ey/Hz/dBzdt output remains the default.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "dolfinx" / "sotem_pipeline.py"

CONSTANT_MARKER = "MAGNETIC6_COMPONENTS = ("
CONFIG_MARKER = 'magnetic_output_contract: str = "legacy"'
FORWARD_MARKER = "# MAGNETIC6_PRODUCTION_COMPONENT_GATE"
SAVE_MARKER = "# MAGNETIC6_PRODUCTION_SAVE"
PARTIAL_SAVE_MARKER = "# MAGNETIC6_PRODUCTION_PARTIAL_SAVE"


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one occurrence, found {count}")
    return text.replace(old, new, 1)


def _lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def _function_node(text: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(text)
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one function {name!r}, found {len(matches)}")
    return matches[0]


def _insert_after_function_docstring(
    text: str,
    function_name: str,
    marker: str,
    body: str,
) -> str:
    if marker in text:
        return text
    node = _function_node(text, function_name)
    insertion_line = node.lineno
    if node.body and isinstance(node.body[0], ast.Expr):
        value = node.body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            insertion_line = int(node.body[0].end_lineno)
    lines = _lines(text)
    snippet = "".join(f"    {line}\n" if line else "\n" for line in body.splitlines())
    lines.insert(insertion_line, snippet)
    return "".join(lines)


def _insert_before_function_exit(
    text: str,
    function_name: str,
    marker: str,
    body: str,
) -> str:
    if marker in text:
        return text
    node = _function_node(text, function_name)
    if not node.body:
        raise RuntimeError(f"function {function_name!r} has no body")
    final_statement = node.body[-1]
    if isinstance(final_statement, ast.Return):
        insertion_index = int(final_statement.lineno) - 1
    else:
        insertion_index = int(node.end_lineno)
    lines = _lines(text)
    snippet = "".join(f"    {line}\n" if line else "\n" for line in body.splitlines())
    lines.insert(insertion_index, snippet)
    return "".join(lines)


def _insert_before_function(
    text: str,
    function_name: str,
    marker: str,
    body: str,
) -> str:
    if marker in text:
        return text
    node = _function_node(text, function_name)
    lines = _lines(text)
    snippet = body.rstrip() + "\n\n\n"
    lines.insert(int(node.lineno) - 1, snippet)
    return "".join(lines)


def _argument_call_end_line(text: str, option: str) -> int:
    tree = ast.parse(text)
    matches: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and first.value == option:
            matches.append(node)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one parser.add_argument({option!r}), found {len(matches)}"
        )
    return int(matches[0].end_lineno)


def _insert_parser_option(text: str) -> str:
    option = "--magnetic-output-contract"
    if option in text:
        return text
    insertion_line = _argument_call_end_line(text, "--magnetic-dbdt-mode")
    lines = _lines(text)
    snippet = (
        "    parser.add_argument(\n"
        "        \"--magnetic-output-contract\",\n"
        "        choices=(\"legacy\", \"magnetic6\"),\n"
        "        default=\"legacy\",\n"
        "        help=(\n"
        "            \"Receiver output contract. magnetic6 writes Hx,Hy,Hz and \"\n"
        "            \"dBxdt,dBydt,dBzdt in a fixed NPZ schema.\"\n"
        "        ),\n"
        "    )\n"
    )
    lines.insert(insertion_line, snippet)
    return "".join(lines)


def _apply(text: str) -> str:
    text = _replace_once(
        text,
        "REQUIRED_NEDELEC_ORDER = 2\n",
        (
            "REQUIRED_NEDELEC_ORDER = 2\n"
            "MAGNETIC6_COMPONENTS = (\n"
            "    \"Hx\",\n"
            "    \"Hy\",\n"
            "    \"Hz\",\n"
            "    \"dBxdt\",\n"
            "    \"dBydt\",\n"
            "    \"dBzdt\",\n"
            ")\n"
            "MAGNETIC6_UNITS = (\"A/m\", \"A/m\", \"A/m\", \"T/s\", \"T/s\", \"T/s\")\n"
        ),
        label="insert magnetic-six constants",
    )
    text = _replace_once(
        text,
        '    magnetic_dbdt_mode: str = "curl"  # curl, biot_rate\n',
        (
            '    magnetic_dbdt_mode: str = "curl"  # curl, biot_rate\n'
            '    magnetic_output_contract: str = "legacy"  # legacy, magnetic6\n'
        ),
        label="insert magnetic output contract config",
    )
    text = _replace_once(
        text,
        (
            "    def output_npz(self) -> Path:\n"
            "        return self.workdir / \"verification_data.npz\"\n"
        ),
        (
            "    def output_npz(self) -> Path:\n"
            "        return self.workdir / \"verification_data.npz\"\n\n"
            "    def magnetic6_output_npz(self) -> Path:\n"
            "        return self.workdir / \"magnetic6_numerical.npz\"\n"
        ),
        label="insert magnetic-six output path",
    )
    text = _insert_parser_option(text)
    text = _insert_after_function_docstring(
        text,
        "_forward_components",
        FORWARD_MARKER,
        """# MAGNETIC6_PRODUCTION_COMPONENT_GATE
contract = str(config.magnetic_output_contract).strip().lower()
if contract not in {"legacy", "magnetic6"}:
    raise ValueError("magnetic_output_contract must be 'legacy' or 'magnetic6'")
if contract == "magnetic6":
    formulation = str(config.formulation).strip().lower()
    receiver_mode = str(config.magnetic_receiver_mode).strip().lower()
    dbdt_mode = str(config.magnetic_dbdt_mode).strip().lower()
    if formulation == "e" and receiver_mode != "biot_current":
        raise ValueError(
            "E-form magnetic6 output requires magnetic_receiver_mode='biot_current' "
            "so H includes both conductive and impressed-wire currents"
        )
    if formulation not in {"e", "h"}:
        raise ValueError("magnetic6 output supports formulation='e' or 'h'")
    if dbdt_mode != "curl":
        raise ValueError(
            "formal magnetic6 output requires magnetic_dbdt_mode='curl' for an "
            "independent Faraday-law dB/dt observable"
        )
    return list(MAGNETIC6_COMPONENTS)""",
    )
    text = _replace_once(
        text,
        '    rec = {"Ex": float(e_val[0]), "Ey": float(e_val[1]), "dBzdt": float(dbdt_val[2])}\n',
        (
            "    rec = {\n"
            '        "Ex": float(e_val[0]),\n'
            '        "Ey": float(e_val[1]),\n'
            '        "dBxdt": float(dbdt_val[0]),\n'
            '        "dBydt": float(dbdt_val[1]),\n'
            '        "dBzdt": float(dbdt_val[2]),\n'
            "    }\n"
        ),
        label="expand curl receiver to dB/dt three components",
    )
    text = _replace_once(
        text,
        (
            "def _assign_biot_receiver_hz(receiver_values: dict[str, float], h_receiver) -> None:\n"
            "    \"\"\"Use Biot-Savart for H while preserving instantaneous Faraday dB/dt.\"\"\"\n\n"
            "    receiver_values[\"Hz\"] = float(h_receiver[2])\n"
        ),
        (
            "def _assign_biot_receiver_hz(receiver_values: dict[str, float], h_receiver) -> None:\n"
            "    \"\"\"Assign total Biot-Savart Hx, Hy, Hz without changing curl dB/dt.\"\"\"\n\n"
            "    values = tuple(float(value) for value in h_receiver)\n"
            "    if len(values) != 3:\n"
            "        raise ValueError(\"h_receiver must contain Hx, Hy, Hz\")\n"
            "    receiver_values[\"Hx\"] = values[0]\n"
            "    receiver_values[\"Hy\"] = values[1]\n"
            "    receiver_values[\"Hz\"] = values[2]\n"
        ),
        label="assign H three components",
    )
    text = _replace_once(
        text,
        (
            "                biot_rate = float(_biot_receiver_dbdt_from_h(H_new_receiver, H_old_receiver, dt=dt)[2])\n"
            "                rec[\"dBzdt_biot_rate\"] = biot_rate\n"
            "                if magnetic_dbdt_mode == \"biot_rate\":\n"
            "                    rec[\"dBzdt\"] = biot_rate\n"
        ),
        (
            "                biot_rate = _biot_receiver_dbdt_from_h(\n"
            "                    H_new_receiver, H_old_receiver, dt=dt\n"
            "                )\n"
            "                for axis, component in enumerate((\"dBxdt\", \"dBydt\", \"dBzdt\")):\n"
            "                    rec[f\"{component}_biot_rate\"] = float(biot_rate[axis])\n"
            "                rec[\"dBzdt_biot_rate\"] = float(biot_rate[2])\n"
            "                if magnetic_dbdt_mode == \"biot_rate\":\n"
            "                    for axis, component in enumerate((\"dBxdt\", \"dBydt\", \"dBzdt\")):\n"
            "                        rec[component] = float(biot_rate[axis])\n"
        ),
        label="expand Biot-rate diagnostic to three components",
    )
    text = _replace_once(
        text,
        (
            '    return {"Ex": float(e_val[0]), "Ey": float(e_val[1]), "Hz": float(h_new[2]), "dBzdt": float(dbdt[2])}\n'
        ),
        (
            "    return {\n"
            '        "Ex": float(e_val[0]),\n'
            '        "Ey": float(e_val[1]),\n'
            '        "Hx": float(h_new[0]),\n'
            '        "Hy": float(h_new[1]),\n'
            '        "Hz": float(h_new[2]),\n'
            '        "dBxdt": float(dbdt[0]),\n'
            '        "dBydt": float(dbdt[1]),\n'
            '        "dBzdt": float(dbdt[2]),\n'
            "    }\n"
        ),
        label="expand H-form receiver to six magnetic components",
    )
    text = _replace_once(
        text,
        '        "magnetic_dbdt_mode": str(config.magnetic_dbdt_mode),\n',
        (
            '        "magnetic_dbdt_mode": str(config.magnetic_dbdt_mode),\n'
            '        "magnetic_output_contract": str(config.magnetic_output_contract),\n'
        ),
        label="record magnetic output contract",
    )

    writer = '''def _write_magnetic6_numerical_npz(config: PipelineConfig, fem_result) -> Path:
    """Write a canonical single-receiver magnetic-six numerical artifact."""

    import numpy as np

    components = tuple(str(value) for value in fem_result["components"])
    if components != MAGNETIC6_COMPONENTS:
        raise ValueError(
            "magnetic-six artifact requires components "
            + ",".join(MAGNETIC6_COMPONENTS)
        )
    data = np.asarray(fem_result["data"], dtype=float)
    if data.ndim == 2:
        data = data[:, None, :]
    if data.ndim != 3 or data.shape[-1] != len(MAGNETIC6_COMPONENTS):
        raise ValueError(
            "magnetic-six data must have shape (time,6) or (time,location,6)"
        )
    times = np.asarray(fem_result["times"], dtype=float).reshape(-1)
    if times.size < data.shape[0]:
        raise ValueError("magnetic-six time axis is shorter than the data rows")
    times = times[: data.shape[0]]
    if data.shape[1] != 1:
        raise ValueError(
            "the current DOLFINx production pipeline supports one configured receiver; "
            "multi-location magnetic-six output must use the array pipeline"
        )
    path = config.magnetic6_output_npz()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_schema": np.asarray("atem3d.dolfinx_magnetic6_numerical.v1"),
        "times": times,
        "data": data,
        "components": np.asarray(MAGNETIC6_COMPONENTS),
        "units": np.asarray(MAGNETIC6_UNITS),
        "receiver_locations": np.asarray([config.receiver], dtype=float),
        "coordinate_system": np.asarray("z_up"),
        "time_origin": np.asarray(str(config.time_origin)),
        "ramp_off_time": np.asarray(float(config.ramp_off_time)),
        "source_current": np.asarray(float(config.source_current)),
        "nedelec_order": np.asarray(int(config.nedelec_order)),
        "formulation": np.asarray(str(config.formulation)),
        "magnetic_receiver_mode": np.asarray(str(config.magnetic_receiver_mode)),
        "magnetic_dbdt_mode": np.asarray(str(config.magnetic_dbdt_mode)),
        "magnetic_output_contract": np.asarray(str(config.magnetic_output_contract)),
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    tmp_path.replace(path)
    print(f"[data] saved {path}", flush=True)
    return path'''
    text = _insert_before_function(
        text,
        "_save_npz",
        "def _write_magnetic6_numerical_npz",
        writer,
    )
    text = _insert_before_function_exit(
        text,
        "_save_npz",
        SAVE_MARKER,
        """# MAGNETIC6_PRODUCTION_SAVE
if tuple(str(value) for value in fem_result["components"]) == MAGNETIC6_COMPONENTS:
    _write_magnetic6_numerical_npz(config, fem_result)""",
    )
    text = _insert_before_function_exit(
        text,
        "_save_forward_partial",
        PARTIAL_SAVE_MARKER,
        """# MAGNETIC6_PRODUCTION_PARTIAL_SAVE
if tuple(str(value) for value in components) == MAGNETIC6_COMPONENTS:
    _write_magnetic6_numerical_npz(
        config,
        {
            "times": times,
            "data": rows,
            "components": components,
        },
    )""",
    )
    return text


def check_policy(text: str | None = None) -> None:
    source = PIPELINE.read_text(encoding="utf-8") if text is None else text
    ast.parse(source)
    required = {
        "constants": CONSTANT_MARKER,
        "config": CONFIG_MARKER,
        "cli": "--magnetic-output-contract",
        "forward gate": FORWARD_MARKER,
        "Hx assignment": 'receiver_values["Hx"] = values[0]',
        "Hy assignment": 'receiver_values["Hy"] = values[1]',
        "dBxdt receiver": '"dBxdt": float(dbdt_val[0])',
        "dBydt receiver": '"dBydt": float(dbdt_val[1])',
        "H-form Hx": '"Hx": float(h_new[0])',
        "writer": "def _write_magnetic6_numerical_npz",
        "final save": SAVE_MARKER,
        "partial save": PARTIAL_SAVE_MARKER,
        "resolved config": '"magnetic_output_contract": str(config.magnetic_output_contract)',
    }
    missing = [label for label, token in required.items() if token not in source]
    if missing:
        raise RuntimeError("magnetic-six production policy incomplete: " + ", ".join(missing))
    forbidden = {
        "z-only curl receiver": (
            'rec = {"Ex": float(e_val[0]), "Ey": float(e_val[1]), '
            '"dBzdt": float(dbdt_val[2])}'
        ),
        "z-only Biot assignment": 'receiver_values["Hz"] = float(h_receiver[2])',
        "z-only H-form return": (
            'return {"Ex": float(e_val[0]), "Ey": float(e_val[1]), '
            '"Hz": float(h_new[2]), "dBzdt": float(dbdt[2])}'
        ),
    }
    present = [label for label, token in forbidden.items() if token in source]
    if present:
        raise RuntimeError("legacy z-only production path remains: " + ", ".join(present))


def apply_policy() -> bool:
    original = PIPELINE.read_text(encoding="utf-8")
    updated = _apply(original)
    check_policy(updated)
    if updated == original:
        return False
    PIPELINE.write_text(updated, encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.apply:
        changed = apply_policy()
        print("updated" if changed else "already-applied")
    else:
        check_policy()
        print("magnetic-six production policy: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
