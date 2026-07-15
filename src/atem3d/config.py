"""YAML configuration loading and simulation construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from discretize import TensorMesh

from .fit import (
    DebyeFitResult,
    fit_cole_cole_conductivity_debye,
    fit_pelton_resistivity_debye,
)
from .boundary import BoundaryConfig, apply_boundary
from .cpml import CPMLConfig
from .hj import HJMagneticSimulation
from .ip import DebyeIPModel, DebyeTerm
from .receivers import PointReceiver, build_receiver
from .source_history_runtime import (
    ChargeConservingInitialPolarizationSourceHistoryCorrection,
    DrivenRecoverySourceHistoryCorrection,
    InitialPolarizationSourceHistoryCorrection,
    SourceDiffusionKernelSourceHistoryCorrection,
    SourceHistoryCorrection,
    SourcePrimaryDelta6SourceHistoryCorrection,
    TimeSeriesSourceHistoryCorrection,
)
from .simulation import TDEMIPSimulation
from .sources import (
    GroundedWireSource,
    LinearRampOffWaveform,
    StepOffWaveform,
    TabulatedWaveform,
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config


def build_simulation(config: dict[str, Any]) -> TDEMIPSimulation | HJMagneticSimulation:
    """Build an EB or H/J simulation from a config dictionary."""

    mesh = _build_mesh(config["mesh"])
    formulation = str(config.get("formulation", "eb")).lower()
    if formulation not in {"eb", "hj"}:
        raise ValueError("formulation must be 'eb' or 'hj'")
    model_cfg = dict(config["model"])
    model_cfg.setdefault("coordinate_system", config.get("coordinate_system", "depth_down"))
    sigma_inf, terms = _build_ip_model_properties(mesh, model_cfg)
    physical_ip_model = DebyeIPModel(
        sigma_inf.copy(),
        [DebyeTerm(delta_sigma=term.delta_sigma.copy(), tau=term.tau) for term in terms],
    )

    boundary_cfg = _build_boundary_config(config)
    if (
        formulation == "hj"
        and boundary_cfg.kind == "cpml"
        and boundary_cfg.thickness_cells > 0
    ):
        raise ValueError("H/J formulation does not support active CPML")
    cpml = _build_cpml_config(config) if boundary_cfg.kind == "cpml" else None
    if cpml is None:
        sigma_inf, term_pairs = apply_boundary(
            mesh,
            sigma_inf,
            [(term.tau, term.delta_sigma) for term in terms],
            boundary_cfg,
        )
    else:
        term_pairs = [(term.tau, term.delta_sigma.copy()) for term in terms]
        sigma_inf = sigma_inf.copy()
    terms = [DebyeTerm(delta_sigma=delta, tau=tau) for tau, delta in term_pairs]

    source = _build_source(config["source"])
    receivers = _build_receivers(config)
    ip_model = DebyeIPModel(sigma_inf, terms)
    initial_ip_model = ip_model if boundary_cfg.apply_to_initial else physical_ip_model

    if formulation == "hj":
        return HJMagneticSimulation(
            mesh=mesh,
            ip_model=ip_model,
            initial_ip_model=initial_ip_model,
            time_steps=_build_time_steps(config["time_steps"]),
            sources=[source],
            receivers=receivers,
            linear_solver=str(config.get("solver", {}).get("type", "direct")),
            cg_tolerance=float(config.get("solver", {}).get("tolerance", 1.0e-8)),
            cg_maxiter=config.get("solver", {}).get("maxiter"),
            cg_preconditioner=str(config.get("solver", {}).get("preconditioner", "jacobi")),
            magnetic_receiver_mode=str(config.get("magnetic_receiver_mode", "stored_h")),
            magnetic_recovery_subdivisions=int(config.get("magnetic_recovery_subdivisions", 1)),
            magnetic_recovery_polarization_scale=config.get(
                "magnetic_recovery_polarization_scale",
                1.0,
            ),
            magnetic_recovery_initial_polarization_scale=float(
                config.get("magnetic_recovery_initial_polarization_scale", 0.0)
            ),
            magnetic_recovery_source_primary_delta6=bool(
                config.get("magnetic_recovery_source_primary_delta6", False)
            ),
            magnetic_recovery_source_primary_delta6_basis=str(
                config.get("magnetic_recovery_source_primary_delta6_basis", "wire")
            ),
            magnetic_recovery_source_history=_build_source_history_correction(config),
        )

    return TDEMIPSimulation(
        mesh=mesh,
        ip_model=ip_model,
        initial_ip_model=initial_ip_model,
        time_steps=_build_time_steps(config["time_steps"]),
        sources=[source],
        receivers=receivers,
        initial_magnetic_mode=str(config.get("initial_magnetic_field", "ampere")),
        linear_solver=str(config.get("solver", {}).get("type", "direct")),
        cg_tolerance=float(config.get("solver", {}).get("tolerance", 1.0e-8)),
        cg_maxiter=config.get("solver", {}).get("maxiter"),
        cg_preconditioner=str(config.get("solver", {}).get("preconditioner", "jacobi")),
        cpml=cpml,
        magnetic_receiver_mode=str(config.get("magnetic_receiver_mode", "stored_b")),
        magnetic_recovery_subdivisions=int(config.get("magnetic_recovery_subdivisions", 1)),
        magnetic_recovery_polarization_scale=config.get(
            "magnetic_recovery_polarization_scale",
            1.0,
        ),
        magnetic_recovery_initial_polarization_scale=float(
            config.get("magnetic_recovery_initial_polarization_scale", 0.0)
        ),
        magnetic_recovery_source_primary_delta6=bool(
            config.get("magnetic_recovery_source_primary_delta6", False)
        ),
        magnetic_recovery_source_primary_delta6_basis=str(
            config.get("magnetic_recovery_source_primary_delta6_basis", "wire")
        ),
        magnetic_recovery_source_history=_build_source_history_correction(config),
    )


def _build_mesh(mesh_cfg: dict[str, Any]) -> TensorMesh:
    h = [_mesh_widths(mesh_cfg["hx"]), _mesh_widths(mesh_cfg["hy"]), _mesh_widths(mesh_cfg["hz"])]
    return TensorMesh(h, origin=mesh_cfg.get("origin"))


def _mesh_widths(widths: Any) -> Any:
    if isinstance(widths, list):
        converted = []
        for item in widths:
            if isinstance(item, list):
                converted.append(tuple(item))
            else:
                converted.append(item)
        return converted
    return widths


def _build_time_steps(time_steps: Any) -> list[float]:
    expanded: list[float] = []
    for item in time_steps:
        if isinstance(item, list):
            if len(item) != 2:
                raise ValueError("time step repeat entries must be [dt, repeat]")
            dt = float(item[0])
            repeat = int(item[1])
            if repeat <= 0:
                raise ValueError("time step repeat counts must be positive")
            expanded.extend([dt] * repeat)
        else:
            expanded.append(float(item))
    return expanded


def _build_source(source_cfg: dict[str, Any]) -> GroundedWireSource:
    waveform_cfg = source_cfg.get("waveform", {"type": "step_off", "off_time": 0.0})
    waveform_type = waveform_cfg.get("type", "step_off")
    source_current = float(source_cfg["current"])
    if waveform_type == "step_off":
        waveform = StepOffWaveform(
            off_time=float(waveform_cfg.get("off_time", 0.0)),
            on_value=float(waveform_cfg.get("on_value", 1.0)),
        )
    elif waveform_type == "linear_ramp_off":
        if source_current == 0.0:
            raise ValueError("current must be nonzero")
        current_initial = float(
            waveform_cfg.get(
                "current_initial",
                source_current * float(waveform_cfg.get("initial_value", 1.0)),
            )
        )
        current_final = float(
            waveform_cfg.get(
                "current_final",
                source_current * float(waveform_cfg.get("final_value", 0.0)),
            )
        )
        waveform = LinearRampOffWaveform(
            off_time=float(waveform_cfg.get("t_off", waveform_cfg.get("off_time", 0.0))),
            initial_value_scale=current_initial / source_current,
            final_value_scale=current_final / source_current,
        )
    elif waveform_type == "tabulated":
        waveform = _build_tabulated_source_waveform(waveform_cfg, source_current)
    else:
        raise ValueError(f"unsupported waveform type: {waveform_type}")

    return GroundedWireSource(
        start=tuple(source_cfg["start"]),
        end=tuple(source_cfg["end"]),
        current=source_current,
        waveform=waveform,
        face_projection=str(source_cfg.get("face_projection", "auto")),
    )


def _build_tabulated_source_waveform(
    waveform_cfg: dict[str, Any],
    source_current: float,
) -> TabulatedWaveform:
    path = waveform_cfg.get("path", waveform_cfg.get("csv_path"))
    if path is not None:
        if source_current == 0.0:
            raise ValueError("current must be nonzero")
        table = np.genfromtxt(path, delimiter=",", names=True)
        if table.dtype.names is None or "time" not in table.dtype.names or "current" not in table.dtype.names:
            raise ValueError("tabulated waveform CSV must contain time,current columns")
        times = np.atleast_1d(np.asarray(table["time"], dtype=float))
        currents = np.atleast_1d(np.asarray(table["current"], dtype=float))
        values = currents / source_current
        if "initial_field_current" in waveform_cfg:
            initial_field_value = float(waveform_cfg["initial_field_current"]) / source_current
        else:
            initial_field_value = float(waveform_cfg.get("initial_field_value", values[0]))
        return TabulatedWaveform(
            times=times,
            values=values,
            initial_field_value=initial_field_value,
        )
    if "currents" in waveform_cfg:
        if source_current == 0.0:
            raise ValueError("current must be nonzero")
        currents = np.asarray(waveform_cfg["currents"], dtype=float)
        values = currents / source_current
        initial_field_value = float(waveform_cfg.get("initial_field_value", values[0]))
    else:
        values = np.asarray(waveform_cfg["values"], dtype=float)
        initial_field_value = float(waveform_cfg.get("initial_field_value", 0.0))
    return TabulatedWaveform(
        times=np.asarray(waveform_cfg["times"], dtype=float),
        values=values,
        initial_field_value=initial_field_value,
    )


def _build_source_history_correction(
    config: dict[str, Any],
) -> (
    SourceHistoryCorrection
    | InitialPolarizationSourceHistoryCorrection
    | ChargeConservingInitialPolarizationSourceHistoryCorrection
    | DrivenRecoverySourceHistoryCorrection
    | SourceDiffusionKernelSourceHistoryCorrection
    | SourcePrimaryDelta6SourceHistoryCorrection
    | TimeSeriesSourceHistoryCorrection
    | tuple[
        SourceHistoryCorrection
        | InitialPolarizationSourceHistoryCorrection
        | ChargeConservingInitialPolarizationSourceHistoryCorrection
        | DrivenRecoverySourceHistoryCorrection
        | SourceDiffusionKernelSourceHistoryCorrection
        | SourcePrimaryDelta6SourceHistoryCorrection
        | TimeSeriesSourceHistoryCorrection,
        ...,
    ]
    | None
):
    cfg = config.get("magnetic_recovery_source_history")
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        raise ValueError("magnetic_recovery_source_history must be a mapping")
    if "terms" in cfg:
        terms = cfg["terms"]
        if not isinstance(terms, list) or not terms:
            raise ValueError("magnetic_recovery_source_history.terms must be a nonempty list")
        return tuple(_source_history_correction_from_mapping(term) for term in terms)
    return _source_history_correction_from_mapping(cfg)


def _source_history_correction_from_mapping(
    cfg: dict[str, Any],
) -> (
    SourceHistoryCorrection
    | InitialPolarizationSourceHistoryCorrection
    | ChargeConservingInitialPolarizationSourceHistoryCorrection
    | DrivenRecoverySourceHistoryCorrection
    | SourceDiffusionKernelSourceHistoryCorrection
    | SourcePrimaryDelta6SourceHistoryCorrection
    | TimeSeriesSourceHistoryCorrection
):
    if not isinstance(cfg, dict):
        raise ValueError("each magnetic_recovery_source_history term must be a mapping")
    kind = str(cfg.get("kind", "prescribed_source_moments"))
    if kind.strip().lower() == "initial_polarization_source_moments":
        return InitialPolarizationSourceHistoryCorrection(
            kind=kind,
            source_moment_degrees=tuple(cfg.get("source_moment_degrees", [0, 2])),
            receiver_matrix=str(cfg.get("receiver_matrix", "auto")),
            projection=str(cfg.get("projection", "receiver_l2")),
            source_edge_atol=float(cfg.get("source_edge_atol", 0.0)),
        )
    if kind.strip().lower() == "charge_conserving_initial_polarization_source_moments":
        return ChargeConservingInitialPolarizationSourceHistoryCorrection(
            kind=kind,
            source_moment_degrees=tuple(cfg.get("source_moment_degrees", [0, 2])),
            receiver_matrix=str(cfg.get("receiver_matrix", "auto")),
            projection=str(cfg.get("projection", "receiver_l2")),
            source_edge_atol=float(cfg.get("source_edge_atol", 0.0)),
        )
    if kind.strip().lower() == "driven_recovery_source_moments":
        return DrivenRecoverySourceHistoryCorrection(
            kind=kind,
            driver_tau=float(cfg["driver_tau"]),
            response_tau=_driven_recovery_response_tau(cfg),
            source_moment_degrees=tuple(cfg.get("source_moment_degrees", [0, 2])),
            coefficients=(
                tuple(cfg["coefficients"]) if "coefficients" in cfg else None
            ),
            normalized_coefficients=(
                tuple(cfg["normalized_coefficients"])
                if "normalized_coefficients" in cfg
                else None
            ),
            receiver_matrix=str(cfg.get("receiver_matrix", "auto")),
            source_edge_atol=float(cfg.get("source_edge_atol", 0.0)),
        )
    if kind.strip().lower() == "time_series_source_moments":
        return TimeSeriesSourceHistoryCorrection(
            kind=kind,
            times=tuple(cfg["times"]),
            source_moment_degrees=tuple(cfg.get("source_moment_degrees", [0, 2])),
            coefficients=tuple(cfg["coefficients"]),
            receiver_matrix=str(cfg.get("receiver_matrix", "auto")),
            source_edge_atol=float(cfg.get("source_edge_atol", 0.0)),
        )
    if kind.strip().lower() == "source_diffusion_kernel_source_moments":
        has_amplitude = "amplitude" in cfg
        has_coefficients = "coefficients" in cfg
        has_normalized = "normalized_amplitude" in cfg
        if sum([has_amplitude, has_coefficients, has_normalized]) != 1:
            raise ValueError(
                "source_diffusion_kernel_source_moments requires exactly one "
                "of amplitude, coefficients, or normalized_amplitude"
            )
        return SourceDiffusionKernelSourceHistoryCorrection(
            kind=kind,
            amplitude=float(cfg.get("amplitude", 0.0)),
            normalized_amplitude=(
                float(cfg["normalized_amplitude"]) if has_normalized else None
            ),
            tau_multiplier=float(cfg.get("tau_multiplier", 1.0)),
            amplitude_time=float(cfg.get("amplitude_time", 0.0)),
            basis_kind=str(cfg.get("basis_kind", "continuous")),
            source_moment_degrees=tuple(cfg.get("source_moment_degrees", [0])),
            coefficients=(
                tuple(cfg["coefficients"]) if has_coefficients else None
            ),
            receiver_matrix=str(cfg.get("receiver_matrix", "auto")),
            source_edge_atol=float(cfg.get("source_edge_atol", 0.0)),
        )
    if kind.strip().lower() == "source_primary_delta6_source_moments":
        return SourcePrimaryDelta6SourceHistoryCorrection(
            kind=kind,
            source_moment_degrees=tuple(cfg.get("source_moment_degrees", [0])),
            receiver_matrix=str(cfg.get("receiver_matrix", "auto")),
            source_edge_atol=float(cfg.get("source_edge_atol", 0.0)),
        )
    return SourceHistoryCorrection(
        kind=kind,
        tau=float(cfg["tau"]),
        max_order=int(cfg.get("max_order", 1)),
        source_moment_degrees=tuple(cfg.get("source_moment_degrees", [0, 2])),
        coefficients=(
            tuple(cfg["coefficients"]) if "coefficients" in cfg else None
        ),
        normalized_coefficients=(
            tuple(cfg["normalized_coefficients"])
            if "normalized_coefficients" in cfg
            else None
        ),
        receiver_matrix=str(cfg.get("receiver_matrix", "auto")),
        source_edge_atol=float(cfg.get("source_edge_atol", 0.0)),
    )


def _driven_recovery_response_tau(cfg: dict[str, Any]) -> Any:
    has_scalar = "response_tau" in cfg
    has_vector = "response_taus" in cfg
    if has_scalar == has_vector:
        raise ValueError(
            "driven_recovery_source_moments requires exactly one of "
            "response_tau or response_taus"
        )
    return cfg["response_tau"] if has_scalar else cfg["response_taus"]


def _build_receivers(config: dict[str, Any]) -> list:
    receivers = [
        build_receiver(
            location=tuple(rx["location"]),
            component=str(rx["component"]),
            receiver_type=str(rx.get("type", rx.get("receiver_type", "point"))),
            radius=rx.get("radius"),
        )
        for rx in config.get("receivers", [])
    ]
    line_cfg = config.get("receiver_line")
    if line_cfg:
        xs = np.asarray(line_cfg["x"], dtype=float)
        y = float(line_cfg["y"])
        z = float(line_cfg["z"])
        components = [str(component) for component in line_cfg["components"]]
        for x in xs:
            for component in components:
                receivers.append(PointReceiver(location=(float(x), y, z), component=component))
    return receivers


def _build_ip_model_properties(mesh: TensorMesh, model_cfg: dict[str, Any]) -> tuple[np.ndarray, list[DebyeTerm]]:
    if "layers" not in model_cfg:
        sigma_value, scalar_terms = _layer_debye_model(model_cfg, model_cfg)
        sigma_inf = _cell_property(mesh, sigma_value, "sigma_infinity")
        delta_by_tau = {
            float(term.tau): _cell_property(mesh, term.delta_sigma[0], "delta_sigma")
            for term in scalar_terms
        }
        sigma_inf, delta_by_tau = _apply_conductivity_boxes(
            mesh, sigma_inf, delta_by_tau, model_cfg
        )
        terms = [DebyeTerm(delta_sigma=values, tau=tau) for tau, values in delta_by_tau.items()]
        return sigma_inf, terms

    layers = model_cfg["layers"]
    _validate_layer_boundaries(mesh, layers, model_cfg)
    sigma_inf = np.zeros(mesh.n_cells, dtype=float)
    layer_models = [_layer_debye_model(layer, model_cfg) for layer in layers]
    tau_values = sorted(
        {
            float(term.tau)
            for _, terms in layer_models
            for term in terms
        }
    )
    delta_by_tau = {tau: np.zeros(mesh.n_cells, dtype=float) for tau in tau_values}
    z = mesh.cell_centers[:, 2]
    assigned = np.zeros(mesh.n_cells, dtype=bool)

    for layer, (layer_sigma_inf, terms) in zip(layers, layer_models):
        mask = _layer_cell_mask(z, layer, str(model_cfg.get("coordinate_system", "depth_down")))
        sigma_inf[mask] = layer_sigma_inf
        assigned |= mask
        for term in terms:
            delta_by_tau[float(term.tau)][mask] = float(term.delta_sigma[0])

    if not np.all(assigned):
        raise ValueError("layer definitions must cover every mesh cell center")

    sigma_inf, delta_by_tau = _apply_conductivity_boxes(
        mesh, sigma_inf, delta_by_tau, model_cfg
    )
    terms = [
        DebyeTerm(delta_sigma=delta_by_tau[tau], tau=tau)
        for tau in tau_values
    ]
    return sigma_inf, terms


def _apply_conductivity_boxes(
    mesh: TensorMesh,
    sigma_inf: np.ndarray,
    delta_by_tau: dict[float, np.ndarray],
    model_cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    centers = np.asarray(mesh.cell_centers, dtype=float)
    for box in model_cfg.get("conductivity_boxes", []):
        bounds = np.asarray(box["bounds"], dtype=float)
        if bounds.shape != (3, 2) or np.any(bounds[:, 1] <= bounds[:, 0]):
            raise ValueError(
                "conductivity box bounds must have shape (3, 2) with upper > lower"
            )
        mask = np.logical_and.reduce(
            [
                (centers[:, axis] >= bounds[axis, 0])
                & (centers[:, axis] <= bounds[axis, 1])
                for axis in range(3)
            ]
        )
        name = str(box.get("name", "<unnamed>"))
        if not np.any(mask):
            raise ValueError(f"conductivity box {name} marks no cells")
        minimum = int(box.get("minimum_cells_per_cross_section", 1))
        if minimum < 1:
            raise ValueError("minimum_cells_per_cross_section must be at least 1")
        for axis in (1, 2):
            count = np.unique(np.round(centers[mask, axis], decimals=12)).size
            if count < minimum:
                raise ValueError(
                    f"conductivity box {name} requires at least {minimum} cells "
                    f"across axis {axis}; found {count}"
                )
        conductivity = float(box["sigma_infinity"])
        if not np.isfinite(conductivity) or conductivity <= 0.0:
            raise ValueError("conductivity box sigma_infinity must be positive and finite")
        sigma_inf[mask] = conductivity
        for values in delta_by_tau.values():
            values[mask] = 0.0
    return sigma_inf, delta_by_tau


def _validate_layer_boundaries(
    mesh: TensorMesh,
    layers: list[dict[str, Any]],
    model_cfg: dict[str, Any],
) -> None:
    if not bool(model_cfg.get("require_layer_boundary_alignment", False)):
        return

    nodes_z = np.asarray(mesh.nodes_z, dtype=float)
    tolerance = float(model_cfg.get("layer_boundary_tolerance", 1.0e-8))
    boundaries = sorted(
        {
            float(value)
            for layer in layers
            for value in (layer["top"], layer["bottom"])
            if np.isfinite(float(value)) and abs(float(value)) < 1.0e8
        }
    )
    for boundary in boundaries:
        if not np.any(np.isclose(nodes_z, boundary, rtol=0.0, atol=tolerance)):
            raise ValueError(
                f"layer boundary z={boundary} must align with a mesh node "
                "when require_layer_boundary_alignment is true"
            )


def _layer_cell_mask(
    z: np.ndarray,
    layer: dict[str, Any],
    coordinate_system: str,
) -> np.ndarray:
    top = float(layer["top"])
    bottom = float(layer["bottom"])
    if coordinate_system == "depth_down":
        return (z >= top) & (z < bottom)
    if coordinate_system == "z_up":
        return (z <= top) & (z > bottom)
    raise ValueError("coordinate_system must be 'z_up' or 'depth_down'")


def _layer_debye_terms(layer: dict[str, Any], model_cfg: dict[str, Any]) -> list[DebyeTerm]:
    _, terms = _layer_debye_model(layer, model_cfg)
    return terms


def _layer_debye_model(layer: dict[str, Any], model_cfg: dict[str, Any]) -> tuple[float, list[DebyeTerm]]:
    sigma_infinity = float(layer["sigma_infinity"])
    terms = [
        DebyeTerm(delta_sigma=float(term_cfg["delta_sigma"]), tau=float(term_cfg["tau"]))
        for term_cfg in layer.get("debye_terms", [])
    ]
    if "ip_model" not in layer:
        return sigma_infinity, terms

    ip_cfg = layer["ip_model"]
    frequencies = np.asarray(ip_cfg.get("fit_frequencies", model_cfg.get("fit_frequencies")), dtype=float)
    if frequencies.size == 0:
        raise ValueError("ip_model requires fit_frequencies in the layer or model")
    tau_grid = ip_cfg.get("tau_grid")
    if tau_grid is not None:
        tau_grid = np.asarray(tau_grid, dtype=float)
    n_terms = int(ip_cfg.get("n_terms", 10))
    model_type = ip_cfg["type"]
    fit: DebyeFitResult
    if model_type == "cole_cole_conductivity":
        fit = fit_cole_cole_conductivity_debye(
            sigma_infinity=float(ip_cfg.get("sigma_infinity", layer["sigma_infinity"])),
            eta=float(ip_cfg["eta"]),
            tau=float(ip_cfg["tau"]),
            c=float(ip_cfg["c"]),
            frequencies=frequencies,
            tau_grid=tau_grid,
            n_terms=n_terms,
        )
    elif model_type == "pelton":
        fit = fit_pelton_resistivity_debye(
            rho0=float(ip_cfg["rho0"]),
            chargeability=float(ip_cfg["chargeability"]),
            tau=float(ip_cfg["tau"]),
            c=float(ip_cfg["c"]),
            frequencies=frequencies,
            tau_grid=tau_grid,
            n_terms=n_terms,
        )
    else:
        raise ValueError("ip_model.type must be 'cole_cole_conductivity' or 'pelton'")
    terms.extend(fit.terms)
    return fit.sigma_infinity, terms


def _cell_property(mesh: TensorMesh, value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(mesh.n_cells, float(array))
    if array.shape == (mesh.n_cells,):
        return array
    raise ValueError(f"{name} must be scalar or have length mesh.n_cells={mesh.n_cells}")


def _build_boundary_config(config: dict[str, Any]) -> BoundaryConfig:
    if "boundary" in config:
        cfg = config["boundary"]
        return BoundaryConfig(
            kind=str(cfg.get("kind", "none")),
            thickness_cells=int(cfg.get("thickness_cells", 0)),
            strength=float(cfg.get("strength", 0.0)),
            power=float(cfg.get("power", 2.0)),
            disable_ip_in_shell=bool(cfg.get("disable_ip_in_shell", True)),
            apply_to_initial=bool(cfg.get("apply_to_initial", True)),
            sides=_boundary_sides(cfg.get("sides")),
        )
    if "sponge" in config:
        cfg = config["sponge"]
        return BoundaryConfig(
            kind="sponge",
            thickness_cells=int(cfg.get("thickness_cells", 0)),
            strength=float(cfg.get("strength", 0.0)),
            power=float(cfg.get("power", 2.0)),
            disable_ip_in_shell=bool(cfg.get("disable_ip_in_sponge", True)),
            apply_to_initial=bool(cfg.get("apply_to_initial", True)),
            sides=_boundary_sides(cfg.get("sides")),
        )
    return BoundaryConfig()


def _boundary_sides(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return tuple(side.strip() for side in value.split(",") if side.strip())
    return tuple(str(side) for side in value)


def _build_cpml_config(config: dict[str, Any]) -> CPMLConfig:
    cfg = dict(config.get("boundary", {}))
    sigma_max = cfg.get("sigma_max", cfg.get("strength", 0.0))
    return CPMLConfig(
        thickness_cells=int(cfg.get("thickness_cells", 0)),
        sigma_max=float(sigma_max),
        alpha_max=float(cfg.get("alpha_max", 0.0)),
        kappa_max=float(cfg.get("kappa_max", 1.0)),
        power=float(cfg.get("power", 2.0)),
    )
