"""H/J-form finite-volume assembly helpers for Debye IP prototypes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.constants import mu_0

from .ip import DebyeIPModel, DebyeTerm
from .magnetic_recovery import (
    biot_savart_h_from_face_basis_cell_ip_currents,
    biot_savart_h_from_face_basis_currents,
    face_current_biot_matrix,
)
from .source_history_runtime import (
    ChargeConservingInitialPolarizationSourceHistoryCorrection,
    DrivenRecoverySourceHistoryCorrection,
    InitialPolarizationSourceHistoryCorrection,
    SourceHistoryCorrection,
    SourcePrimaryDelta6SourceHistoryCorrection,
    source_history_correction_field,
    source_history_correction_requires_ip,
    source_history_ip_model_has_contrast,
    source_history_correction_terms,
)


@dataclass(frozen=True)
class HJMagneticStepResult:
    """Fields from one H/J magnetic-field prototype time step."""

    h: np.ndarray
    e: np.ndarray
    memories: list[np.ndarray]


@dataclass(frozen=True)
class HJMagneticSimulationResult:
    """Time-node fields from the H/J magnetic-field prototype."""

    times: np.ndarray
    h: np.ndarray
    e: np.ndarray
    memories: list[np.ndarray]
    data: np.ndarray


@dataclass(frozen=True)
class HJReceiverDataResult:
    """Receiver-only data returned by memory-saving H/J runs."""

    times: np.ndarray
    data: np.ndarray
    memories: list[np.ndarray]


@dataclass
class HJMagneticSimulation:
    """Minimal multi-step H/J magnetic-field prototype."""

    mesh: object
    ip_model: DebyeIPModel
    time_steps: Sequence[float]
    initial_ip_model: DebyeIPModel | None = None
    sources: Sequence[object] | None = None
    receivers: Sequence[object] | None = None
    initial_h: np.ndarray | None = None
    initial_e: np.ndarray | None = None
    initial_memories: list[np.ndarray] | None = None
    mu: float = mu_0
    linear_solver: str = "direct"
    cg_tolerance: float = 1.0e-8
    cg_maxiter: int | None = None
    cg_preconditioner: str = "jacobi"
    magnetic_receiver_mode: str = "stored_h"
    magnetic_recovery_subdivisions: int = 1
    magnetic_recovery_polarization_scale: float | str | np.ndarray = 1.0
    magnetic_recovery_initial_polarization_scale: float = 0.0
    magnetic_recovery_source_primary_delta6: bool = False
    magnetic_recovery_source_primary_delta6_basis: str = "wire"
    magnetic_recovery_source_history: (
        SourceHistoryCorrection
        | InitialPolarizationSourceHistoryCorrection
        | ChargeConservingInitialPolarizationSourceHistoryCorrection
        | DrivenRecoverySourceHistoryCorrection
        | SourcePrimaryDelta6SourceHistoryCorrection
        | Sequence[
            SourceHistoryCorrection
            | InitialPolarizationSourceHistoryCorrection
            | ChargeConservingInitialPolarizationSourceHistoryCorrection
            | DrivenRecoverySourceHistoryCorrection
            | SourcePrimaryDelta6SourceHistoryCorrection
        ]
        | None
    ) = None

    def __post_init__(self) -> None:
        self.time_steps = np.asarray(self.time_steps, dtype=float)
        if self.time_steps.ndim != 1 or self.time_steps.size == 0:
            raise ValueError("time_steps must be a nonempty 1D sequence")
        if np.any(self.time_steps <= 0.0):
            raise ValueError("all time steps must be positive")
        if self.initial_ip_model is None:
            self.initial_ip_model = self.ip_model
        self.sources = list(self.sources or [])
        self.receivers = list(self.receivers or [])
        if self.initial_h is not None and np.asarray(self.initial_h).shape != (self.mesh.n_edges,):
            raise ValueError("initial_h must have shape (mesh.n_edges,)")
        if self.initial_e is not None and np.asarray(self.initial_e).shape != (self.mesh.n_faces,):
            raise ValueError("initial_e must have shape (mesh.n_faces,)")
        if self.initial_memories is not None:
            _validate_face_memories(self.mesh, self.ip_model, self.initial_memories)
        if self.linear_solver not in {"direct", "cg", "pardiso"}:
            raise ValueError("linear_solver must be 'direct', 'cg', or 'pardiso'")
        if self.cg_tolerance <= 0.0:
            raise ValueError("cg_tolerance must be positive")
        if self.cg_maxiter is not None and self.cg_maxiter <= 0:
            raise ValueError("cg_maxiter must be positive or None")
        if self.cg_preconditioner not in {"none", "jacobi"}:
            raise ValueError("cg_preconditioner must be 'none' or 'jacobi'")
        if self.magnetic_receiver_mode not in {
            "stored_h",
            "current_biot",
            "face_basis_biot",
            "face_basis_cell_biot",
        }:
            raise ValueError(
                "magnetic_receiver_mode must be 'stored_h', 'current_biot', "
                "'face_basis_biot', or 'face_basis_cell_biot'"
            )
        if (
            self.magnetic_recovery_source_history is not None
            and self.magnetic_receiver_mode == "stored_h"
        ):
            raise ValueError(
                "magnetic_recovery_source_history requires a Biot magnetic receiver mode"
            )
        if self.magnetic_recovery_source_history is not None:
            source_history_correction_terms(self.magnetic_recovery_source_history)
        self._source_history_runtime_cache = {}
        self.magnetic_recovery_subdivisions = int(self.magnetic_recovery_subdivisions)
        if self.magnetic_recovery_subdivisions <= 0:
            raise ValueError("magnetic_recovery_subdivisions must be positive")
        self.magnetic_recovery_polarization_scale = _normalize_polarization_scale(
            self.magnetic_recovery_polarization_scale
        )
        self.magnetic_recovery_initial_polarization_scale = (
            _normalize_initial_polarization_scale(
                self.magnetic_recovery_initial_polarization_scale
            )
        )
        self.magnetic_recovery_source_primary_delta6 = bool(
            self.magnetic_recovery_source_primary_delta6
        )
        self.magnetic_recovery_source_primary_delta6_basis = str(
            self.magnetic_recovery_source_primary_delta6_basis
        ).strip().lower()
        if self.magnetic_recovery_source_primary_delta6_basis not in {
            "wire",
            "face_current",
        }:
            raise ValueError(
                "magnetic_recovery_source_primary_delta6_basis must be "
                "'wire' or 'face_current' for H/J"
            )
        if (
            self.magnetic_recovery_source_primary_delta6
            and self.magnetic_receiver_mode == "stored_h"
        ):
            raise ValueError(
                "magnetic_recovery_source_primary_delta6 requires a Biot magnetic "
                "receiver mode"
            )
        self._matrix_cache: dict[int, sp.csr_matrix] = {}
        self._factor_cache: dict[int, object] = {}
        self._current_biot_matrix_cache: dict[tuple[int, tuple[int, int], tuple[float, ...]], np.ndarray] = {}

    @property
    def times(self) -> np.ndarray:
        return np.r_[0.0, np.cumsum(self.time_steps)]

    def electric_source_term(self, time: float) -> np.ndarray:
        source = np.zeros(self.mesh.n_faces, dtype=float)
        for item in self.sources:
            source -= item.face_vector_at(self.mesh, time)
        return source

    def run(self) -> HJMagneticSimulationResult:
        n_times = self.time_steps.size + 1
        h = np.zeros((n_times, self.mesh.n_edges), dtype=float)
        e = np.zeros((n_times, self.mesh.n_faces), dtype=float)
        data = np.zeros((n_times, len(self.receivers)), dtype=float)
        if self.initial_h is not None:
            h[0] = np.asarray(self.initial_h, dtype=float)
        else:
            h[0] = hj_mmr_initial_magnetic_field(
                self.mesh,
                self.initial_ip_model,
                self.sources,
                mu=self.mu,
            )
        if self.initial_e is not None:
            e[0] = np.asarray(self.initial_e, dtype=float)
        else:
            e[0] = hj_dc_initial_electric_field(
                self.mesh,
                self.initial_ip_model,
                self.sources,
            )

        memories = self._initial_memories(e[0])
        initial_memories = [memory.copy() for memory in memories]
        memory_history = [np.zeros((n_times, self.mesh.n_faces), dtype=float) for _ in memories]
        for index, memory in enumerate(memories):
            memory_history[index][0] = memory
        data[0] = self._sample_receivers(
            e[0],
            h[0],
            time=0.0,
            memories=memories,
            initial_memories=initial_memories,
        )

        for step_index, dt in enumerate(self.time_steps):
            new_time = float(self.times[step_index + 1])
            step = self._magnetic_step(
                step_index,
                h[step_index],
                memories,
                float(dt),
                electric_source=self.electric_source_term(new_time),
            )
            h[step_index + 1] = step.h
            e[step_index + 1] = step.e
            data[step_index + 1] = self._sample_receivers_time_derivative(
                e[step_index + 1],
                h[step_index + 1],
                h[step_index],
                float(dt),
                time=new_time,
                memories=step.memories,
                initial_memories=initial_memories,
            )
            memories = step.memories
            for index, memory in enumerate(memories):
                memory_history[index][step_index + 1] = memory

        return HJMagneticSimulationResult(self.times, h, e, memory_history, data)

    def run_data_only(self) -> HJReceiverDataResult:
        """Run the H/J simulation while storing receiver data but not field histories."""

        n_times = self.time_steps.size + 1
        data = np.zeros((n_times, len(self.receivers)), dtype=float)

        if self.initial_h is not None:
            h_old = np.asarray(self.initial_h, dtype=float).copy()
        else:
            h_old = hj_mmr_initial_magnetic_field(
                self.mesh,
                self.initial_ip_model,
                self.sources,
                mu=self.mu,
            )
        if self.initial_e is not None:
            e_old = np.asarray(self.initial_e, dtype=float).copy()
        else:
            e_old = hj_dc_initial_electric_field(
                self.mesh,
                self.initial_ip_model,
                self.sources,
            )

        memories = self._initial_memories(e_old)
        initial_memories = [memory.copy() for memory in memories]
        data[0] = self._sample_receivers(
            e_old,
            h_old,
            time=0.0,
            memories=memories,
            initial_memories=initial_memories,
        )

        times = self.times
        for step_index, dt in enumerate(self.time_steps):
            new_time = float(times[step_index + 1])
            step = self._magnetic_step(
                step_index,
                h_old,
                memories,
                float(dt),
                electric_source=self.electric_source_term(new_time),
            )
            data[step_index + 1] = self._sample_receivers_time_derivative(
                step.e,
                step.h,
                h_old,
                float(dt),
                time=new_time,
                memories=step.memories,
                initial_memories=initial_memories,
            )
            h_old = step.h
            e_old = step.e
            memories = step.memories

        return HJReceiverDataResult(times=times, data=data, memories=memories)

    def _magnetic_step(
        self,
        step_index: int,
        h_old: np.ndarray,
        memories: list[np.ndarray],
        dt: float,
        electric_source: np.ndarray | None = None,
        magnetic_source: np.ndarray | None = None,
    ) -> HJMagneticStepResult:
        matrix = self._system_matrix_for_step(step_index, memories, dt)
        rhs = hj_magnetic_rhs(
            self.mesh,
            h_old,
            self.ip_model,
            memories,
            dt,
            electric_source=electric_source,
            magnetic_source=magnetic_source,
            mu=self.mu,
        )
        h_new = self._solver_for_step(step_index, matrix)(rhs)
        current = self.mesh.edge_curl @ h_new
        if electric_source is not None:
            current = current - np.asarray(electric_source, dtype=float)
        e_new, new_memories = _electric_from_hj_current(
            self.mesh,
            self.ip_model,
            memories,
            current,
            dt,
        )
        return HJMagneticStepResult(np.asarray(h_new, dtype=float), e_new, new_memories)

    def _initial_memories(self, initial_e: np.ndarray) -> list[np.ndarray]:
        if self.initial_memories is not None:
            return [np.asarray(memory, dtype=float).copy() for memory in self.initial_memories]
        return self.ip_model.initial_memory(self.mesh.n_faces, initial_e)

    def _system_matrix_for_step(
        self,
        step_index: int,
        memories: list[np.ndarray],
        dt: float,
    ) -> sp.csr_matrix:
        key = self._time_step_cache_key(step_index)
        if key not in self._matrix_cache:
            self._matrix_cache[key] = hj_magnetic_system_matrix(
                self.mesh,
                self.ip_model,
                memories,
                dt,
                mu=self.mu,
            )
        return self._matrix_cache[key]

    def _solver_for_step(self, step_index: int, matrix: sp.csr_matrix):
        if self.linear_solver == "cg":
            preconditioner = self._cg_preconditioner(matrix)

            def solve(rhs):
                solution, info = spla.cg(
                    matrix,
                    rhs,
                    rtol=float(self.cg_tolerance),
                    atol=0.0,
                    maxiter=self.cg_maxiter,
                    M=preconditioner,
                )
                if info != 0:
                    raise RuntimeError(f"H/J CG solver failed to converge with info={info}")
                return solution

            return solve

        key = self._time_step_cache_key(step_index)
        if key not in self._factor_cache:
            if self.linear_solver == "pardiso":
                self._factor_cache[key] = self._factorize_pardiso(matrix)
            else:
                self._factor_cache[key] = spla.factorized(matrix.tocsc())
        return self._factor_cache[key]

    def _cg_preconditioner(self, matrix: sp.spmatrix):
        if self.cg_preconditioner == "none":
            return None
        diagonal = np.asarray(matrix.diagonal(), dtype=float)
        inverse = np.zeros_like(diagonal)
        mask = np.abs(diagonal) > 0.0
        inverse[mask] = 1.0 / diagonal[mask]
        return spla.LinearOperator(matrix.shape, matvec=lambda x: inverse * x)

    def _factorize_pardiso(self, matrix: sp.csr_matrix):
        try:
            from pymatsolver import Pardiso  # noqa: PLC0415
        except ImportError as err:
            raise RuntimeError("linear_solver='pardiso' requires pymatsolver") from err

        solver = Pardiso(matrix.tocsr(), is_symmetric=True, is_positive_definite=True)
        return solver.solve

    def _time_step_cache_key(self, step_index: int) -> int:
        dt = float(self.time_steps[step_index])
        for index, old_dt in enumerate(self.time_steps[: step_index + 1]):
            if np.isclose(float(old_dt), dt, rtol=0.0, atol=1.0e-15):
                return int(index)
        return int(step_index)

    def _sample_receivers(
        self,
        e: np.ndarray,
        h: np.ndarray,
        time: float = 0.0,
        memories: list[np.ndarray] | None = None,
        initial_memories: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        values = np.array(
            [receiver.sample_hj(self.mesh, e, h, self.mu) for receiver in self.receivers],
            dtype=float,
        )
        return self._replace_magnetic_receivers(
            values,
            e,
            h,
            time,
            memories or [],
            initial_memories or [],
        )

    def _sample_receivers_time_derivative(
        self,
        e: np.ndarray,
        h_new: np.ndarray,
        h_old: np.ndarray,
        dt: float,
        time: float,
        memories: list[np.ndarray] | None = None,
        initial_memories: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        values = np.array(
            [
                receiver.sample_hj_time_derivative(self.mesh, e, h_new, h_old, dt, self.mu)
                for receiver in self.receivers
            ],
            dtype=float,
        )
        return self._replace_magnetic_receivers(
            values,
            e,
            h_new,
            time,
            memories or [],
            initial_memories or [],
        )

    def _replace_magnetic_receivers(
        self,
        values: np.ndarray,
        e: np.ndarray,
        h: np.ndarray,
        time: float,
        memories: list[np.ndarray],
        initial_memories: list[np.ndarray],
    ) -> np.ndarray:
        if self.magnetic_receiver_mode == "stored_h":
            return values
        magnetic_indices = [
            index
            for index, receiver in enumerate(self.receivers)
            if receiver.uses_magnetic_field_vector
        ]
        if not magnetic_indices:
            return values
        locations = np.array([self.receivers[index].location for index in magnetic_indices])
        if self.magnetic_receiver_mode == "current_biot":
            recovered_h = self._current_biot_h(h, locations, time)
        elif self.magnetic_receiver_mode == "face_basis_biot":
            recovered_h = self._face_basis_biot_h(h, locations, time)
        else:
            recovered_h = self._face_basis_cell_biot_h(
                e,
                memories,
                initial_memories,
                locations,
            )
        if self.magnetic_recovery_source_history is not None:
            recovered_h += self._source_history_magnetic_field(
                locations,
                time,
                initial_memories,
            )
        if self.magnetic_recovery_source_primary_delta6:
            recovered_h += self._source_primary_delta6_magnetic_field(locations, time)
        values = np.asarray(values, dtype=float).copy()
        for local_index, receiver_index in enumerate(magnetic_indices):
            values[receiver_index] = self.receivers[receiver_index].sample_magnetic_field_vector(
                recovered_h[local_index],
                self.mu,
            )
        return values

    def _current_biot_h(self, h: np.ndarray, locations: np.ndarray, time: float) -> np.ndarray:
        face_current = self.mesh.edge_curl @ np.asarray(h, dtype=float)
        face_current = face_current - self.electric_source_term(float(time))
        receiver_matrix = self._current_biot_receiver_matrix(locations)
        return np.einsum("kcf,f->kc", receiver_matrix, face_current)

    def _current_biot_receiver_matrix(self, locations: np.ndarray) -> np.ndarray:
        locations = np.asarray(locations, dtype=float)
        if locations.ndim != 2 or locations.shape[1] != 3:
            raise ValueError("locations must have shape (n_locations, 3)")
        key = (
            int(self.magnetic_recovery_subdivisions),
            tuple(int(v) for v in locations.shape),
            tuple(float(v) for v in locations.ravel()),
        )
        if key not in self._current_biot_matrix_cache:
            self._current_biot_matrix_cache[key] = face_current_biot_matrix(
                self.mesh,
                locations,
                subdivisions=self.magnetic_recovery_subdivisions,
            )
        return self._current_biot_matrix_cache[key]

    def _face_basis_biot_h(self, h: np.ndarray, locations: np.ndarray, time: float) -> np.ndarray:
        face_current = self.mesh.edge_curl @ np.asarray(h, dtype=float)
        face_current = face_current - self.electric_source_term(float(time))
        return biot_savart_h_from_face_basis_currents(
            self.mesh,
            face_current,
            locations,
            subdivisions=self.magnetic_recovery_subdivisions,
        )

    def _face_basis_cell_biot_h(
        self,
        e: np.ndarray,
        memories: list[np.ndarray],
        initial_memories: list[np.ndarray],
        locations: np.ndarray,
    ) -> np.ndarray:
        return biot_savart_h_from_face_basis_cell_ip_currents(
            self.mesh,
            e,
            self.ip_model.sigma_infinity,
            self.ip_model.terms,
            memories,
            locations,
            subdivisions=self.magnetic_recovery_subdivisions,
            polarization_scale=self.magnetic_recovery_polarization_scale,
            initial_polarization_scale=(
                self.magnetic_recovery_initial_polarization_scale
            ),
            initial_memories=initial_memories,
        )

    def _source_history_magnetic_field(
        self,
        locations: np.ndarray,
        time: float,
        initial_memories: list[np.ndarray],
    ) -> np.ndarray:
        """Evaluate prescribed H/J face-source history magnetic correction."""

        field = np.zeros((locations.shape[0], 3), dtype=float)
        if self.magnetic_recovery_source_history is None:
            return field
        corrections = source_history_correction_terms(
            self.magnetic_recovery_source_history
        )
        if not source_history_ip_model_has_contrast(self.ip_model):
            corrections = tuple(
                correction
                for correction in corrections
                if not source_history_correction_requires_ip(correction)
            )
            if not corrections:
                return field
        for correction in corrections:
            field += source_history_correction_field(
                self.mesh,
                self.sources,
                self.time_steps,
                time,
                locations,
                correction=correction,
                magnetic_receiver_mode=self.magnetic_receiver_mode,
                subdivisions=self.magnetic_recovery_subdivisions,
                field_location="face",
                ip_model=self.ip_model,
                initial_ip_model=self.initial_ip_model,
                initial_memories=initial_memories,
                mu=self.mu,
                cache=self._source_history_runtime_cache,
            )
        return field

    def _source_primary_delta6_magnetic_field(
        self,
        locations: np.ndarray,
        time: float,
    ) -> np.ndarray:
        """Diagnostic H/J source-primary IP correction for magnetic recovery."""

        field = np.zeros((locations.shape[0], 3), dtype=float)
        if time <= 0.0 or not self.ip_model.terms:
            return field
        line_current_cls = None
        if self.magnetic_recovery_source_primary_delta6_basis == "wire":
            try:
                from geoana.em.static import LineCurrentWholeSpace as line_current_cls  # noqa: PLC0415
            except ImportError as err:
                raise RuntimeError(
                    "magnetic_recovery_source_primary_delta6 with basis='wire' "
                    "requires geoana"
                ) from err

        for source in self.sources:
            initial_current = float(source.current * source.waveform.initial_value())
            if initial_current == 0.0:
                continue
            source_locations = source.locations
            length = float(np.linalg.norm(source_locations[-1] - source_locations[0]))
            if length == 0.0:
                continue
            kernel = 0.0
            for term in self.ip_model.terms:
                delta_sigma = self._source_delta_sigma(term, source)
                if delta_sigma > 0.0:
                    kernel += (
                        -6.0
                        * self.mu
                        * delta_sigma
                        * length**2
                        * np.exp(-float(time) / (2.0 * float(term.tau)))
                    )
            if kernel == 0.0:
                continue
            if self.magnetic_recovery_source_primary_delta6_basis == "face_current":
                source_field = np.einsum(
                    "kcf,f->kc",
                    self._current_biot_receiver_matrix(locations),
                    -source.initial_face_vector(self.mesh),
                )
            else:
                line_current = line_current_cls(
                    source_locations,
                    current=initial_current,
                    mu=self.mu,
                )
                source_field = np.asarray(line_current.magnetic_field(locations), dtype=float)
            field += kernel * source_field
        return field

    def _source_delta_sigma(self, term: DebyeTerm, source) -> float:
        delta = np.asarray(term.delta_sigma, dtype=float)
        if delta.size == 1:
            return float(delta.reshape(-1)[0])
        if delta.shape == (self.mesh.n_cells,):
            return float(delta[self._source_midpoint_cell_index(source)])
        if delta.shape == (self.mesh.n_faces,):
            source_vector = np.abs(source.initial_face_vector(self.mesh))
            weight = float(np.sum(source_vector))
            if weight == 0.0:
                return 0.0
            return float(np.dot(delta, source_vector) / weight)
        raise ValueError("delta_sigma must be scalar, cell-centered, or face-centered")

    def _source_midpoint_cell_index(self, source) -> int:
        midpoint = np.mean(source.locations, axis=0).reshape(1, 3)
        if hasattr(self.mesh, "closest_points_index"):
            index = self.mesh.closest_points_index(midpoint, "CC")
            return int(np.asarray(index).reshape(-1)[0])
        centers = np.asarray(self.mesh.cell_centers, dtype=float)
        distances = np.sum((centers - midpoint[0]) ** 2, axis=1)
        return int(np.argmin(distances))


def face_project_debye_model(mesh, ip_model: DebyeIPModel) -> DebyeIPModel:
    """Project cell-centered Debye material parameters onto H/J faces."""

    sigma = _face_project_property(mesh, ip_model.sigma_infinity, "sigma_infinity")
    terms = [
        DebyeTerm(
            delta_sigma=_face_project_property(mesh, term.delta_sigma, "delta_sigma"),
            tau=term.tau,
        )
        for term in ip_model.terms
    ]
    return DebyeIPModel(sigma, terms)


def hj_dc_initial_current_density(
    mesh,
    ip_model: DebyeIPModel,
    sources: Sequence[object] | None = None,
) -> np.ndarray:
    """Return long-on-time H/J face current from the DC potential problem."""

    source = _initial_face_source(mesh, sources)
    if np.linalg.norm(source) == 0.0:
        return np.zeros(mesh.n_faces, dtype=float)

    low_frequency_rho = 1.0 / ip_model.low_frequency_sigma()
    face_rho = mesh.get_face_inner_product(low_frequency_rho).tocsr()
    face_rho_inverse = _inverse_diagonal_matrix(face_rho, "face resistivity mass")
    divergence = sp.diags(mesh.cell_volumes, format="csr") @ mesh.face_divergence
    gradient = divergence.T.tocsr()
    matrix = (divergence @ face_rho_inverse @ gradient).tocsc()
    rhs = divergence @ source

    phi = spla.spsolve(matrix, rhs)
    return -np.asarray(face_rho_inverse @ (gradient @ phi), dtype=float)


def hj_dc_initial_electric_field(
    mesh,
    ip_model: DebyeIPModel,
    sources: Sequence[object] | None = None,
) -> np.ndarray:
    """Return the long-on-time H/J face electric field."""

    current = hj_dc_initial_current_density(mesh, ip_model, sources)
    if np.linalg.norm(current) == 0.0:
        return np.zeros(mesh.n_faces, dtype=float)

    low_frequency_rho = 1.0 / ip_model.low_frequency_sigma()
    face_rho = mesh.get_face_inner_product(low_frequency_rho).tocsr()
    unit_face = _unit_face_inner_product(mesh)
    electric = (face_rho @ current) / unit_face.diagonal()
    return np.asarray(electric, dtype=float)


def hj_mmr_initial_magnetic_field(
    mesh,
    ip_model: DebyeIPModel,
    sources: Sequence[object] | None = None,
    mu: float = mu_0,
) -> np.ndarray:
    """Return grounded-source long-on-time H/J magnetic field via MMR."""

    source = _initial_face_source(mesh, sources)
    if np.linalg.norm(source) == 0.0:
        return np.zeros(mesh.n_edges, dtype=float)

    current = hj_dc_initial_current_density(mesh, ip_model, sources)
    edge_mu = mesh.get_edge_inner_product(np.full(mesh.n_cells, float(mu))).tocsr()
    edge_mu_inverse = _inverse_diagonal_matrix(edge_mu, "edge permeability mass")
    curl = mesh.edge_curl.tocsr()
    divergence = sp.diags(mesh.cell_volumes, format="csr") @ mesh.face_divergence
    stabilization = (
        divergence.T
        @ sp.diags(np.full(mesh.n_cells, 1.0 / float(mu)) / mesh.cell_volumes, format="csr")
        @ divergence
    )
    matrix = (curl @ edge_mu_inverse @ curl.T - stabilization).tocsc()
    vector_potential = spla.spsolve(matrix, source + current)
    h_initial = edge_mu_inverse @ (curl.T @ vector_potential)
    return np.asarray(h_initial, dtype=float)


def hj_magnetic_system_matrix(
    mesh,
    ip_model: DebyeIPModel,
    memories: list[np.ndarray],
    dt: float,
    mu: float = mu_0,
) -> sp.csr_matrix:
    """Return the H/J magnetic-field system matrix for one time step.

    The assembled matrix is
    ``C.T M_f(rho_eff) C + M_e(mu) / dt``.  With no IP this reduces to
    SimPEG's `Simulation3DMagneticField.getAdiag` matrix.
    """

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    rho_eff, _ = _hj_inverse_constitutive_coefficients(mesh, ip_model, memories, dt)
    face_rho = _face_inner_product_from_coefficients(mesh, rho_eff)
    edge_mu = mesh.get_edge_inner_product(np.full(mesh.n_cells, float(mu))).tocsr()
    curl = mesh.edge_curl.tocsr()
    return (curl.T @ face_rho @ curl + (1.0 / dt) * edge_mu).tocsr()


def hj_magnetic_rhs(
    mesh,
    h_old: np.ndarray,
    ip_model: DebyeIPModel,
    memories: list[np.ndarray],
    dt: float,
    electric_source: np.ndarray | None = None,
    magnetic_source: np.ndarray | None = None,
    mu: float = mu_0,
) -> np.ndarray:
    """Return the H/J magnetic-field right-hand side for one time step."""

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    h_old = np.asarray(h_old, dtype=float)
    if h_old.shape != (mesh.n_edges,):
        raise ValueError("h_old must have shape (mesh.n_edges,)")

    rho_eff, electric_history = _hj_inverse_constitutive_coefficients(
        mesh,
        ip_model,
        memories,
        dt,
    )
    edge_mu = mesh.get_edge_inner_product(np.full(mesh.n_cells, float(mu))).tocsr()
    rhs = (edge_mu @ h_old) / dt

    curl = mesh.edge_curl.tocsr()
    if electric_source is not None:
        electric_source = np.asarray(electric_source, dtype=float)
        if electric_source.shape != (mesh.n_faces,):
            raise ValueError("electric_source must have shape (mesh.n_faces,)")
        rhs += curl.T @ (_face_inner_product_from_coefficients(mesh, rho_eff) @ electric_source)

    if np.any(electric_history):
        rhs -= curl.T @ (_unit_face_inner_product(mesh) @ _face_vector(mesh, electric_history))

    if magnetic_source is not None:
        magnetic_source = np.asarray(magnetic_source, dtype=float)
        if magnetic_source.shape != (mesh.n_edges,):
            raise ValueError("magnetic_source must have shape (mesh.n_edges,)")
        rhs += magnetic_source
    return np.asarray(rhs, dtype=float)


def hj_magnetic_step(
    mesh,
    h_old: np.ndarray,
    ip_model: DebyeIPModel,
    memories: list[np.ndarray],
    dt: float,
    electric_source: np.ndarray | None = None,
    magnetic_source: np.ndarray | None = None,
    mu: float = mu_0,
) -> HJMagneticStepResult:
    """Advance one H/J magnetic-field prototype step with Debye IP memories."""

    matrix = hj_magnetic_system_matrix(mesh, ip_model, memories, dt, mu=mu)
    rhs = hj_magnetic_rhs(
        mesh,
        h_old,
        ip_model,
        memories,
        dt,
        electric_source=electric_source,
        magnetic_source=magnetic_source,
        mu=mu,
    )
    h_new = spla.spsolve(matrix.tocsc(), rhs)
    current = mesh.edge_curl @ h_new
    if electric_source is not None:
        current = current - np.asarray(electric_source, dtype=float)
    e_new, new_memories = _electric_from_hj_current(mesh, ip_model, memories, current, dt)
    return HJMagneticStepResult(
        h=np.asarray(h_new, dtype=float),
        e=e_new,
        memories=new_memories,
    )


def _validate_face_memories(mesh, ip_model: DebyeIPModel, memories: list[np.ndarray]) -> None:
    if len(memories) != len(ip_model.terms):
        raise ValueError("one memory vector is required for each Debye term")
    for memory in memories:
        if np.asarray(memory).shape != (mesh.n_faces,):
            raise ValueError("each H/J memory vector must have shape (mesh.n_faces,)")


def _initial_face_source(mesh, sources: Sequence[object] | None) -> np.ndarray:
    source = np.zeros(mesh.n_faces, dtype=float)
    for item in sources or []:
        source -= item.initial_face_vector(mesh)
    return source


def _inverse_diagonal_matrix(matrix: sp.spmatrix, name: str) -> sp.csr_matrix:
    matrix = matrix.tocsr()
    off_diagonal = matrix - sp.diags(matrix.diagonal(), format="csr")
    if off_diagonal.nnz:
        raise ValueError(f"{name} must be diagonal")
    diagonal = np.asarray(matrix.diagonal(), dtype=float)
    if np.any(diagonal == 0.0):
        raise ValueError(f"{name} contains zero diagonal entries")
    return sp.diags(1.0 / diagonal, format="csr")


def _face_project_property(mesh, values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return np.full(mesh.n_faces, float(values[0]))
    if values.shape == (mesh.n_faces,):
        return values.copy()
    if values.shape == (mesh.n_cells,):
        unit_face = _unit_face_inner_product(mesh)
        weighted_face = mesh.get_face_inner_product(values).tocsr()
        return np.asarray(weighted_face.diagonal(), dtype=float) / unit_face.diagonal()
    raise ValueError(f"{name} must be scalar, cell-centered, or face-centered")


def _face_inner_product_from_coefficients(mesh, coefficients: np.ndarray) -> sp.csr_matrix:
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape == (mesh.n_cells,):
        return mesh.get_face_inner_product(coefficients).tocsr()
    if coefficients.shape == (mesh.n_faces,):
        return (_unit_face_inner_product(mesh) @ sp.diags(coefficients, format="csr")).tocsr()
    if coefficients.size == 1:
        return mesh.get_face_inner_product(np.full(mesh.n_cells, float(coefficients[0]))).tocsr()
    raise ValueError("face coefficients must be scalar, cell-centered, or face-centered")


def _electric_from_hj_current(
    mesh,
    ip_model: DebyeIPModel,
    memories: list[np.ndarray],
    current: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    current = np.asarray(current, dtype=float)
    if current.shape != (mesh.n_faces,):
        raise ValueError("current must have shape (mesh.n_faces,)")
    if ip_model.terms:
        rho_eff, electric_history = _hj_inverse_constitutive_coefficients(
            mesh,
            ip_model,
            memories,
            dt,
        )
        electric = _apply_face_resistivity(mesh, rho_eff, current) + electric_history
        return np.asarray(electric, dtype=float), ip_model.update_memory(memories, electric, dt)
    if ip_model.sigma_infinity.size in (1, mesh.n_faces):
        return ip_model.inverse_constitutive_update(current, memories, dt)

    if ip_model.sigma_infinity.size != mesh.n_cells:
        raise ValueError("sigma_infinity must be scalar, cell-centered, or face-centered")
    face_rho = mesh.get_face_inner_product(1.0 / ip_model.sigma_infinity).tocsr()
    unit_face = _unit_face_inner_product(mesh)
    electric = (face_rho @ current) / unit_face.diagonal()
    return np.asarray(electric, dtype=float), []


def _hj_inverse_constitutive_coefficients(
    mesh,
    ip_model: DebyeIPModel,
    memories: list[np.ndarray],
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if not ip_model.terms:
        sigma = np.asarray(ip_model.sigma_infinity, dtype=float)
        if sigma.size == mesh.n_cells:
            return 1.0 / sigma, np.zeros(mesh.n_faces, dtype=float)
        return ip_model.inverse_constitutive_coefficients(memories, dt, n_dofs=mesh.n_faces)

    _validate_face_memories(mesh, ip_model, memories)
    sigma = np.asarray(ip_model.sigma_infinity, dtype=float)
    if sigma.size in (1, mesh.n_faces):
        return ip_model.inverse_constitutive_coefficients(memories, dt, n_dofs=mesh.n_faces)
    if sigma.size != mesh.n_cells:
        raise ValueError("sigma_infinity must be scalar, cell-centered, or face-centered")

    sigma_eff = sigma.copy()
    alpha_delta_terms: list[np.ndarray] = []
    for term in ip_model.terms:
        alpha, beta = term.coefficients(dt)
        delta = _cell_property_from_term(mesh, term)
        sigma_eff -= beta * delta
        alpha_delta_terms.append(alpha * delta)
    if np.any(sigma_eff <= 0.0):
        raise ValueError("H/J effective conductivity must remain positive")

    rho_eff = 1.0 / sigma_eff
    unit_face = _unit_face_inner_product(mesh)
    unit_diagonal = unit_face.diagonal()
    electric_history = np.zeros(mesh.n_faces, dtype=float)
    for alpha_delta, memory in zip(alpha_delta_terms, memories):
        history_mass = mesh.get_face_inner_product(rho_eff * alpha_delta).tocsr()
        electric_history += history_mass.diagonal() / unit_diagonal * memory
    return rho_eff, electric_history


def _apply_face_resistivity(mesh, coefficients: np.ndarray, current: np.ndarray) -> np.ndarray:
    current = np.asarray(current, dtype=float)
    if current.shape != (mesh.n_faces,):
        raise ValueError("current must have shape (mesh.n_faces,)")
    face_rho = _face_inner_product_from_coefficients(mesh, coefficients)
    unit_face = _unit_face_inner_product(mesh)
    return np.asarray((face_rho @ current) / unit_face.diagonal(), dtype=float)


def _cell_property_from_term(mesh, term: DebyeTerm) -> np.ndarray:
    delta = np.asarray(term.delta_sigma, dtype=float)
    if delta.size == 1:
        return np.full(mesh.n_cells, float(delta[0]))
    if delta.shape == (mesh.n_cells,):
        return delta
    raise ValueError("delta_sigma must be scalar or cell-centered for cell H/J IP")


def _normalize_polarization_scale(value: float | str | np.ndarray) -> float | str | np.ndarray:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized != "low_frequency_ratio":
            raise ValueError(
                "magnetic_recovery_polarization_scale must be nonnegative "
                "or 'low_frequency_ratio'"
            )
        return normalized
    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        if array.shape != (3,):
            raise ValueError("component magnetic_recovery_polarization_scale must have length 3")
        if np.any(array < 0.0):
            raise ValueError("magnetic_recovery_polarization_scale must be nonnegative")
        return array.copy()
    scale = float(array)
    if scale < 0.0:
        raise ValueError("magnetic_recovery_polarization_scale must be nonnegative")
    return scale


def _normalize_initial_polarization_scale(value: float) -> float:
    scale = float(value)
    if not np.isfinite(scale):
        raise ValueError("magnetic_recovery_initial_polarization_scale must be finite")
    return scale


def _unit_face_inner_product(mesh) -> sp.csr_matrix:
    matrix = mesh.get_face_inner_product(np.ones(mesh.n_cells)).tocsr()
    off_diagonal = matrix - sp.diags(matrix.diagonal(), format="csr")
    if off_diagonal.nnz:
        raise ValueError("face-centered H/J prototype requires diagonal face mass")
    diagonal = np.asarray(matrix.diagonal(), dtype=float)
    if np.any(diagonal == 0.0):
        raise ValueError("unit face mass contains zero diagonal entries")
    return matrix


def _face_vector(mesh, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape == (mesh.n_faces,):
        return values
    if values.shape == (mesh.n_cells,):
        if not np.any(values):
            return np.zeros(mesh.n_faces, dtype=float)
    if values.size == 1:
        return np.full(mesh.n_faces, float(values[0]))
    raise ValueError("face vector must be scalar or face-centered")
