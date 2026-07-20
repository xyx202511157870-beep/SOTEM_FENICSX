"""Implicit finite-volume TDEM solver with Debye IP memory variables."""

from __future__ import annotations

import json
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.constants import mu_0

from .cpml import (
    CPMLConfig,
    CPMLProfiles,
    CPMLState,
    CurlSplit,
    build_cpml_profiles,
    effective_stretched_edge_curl_matrix,
    effective_stretched_face_curl_transpose_matrix,
    split_edge_curl,
    stretched_edge_curl,
    stretched_face_curl_transpose,
)
from .ip import DebyeIPModel
from .magnetic_recovery import (
    biot_savart_h_from_cell_currents,
    biot_savart_h_from_edge_basis_cell_ip_currents,
    biot_savart_h_from_edge_basis_currents,
    biot_savart_h_from_edge_current_moments,
)
from .receivers import PointReceiver
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
from .sources import GroundedWireSource


@dataclass(frozen=True)
class SimulationResult:
    """Fields and receiver data returned by :class:`TDEMIPSimulation`."""

    times: np.ndarray
    e: np.ndarray
    b: np.ndarray
    data: np.ndarray
    memories: list[np.ndarray]
    memory_history: list[np.ndarray] | None = None


@dataclass(frozen=True)
class ReceiverDataResult:
    """Receiver-only data returned by memory-saving simulation runs."""

    times: np.ndarray
    data: np.ndarray
    memories: list[np.ndarray]


@dataclass
class TDEMIPSimulation:
    """Backward-Euler EB finite-volume TDEM-IP simulation.

    Electric fields live on mesh edges and magnetic flux density lives on mesh
    faces. IP is represented by finite Debye memories in conductivity form.
    """

    mesh: object
    ip_model: DebyeIPModel
    time_steps: Sequence[float]
    initial_ip_model: DebyeIPModel | None = None
    sources: Sequence[GroundedWireSource] | None = None
    receivers: Sequence[PointReceiver] | None = None
    mu: float = mu_0
    initial_magnetic_mode: str = "ampere"
    linear_solver: str = "direct"
    cg_tolerance: float = 1.0e-8
    cg_maxiter: int | None = None
    cg_preconditioner: str = "jacobi"
    cpml: CPMLConfig | None = None
    magnetic_receiver_mode: str = "stored_b"
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
        if self.initial_magnetic_mode not in {"ampere", "zero", "biot_savart_wire"}:
            raise ValueError(
                "initial_magnetic_mode must be 'ampere', 'zero', or 'biot_savart_wire'"
            )
        if self.linear_solver not in {"direct", "cg", "pardiso"}:
            raise ValueError("linear_solver must be 'direct', 'cg', or 'pardiso'")
        if (
            isinstance(self.cg_tolerance, bool)
            or not isinstance(self.cg_tolerance, Real)
            or not np.isfinite(self.cg_tolerance)
            or self.cg_tolerance <= 0.0
        ):
            raise ValueError("cg_tolerance must be finite and positive")
        self.cg_tolerance = float(self.cg_tolerance)
        if self.cg_maxiter is not None:
            if (
                isinstance(self.cg_maxiter, bool)
                or not isinstance(self.cg_maxiter, Integral)
                or self.cg_maxiter <= 0
            ):
                raise ValueError(
                    "cg_maxiter must be a positive integer excluding bool or None"
                )
            self.cg_maxiter = int(self.cg_maxiter)
        if self.cg_preconditioner not in {"none", "jacobi"}:
            raise ValueError("cg_preconditioner must be 'none' or 'jacobi'")
        if self.magnetic_receiver_mode not in {
            "stored_b",
            "current_biot",
            "edge_basis_biot",
            "edge_basis_cell_biot",
            "edge_current_biot",
        }:
            raise ValueError(
                "magnetic_receiver_mode must be 'stored_b', 'current_biot', "
                "'edge_basis_biot', 'edge_basis_cell_biot', or 'edge_current_biot'"
            )
        self.magnetic_recovery_subdivisions = int(self.magnetic_recovery_subdivisions)
        if self.magnetic_recovery_subdivisions <= 0:
            raise ValueError("magnetic_recovery_subdivisions must be positive")
        self.magnetic_recovery_polarization_scale = self._normalize_polarization_scale(
            self.magnetic_recovery_polarization_scale
        )
        self.magnetic_recovery_initial_polarization_scale = (
            self._normalize_initial_polarization_scale(
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
            "edge_current",
        }:
            raise ValueError(
                "magnetic_recovery_source_primary_delta6_basis must be "
                "'wire' or 'edge_current'"
            )
        if (
            self.magnetic_recovery_source_primary_delta6
            and self.magnetic_receiver_mode == "stored_b"
        ):
            raise ValueError(
                "magnetic_recovery_source_primary_delta6 requires a Biot magnetic "
                "receiver mode"
            )
        if (
            self.magnetic_recovery_source_history is not None
            and self.magnetic_receiver_mode == "stored_b"
        ):
            raise ValueError(
                "magnetic_recovery_source_history requires a Biot magnetic receiver mode"
            )
        if self.magnetic_recovery_source_history is not None:
            source_history_correction_terms(self.magnetic_recovery_source_history)
        self._source_history_runtime_cache = {}
        if (
            self.cpml is not None
            and self.cpml.thickness_cells > 0
            and self.linear_solver == "cg"
        ):
            raise ValueError("active CPML currently requires linear_solver='direct' or 'pardiso'")
        self.sources = list(self.sources or [])
        self.receivers = list(self.receivers or [])
        self.ip_model = self.ip_model.expand_to_cells(self.mesh.n_cells)
        if self.initial_ip_model is None:
            self.initial_ip_model = self.ip_model
        else:
            self.initial_ip_model = self.initial_ip_model.expand_to_cells(self.mesh.n_cells)
        self._face_mu_inverse_matrix: sp.csr_matrix | None = None
        self._unit_edge_mass_diagonal: np.ndarray | None = None
        self._matrix_cache: dict[int, sp.csr_matrix] = {}
        self._factor_cache: dict[int, object] = {}
        self._cpml_profiles_cache: dict[int, CPMLProfiles] = {}
        self._cpml_curl_split_cache: CurlSplit | None = None

    @property
    def times(self) -> np.ndarray:
        return np.r_[0.0, np.cumsum(self.time_steps)]

    @property
    def face_mu_inverse_matrix(self) -> sp.csr_matrix:
        if self._face_mu_inverse_matrix is None:
            mui = np.full(self.mesh.n_cells, 1.0 / self.mu)
            self._face_mu_inverse_matrix = self.mesh.get_face_inner_product(mui).tocsr()
        return self._face_mu_inverse_matrix

    @property
    def unit_edge_mass_diagonal(self) -> np.ndarray:
        if self._unit_edge_mass_diagonal is None:
            matrix = self.mesh.get_edge_inner_product(np.ones(self.mesh.n_cells)).tocsr()
            off_diagonal = matrix - sp.diags(matrix.diagonal(), format="csr")
            if off_diagonal.nnz:
                raise ValueError("mass-consistent current recovery requires diagonal edge mass")
            diagonal = np.asarray(matrix.diagonal(), dtype=float)
            if np.any(diagonal == 0.0):
                raise ValueError("unit edge mass contains zero diagonal entries")
            self._unit_edge_mass_diagonal = diagonal
        return self._unit_edge_mass_diagonal

    def system_matrix(self, step_index: int) -> sp.csr_matrix:
        """Return the matrix for one time step."""

        key = self._time_step_cache_key(step_index)
        if key not in self._matrix_cache:
            dt = float(self.time_steps[step_index])
            sigma_eff = self.ip_model.effective_sigma(dt)
            me_sigma_eff = self.mesh.get_edge_inner_product(sigma_eff).tocsr()
            if self.cpml is None:
                curl = self.mesh.edge_curl
                matrix = curl.T @ self.face_mu_inverse_matrix @ curl + (1.0 / dt) * me_sigma_eff
            else:
                profiles = self._cpml_profiles(step_index)
                curl_split = self._cpml_curl_split()
                faraday_curl = effective_stretched_edge_curl_matrix(
                    self.mesh,
                    profiles,
                    split=curl_split,
                )
                ampere_curl = effective_stretched_face_curl_transpose_matrix(
                    self.mesh,
                    profiles,
                    self.face_mu_inverse_matrix,
                    split=curl_split,
                )
                matrix = ampere_curl @ faraday_curl + (1.0 / dt) * me_sigma_eff
            self._matrix_cache[key] = matrix.tocsr()
        return self._matrix_cache[key]

    def initial_electric_field(self) -> np.ndarray:
        """Compute long-on-time grounded-source DC electric initial fields."""

        source_vec = np.zeros(self.mesh.n_edges, dtype=float)
        for source in self.sources:
            source_vec += source.initial_edge_vector(self.mesh)
        if np.linalg.norm(source_vec) == 0.0:
            return np.zeros(self.mesh.n_edges, dtype=float)

        gradient = self.mesh.nodal_gradient.tocsr()
        me_sigma0 = self.mesh.get_edge_inner_product(
            self.initial_ip_model.low_frequency_sigma()
        ).tocsr()
        adc = gradient.T @ me_sigma0 @ gradient
        rhs = gradient.T @ source_vec

        # Fix one scalar-potential gauge node to remove the null space.
        keep = np.arange(adc.shape[0] - 1)
        phi = np.zeros(adc.shape[0], dtype=float)
        phi[keep] = spla.spsolve(adc[keep][:, keep].tocsc(), rhs[keep])
        return -gradient @ phi

    def initial_magnetic_flux_density(self, e_initial: np.ndarray | None = None) -> np.ndarray:
        """Compute static magnetic flux density consistent with initial current.

        The long-on-time current must satisfy the discrete Ampere balance
        ``C.T M_mu^-1 b0 = M_sigma0 e0 + s0`` before the step-off.  We recover a
        divergence-free ``b0`` through an edge vector potential ``a`` with
        ``b0 = C a`` and a Coulomb-style gauge stabilization.
        """

        source_vec = np.zeros(self.mesh.n_edges, dtype=float)
        for source in self.sources:
            source_vec += source.initial_edge_vector(self.mesh)
        if np.linalg.norm(source_vec) == 0.0 or self.initial_magnetic_mode == "zero":
            return np.zeros(self.mesh.n_faces, dtype=float)
        if self.initial_magnetic_mode == "biot_savart_wire":
            return self._biot_savart_wire_initial_magnetic_flux_density()

        if e_initial is None:
            e_initial = self.initial_electric_field()

        me_sigma0 = self.mesh.get_edge_inner_product(
            self.initial_ip_model.low_frequency_sigma()
        ).tocsr()
        current = me_sigma0 @ e_initial + source_vec
        if np.linalg.norm(current) == 0.0:
            return np.zeros(self.mesh.n_faces, dtype=float)

        curl = self.mesh.edge_curl.tocsr()
        gradient = self.mesh.nodal_gradient.tocsr()
        stiffness = curl.T @ self.face_mu_inverse_matrix @ curl
        gauge = gradient @ gradient.T
        matrix = (stiffness + gauge).tocsc()
        vector_potential = spla.spsolve(matrix, current)
        return curl @ vector_potential

    def _biot_savart_wire_initial_magnetic_flux_density(self) -> np.ndarray:
        try:
            from geoana.em.static import LineCurrentWholeSpace  # noqa: PLC0415
        except ImportError as err:
            raise RuntimeError(
                "initial_magnetic_mode='biot_savart_wire' requires geoana"
            ) from err

        vector_potential = np.zeros(self.mesh.n_edges, dtype=float)
        for source in self.sources:
            if not source.waveform.has_initial_fields:
                continue
            current = float(source.current * source.waveform.initial_value())
            if current == 0.0:
                continue
            line_current = LineCurrentWholeSpace(source.locations, current=current, mu=self.mu)
            sampled = line_current.vector_potential(self.mesh.edges)
            sampled = np.nan_to_num(sampled, nan=0.0, posinf=0.0, neginf=0.0)
            vector_potential += self.mesh.project_edge_vector(sampled)
        return self.mesh.edge_curl @ vector_potential

    def electric_source_term(self, time: float) -> np.ndarray:
        """Integrated electric source term on edges at a time node."""

        value = np.zeros(self.mesh.n_edges, dtype=float)
        for source in self.sources:
            value += source.edge_vector_at(self.mesh, time)
        return value

    def source_rhs(self, step_index: int) -> np.ndarray:
        """Return source contribution for ``e^(n+1)``."""

        old_time = float(self.times[step_index])
        new_time = float(self.times[step_index + 1])
        value = np.zeros(self.mesh.n_edges, dtype=float)
        for source in self.sources:
            value -= source.edge_vector_interval_average_didt(self.mesh, old_time, new_time)
        return value

    def previous_electric_source_term(self, time: float) -> np.ndarray:
        """Integrated electric source left-limit on edges at a time node."""

        value = np.zeros(self.mesh.n_edges, dtype=float)
        for source in self.sources:
            value += source.previous_edge_vector_at(self.mesh, time)
        return value

    def history_rhs(
        self,
        e_old: np.ndarray,
        memories: list[np.ndarray],
        step_index: int,
    ) -> np.ndarray:
        """Return conductivity and IP history terms."""

        dt = float(self.time_steps[step_index])
        rhs = self.mesh.get_edge_inner_product(self.ip_model.sigma_infinity).tocsr() @ e_old
        for term, memory in zip(self.ip_model.terms, memories):
            _, beta = term.coefficients(dt)
            rhs -= beta * (self.mesh.get_edge_inner_product(term.delta_sigma).tocsr() @ memory)
        return rhs / dt

    def magnetic_history_rhs(
        self,
        b_old: np.ndarray,
        memories: list[np.ndarray],
        step_index: int,
        cpml_state: CPMLState | None = None,
    ) -> np.ndarray:
        """Return the time-step RHS using the previous magnetic flux directly."""

        dt = float(self.time_steps[step_index])
        if self.cpml is None:
            rhs = self.mesh.edge_curl.T @ self.face_mu_inverse_matrix @ b_old
        else:
            if cpml_state is None:
                raise ValueError("cpml_state is required when CPML is enabled")
            profiles = self._cpml_profiles(step_index)
            ampere_curl = effective_stretched_face_curl_transpose_matrix(
                self.mesh,
                profiles,
                self.face_mu_inverse_matrix,
                split=self._cpml_curl_split(),
            )
            face_history = np.sum(profiles.face_b * cpml_state.face_curl_memory, axis=0)
            edge_history = np.sum(profiles.edge_b * cpml_state.edge_curl_memory, axis=0)
            rhs = ampere_curl @ b_old - dt * (ampere_curl @ face_history) + edge_history
        for term, memory in zip(self.ip_model.terms, memories):
            alpha, _ = term.coefficients(dt)
            rhs += alpha * (self.mesh.get_edge_inner_product(term.delta_sigma).tocsr() @ memory)
        rhs -= self.electric_source_term(float(self.times[step_index + 1]))
        return rhs / dt

    def run(self) -> SimulationResult:
        """Run the simulation and return all time-node fields."""

        n_times = self.time_steps.size + 1
        e = np.zeros((n_times, self.mesh.n_edges), dtype=float)
        b = np.zeros((n_times, self.mesh.n_faces), dtype=float)
        data = np.zeros((n_times, len(self.receivers)), dtype=float)

        e[0] = self.initial_electric_field()
        if self.initial_magnetic_mode == "zero":
            b[0] = np.zeros(self.mesh.n_faces, dtype=float)
        else:
            b[0] = self.initial_magnetic_flux_density(e[0])
        initial_memories = self.ip_model.initial_memory(self.mesh.n_edges, e[0])
        memories = [memory.copy() for memory in initial_memories]
        memory_history = [
            np.zeros((n_times, self.mesh.n_edges), dtype=float) for _ in memories
        ]
        for index, memory in enumerate(memories):
            memory_history[index][0] = memory
        cpml_state = CPMLState.zeros(self.mesh) if self.cpml is not None else None
        data[0] = self._sample_receivers(e[0], b[0])

        curl = self.mesh.edge_curl
        for step_index, dt in enumerate(self.time_steps):
            solver = self._solver_for_step(step_index)
            if cpml_state is None:
                rhs = self.history_rhs(e[step_index], memories, step_index) + self.source_rhs(
                    step_index
                )
            else:
                rhs = self.magnetic_history_rhs(b[step_index], memories, step_index, cpml_state)
            e_new = solver(rhs)
            if cpml_state is None:
                b_new = b[step_index] - float(dt) * (curl @ e_new)
            else:
                profiles = self._cpml_profiles(step_index)
                curl_e, cpml_state = stretched_edge_curl(
                    self.mesh,
                    profiles,
                    cpml_state,
                    e_new,
                    split=self._cpml_curl_split(),
                )
                b_new = b[step_index] - float(dt) * curl_e
                face_field = self.face_mu_inverse_matrix @ b_new
                _, cpml_state = stretched_face_curl_transpose(
                    self.mesh,
                    profiles,
                    cpml_state,
                    face_field,
                    split=self._cpml_curl_split(),
                )
            new_memories = self.ip_model.update_memory(memories, e_new, float(dt))

            e[step_index + 1] = e_new
            b[step_index + 1] = b_new
            if self.magnetic_receiver_mode in {
                "current_biot",
                "edge_basis_biot",
                "edge_basis_cell_biot",
                "edge_current_biot",
            }:
                data[step_index + 1] = self._sample_receivers_with_current_biot(
                    e_new,
                    b_new,
                    b[step_index],
                    float(dt),
                    new_memories,
                    float(self.times[step_index + 1]),
                    initial_memories,
                )
            else:
                data[step_index + 1] = self._sample_receivers_with_previous_b(
                    e_new,
                    b_new,
                    b[step_index],
                    float(dt),
                )
            memories = new_memories
            for index, memory in enumerate(memories):
                memory_history[index][step_index + 1] = memory

        return SimulationResult(
            times=self.times,
            e=e,
            b=b,
            data=data,
            memories=memories,
            memory_history=memory_history,
        )

    def run_data_only(self) -> ReceiverDataResult:
        """Run the simulation while storing receiver data but not field histories."""

        n_times = self.time_steps.size + 1
        data = np.zeros((n_times, len(self.receivers)), dtype=float)

        e_old = self.initial_electric_field()
        if self.initial_magnetic_mode == "zero":
            b_old = np.zeros(self.mesh.n_faces, dtype=float)
        else:
            b_old = self.initial_magnetic_flux_density(e_old)
        initial_memories = self.ip_model.initial_memory(self.mesh.n_edges, e_old)
        memories = [memory.copy() for memory in initial_memories]
        cpml_state = CPMLState.zeros(self.mesh) if self.cpml is not None else None
        data[0] = self._sample_receivers(e_old, b_old)

        curl = self.mesh.edge_curl
        for step_index, dt in enumerate(self.time_steps):
            solver = self._solver_for_step(step_index)
            if cpml_state is None:
                rhs = self.history_rhs(e_old, memories, step_index) + self.source_rhs(
                    step_index
                )
            else:
                rhs = self.magnetic_history_rhs(b_old, memories, step_index, cpml_state)
            e_new = solver(rhs)
            if cpml_state is None:
                b_new = b_old - float(dt) * (curl @ e_new)
            else:
                profiles = self._cpml_profiles(step_index)
                curl_e, cpml_state = stretched_edge_curl(
                    self.mesh,
                    profiles,
                    cpml_state,
                    e_new,
                    split=self._cpml_curl_split(),
                )
                b_new = b_old - float(dt) * curl_e
                face_field = self.face_mu_inverse_matrix @ b_new
                _, cpml_state = stretched_face_curl_transpose(
                    self.mesh,
                    profiles,
                    cpml_state,
                    face_field,
                    split=self._cpml_curl_split(),
                )
            new_memories = self.ip_model.update_memory(memories, e_new, float(dt))

            if self.magnetic_receiver_mode in {
                "current_biot",
                "edge_basis_biot",
                "edge_basis_cell_biot",
                "edge_current_biot",
            }:
                data[step_index + 1] = self._sample_receivers_with_current_biot(
                    e_new,
                    b_new,
                    b_old,
                    float(dt),
                    new_memories,
                    float(self.times[step_index + 1]),
                    initial_memories,
                )
            else:
                data[step_index + 1] = self._sample_receivers_with_previous_b(
                    e_new,
                    b_new,
                    b_old,
                    float(dt),
                )
            e_old = e_new
            b_old = b_new
            memories = new_memories

        return ReceiverDataResult(times=self.times, data=data, memories=memories)

    def _sample_receivers(self, e: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.array([receiver.sample(self.mesh, e, b, self.mu) for receiver in self.receivers])

    def _sample_receivers_with_previous_b(
        self,
        e: np.ndarray,
        b_new: np.ndarray,
        b_old: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        return np.array(
            [
                receiver.sample_time_derivative(self.mesh, e, b_new, b_old, dt, self.mu)
                for receiver in self.receivers
            ]
        )

    def _sample_receivers_with_current_biot(
        self,
        e: np.ndarray,
        b_new: np.ndarray,
        b_old: np.ndarray,
        dt: float,
        memories: list[np.ndarray],
        time: float,
        initial_memories: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        magnetic_indices = [
            index
            for index, receiver in enumerate(self.receivers)
            if receiver.uses_magnetic_field_vector
        ]
        if not magnetic_indices:
            return self._sample_receivers_with_previous_b(e, b_new, b_old, dt)

        locations = np.array([self.receivers[index].location for index in magnetic_indices])
        if self.magnetic_receiver_mode == "edge_current_biot":
            edge_current = self._edge_current_moments(e, memories, initial_memories)
            recovered_h = biot_savart_h_from_edge_current_moments(
                self.mesh,
                edge_current,
                locations,
            )
        elif self.magnetic_receiver_mode == "edge_basis_biot":
            edge_current = self._edge_current_moments(e, memories, initial_memories)
            edge_current_field = edge_current / self.unit_edge_mass_diagonal
            recovered_h = biot_savart_h_from_edge_basis_currents(
                self.mesh,
                edge_current_field,
                locations,
                subdivisions=self.magnetic_recovery_subdivisions,
            )
        elif self.magnetic_receiver_mode == "edge_basis_cell_biot":
            recovered_h = biot_savart_h_from_edge_basis_cell_ip_currents(
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
        else:
            current_density = self._cell_current_density(e, memories, initial_memories)
            recovered_h = biot_savart_h_from_cell_currents(
                self.mesh,
                current_density,
                locations,
                subdivisions=self.magnetic_recovery_subdivisions,
            )
        recovered_h += self._source_magnetic_field(locations, time)
        if self.magnetic_recovery_source_primary_delta6:
            recovered_h += self._source_primary_delta6_magnetic_field(locations, time)
        if self.magnetic_recovery_source_history is not None:
            recovered_h += self._source_history_magnetic_field(
                locations,
                time,
                initial_memories,
            )

        values = self._sample_receivers_with_previous_b(e, b_new, b_old, dt)
        for local_index, receiver_index in enumerate(magnetic_indices):
            values[receiver_index] = self.receivers[receiver_index].sample_magnetic_field_vector(
                recovered_h[local_index],
                self.mu,
            )
        return values

    def _cell_current_density(
        self,
        e: np.ndarray,
        memories: list[np.ndarray],
        initial_memories: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        edge_current = self._edge_current_moments(e, memories, initial_memories)
        edge_current_field = edge_current / self.unit_edge_mass_diagonal
        current = self.mesh.average_edge_to_cell_vector @ edge_current_field
        current = np.asarray(current).reshape((self.mesh.n_cells, 3), order="F")
        return current

    def _edge_current_moments(
        self,
        e: np.ndarray,
        memories: list[np.ndarray],
        initial_memories: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        edge_current = self.mesh.get_edge_inner_product(self.ip_model.sigma_infinity).tocsr() @ e
        for term, memory in zip(self.ip_model.terms, memories):
            edge_current -= self._polarization_current_matrix(term) @ memory
        if self.magnetic_recovery_initial_polarization_scale != 0.0:
            if initial_memories is None:
                raise ValueError(
                    "initial_memories are required when "
                    "magnetic_recovery_initial_polarization_scale is nonzero"
                )
            self.ip_model._validate_memories(initial_memories, self.mesh.n_edges)
            for term, initial_memory in zip(self.ip_model.terms, initial_memories):
                edge_current += (
                    self.magnetic_recovery_initial_polarization_scale
                    * (self.mesh.get_edge_inner_product(term.delta_sigma).tocsr() @ initial_memory)
                )
        return np.asarray(edge_current, dtype=float)

    def _polarization_current_matrix(self, term) -> sp.csr_matrix:
        scale = self.magnetic_recovery_polarization_scale
        if isinstance(scale, str) and scale == "low_frequency_ratio":
            ratio = self.ip_model.sigma_infinity / self.ip_model.low_frequency_sigma()
            return self.mesh.get_edge_inner_product(ratio * term.delta_sigma).tocsr()
        if isinstance(scale, np.ndarray):
            edge_scale = np.r_[
                np.full(self.mesh.n_edges_x, scale[0]),
                np.full(self.mesh.n_edges_y, scale[1]),
                np.full(self.mesh.n_edges_z, scale[2]),
            ]
            return sp.diags(edge_scale, format="csr") @ self.mesh.get_edge_inner_product(
                term.delta_sigma
            ).tocsr()
        return float(scale) * self.mesh.get_edge_inner_product(term.delta_sigma).tocsr()

    def _normalize_polarization_scale(self, value: float | str | np.ndarray) -> float | str | np.ndarray:
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
                raise ValueError(
                    "component magnetic_recovery_polarization_scale must have length 3"
                )
            if np.any(array < 0.0):
                raise ValueError("magnetic_recovery_polarization_scale must be nonnegative")
            return array.copy()
        scale = float(array)
        if scale < 0.0:
            raise ValueError("magnetic_recovery_polarization_scale must be nonnegative")
        return scale

    def _normalize_initial_polarization_scale(self, value: float) -> float:
        scale = float(value)
        if not np.isfinite(scale):
            raise ValueError("magnetic_recovery_initial_polarization_scale must be finite")
        return scale

    def _source_magnetic_field(self, locations: np.ndarray, time: float) -> np.ndarray:
        field = np.zeros((locations.shape[0], 3), dtype=float)
        for source in self.sources:
            current = float(source.current * source.waveform.value(time))
            if current == 0.0:
                continue
            try:
                from geoana.em.static import LineCurrentWholeSpace  # noqa: PLC0415
            except ImportError as err:
                raise RuntimeError("magnetic_receiver_mode='current_biot' requires geoana") from err
            line_current = LineCurrentWholeSpace(source.locations, current=current, mu=self.mu)
            field += np.asarray(line_current.magnetic_field(locations), dtype=float)
        return field

    def _source_primary_delta6_magnetic_field(
        self,
        locations: np.ndarray,
        time: float,
    ) -> np.ndarray:
        """Diagnostic non-fitted source-primary IP correction for magnetic recovery."""

        field = np.zeros((locations.shape[0], 3), dtype=float)
        if not self.ip_model.terms:
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
            cell_index = self._source_midpoint_cell_index(source)
            kernel = 0.0
            for term in self.ip_model.terms:
                delta_sigma = float(term.delta_sigma[cell_index])
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
            if self.magnetic_recovery_source_primary_delta6_basis == "edge_current":
                source_field = biot_savart_h_from_edge_current_moments(
                    self.mesh,
                    source.initial_edge_vector(self.mesh),
                    locations,
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

    def _source_history_magnetic_field(
        self,
        locations: np.ndarray,
        time: float,
        initial_memories: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        """Evaluate prescribed source-history magnetic recovery correction."""

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
                ip_model=self.ip_model,
                initial_ip_model=self.initial_ip_model,
                initial_memories=initial_memories,
                mu=self.mu,
                cache=self._source_history_runtime_cache,
            )
        return field

    def _source_midpoint_cell_index(self, source: GroundedWireSource) -> int:
        midpoint = np.mean(source.locations, axis=0).reshape(1, 3)
        if hasattr(self.mesh, "closest_points_index"):
            index = self.mesh.closest_points_index(midpoint, "CC")
            return int(np.asarray(index).reshape(-1)[0])
        centers = np.asarray(self.mesh.cell_centers, dtype=float)
        distances = np.sum((centers - midpoint[0]) ** 2, axis=1)
        return int(np.argmin(distances))

    def _solver_for_step(self, step_index: int):
        if self.linear_solver == "cg":
            matrix = self.system_matrix(step_index)
            preconditioner = self._cg_preconditioner(matrix)

            def solve(rhs):
                rhs = np.asarray(rhs, dtype=float)
                rhs_norm = float(np.linalg.norm(rhs))
                initial_relative_residual = 0.0 if rhs_norm == 0.0 else 1.0
                iterations = 0

                def relative_true_residual(iterate: np.ndarray) -> float:
                    residual_norm = float(np.linalg.norm(rhs - matrix @ iterate))
                    if rhs_norm == 0.0:
                        return 0.0 if residual_norm == 0.0 else float("inf")
                    return residual_norm / rhs_norm

                def count_iteration(_iterate: np.ndarray) -> None:
                    nonlocal iterations
                    iterations += 1

                solution, info = spla.cg(
                    matrix,
                    rhs,
                    rtol=float(self.cg_tolerance),
                    atol=0.0,
                    maxiter=self.cg_maxiter,
                    M=preconditioner,
                    callback=count_iteration,
                )
                final_relative_residual = relative_true_residual(solution)
                external_gate_pass = bool(
                    np.isfinite(final_relative_residual)
                    and final_relative_residual <= float(self.cg_tolerance)
                )
                if info != 0 or not external_gate_pass:
                    reason = (
                        "backend_failed"
                        if info != 0
                        else "external_true_residual_above_tolerance"
                    )
                    diagnostics = self._cg_failure_diagnostics(
                        matrix,
                        step_index=step_index,
                        reason=reason,
                        backend_info=info,
                        iterations=iterations,
                        initial_relative_residual=initial_relative_residual,
                        final_relative_residual=final_relative_residual,
                    )
                    raise RuntimeError(
                        "CG solver failed convergence gate: "
                        + json.dumps(
                            diagnostics,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                    )
                return solution

            return solve

        if self.linear_solver == "pardiso":
            key = self._time_step_cache_key(step_index)
            if key not in self._factor_cache:
                self._factor_cache[key] = self._factorize_pardiso(self.system_matrix(step_index))
            return self._factor_cache[key]

        key = self._time_step_cache_key(step_index)
        if key not in self._factor_cache:
            self._factor_cache[key] = self._factorize(self.system_matrix(step_index))
        return self._factor_cache[key]

    def _cg_preconditioner(self, matrix: sp.spmatrix):
        if self.cg_preconditioner == "none":
            return None
        diagonal = np.asarray(matrix.diagonal(), dtype=float)
        inverse = np.zeros_like(diagonal)
        mask = np.abs(diagonal) > 0.0
        inverse[mask] = 1.0 / diagonal[mask]
        return spla.LinearOperator(matrix.shape, matvec=lambda x: inverse * x)

    def _cg_failure_diagnostics(
        self,
        matrix: sp.spmatrix,
        *,
        step_index: int,
        reason: str,
        backend_info: int,
        iterations: int,
        initial_relative_residual: float,
        final_relative_residual: float,
    ) -> dict[str, object]:
        matrix = matrix.tocsr()
        diagonal = np.asarray(matrix.diagonal(), dtype=float)
        difference = matrix - matrix.T
        matrix_max = float(np.max(np.abs(matrix.data))) if matrix.nnz else 0.0
        difference_max = (
            float(np.max(np.abs(difference.data))) if difference.nnz else 0.0
        )

        def finite_value(value: float) -> float | str:
            value = float(value)
            if np.isfinite(value):
                return value
            return "nan" if np.isnan(value) else ("inf" if value > 0.0 else "-inf")

        return {
            "diagnostic_schema": "atem3d.cg-convergence-diagnostic",
            "diagnostic_schema_version": 2,
            "reason": reason,
            "solver": "scipy_cg",
            "preconditioner": self.cg_preconditioner,
            "step_index": int(step_index),
            "dt_s": finite_value(self.time_steps[step_index]),
            "iterations": int(iterations),
            "backend_info": int(backend_info),
            "backend_reported_converged": bool(backend_info == 0),
            "rtol": float(self.cg_tolerance),
            "atol": 0.0,
            "maxiter": self.cg_maxiter,
            "relative_true_residual": {
                "initial": finite_value(initial_relative_residual),
                "final": finite_value(final_relative_residual),
                "best": None,
                "history_available": False,
            },
            "matrix": {
                "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
                "nnz": int(matrix.nnz),
                "diagonal_min": finite_value(np.min(diagonal)),
                "diagonal_max": finite_value(np.max(diagonal)),
                "relative_max_abs_transpose_difference": finite_value(
                    difference_max / matrix_max if matrix_max > 0.0 else 0.0
                ),
            },
        }

    def _factorize(self, matrix: sp.csr_matrix):
        return spla.factorized(matrix.tocsc())

    def _factorize_pardiso(self, matrix: sp.csr_matrix):
        try:
            from pymatsolver import Pardiso  # noqa: PLC0415
        except ImportError as err:
            raise RuntimeError(
                "linear_solver='pardiso' requires pymatsolver with a Pardiso backend"
            ) from err

        active_cpml = self.cpml is not None and self.cpml.thickness_cells > 0
        solver = Pardiso(
            matrix.tocsr(),
            is_symmetric=False if active_cpml else True,
            is_positive_definite=False if active_cpml else True,
        )
        return solver.solve

    def _time_step_cache_key(self, step_index: int) -> int:
        dt = float(self.time_steps[step_index])
        for i, old_dt in enumerate(self.time_steps[: step_index + 1]):
            if np.isclose(float(old_dt), dt, rtol=0.0, atol=1.0e-15):
                return int(i)
        return int(step_index)

    def _cpml_profiles(self, step_index: int) -> CPMLProfiles:
        if self.cpml is None:
            raise ValueError("CPML is not enabled")
        key = self._time_step_cache_key(step_index)
        if key not in self._cpml_profiles_cache:
            self._cpml_profiles_cache[key] = build_cpml_profiles(
                self.mesh,
                self.cpml,
                float(self.time_steps[step_index]),
            )
        return self._cpml_profiles_cache[key]

    def _cpml_curl_split(self) -> CurlSplit:
        if self.cpml is None:
            raise ValueError("CPML is not enabled")
        if self._cpml_curl_split_cache is None:
            self._cpml_curl_split_cache = split_edge_curl(self.mesh)
        return self._cpml_curl_split_cache
