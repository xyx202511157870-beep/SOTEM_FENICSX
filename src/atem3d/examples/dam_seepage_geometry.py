"""Frozen geometry contract for the HTML dam-seepage model.

The numbers come from the interactive HTML schematic
「堤坝渗流通道交互模型_三维与侧剖面」.  This module is the unique
geometry source for the H3 forward task.  It does not run FEM.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


HTML_SOURCE_NAME = "堤坝渗流通道交互模型_三维与侧剖面.html"
COORDINATE_SYSTEM = "z_up"
LENGTH_UNIT = "m"

DAM_CREST_Z = 100.0
COVER_TOP_Z = 50.0
WATER_SURFACE_Z = 85.0
VALLEY_FLOOR_Z = 0.0
HILL_TOP_Z = 150.0
DAM_X_HALF_WIDTH = 200.0

CHANNEL_P1 = (30.0, -128.5, 65.0)
CHANNEL_P2 = (30.0, 0.0, 40.0)
CHANNEL_P3 = (30.0, 276.0, 10.0)

WIRE_POLYLINE = (
    (0.0, -240.0, 50.0),
    (0.0, -240.0, 95.0),
    (-50.0, -240.0, 120.0),
    (-200.0, -240.0, 120.0),
    (-275.0, -240.0, 175.0),
    (-350.0, -240.0, 190.0),
    (-350.0, 330.0, 190.0),
    (-275.0, 330.0, 175.0),
    (-200.0, 330.0, 120.0),
    (-50.0, 330.0, 20.0),
    (0.0, 330.0, 5.0),
)
ELECTRODE_A = WIRE_POLYLINE[0]
ELECTRODE_B = WIRE_POLYLINE[-1]
SOURCE_CURRENT_A = 10.0

UAV_LINE_Y = 0.0
UAV_LINE_X_MIN = -150.0
UAV_LINE_X_MAX = 150.0
UAV_LINE_DX = 5.0
UAV_FLIGHT_HEIGHT_M = 0.5
HTML_DISPLAY_RECEIVER_Z = 100.3

CHANNEL_EQUIVALENT_RADIUS_M = 2.0
CHANNEL_RADIUS_STATUS = "proposed_not_printed_in_html"

RESISTIVITY_OHM_M = {
    "air": 1.0e8,
    "water": 20.0,
    "dam": 200.0,
    "cover": 80.0,
    "channel": 10.0,
    "foundation": 300.0,
    "hill": 300.0,
}
RESISTIVITY_STATUS = "project_convention_not_printed_in_html"

PHYSICAL_MARKERS = {
    "air": 100,
    "foundation": 101,
    "dam": 102,
    "water": 103,
    "cover": 104,
    "channel": 105,
    "hill": 106,
    "outer_boundary": 201,
    "terrain_surface": 202,
    "source_wire": 301,
}


def valley_floor_z(x: float | np.ndarray) -> np.ndarray:
    """Valley / hill elevation used by the HTML ``ground(x)`` function."""

    a = np.abs(np.asarray(x, dtype=float))
    z = np.empty_like(a, dtype=float)
    z[a <= 50.0] = 0.0
    mid = (a > 50.0) & (a < 275.0)
    z[mid] = (a[mid] - 50.0) / 1.5
    z[a >= 275.0] = 150.0
    return z if z.shape else float(z)


def dam_upstream_y(z: float | np.ndarray) -> np.ndarray:
    """Upstream dam face ``y = -356 + 3.5 z``."""

    return -356.0 + 3.5 * np.asarray(z, dtype=float)


def dam_downstream_y(z: float | np.ndarray) -> np.ndarray:
    """Downstream dam face ``y = 306 - 3 z``."""

    return 306.0 - 3.0 * np.asarray(z, dtype=float)


def uav_line_x() -> np.ndarray:
    return np.arange(UAV_LINE_X_MIN, UAV_LINE_X_MAX + 0.5 * UAV_LINE_DX, UAV_LINE_DX)


def uav_receiver_locations(*, flight_height_m: float = UAV_FLIGHT_HEIGHT_M) -> np.ndarray:
    """HTML crest-line planform at y=0; production height is local topo + 0.5 m."""

    x = uav_line_x()
    y = np.full(x.shape, UAV_LINE_Y)
    on_dam = np.abs(x) <= DAM_X_HALF_WIDTH
    z_topo = np.where(on_dam, DAM_CREST_Z, valley_floor_z(x))
    return np.column_stack([x, y, z_topo + float(flight_height_m)])


@dataclass(frozen=True)
class DamSeepageGeometry:
    """HTML dam-seepage geometry frozen for the H3 forward task."""

    channel_points: tuple[tuple[float, float, float], ...] = (CHANNEL_P1, CHANNEL_P2, CHANNEL_P3)
    wire_polyline: tuple[tuple[float, float, float], ...] = WIRE_POLYLINE
    source_current: float = SOURCE_CURRENT_A
    flight_height_m: float = UAV_FLIGHT_HEIGHT_M
    channel_radius_m: float = CHANNEL_EQUIVALENT_RADIUS_M
    resistivities_ohm_m: dict[str, float] = field(default_factory=lambda: dict(RESISTIVITY_OHM_M))

    @property
    def electrode_a(self) -> tuple[float, float, float]:
        return self.wire_polyline[0]

    @property
    def electrode_b(self) -> tuple[float, float, float]:
        return self.wire_polyline[-1]

    def uav_receiver_locations(self) -> np.ndarray:
        return uav_receiver_locations(flight_height_m=self.flight_height_m)

    def validate(self) -> dict[str, float | int | str]:
        """Fail closed if HTML invariants are violated."""

        p3 = np.asarray(self.channel_points[2], dtype=float)
        a = np.asarray(self.electrode_a, dtype=float)
        b = np.asarray(self.electrode_b, dtype=float)
        if not np.allclose(p3, CHANNEL_P3):
            raise ValueError("P3 must remain (30, 276, 10) m")
        if not np.allclose(b, ELECTRODE_B):
            raise ValueError("electrode B must remain (0, 330, 5) m inside the cover")
        if not np.allclose(a, ELECTRODE_A):
            raise ValueError("electrode A must remain (0, -240, 50) m")
        if abs(COVER_TOP_Z - p3[2] - 40.0) > 1.0e-9:
            raise ValueError("cover top must be 40 m above the channel outlet")
        if not (0.0 < b[2] < COVER_TOP_Z):
            raise ValueError("electrode B must sit inside the cover interval 0 < z < 50 m")
        if float(self.source_current) <= 0.0:
            raise ValueError("source current must be positive from A to B")
        if abs(float(self.flight_height_m) - UAV_FLIGHT_HEIGHT_M) > 1.0e-12:
            raise ValueError("production UAV height must be 0.5 m AGL")
        if float(self.channel_radius_m) <= 0.0:
            raise ValueError("channel radius must be positive")
        stations = self.uav_receiver_locations()
        if stations.shape != (int(round((UAV_LINE_X_MAX - UAV_LINE_X_MIN) / UAV_LINE_DX)) + 1, 3):
            raise ValueError("UAV line must follow x=-150:5:150 at y=0")
        if not np.allclose(stations[:, 2] - DAM_CREST_Z, self.flight_height_m):
            raise ValueError("crest-line UAV stations must sit 0.5 m above z=100")
        wire = np.asarray(self.wire_polyline, dtype=float)
        wire_length = float(np.sum(np.linalg.norm(np.diff(wire, axis=0), axis=1)))
        return {
            "html_source": HTML_SOURCE_NAME,
            "coordinate_system": COORDINATE_SYSTEM,
            "n_wire_vertices": int(wire.shape[0]),
            "wire_length_m": wire_length,
            "n_uav_stations": int(stations.shape[0]),
            "flight_height_m": float(self.flight_height_m),
            "channel_radius_m": float(self.channel_radius_m),
            "channel_radius_status": CHANNEL_RADIUS_STATUS,
            "resistivity_status": RESISTIVITY_STATUS,
            "cover_minus_outlet_m": float(COVER_TOP_Z - p3[2]),
        }


def build_dam_seepage_geometry() -> DamSeepageGeometry:
    geometry = DamSeepageGeometry()
    geometry.validate()
    return geometry
