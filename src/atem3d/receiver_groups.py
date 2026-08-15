"""Convenience layouts for three-component magnetic-field receivers.

中文说明：本模块固定三分量磁场强度的输出顺序为 ``Hx, Hy, Hz``，单位
为 A/m。对于有限面积接收器，每个分量对应一只法向分别沿 x、y、z 的正交
线圈；对于点接收器，直接采样全局坐标系中的 H 三分量。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .receivers import AverageReceiver, PointReceiver, build_receiver


H3_COMPONENTS: tuple[str, str, str] = ("Hx", "Hy", "Hz")


def build_h3_receivers(
    *,
    location: tuple[float, float, float],
    receiver_type: str = "point",
    radius: float | None = None,
) -> tuple[PointReceiver | AverageReceiver, ...]:
    """Build one colocated ``Hx, Hy, Hz`` receiver triplet.

    Parameters
    ----------
    location
        Receiver centre in the global Cartesian coordinate system.
    receiver_type
        ``"point"``, ``"disk_average"`` or ``"volume_average"``.
    radius
        Required for finite-area/finite-volume receiver types.

    Returns
    -------
    tuple
        Receivers ordered strictly as ``Hx, Hy, Hz``.  The corresponding data
        unit is A/m.  For ``disk_average``, the three receiver normals are the
        global x, y and z axes, respectively.
    """

    return tuple(
        build_receiver(
            location=location,
            component=component,
            receiver_type=receiver_type,
            radius=radius,
        )
        for component in H3_COMPONENTS
    )


def build_h3_receiver_array(
    *,
    locations: Sequence[Sequence[float]],
    receiver_type: str = "point",
    radius: float | None = None,
) -> tuple[PointReceiver | AverageReceiver, ...]:
    """Build location-major H3 receivers for a survey array.

    The flattened order is

    ``location_0: Hx, Hy, Hz; location_1: Hx, Hy, Hz; ...``.
    """

    normalized_locations = np.asarray(locations, dtype=float)
    if normalized_locations.ndim != 2 or normalized_locations.shape[1] != 3:
        raise ValueError("locations must have shape (n_locations, 3)")
    if not np.all(np.isfinite(normalized_locations)):
        raise ValueError("locations must contain finite values")

    receivers: list[PointReceiver | AverageReceiver] = []
    for location in normalized_locations:
        receivers.extend(
            build_h3_receivers(
                location=tuple(float(value) for value in location),
                receiver_type=receiver_type,
                radius=radius,
            )
        )
    return tuple(receivers)


def reshape_h3_data(data, *, n_locations: int) -> np.ndarray:
    """Reshape flattened receiver data to ``(..., n_locations, 3)``.

    The last axis is ordered ``Hx, Hy, Hz``.  This helper accepts both one-time
    vectors and multi-time arrays as long as their final dimension contains
    exactly ``3 * n_locations`` receiver channels.
    """

    if isinstance(n_locations, bool) or int(n_locations) <= 0:
        raise ValueError("n_locations must be a positive integer")
    n_locations = int(n_locations)
    array = np.asarray(data, dtype=float)
    expected_channels = 3 * n_locations
    if array.ndim == 0 or array.shape[-1] != expected_channels:
        raise ValueError(
            "the final data dimension must equal 3 * n_locations "
            f"({expected_channels})"
        )
    return array.reshape(*array.shape[:-1], n_locations, 3)
