"""Optional empymod reference-response wrapper."""

from __future__ import annotations

import numpy as np

from .survey import FiniteWireSurvey, LayerModel


def empymod_bipole_reference(
    times,
    survey: FiniteWireSurvey,
    layers: LayerModel,
    *,
    components=("Ex", "Ey"),
    srcpts: int = 21,
    signal: int = -1,
):
    """Compute a finite-wire empymod reference if empymod is installed.

    This wrapper intentionally stays small. It is meant for 1D layered references;
    full 3D FEM/DOLFINx validation belongs in a separate modelling repository.
    """

    try:
        import empymod
    except ImportError as exc:
        raise RuntimeError("empymod is required for empymod_bipole_reference") from exc

    times = np.asarray(times, dtype=float)
    depth, res = layers.empymod_depth_res()
    src = [
        survey.source_start[0],
        survey.source_end[0],
        survey.source_start[1],
        survey.source_end[1],
        -survey.source_start[2],
        -survey.source_end[2],
    ]
    x, y, z = survey.receiver
    rec_map = {
        "Ex": ([x, y, -z, 0.0, 0.0], False, 1.0),
        "Ey": ([x, y, -z, 90.0, 0.0], False, 1.0),
        "Hz": ([x, y, -z, 0.0, 90.0], True, 1.0),
    }
    cols = []
    for component in components:
        if component not in rec_map:
            raise ValueError(f"unsupported component {component!r}")
        rec, mrec, factor = rec_map[component]
        values = empymod.bipole(
            src=src,
            rec=rec,
            depth=depth,
            res=res,
            freqtime=times,
            signal=signal,
            strength=survey.current,
            srcpts=int(srcpts),
            mrec=mrec,
            verb=1,
        )
        cols.append(np.asarray(values, dtype=float).reshape(-1) * factor)
    return np.column_stack(cols)

