from pathlib import Path

import numpy as np
import pytest

from atem3d.cli import _read_response_csv


def test_empymod_reference_table_reader_returns_time_and_component_matrix(tmp_path: Path):
    path = tmp_path / "reference_empymod_or_1d.csv"
    path.write_text(
        "time_obs,Ex,Ey,dBzdt\n"
        "1e-5,1.0,2.0,3.0\n"
        "1.0,4.0,5.0,6.0\n",
        encoding="utf-8",
    )

    times, values = _read_response_csv(path, ["Ex", "Ey", "dBzdt"])

    np.testing.assert_allclose(times, [1.0e-5, 1.0])
    np.testing.assert_allclose(values, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_empymod_reference_table_reader_rejects_missing_component(tmp_path: Path):
    path = tmp_path / "reference_empymod_or_1d.csv"
    path.write_text("time_obs,Ex,dBzdt\n1e-5,1.0,3.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing components"):
        _read_response_csv(path, ["Ex", "Ey", "dBzdt"])
