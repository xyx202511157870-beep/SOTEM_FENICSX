"""Result persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import yaml


def save_result_hdf5(path: str | Path, result: Any, config: dict[str, Any]) -> None:
    """Save fields, receiver data, and YAML metadata to HDF5."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("times", data=result.times)
        if hasattr(result, "e"):
            h5.create_dataset("e", data=result.e)
            h5.attrs["receiver_data_only"] = False
            if hasattr(result, "b"):
                h5.create_dataset("b", data=result.b)
                h5.attrs["formulation"] = "eb"
                h5.attrs["electric_field_location"] = "edges"
                h5.attrs["magnetic_field_location"] = "faces"
            elif hasattr(result, "h"):
                h5.create_dataset("h", data=result.h)
                h5.attrs["formulation"] = "hj"
                h5.attrs["electric_field_location"] = "faces"
                h5.attrs["magnetic_field_location"] = "edges"
            else:
                raise ValueError("field-history result must contain either b or h")
        else:
            h5.attrs["receiver_data_only"] = True
            h5.attrs["formulation"] = str(config.get("formulation", "eb")).lower()
            h5.attrs["electric_field_location"] = "not_saved"
            h5.attrs["magnetic_field_location"] = "not_saved"
        h5.create_dataset("data", data=result.data)
        h5.attrs["config_yaml"] = yaml.safe_dump(config, sort_keys=True)
