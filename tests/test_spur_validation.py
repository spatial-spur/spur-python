from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spur import spurtransform
from spur.utils.matrix import cluster_matrix


def test_spurtransform_validation_errors_are_value_errors(capsys) -> None:
    df = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "lon": [10.0, 11.0, 12.0],
            "lat": [20.0, 21.0, 22.0],
            "sx": [0.0, 1.0, 2.0],
            "sy": [0.0, 0.5, 1.0],
            "grp": ["a", None, "b"],
        }
    )

    with pytest.raises(ValueError, match="radius must be specified"):
        spurtransform("y ~ 1", df, lon="lon", lat="lat", transformation="iso")

    with pytest.raises(ValueError, match="clustvar must be specified"):
        spurtransform("y ~ 1", df, transformation="cluster")

    with pytest.raises(ValueError, match="Cluster variable 'nope'"):
        spurtransform("y ~ 1", df, transformation="cluster", clustvar="nope")

    with pytest.raises(ValueError, match="missing values"):
        spurtransform("y ~ 1", df, transformation="cluster", clustvar="grp")

    with pytest.raises(ValueError, match="Longitude"):
        bad_lon = df.copy()
        bad_lon["lon"] = [200.0, 10.0, 11.0]
        spurtransform("y ~ 1", bad_lon, lon="lon", lat="lat", transformation="nn")

    with pytest.raises(ValueError, match="No variables"):
        spurtransform("1 ~ 1", df, lon="lon", lat="lat", transformation="nn")

    out = cluster_matrix(np.array([0, 0, 1]))
    assert out.shape == (3, 3)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("backend", ["fallback", "rust"])
@pytest.mark.parametrize("bad_radius", [0.0, -5.0, np.nan, np.inf, True])
def test_spurtransform_iso_rejects_invalid_radius(
    backend: str, bad_radius: float | bool
) -> None:
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0], "sx": [0.0, 1.0, 2.0], "sy": [0.0, 0.0, 1.0]})
    with pytest.raises(ValueError, match="radius"):
        spurtransform(
            "y ~ 1",
            df,
            coords_euclidean=("sx", "sy"),
            transformation="iso",
            radius=bad_radius,
            backend=backend,
        )


@pytest.mark.parametrize("backend", ["fallback", "rust"])
@pytest.mark.parametrize("transformation", ["nn", "iso", "lbmgls"])
def test_spurtransform_spatial_rejects_single_observation(
    backend: str, transformation: str
) -> None:
    df = pd.DataFrame({"y": [1.0], "sx": [0.0], "sy": [0.0]})
    with pytest.raises(ValueError, match="at least 2 observations"):
        spurtransform(
            "y ~ 1",
            df,
            coords_euclidean=("sx", "sy"),
            transformation=transformation,
            radius=1.0 if transformation == "iso" else None,
            backend=backend,
        )
