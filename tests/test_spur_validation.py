from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spur import spurtransform
from spur.core import load_rust_backend
from spur.utils.matrix import cluster_matrix

TRANSFORM_BACKENDS = (
    "fallback",
    pytest.param(
        "rust",
        marks=pytest.mark.skipif(
            load_rust_backend() is None, reason="Rust backend is unavailable"
        ),
    ),
)


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


@pytest.mark.parametrize("backend", TRANSFORM_BACKENDS)
@pytest.mark.parametrize("bad_radius", [0.0, -5.0, np.nan, np.inf, True])
def test_spurtransform_iso_rejects_invalid_radius(
    backend: str, bad_radius: float | bool
) -> None:
    df = pd.DataFrame(
        {"y": [1.0, 2.0, 3.0], "sx": [0.0, 1.0, 2.0], "sy": [0.0, 0.0, 1.0]}
    )
    with pytest.raises(ValueError, match="radius"):
        spurtransform(
            "y ~ 1",
            df,
            coords_euclidean=("sx", "sy"),
            transformation="iso",
            radius=bad_radius,
            backend=backend,
        )


@pytest.mark.parametrize("backend", TRANSFORM_BACKENDS)
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


def test_residual_no_intercept_invariance_and_transform_helper_alignment(capsys) -> None:
    import spur
    from spur.utils.dist import haversine_distance
    from spur.utils.matrix import get_transformation_stats, iso_matrix, transform

    rng = np.random.default_rng(303)
    n = 42
    sx = np.linspace(0.0, 1.0, n)
    sy = np.sin(np.arange(n) * 0.4)
    x1 = 1.5 + rng.normal(size=n)
    eps = np.sin(np.arange(n) * 0.3) + 0.1 * np.arange(n)
    df = pd.DataFrame({"y": eps, "x1": x1, "sx": sx, "sy": sy})
    df_shift = df.copy()
    df_shift["y"] = eps + 1e6 * x1

    for backend in ("fallback", "rust"):
        for fn in (spur.spurtest_i0resid, spur.spurtest_i1resid):
            r1 = fn("y ~ 0 + x1", df, coords_euclidean=("sx", "sy"), q=6, nrep=120, seed=17, backend=backend)
            r2 = fn("y ~ 0 + x1", df_shift, coords_euclidean=("sx", "sy"), q=6, nrep=120, seed=17, backend=backend)
            np.testing.assert_allclose(r1.LR, r2.LR, rtol=1e-8, atol=1e-8)
            np.testing.assert_allclose(r1.pvalue, r2.pvalue, rtol=1e-12, atol=1e-12)

    # Geographic exact boundary iso & Euclidean exact boundary match Rust vs fallback and stay silent
    geo_radius = float(haversine_distance(0.0, 0.0, 0.0, 1.0))
    df_geo = pd.DataFrame(
        {"y": [10.0, 20.0, 50.0], "lat": [0.0, 0.0, 15.0], "lon": [0.0, 1.0, 15.0]}
    )
    geo_fb = spur.spurtransform("y ~ 1", df_geo, lon="lon", lat="lat", transformation="iso", radius=geo_radius, backend="fallback")
    assert capsys.readouterr().out == ""
    geo_ru = spur.spurtransform("y ~ 1", df_geo, lon="lon", lat="lat", transformation="iso", radius=geo_radius, backend="rust")
    np.testing.assert_allclose(geo_ru["h_y"], geo_fb["h_y"], rtol=1e-12, atol=1e-12)

    _ = iso_matrix(df_geo[["lat", "lon"]].to_numpy(), radius=geo_radius, latlon=True)
    assert capsys.readouterr().out == ""

    # Stats helper aligns with positive-distance NN and inclusive iso radius
    coords_stats = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.5, 0.0]])
    st_nn = get_transformation_stats(coords_stats, method="nn", latlon=False)
    assert np.isclose(st_nn["nn_dist_mean"], (1.0 + 1.0 + 1.0 + 1.5) / 4.0)
    st_iso = get_transformation_stats(coords_stats, method="iso", radius=1.0, latlon=False)
    assert st_iso["n_isolated"] == 1

    # Helper transform(..., method="lbmgls") centers output
    tr_lbm = transform(
        np.array([1.0, 9.0, 20.0]) + 1e4,
        coords_stats[[0, 2, 3]],
        method="lbmgls",
        latlon=False,
    )
    assert tr_lbm.shape == (3,)
    assert abs(float(np.mean(tr_lbm))) < 1e-12
    assert abs(float(np.mean(transform(np.array([1.0, 3.0, 9.0]) + 1e6, np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]]), method="lbmgls", latlon=False)))) < 1e-12

    import spatial_spur_rust as native
    with pytest.raises(TypeError, match="bool"):
        native.run_spur_transform(
            "iso",
            3,
            y=np.array([1.0, 2.0, 3.0]),
            coords=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            radius=True,
        )
