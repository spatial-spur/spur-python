from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from spur import spurtest_i0resid, spurtest_i1resid, spurtransform
from spur.utils.dist import haversine_distance
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


@pytest.mark.parametrize("backend", TRANSFORM_BACKENDS)
@pytest.mark.parametrize("formula", ["y ~ 0 + x1", "y ~ x1"])
def test_residual_projection_is_invariant_to_nuisance_shifts(
    backend: str, formula: str
) -> None:
    rng = np.random.default_rng(303)
    n = 42
    sx = np.linspace(0.0, 1.0, n)
    sy = np.sin(np.arange(n) * 0.4)
    x1 = 1.5 + rng.normal(size=n)
    y = (np.arange(n) ** 2 % 11).astype(float)
    shift = 1e6 * x1 if formula == "y ~ 0 + x1" else 1e13
    data = pd.DataFrame({"y": y, "x1": x1, "sx": sx, "sy": sy})
    shifted = data.assign(y=y + shift)
    opts: dict[str, Any] = dict(
        coords_euclidean=("sx", "sy"), q=8, nrep=1000, seed=17, backend=backend
    )
    for fn in (spurtest_i0resid, spurtest_i1resid):
        base = fn(formula, data, **opts)
        result = fn(formula, shifted, **opts)
        np.testing.assert_allclose(base.LR, result.LR, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(base.pvalue, result.pvalue, rtol=0, atol=0)


@pytest.mark.parametrize("backend", TRANSFORM_BACKENDS)
@pytest.mark.parametrize(
    "site1,site2",
    [
        ((0.0, 0.0), (0.0, 1.0)),
        ((0.0, -178.6), (0.0, -178.5)),
        (
            (-1.0036422005124446, 10.076763636412846),
            (37.11177842377785, -142.9944155287363),
        ),
    ],
)
def test_spurtransform_geographic_iso_includes_exact_radius(
    backend: str, site1: tuple[float, float], site2: tuple[float, float], capsys
) -> None:
    lat1, lon1 = site1
    lat2, lon2 = site2
    data = pd.DataFrame({"y": [1.0, 4.0], "lat": [lat1, lat2], "lon": [lon1, lon2]})
    radius = float(haversine_distance(*site1, *site2))
    opts: dict[str, Any] = dict(
        lon="lon", lat="lat", transformation="iso", backend=backend
    )

    inclusive = spurtransform("y ~ 1", data, radius=radius, **opts)
    outside = spurtransform("y ~ 1", data, radius=radius * (1.0 - 1e-12), **opts)
    np.testing.assert_array_equal(inclusive["h_y"], [-3.0, 3.0])
    np.testing.assert_array_equal(outside["h_y"], 0.0)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("backend", TRANSFORM_BACKENDS)
def test_spurtransform_centers_lbmgls_output(backend: str) -> None:
    data = pd.DataFrame({"y": [1e6 + 1.0, 1e6 + 3.0, 1e6 + 9.0], "x": [0.0, 1.0, 3.0]})
    result = spurtransform(
        "y ~ 1", data, coords_euclidean=("x",), transformation="lbmgls", backend=backend
    )
    assert float(result["h_y"].mean()) == pytest.approx(0.0, abs=1e-12)


def test_native_iso_rejects_boolean_radius() -> None:
    native = load_rust_backend()
    if native is None:
        pytest.skip("Rust backend is unavailable")
    with pytest.raises(TypeError, match="bool"):
        native.run_spur_transform(
            "iso",
            3,
            y=np.array([1.0, 2.0, 3.0]),
            coords=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            radius=True,
        )
