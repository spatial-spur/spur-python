from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import spur


@pytest.mark.parametrize("backend", ["fallback", "rust"])
def test_residual_dimension_and_units(backend):
    rng = np.random.default_rng(788)
    d = pd.DataFrame(rng.normal(size=(12, 3)), columns=["y", "sx", "sy"])
    d["x"] = rng.normal(size=12)
    kw = dict(coords_euclidean=["sx", "sy"], nrep=100, seed=3, backend=backend)
    with pytest.raises(ValueError, match="residual dimension"):
        spur.spurtest_i0resid("y ~ x", d, q=11, **kw)
    with pytest.raises(ValueError, match=">= 2"):
        spur.spurhalflife("y", d, q=1, **kw)
    with pytest.raises(ValueError, match="variation|constant"):
        spur.spurhalflife("constant", d.assign(constant=5.0), q=4, **kw)


@pytest.mark.parametrize("backend", ["fallback", "rust"])
def test_regressor_scale_and_half_life_units(backend):
    rng = np.random.default_rng(2026)
    d = pd.DataFrame({k: rng.normal(size=100) for k in ["y", "x", "sx", "sy"]})
    kw = dict(coords_euclidean=["sx", "sy"], q=8, nrep=100, seed=7, backend=backend)
    a = spur.spurtest_i0resid("y ~ x", d, **kw)
    b = spur.spurtest_i0resid("y ~ x", d.assign(x=d.x * 1e-12), **kw)
    np.testing.assert_allclose(
        [a.LR, a.ha_param, a.pvalue], [b.LR, b.ha_param, b.pvalue], atol=1e-9
    )
    line = pd.DataFrame(
        {"sx": np.arange(30.0), "sy": 0.0, "y": np.sin(np.arange(30.0))}
    )
    opts = dict(coords_euclidean=["sx", "sy"], q=4, nrep=99, seed=42, backend=backend)
    h = spur.spurhalflife("y", line, normdist=True, **opts)
    absolute = spur.spurhalflife("y", line, normdist=False, **opts)
    scaled = spur.spurhalflife(
        "y", line.assign(sx=line.sx * 10), normdist=False, **opts
    )
    assert absolute.max_dist == pytest.approx(29)
    assert absolute.ci_lower == pytest.approx(h.ci_lower * 29)
    assert scaled.ci_lower == pytest.approx(absolute.ci_lower * 10)
    assert scaled.max_dist == pytest.approx(290)


def test_public_seeded_residual_parity_and_host_rng():
    d = pd.read_csv(
        Path(__file__).resolve().parents[2] / "verification/data/bench_200.csv"
    )
    kw = dict(coords_euclidean=["s_1", "s_2"], q=8, nrep=1000, seed=42)
    np.random.seed(101)
    state = np.random.get_state()
    a = spur.spurtest_i0resid("y ~ x1 + x2", d, backend="fallback", **kw)
    middle = np.random.get_state()
    b = spur.spurtest_i0resid("y ~ x1 + x2", d, backend="rust", **kw)
    after = np.random.get_state()
    for result in [middle, after]:
        assert result[0] == state[0] and result[2:] == state[2:]
        np.testing.assert_array_equal(result[1], state[1])
    np.testing.assert_allclose(
        [a.LR, a.ha_param, a.pvalue], [b.LR, b.ha_param, b.pvalue], atol=1e-9
    )


@pytest.mark.parametrize("backend", ["fallback", "rust"])
def test_uninformative_symmetric_geometry(backend):
    d = pd.DataFrame(
        {"sx": [0.0, 1.0, 0.5], "sy": [0.0, 0.0, np.sqrt(3) / 2], "y": [1.0, 2.0, 4.0]}
    )
    options = dict(coords_euclidean=["sx", "sy"], q=2, nrep=20, seed=3, backend=backend)
    for fn in (spur.spurtest_i0, spur.spurtest_i1, spur.spurhalflife):
        with pytest.raises(ValueError, match="uninformative"):
            fn("y", d, **options)


@pytest.mark.parametrize("backend", ["fallback", "rust"])
def test_geographic_half_life_uses_metres(backend):
    d = pd.DataFrame({"lon": np.arange(30.0), "lat": 0.0, "y": np.sin(np.arange(30.0))})
    opts = dict(lon="lon", lat="lat", q=4, nrep=99, seed=42, backend=backend)
    normalized = spur.spurhalflife("y", d, normdist=True, **opts)
    absolute = spur.spurhalflife("y", d, normdist=False, **opts)
    assert absolute.max_dist == pytest.approx(6_371_000 * np.deg2rad(29))
    assert absolute.ci_lower == pytest.approx(normalized.ci_lower * absolute.max_dist)


@pytest.mark.parametrize("backend", ["fallback", "rust"])
def test_residual_dimension_boundary_is_valid(backend):
    rng = np.random.default_rng(788)
    d = pd.DataFrame(rng.normal(size=(12, 3)), columns=["y", "sx", "sy"])
    d["x"] = rng.normal(size=12)
    result = spur.spurtest_i0resid(
        "y ~ x",
        d,
        coords_euclidean=["sx", "sy"],
        q=10,
        nrep=500,
        seed=3,
        backend=backend,
    )
    assert result.pvalue == pytest.approx(0.72)


def test_power_calibration_reports_unbracketable_experiment(monkeypatch):
    from spur.utils import inference

    monkeypatch.setattr(inference, "get_pow_qf", lambda *args: 0.0)
    with pytest.raises(RuntimeError, match="step-2 bracketing did not converge in 200"):
        inference.get_ha_param_i0(np.eye(2), np.eye(2), np.eye(2), np.ones((2, 10)))


@pytest.mark.parametrize("backend", ["fallback", "rust"])
def test_response_shift_does_not_change_inference(backend):
    d = pd.read_csv(
        Path(__file__).resolve().parents[2] / "verification/data/bench_200.csv"
    )
    options = dict(
        coords_euclidean=["s_1", "s_2"], q=8, nrep=1000, seed=42, backend=backend
    )
    baseline = spur.spurtest_i0resid("y ~ x1 + x2", d, **options)
    shifted = spur.spurtest_i0resid("y ~ x1 + x2", d.assign(y=d.y + 1024.0), **options)
    np.testing.assert_allclose(
        [baseline.LR, baseline.ha_param, baseline.pvalue],
        [shifted.LR, shifted.ha_param, shifted.pvalue],
        atol=1e-9,
    )


@pytest.mark.parametrize("backend", ["fallback", "rust"])
def test_insufficient_positive_covariance_rank_is_rejected(backend):
    rng = np.random.default_rng(83)
    d = pd.DataFrame(
        {"sx": np.repeat(np.arange(6.0), 2), "sy": 0.0, "y": rng.normal(size=12)}
    )
    with pytest.raises(ValueError, match="positive rank"):
        spur.spurtest_i0(
            "y",
            d,
            coords_euclidean=["sx", "sy"],
            q=6,
            nrep=100,
            seed=3,
            backend=backend,
        )


@pytest.mark.parametrize("backend", ["fallback", "rust"])
def test_zero_projected_signal_is_rejected(backend):
    from spur.utils.dist import normalized_distmat
    from spur.utils.matrix import demean_matrix, get_r, get_sigma_lbm

    rng = np.random.default_rng(603)
    coords = rng.normal(size=(20, 2))
    r = get_r(demean_matrix(get_sigma_lbm(normalized_distmat(coords, latlon=False))), 4)
    _, _, vt = np.linalg.svd(np.column_stack((np.ones(20), r)).T, full_matrices=True)
    d = pd.DataFrame({"sx": coords[:, 0], "sy": coords[:, 1], "y": vt[-1]})
    with pytest.raises(ValueError, match="[Pp]roject"):
        spur.spurtest_i0(
            "y", d, coords_euclidean=["sx", "sy"], q=4, nrep=100, backend=backend
        )


def test_stale_spur_extension_falls_back_without_disabling_scpc(monkeypatch):
    from types import SimpleNamespace
    import importlib
    import scpc.core as scpc_core
    import spur.core as spur_core

    stale = SimpleNamespace(API_VERSION=5, SPUR_API_VERSION=1)
    original_import = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name, *args, **kwargs: (
            stale
            if name == "spatial_spur_rust"
            else original_import(name, *args, **kwargs)
        ),
    )
    assert spur_core.load_rust_backend() is None
    assert scpc_core.load_rust_backend() is stale

    rng = np.random.default_rng(788)
    d = pd.DataFrame(rng.normal(size=(12, 3)), columns=["y", "sx", "sy"])
    d["x"] = rng.normal(size=12)
    opts = dict(coords_euclidean=["sx", "sy"], q=10, nrep=500, seed=3, nthreads=1)
    result = spur.spurtest_i0resid("y ~ x", d, backend="auto", **opts)
    assert result.backend == "fallback"
    with pytest.raises(RuntimeError, match="unavailable or incompatible"):
        spur.spurtest_i0resid("y ~ x", d, backend="rust", **opts)
    with pytest.raises(ValueError, match="residual dimension"):
        spur.spurtest_i0resid("y ~ x", d, backend="auto", **(opts | {"q": 11}))
    triangle = pd.DataFrame(
        {"sx": [0.0, 1.0, 0.5], "sy": [0.0, 0.0, np.sqrt(3) / 2], "y": [1.0, 2.0, 4.0]}
    )
    with pytest.raises(ValueError, match="uninformative"):
        spur.spurtest_i0("y", triangle, backend="auto", **(opts | {"q": 2}))


@pytest.mark.parametrize("scale", [1e200, -1e200])
def test_large_response_scale_native_fallback_parity(scale):
    d = pd.read_csv(
        Path(__file__).resolve().parents[2] / "verification/data/bench_200.csv"
    )
    options = dict(coords_euclidean=["s_1", "s_2"], q=8, nrep=1000, seed=42)
    baseline = spur.spurtest_i0resid("y ~ x1 + x2", d, backend="fallback", **options)
    data_scaled = d.assign(y=d.y * scale)
    for backend in ("fallback", "rust"):
        actual = spur.spurtest_i0resid(
            "y ~ x1 + x2", data_scaled, backend=backend, **options
        )
        np.testing.assert_allclose(
            [actual.LR, actual.ha_param, actual.pvalue],
            [baseline.LR, baseline.ha_param, baseline.pvalue],
            atol=1e-9,
        )
