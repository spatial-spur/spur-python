from typing import Any

import numpy as np
import pandas as pd
import pytest

import spur
from spur.core import load_rust_backend
from tests.config import BENCH_200


RUST_AVAILABLE = load_rust_backend() is not None
RUST_PARAMETER = pytest.param(
    "rust",
    marks=pytest.mark.skipif(
        not RUST_AVAILABLE, reason="compatible native extension unavailable"
    ),
)


@pytest.mark.parametrize("backend", ["fallback", RUST_PARAMETER])
def test_nearly_constant_design_preserves_residual_inference(backend):
    s = np.linspace(0.0, 1.0, 60)
    x = 1.0 + 1e-8 * s
    y = np.random.default_rng(123).normal(size=s.size)
    opts: dict[str, Any] = dict(
        coords_euclidean=["s"], q=8, nrep=3000, seed=42, nthreads=1
    )
    for test in (spur.spurtest_i0resid, spur.spurtest_i1resid):
        base = test(
            "y ~ 0 + x", pd.DataFrame(dict(y=y, s=s, x=x)), backend=backend, **opts
        )
        shifted = test(
            "y ~ 0 + x",
            pd.DataFrame(dict(y=y + 1e9 * x, s=s, x=x)),
            backend=backend,
            **opts,
        )
        assert shifted.LR == pytest.approx(base.LR, rel=1e-6)
        assert shifted.pvalue == base.pvalue


@pytest.mark.parametrize("backend", ["fallback", RUST_PARAMETER])
def test_residual_inference_agrees_across_equivalent_large_offset_designs(backend):
    rng = np.random.default_rng(86)
    data = pd.DataFrame(
        {
            "s": rng.uniform(size=36),
            "g": np.repeat(["a", "b", "c"], 12),
            "y": 1e13 + 0.25 * rng.integers(-12, 13, size=36),
        }
    )
    opts = dict(coords_euclidean=["s"], q=4, nrep=499, seed=42, backend=backend)
    for test in (spur.spurtest_i0resid, spur.spurtest_i1resid):
        baseline = test("y ~ C(g)", data, **opts)
        all_indicators = test("y ~ 0 + C(g)", data, **opts)
        assert all_indicators.LR == pytest.approx(baseline.LR, rel=1e-12)
        assert all_indicators.pvalue == baseline.pvalue


@pytest.mark.parametrize("backend", ["fallback", RUST_PARAMETER])
def test_i0_and_i1_calibrate_over_attainable_colocated_covariances(backend):
    sites = np.array([0.0, 0.0, 0.04, 0.17, 0.19, 0.25, 0.42, 0.6, 0.78, 1.0])
    data = pd.DataFrame({"s": sites, "y": np.sin(7 * sites)})
    opts = dict(coords_euclidean=["s"], q=2, nrep=199, seed=17, backend=backend)
    i0 = spur.spurtest_i0("y", data, **opts)
    assert i0.rho_grid is not None
    assert i0.rho_grid[0] == pytest.approx(1 / 45)
    assert i0.rho_grid[-1] == 0.03
    assert np.all(np.diff(i0.rho_grid) > 0)
    assert np.isfinite(i0.LR)
    assert np.isfinite(spur.spurtest_i1("y", data, **opts).LR)


@pytest.mark.parametrize("backend", ["fallback", RUST_PARAMETER])
def test_residual_dimension_and_units(backend):
    rng = np.random.default_rng(788)
    d = pd.DataFrame(rng.normal(size=(12, 3)), columns=["y", "sx", "sy"])
    d["x"] = rng.normal(size=12)
    kw: dict[str, Any] = dict(
        coords_euclidean=["sx", "sy"], nrep=100, seed=3, backend=backend
    )
    with pytest.raises(ValueError, match="residual dimension"):
        spur.spurtest_i0resid("y ~ x", d, q=11, **kw)
    with pytest.raises(ValueError, match=">= 2"):
        spur.spurhalflife("y", d, q=1, **kw)
    with pytest.raises(ValueError, match="variation|constant"):
        spur.spurhalflife("constant", d.assign(constant=5.0), q=4, **kw)


@pytest.mark.parametrize("backend", ["fallback", RUST_PARAMETER])
def test_regressor_scale_and_half_life_units(backend):
    rng = np.random.default_rng(2026)
    d = pd.DataFrame({k: rng.normal(size=100) for k in ["y", "x", "sx", "sy"]})
    kw: dict[str, Any] = dict(
        coords_euclidean=["sx", "sy"], q=8, nrep=100, seed=7, backend=backend
    )
    a = spur.spurtest_i0resid("y ~ x", d, **kw)
    b = spur.spurtest_i0resid("y ~ x", d.assign(x=d.x * 1e-12), **kw)
    np.testing.assert_allclose(
        [a.LR, a.ha_param, a.pvalue], [b.LR, b.ha_param, b.pvalue], atol=1e-9
    )
    line = pd.DataFrame(
        {"sx": np.arange(30.0), "sy": 0.0, "y": np.sin(np.arange(30.0))}
    )
    opts: dict[str, Any] = dict(
        coords_euclidean=["sx", "sy"], q=4, nrep=99, seed=42, backend=backend
    )
    h = spur.spurhalflife("y", line, normdist=True, **opts)
    absolute = spur.spurhalflife("y", line, normdist=False, **opts)
    scaled = spur.spurhalflife(
        "y", line.assign(sx=line.sx * 10), normdist=False, **opts
    )
    assert absolute.max_dist == pytest.approx(29)
    assert absolute.ci_lower == pytest.approx(h.ci_lower * 29)
    assert scaled.ci_lower == pytest.approx(absolute.ci_lower * 10)
    assert scaled.max_dist == pytest.approx(290)


@pytest.mark.parametrize("backend", ["fallback", RUST_PARAMETER])
def test_public_seeded_residual_parity_and_host_rng(backend):
    d = pd.read_csv(BENCH_200)
    kw: dict[str, Any] = dict(coords_euclidean=["s_1", "s_2"], q=8, nrep=1000, seed=42)
    np.random.seed(101)
    state = np.random.get_state(legacy=True)
    assert isinstance(state, tuple)
    a = spur.spurtest_i0resid("y ~ x1 + x2", d, backend="fallback", **kw)
    middle = np.random.get_state(legacy=True)
    b = spur.spurtest_i0resid("y ~ x1 + x2", d, backend=backend, **kw)
    after = np.random.get_state(legacy=True)
    for result in [middle, after]:
        assert isinstance(result, tuple)
        assert result[0] == state[0] and result[2:] == state[2:]
        np.testing.assert_array_equal(result[1], state[1])
    np.testing.assert_allclose(
        [a.LR, a.ha_param, a.pvalue], [b.LR, b.ha_param, b.pvalue], atol=1e-9
    )


@pytest.mark.parametrize("backend", ["fallback", RUST_PARAMETER])
def test_uninformative_symmetric_geometry(backend):
    d = pd.DataFrame(
        {"sx": [0.0, 1.0, 0.5], "sy": [0.0, 0.0, np.sqrt(3) / 2], "y": [1.0, 2.0, 4.0]}
    )
    options: dict[str, Any] = dict(
        coords_euclidean=["sx", "sy"], q=2, nrep=20, seed=3, backend=backend
    )
    for fn in (spur.spurtest_i0, spur.spurtest_i1, spur.spurhalflife):
        with pytest.raises(ValueError, match="uninformative"):
            fn("y", d, **options)


@pytest.mark.parametrize("backend", ["fallback", RUST_PARAMETER])
def test_geographic_half_life_uses_metres(backend):
    d = pd.DataFrame({"lon": np.arange(30.0), "lat": 0.0, "y": np.sin(np.arange(30.0))})
    opts: dict[str, Any] = dict(
        lon="lon", lat="lat", q=4, nrep=99, seed=42, backend=backend
    )
    normalized = spur.spurhalflife("y", d, normdist=True, **opts)
    absolute = spur.spurhalflife("y", d, normdist=False, **opts)
    assert absolute.max_dist == pytest.approx(6_371_000 * np.deg2rad(29))
    assert absolute.ci_lower == pytest.approx(normalized.ci_lower * absolute.max_dist)


@pytest.mark.parametrize("backend", ["fallback", RUST_PARAMETER])
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
    assert result.pvalue == pytest.approx(330 / 501)


def test_power_calibration_returns_best_trial_when_target_is_unreachable(monkeypatch):
    from spur.utils import inference

    monkeypatch.setattr(inference, "get_pow_qf", lambda *args: 0.0)
    assert (
        inference.get_ha_param_i0(np.eye(2), np.eye(2), np.eye(2), np.ones((2, 3)))
        == 30.0
    )


@pytest.mark.parametrize("backend", ["fallback", RUST_PARAMETER])
def test_response_shift_does_not_change_inference(backend):
    d = pd.read_csv(BENCH_200)
    options: dict[str, Any] = dict(
        coords_euclidean=["s_1", "s_2"], q=8, nrep=1000, seed=42, backend=backend
    )
    baseline = spur.spurtest_i0resid("y ~ x1 + x2", d, **options)
    shifted = spur.spurtest_i0resid("y ~ x1 + x2", d.assign(y=d.y + 1024.0), **options)
    np.testing.assert_allclose(
        [baseline.LR, baseline.ha_param, baseline.pvalue],
        [shifted.LR, shifted.ha_param, shifted.pvalue],
        atol=1e-9,
    )


@pytest.mark.parametrize("backend", ["fallback", RUST_PARAMETER])
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


@pytest.mark.parametrize("backend", ["fallback", RUST_PARAMETER])
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


@pytest.mark.parametrize("spur_api, spec", [(4, "0.3.8"), (5, "0.3.7")])
def test_incompatible_spur_extension_uses_fallback(monkeypatch, spur_api, spec):
    from types import SimpleNamespace
    import importlib
    import scpc.core as scpc_core
    import spur.core as spur_core

    stale = SimpleNamespace(
        API_VERSION=7,
        SPUR_API_VERSION=spur_api,
        build_info=lambda: {"spec_version": spec},
    )
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
    if getattr(scpc_core, "RUST_API_VERSION", None) == 7:
        assert scpc_core.load_rust_backend() is (stale if spur_api == 4 else None)

    rng = np.random.default_rng(788)
    d = pd.DataFrame(rng.normal(size=(12, 3)), columns=["y", "sx", "sy"])
    d["x"] = rng.normal(size=12)
    opts: dict[str, Any] = dict(
        coords_euclidean=["sx", "sy"], nrep=500, seed=3, nthreads=1
    )
    result = spur.spurtest_i0resid("y ~ x", d, q=10, backend="auto", **opts)
    assert result.backend == "fallback"
    assert result.fallback_reason == "missing_or_incompatible_native_extension"
    with pytest.raises(RuntimeError, match="SPUR API 5"):
        spur.spurtest_i0resid("y ~ x", d, q=10, backend="rust", **opts)
    with pytest.raises(ValueError, match="residual dimension"):
        spur.spurtest_i0resid("y ~ x", d, q=11, backend="auto", **opts)
    triangle = pd.DataFrame(
        {"sx": [0.0, 1.0, 0.5], "sy": [0.0, 0.0, np.sqrt(3) / 2], "y": [1.0, 2.0, 4.0]}
    )
    with pytest.raises(ValueError, match="uninformative"):
        spur.spurtest_i0("y", triangle, q=2, backend="auto", **opts)


@pytest.mark.parametrize("scale", [1e200, -1e200])
def test_large_response_scale_native_fallback_parity(scale):
    d = pd.read_csv(BENCH_200)
    options: dict[str, Any] = dict(
        coords_euclidean=["s_1", "s_2"], q=8, nrep=1000, seed=42
    )
    baseline = spur.spurtest_i0resid("y ~ x1 + x2", d, backend="fallback", **options)
    data_scaled = d.assign(y=d.y * scale)
    for backend in ("fallback", "rust") if RUST_AVAILABLE else ("fallback",):
        actual = spur.spurtest_i0resid(
            "y ~ x1 + x2", data_scaled, backend=backend, **options
        )
        np.testing.assert_allclose(
            [actual.LR, actual.ha_param, actual.pvalue],
            [baseline.LR, baseline.ha_param, baseline.pvalue],
            atol=1e-9,
        )


@pytest.mark.parametrize("backend", ["fallback", RUST_PARAMETER])
def test_tiny_coordinate_scale_invariance(backend):
    d = pd.read_csv(BENCH_200)
    d_tiny = d.assign(s_1=d.s_1 * 1e-12, s_2=d.s_2 * 1e-12)
    test_opts: dict[str, Any] = dict(
        coords_euclidean=["s_1", "s_2"], q=8, nrep=500, seed=42, backend=backend
    )
    res_orig = spur.spurtest_i0resid("y ~ x1 + x2", d, **test_opts)
    res_tiny = spur.spurtest_i0resid("y ~ x1 + x2", d_tiny, **test_opts)
    assert res_orig.backend == backend
    assert res_tiny.backend == backend
    np.testing.assert_allclose(
        [res_tiny.LR, res_tiny.ha_param, res_tiny.pvalue],
        [res_orig.LR, res_orig.ha_param, res_orig.pvalue],
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(res_tiny.cv, res_orig.cv, rtol=1e-10, atol=1e-12)

    hl_opts: dict[str, Any] = dict(
        coords_euclidean=["s_1", "s_2"], q=6, nrep=200, seed=42, backend=backend
    )
    for normdist in (True, False):
        hl_orig = spur.spurhalflife("y", d, normdist=normdist, **hl_opts)
        hl_tiny = spur.spurhalflife("y", d_tiny, normdist=normdist, **hl_opts)
        assert hl_orig.backend == backend
        assert hl_tiny.backend == backend
        assert hl_tiny.max_dist < 1e-10
        np.testing.assert_allclose(
            hl_tiny.max_dist, hl_orig.max_dist * 1e-12, rtol=1e-10, atol=0.0
        )
        scale = 1.0 if normdist else 1e-12
        if np.isnan(hl_orig.ci_lower):
            assert np.isnan(hl_tiny.ci_lower)
        else:
            np.testing.assert_allclose(
                hl_tiny.ci_lower, hl_orig.ci_lower * scale, rtol=1e-10, atol=0.0
            )
        if np.isinf(hl_orig.ci_upper):
            assert np.isinf(hl_tiny.ci_upper)
        else:
            np.testing.assert_allclose(
                hl_tiny.ci_upper, hl_orig.ci_upper * scale, rtol=1e-10, atol=0.0
            )

    tr_orig = spur.spurtransform(
        "y ~ x1",
        d,
        coords_euclidean=["s_1", "s_2"],
        transformation="lbmgls",
        backend=backend,
    )
    tr_tiny = spur.spurtransform(
        "y ~ x1",
        d_tiny,
        coords_euclidean=["s_1", "s_2"],
        transformation="lbmgls",
        backend=backend,
    )
    np.testing.assert_allclose(
        tr_tiny["h_y"].to_numpy(), tr_orig["h_y"].to_numpy(), rtol=1e-9, atol=1e-9
    )
