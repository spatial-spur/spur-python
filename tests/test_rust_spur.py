from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import spur
import spur.core as spur_core
from spur.utils.dist import get_cbar, normalized_distmat
from spur.utils.matrix import (
    demean_matrix,
    get_r,
    get_sigma_dm,
    get_sigma_lbm,
    get_sigma_residual,
)

BENCH_200 = (
    Path(__file__).resolve().parents[2] / "verification" / "data" / "bench_200.csv"
)


def _load_bench_df() -> pd.DataFrame:
    return pd.read_csv(BENCH_200)


def _sign_alignment(r_ref: np.ndarray) -> np.ndarray:
    """Return +1/-1 column signs matching spatial-core's canonical eigenvector sign rule."""
    signs = np.ones(r_ref.shape[1], dtype=float)
    for j in range(r_ref.shape[1]):
        idx = int(np.argmax(np.abs(r_ref[:, j])))
        if r_ref[idx, j] < 0.0:
            signs[j] = -1.0
    return signs


def _aligned_emat_pair(
    emat: np.ndarray,
    coords: np.ndarray,
    *,
    latlong: bool,
    q: int,
    kind: str,
    x_in: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Adjust fallback vs Rust emat row signs to neutralize eigenvector sign ambiguity."""
    distmat = normalized_distmat(coords, latlon=latlong)
    if kind == "lbm":
        sig = demean_matrix(get_sigma_lbm(distmat))
    elif kind == "resid":
        assert x_in is not None
        m_mat = np.eye(len(distmat)) - x_in @ np.linalg.solve(x_in.T @ x_in, x_in.T)
        sig = get_sigma_residual(distmat, get_cbar(0.999, distmat), m_mat)
    elif kind == "hl":
        sig = get_sigma_dm(distmat, get_cbar(0.999, distmat))
    else:
        raise ValueError(kind)
    r_fb = get_r(sig, q)
    signs = _sign_alignment(r_fb)
    return emat, emat * signs[:, None]


def test_spur_diagnostics_frozen_emat_parity_geodesic_and_euclidean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _load_bench_df()
    q = 8
    nrep = 4000
    rng = np.random.default_rng(11)
    base_emat = rng.standard_normal((q, nrep))

    x_in = np.column_stack([np.ones(len(df)), df["x1"].to_numpy(), df["x2"].to_numpy()])
    configs: list[tuple[bool, str | None, str | None, list[str] | None, np.ndarray]] = [
        (True, "s_1", "s_2", None, df[["s_2", "s_1"]].to_numpy()),
        (False, None, None, ["s_1", "s_2"], df[["s_1", "s_2"]].to_numpy()),
    ]
    for latlong, lon_c, lat_c, euc_c, coords_mat in configs:
        e_lbm_fb, e_lbm_ru = _aligned_emat_pair(
            base_emat, coords_mat, latlong=latlong, q=q, kind="lbm"
        )
        e_res_fb, e_res_ru = _aligned_emat_pair(
            base_emat, coords_mat, latlong=latlong, q=q, kind="resid", x_in=x_in
        )

        # I(0) and I(1)
        for test_name, fn in [("i0", spur.spurtest_i0), ("i1", spur.spurtest_i1)]:
            monkeypatch.setattr(
                spur_core, "_draw_emat", lambda _s, _q, _n, e=e_lbm_fb: e
            )
            fb = fn(
                "y",
                df,
                lon=lon_c,
                lat=lat_c,
                coords_euclidean=euc_c,
                q=q,
                nrep=nrep,
                seed=7,
                backend="fallback",
            )
            monkeypatch.setattr(
                spur_core, "_draw_emat", lambda _s, _q, _n, e=e_lbm_ru: e
            )
            ru = fn(
                "y",
                df,
                lon=lon_c,
                lat=lat_c,
                coords_euclidean=euc_c,
                q=q,
                nrep=nrep,
                seed=7,
                backend="rust",
                nthreads=2,
            )

            assert fb.backend == "fallback"
            assert ru.backend == "rust"
            assert not ru.tied
            npt.assert_allclose(ru.LR, fb.LR, rtol=1e-10, atol=1e-10)
            npt.assert_allclose(ru.ha_param, fb.ha_param, rtol=1e-10, atol=1e-10)
            npt.assert_allclose(ru.cv, fb.cv, rtol=1e-9, atol=1e-9)
            npt.assert_allclose(ru.pvalue, fb.pvalue, rtol=0.0, atol=1e-12)
            if test_name == "i0":
                assert ru.rho_grid is not None and fb.rho_grid is not None
                assert ru.pvalue_mat is not None and fb.pvalue_mat is not None
                assert ru.cvalue_mat is not None and fb.cvalue_mat is not None
                assert len(ru.rho_grid) == 30
                npt.assert_allclose(ru.rho_grid, fb.rho_grid, rtol=1e-12, atol=1e-12)
                npt.assert_allclose(ru.pvalue_mat, fb.pvalue_mat, rtol=0.0, atol=1e-12)
                npt.assert_allclose(ru.cvalue_mat, fb.cvalue_mat, rtol=1e-9, atol=1e-9)
                assert ru.pvalue == pytest.approx(
                    float(np.max(ru.pvalue_mat)), abs=1e-15
                )
                npt.assert_allclose(
                    ru.cv, np.max(ru.cvalue_mat, axis=0), rtol=0.0, atol=1e-15
                )

        # I(0) resid and I(1) resid
        for test_name, rfn in [
            ("i0resid", spur.spurtest_i0resid),
            ("i1resid", spur.spurtest_i1resid),
        ]:
            monkeypatch.setattr(
                spur_core, "_draw_emat", lambda _s, _q, _n, e=e_res_fb: e
            )
            fb = rfn(
                "y ~ x1 + x2",
                df,
                lon=lon_c,
                lat=lat_c,
                coords_euclidean=euc_c,
                q=q,
                nrep=nrep,
                seed=7,
                backend="fallback",
            )
            monkeypatch.setattr(
                spur_core, "_draw_emat", lambda _s, _q, _n, e=e_res_ru: e
            )
            ru = rfn(
                "y ~ x1 + x2",
                df,
                lon=lon_c,
                lat=lat_c,
                coords_euclidean=euc_c,
                q=q,
                nrep=nrep,
                seed=7,
                backend="rust",
                nthreads=2,
            )

            assert ru.backend == "rust"
            assert not ru.tied
            npt.assert_allclose(ru.LR, fb.LR, rtol=1e-9, atol=1e-9)
            npt.assert_allclose(ru.ha_param, fb.ha_param, rtol=1e-9, atol=1e-9)
            npt.assert_allclose(ru.cv, fb.cv, rtol=1e-9, atol=1e-9)
            npt.assert_allclose(ru.pvalue, fb.pvalue, rtol=0.0, atol=1e-12)
            if test_name == "i0resid":
                assert ru.pvalue_mat is not None and fb.pvalue_mat is not None
                assert ru.pvalue == pytest.approx(
                    float(np.max(ru.pvalue_mat)), abs=1e-15
                )


def test_spur_halflife_frozen_emat_parity_and_invariants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = _load_bench_df().iloc[:120].reset_index(drop=True)
    q = 6
    nrep = 200
    base_emat = np.random.default_rng(77).standard_normal((q, nrep))

    hl_configs: list[
        tuple[bool, str | None, str | None, list[str] | None, np.ndarray]
    ] = [
        (True, "s_1", "s_2", None, df[["s_2", "s_1"]].to_numpy()),
        (False, None, None, ["s_1", "s_2"], df[["s_1", "s_2"]].to_numpy()),
    ]
    for latlong, lon_c, lat_c, euc_c, coords_mat in hl_configs:
        e_fb, e_ru = _aligned_emat_pair(
            base_emat, coords_mat, latlong=latlong, q=q, kind="hl"
        )
        for normdist in (False, True):
            monkeypatch.setattr(spur_core, "_draw_emat", lambda _s, _q, _n, e=e_fb: e)
            fb = spur.spurhalflife(
                "y",
                df,
                lon=lon_c,
                lat=lat_c,
                coords_euclidean=euc_c,
                q=q,
                nrep=nrep,
                level=95,
                normdist=normdist,
                backend="fallback",
            )
            monkeypatch.setattr(spur_core, "_draw_emat", lambda _s, _q, _n, e=e_ru: e)
            ru = spur.spurhalflife(
                "y",
                df,
                lon=lon_c,
                lat=lat_c,
                coords_euclidean=euc_c,
                q=q,
                nrep=nrep,
                level=95,
                normdist=normdist,
                backend="rust",
                nthreads=2,
            )

            assert ru.backend == "rust"
            assert ru.pv_mat is not None and len(ru.pv_mat) == 131
            assert ru.hl_grid is not None and len(ru.hl_grid) == 131
            npt.assert_allclose(ru.max_dist, fb.max_dist, rtol=1e-12, atol=1e-12)
            npt.assert_allclose(ru.ci_lower, fb.ci_lower, rtol=1e-12, atol=1e-12)
            if np.isinf(fb.ci_upper):
                assert np.isinf(ru.ci_upper)
            else:
                npt.assert_allclose(ru.ci_upper, fb.ci_upper, rtol=1e-12, atol=1e-12)


def test_spur_transforms_rust_fallback_parity_and_colocation() -> None:
    df = _load_bench_df().iloc[:80].reset_index(drop=True)
    # Introduce co-located pair + an isolated cluster label for cluster transform:
    df.loc[1, ["s_1", "s_2"]] = df.loc[0, ["s_1", "s_2"]]

    tr_configs: list[tuple[str | None, str | None, list[str] | None, float]] = [
        ("s_1", "s_2", None, 250_000.0),
        (None, None, ["s_1", "s_2"], 1.5),
    ]
    for lon_c, lat_c, euc_c, radius in tr_configs:
        for tr_kind in ("lbmgls", "nn", "iso"):
            rad_arg = radius if tr_kind == "iso" else None
            fb = spur.spurtransform(
                "y ~ x1",
                df,
                lon=lon_c,
                lat=lat_c,
                coords_euclidean=euc_c,
                transformation=tr_kind,
                radius=rad_arg,
                backend="fallback",
            )
            ru = spur.spurtransform(
                "y ~ x1",
                df,
                lon=lon_c,
                lat=lat_c,
                coords_euclidean=euc_c,
                transformation=tr_kind,
                radius=rad_arg,
                backend="rust",
                nthreads=2,
            )
            npt.assert_allclose(
                ru["h_y"].to_numpy(), fb["h_y"].to_numpy(), rtol=1e-9, atol=1e-9
            )
            npt.assert_allclose(
                ru["h_x1"].to_numpy(), fb["h_x1"].to_numpy(), rtol=1e-9, atol=1e-9
            )
            # Co-located rows in LBM-GLS must have zero variance in constant vector / equal rows
            if tr_kind == "lbmgls":
                const_df = df.assign(ones=1.0)
                const_ru = spur.spurtransform(
                    "ones ~ 1",
                    const_df,
                    lon=lon_c,
                    lat=lat_c,
                    coords_euclidean=euc_c,
                    transformation="lbmgls",
                    backend="rust",
                )
                npt.assert_allclose(const_ru["h_ones"].to_numpy(), 0.0, atol=1e-8)
                npt.assert_allclose(
                    ru["h_y"].iloc[0], ru["h_y"].iloc[1], rtol=1e-9, atol=1e-9
                )

    # Cluster transform with missing Y propagation
    df_cl = df.iloc[:12].copy()
    df_cl["grp"] = ["A", "A", "A", "B", "B", "B", "C", "C", "C", "D", "D", "D"]
    df_cl.loc[4, "y"] = np.nan
    fb_cl = spur.spurtransform(
        "y ~ x1", df_cl, transformation="cluster", clustvar="grp", backend="fallback"
    )
    ru_cl = spur.spurtransform(
        "y ~ x1", df_cl, transformation="cluster", clustvar="grp", backend="rust"
    )
    npt.assert_allclose(
        ru_cl["h_x1"].to_numpy(), fb_cl["h_x1"].to_numpy(), rtol=1e-13, atol=1e-13
    )
    npt.assert_allclose(
        ru_cl["h_y"].to_numpy(),
        fb_cl["h_y"].to_numpy(),
        rtol=1e-13,
        atol=1e-13,
        equal_nan=True,
    )


def test_spur_pipeline_parity_and_thread_invariance() -> None:
    df = _load_bench_df().iloc[:120].reset_index(drop=True)
    res_t1 = spur_core.spur(
        "y ~ x1 + x2",
        df,
        lon="s_1",
        lat="s_2",
        q=8,
        nrep=400,
        seed=42,
        backend="rust",
        nthreads=1,
    )
    res_t4 = spur_core.spur(
        "y ~ x1 + x2",
        df,
        lon="s_1",
        lat="s_2",
        q=8,
        nrep=400,
        seed=42,
        backend="rust",
        nthreads=4,
    )
    assert res_t1.backend == "rust"
    assert res_t1.fits.levels.scpc.backend == "rust"
    assert res_t1.fits.transformed.scpc.backend == "rust"

    for name in ("i0", "i1", "i0resid", "i1resid"):
        a = getattr(res_t1.tests, name)
        b = getattr(res_t4.tests, name)
        npt.assert_allclose(a.LR, b.LR, rtol=1e-13, atol=1e-13)
        npt.assert_allclose(a.ha_param, b.ha_param, rtol=1e-13, atol=1e-13)
        npt.assert_allclose(a.pvalue, b.pvalue, rtol=0.0, atol=1e-15)
        npt.assert_allclose(a.cv, b.cv, rtol=1e-13, atol=1e-13)

    npt.assert_allclose(
        res_t1.fits.transformed.scpc.scpcstats,
        res_t4.fits.transformed.scpc.scpcstats,
        rtol=1e-13,
        atol=1e-13,
    )


def test_tied_eigenspace_triggers_auto_fallback_and_keeps_rust_explicit() -> None:
    # A regular 16-gon has degenerate 2D cos/sin eigenspaces across odd q=1, 3, ... (and even q=2 for LBM).
    k = np.arange(16)
    df = pd.DataFrame(
        {
            "x_c": np.cos(2.0 * np.pi * k / 16.0),
            "y_c": np.sin(2.0 * np.pi * k / 16.0),
            "y": np.sin(k * 1.3) + np.cos(k * 0.7),
        }
    )
    ru = spur.spurtest_i0(
        "y",
        df,
        coords_euclidean=["x_c", "y_c"],
        q=2,
        nrep=500,
        seed=42,
        backend="rust",
    )
    auto_res = spur.spurtest_i0(
        "y",
        df,
        coords_euclidean=["x_c", "y_c"],
        q=2,
        nrep=500,
        seed=42,
        backend="auto",
    )
    assert ru.backend == "rust"
    assert ru.tied is True
    assert auto_res.backend == "fallback"
    assert auto_res.tied is True

    hl_ru = spur.spurhalflife(
        "y",
        df,
        coords_euclidean=["x_c", "y_c"],
        q=3,
        nrep=300,
        seed=42,
        backend="rust",
    )
    hl_auto = spur.spurhalflife(
        "y",
        df,
        coords_euclidean=["x_c", "y_c"],
        q=3,
        nrep=300,
        seed=42,
        backend="auto",
    )
    assert hl_ru.backend == "rust"
    assert hl_ru.tied is True
    assert hl_auto.backend == "fallback"
    assert hl_auto.tied is True


def test_backend_validation_and_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _load_bench_df()
    monkeypatch.setenv("SPATIAL_BACKEND", "fallback")
    res_fb = spur.spurtest_i1("y", df, lon="s_1", lat="s_2", q=8, nrep=400, seed=42)
    assert res_fb.backend == "fallback"

    monkeypatch.setenv("SPATIAL_BACKEND", "rust")
    res_ru = spur.spurtest(
        "y", df, test="i1", lon="s_1", lat="s_2", q=8, nrep=400, seed=42
    )
    assert res_ru.backend == "rust"

    with pytest.raises(ValueError, match="backend"):
        spur.spurtest_i1(
            "y", df, lon="s_1", lat="s_2", q=8, nrep=400, backend="invalid"
        )

    with pytest.raises(ValueError, match="nthreads"):
        spur.spurtest_i1("y", df, lon="s_1", lat="s_2", q=8, nrep=400, nthreads=0)

    monkeypatch.setenv("SPATIAL_BACKEND", "bogus")
    with pytest.raises(ValueError, match="SPATIAL_BACKEND"):
        spur.spurtest_i1("y", df, lon="s_1", lat="s_2", q=8, nrep=400)

    monkeypatch.setenv("SPATIAL_BACKEND", "auto")
    monkeypatch.setattr(spur_core, "load_rust_backend", lambda: None)
    auto_res = spur.spurtest_i1("y", df, lon="s_1", lat="s_2", q=8, nrep=400)
    assert auto_res.backend == "fallback"
    with pytest.raises(RuntimeError, match="spatial_spur_rust"):
        spur.spurtest_i1("y", df, lon="s_1", lat="s_2", q=8, nrep=400, backend="rust")
    _: Any = res_fb
