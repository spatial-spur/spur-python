from __future__ import annotations

from collections.abc import Callable, Sequence
import importlib
from inspect import signature
import math
import os
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from scpc import scpc
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import MissingDataError

from .types import (
    Fits,
    HalfLifeResult,
    PipelineResult,
    RegressionResult,
    TestResult,
    Tests,
)
from .utils.dist import (
    get_cbar,
    get_distance_matrix,
    normalized_distmat,
    resolve_spur_coords,
)
from .utils.formula import (
    evaluate_formula,
    parse_residual_formula,
    parse_single_var_formula,
)
from .utils.inference import (
    get_ha_param_i0,
    get_ha_param_i1,
    get_ha_param_i1_residual,
    spur_persistence,
)
from .utils.matrix import (
    cholesky_upper,
    cluster_matrix,
    demean_matrix,
    get_r,
    get_sigma_dm,
    get_sigma_lbm,
    get_sigma_residual,
    iso_matrix,
    lbmgls_matrix,
    nn_matrix,
    orthonormal_design_basis,
    validate_and_project_y,
    validate_y_variation,
)

VALID_BACKENDS = {"auto", "rust", "fallback"}
RUST_API_VERSION = 6
SPUR_RUST_API_VERSION = 3
EARTH_RADIUS_M = 6_371_000.0
EARTH_CIRCUMFERENCE_M = 2.0 * math.pi * EARTH_RADIUS_M


def _different_covariance_shapes(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.max(np.abs(a / np.trace(a) - b / np.trace(b))) > 1e-12)


def _validate_i1_shapes(
    distmat: np.ndarray, r: np.ndarray, m: np.ndarray | None = None
) -> None:
    def omega(rho: float) -> np.ndarray:
        c = get_cbar(rho, distmat)
        sigma = (
            get_sigma_dm(distmat, c) if m is None else get_sigma_residual(distmat, c, m)
        )
        return r.T @ sigma @ r

    lo, mid, hi = (omega(rho) for rho in (0.01, 0.5, 0.95))
    if not _different_covariance_shapes(lo, mid) and not _different_covariance_shapes(
        lo, hi
    ):
        raise ValueError(
            "SPUR experiment is uninformative: covariance shapes are proportional."
        )


def resolve_backend(backend: object) -> str:
    """Resolve the execution backend from explicit argument or `SPATIAL_BACKEND`."""
    if backend is None:
        env_backend = os.environ.get("SPATIAL_BACKEND", "auto") or "auto"
        if env_backend not in VALID_BACKENDS:
            raise ValueError(
                '`SPATIAL_BACKEND` must be one of "auto", "rust", or "fallback".'
            )
        return env_backend
    if not isinstance(backend, str) or backend not in VALID_BACKENDS:
        raise ValueError('`backend` must be one of "auto", "rust", or "fallback".')
    return backend


def validate_nthreads(nthreads: object) -> int | None:
    """Validate optional thread budget (`None` or integer >= 1)."""
    if nthreads is None:
        return None
    if (
        isinstance(nthreads, bool)
        or not isinstance(nthreads, (int, np.integer))
        or int(nthreads) < 1
    ):
        raise ValueError("`nthreads` must be a positive integer or None.")
    return int(nthreads)


def load_rust_backend() -> Any | None:
    """Import the compiled Rust extension module if present and SPUR-compatible."""
    try:
        rust_mod = importlib.import_module("spatial_spur_rust")
    except ImportError:
        return None
    if getattr(rust_mod, "API_VERSION", None) != RUST_API_VERSION:
        return None
    if getattr(rust_mod, "SPUR_API_VERSION", None) != SPUR_RUST_API_VERSION:
        return None
    return rust_mod


def _require_rust_if_requested(resolved_backend: str) -> Any | None:
    """Return the Rust module when enabled, raising if `backend='rust'` cannot load it."""
    if resolved_backend == "fallback":
        return None
    rust_mod = load_rust_backend()
    if rust_mod is None and resolved_backend == "rust":
        raise RuntimeError(
            "Rust backend requested, but `spatial_spur_rust` with compatible SPUR support is unavailable. Install spatial-spur-rust >= 0.2.3."
        )
    return rust_mod


def _draw_emat(seed: int, q: int, nrep: int) -> np.ndarray:
    """Generate the Monte Carlo normal shock matrix `q x nrep`."""
    return np.random.default_rng(seed).standard_normal((q, nrep))


def _rust_coords(coords: np.ndarray, latlong: bool) -> np.ndarray:
    """Convert `[lat, lon]` Python SPUR coordinates into Rust `[lon, lat]` row-major float64."""
    arr = np.asarray(coords, dtype=np.float64)
    if latlong:
        arr = arr[:, [1, 0]]
    return np.ascontiguousarray(arr, dtype=np.float64)


def _test_result_from_rust(
    test_type: str,
    rust_res: dict[str, Any],
    *,
    is_i0: bool,
) -> TestResult:
    """Construct a `TestResult` from a `spatial_spur_rust` dictionary."""
    lr_arr = np.asarray(rust_res["lr"], dtype=float).ravel()
    pval_arr = np.asarray(rust_res["pvalue"], dtype=float).ravel()
    cv_arr = np.asarray(rust_res["cvalue"], dtype=float).ravel()
    if is_i0:
        pval_mat = np.asarray(rust_res["pvalue_mat"], dtype=float).ravel()
        cval_mat = np.asarray(rust_res["cvalue_mat"], dtype=float)
        rho_grid = np.asarray(rust_res["rho_grid"], dtype=float).ravel()
    else:
        pval_mat = None
        cval_mat = None
        rho_grid = None
    return TestResult(
        test_type=test_type,
        LR=float(lr_arr[0]),
        pvalue=float(pval_arr[0]),
        cv=cv_arr,
        ha_param=float(rust_res["ha_param"]),
        pvalue_mat=pval_mat,
        cvalue_mat=cval_mat,
        rho_grid=rho_grid,
        backend="rust",
        tied=bool(rust_res["tied"]),
    )


def _try_rust_spur_test(
    test_type: str,
    *,
    coords: np.ndarray,
    latlong: bool,
    y: np.ndarray,
    emat: np.ndarray,
    q: int,
    nrep: int,
    resolved_backend: str,
    resolved_nthreads: int | None,
    x_in: np.ndarray | None = None,
) -> tuple[TestResult | None, bool]:
    """Execute one SPUR diagnostic test in Rust unless auto-falling back on tied eigenspaces."""
    rust_mod = _require_rust_if_requested(resolved_backend)
    if rust_mod is None:
        return None, False

    coords_rust = _rust_coords(coords, latlong)
    y_rust = np.ascontiguousarray(y, dtype=np.float64).ravel()
    emat_rust = np.ascontiguousarray(emat, dtype=np.float64)
    is_i0 = test_type in ("i0", "i0resid")

    if x_in is None:
        fn = rust_mod.run_spur_i0 if is_i0 else rust_mod.run_spur_i1
        rust_res = fn(
            coords_rust,
            y_rust,
            emat_rust,
            int(q),
            int(nrep),
            latlong=bool(latlong),
            nthreads=resolved_nthreads,
        )
    else:
        fn = rust_mod.run_spur_i0_resid if is_i0 else rust_mod.run_spur_i1_resid
        x_rust = np.ascontiguousarray(x_in, dtype=np.float64)
        rust_res = fn(
            coords_rust,
            y_rust,
            x_rust,
            emat_rust,
            int(q),
            int(nrep),
            latlong=bool(latlong),
            nthreads=resolved_nthreads,
        )

    tied = bool(rust_res["tied"])
    if tied and resolved_backend == "auto":
        return None, True
    return _test_result_from_rust(test_type, rust_res, is_i0=is_i0), tied


def _fallback_reason(backend: str, tied: bool = False) -> str | None:
    if tied:
        return "tied_eigenspace"
    return "missing_or_incompatible_native_extension" if backend == "auto" else None


def _validate_test_dimensions(
    q: int, nrep: int, n: int, design_basis: np.ndarray | None = None
) -> None:
    if q >= n:
        raise ValueError(f"q={q} must be less than n={n}. Use q <= {n - 1}.")
    if q < 2:
        raise ValueError(f"q={q} must be >= 2 for informative SPUR inference.")
    if nrep < 1:
        raise ValueError(f"nrep={nrep} must be >= 1.")
    if design_basis is not None and q > n - design_basis.shape[1]:
        raise ValueError(
            f"q={q} exceeds the residual dimension {n - design_basis.shape[1]}."
        )


def _run_spur_test(
    test_type: str,
    coords: np.ndarray,
    latlong: bool,
    y: np.ndarray,
    q: int,
    nrep: int,
    seed: int,
    backend: str,
    nthreads: int | None,
    *,
    x_in: np.ndarray | None = None,
) -> TestResult:
    design_basis = None if x_in is None else orthonormal_design_basis(x_in)
    _validate_test_dimensions(q, nrep, len(y), design_basis)
    emat = _draw_emat(seed, q, nrep)
    rust_res, tied = _try_rust_spur_test(
        test_type,
        coords=coords,
        latlong=latlong,
        y=y,
        emat=emat,
        q=q,
        nrep=nrep,
        resolved_backend=backend,
        resolved_nthreads=nthreads,
        x_in=x_in,
    )
    if rust_res is not None:
        return rust_res
    return _fallback_spur_test(
        test_type,
        y,
        normalized_distmat(coords, latlon=latlong),
        emat,
        design_basis,
        tied=tied,
        fallback_reason=_fallback_reason(backend, tied),
    )


def _fallback_spur_test(
    test_type: str,
    y: np.ndarray,
    distmat: np.ndarray,
    emat: np.ndarray,
    design_basis: np.ndarray | None,
    *,
    tied: bool,
    fallback_reason: str | None,
) -> TestResult:
    q = emat.shape[0]
    projection = (
        None if design_basis is None else np.eye(len(y)) - design_basis @ design_basis.T
    )

    def sigma(c: float) -> np.ndarray:
        if projection is None:
            return get_sigma_dm(distmat, c)
        return get_sigma_residual(distmat, c, projection)

    sigdm_bm = (
        demean_matrix(get_sigma_lbm(distmat))
        if projection is None
        else sigma(get_cbar(0.999, distmat))
    )
    r = get_r(sigdm_bm, q)
    P = validate_and_project_y(y, r)
    if test_type.startswith("i0"):
        return _fallback_spur_i0(
            test_type, distmat, emat, r, P, sigdm_bm, sigma, tied, fallback_reason
        )
    return _fallback_spur_i1(
        test_type,
        distmat,
        emat,
        r,
        P,
        sigdm_bm,
        sigma,
        projection,
        tied,
        fallback_reason,
    )


def _fallback_spur_i0(
    test_type: str,
    distmat: np.ndarray,
    emat: np.ndarray,
    r: np.ndarray,
    P: np.ndarray,
    sigdm_bm: np.ndarray,
    sigma: Callable[[float], np.ndarray],
    tied: bool,
    fallback_reason: str | None,
) -> TestResult:
    q = emat.shape[0]
    rho = 0.001
    c = get_cbar(rho, distmat)
    sigdm_rho = sigma(c)

    om_rho = r.T @ sigdm_rho @ r
    om_bm = r.T @ sigdm_bm @ r
    if not _different_covariance_shapes(om_rho, om_bm):
        raise ValueError(
            "SPUR experiment is uninformative: covariance shapes are proportional."
        )

    om_i0 = om_rho
    om_ho = om_rho
    ha_parm = get_ha_param_i0(om_ho, om_i0, om_bm, emat)
    om_ha = om_i0 + ha_parm * om_bm

    ch_omi_ho = cholesky_upper(np.linalg.inv(om_ho))
    ch_omi_ha = cholesky_upper(np.linalg.inv(om_ha))

    y_P_ho = ch_omi_ho @ P
    y_P_ha = ch_omi_ha @ P
    q_P_ho = np.sum(y_P_ho**2)
    q_P_ha = np.sum(y_P_ha**2)
    LR = float(q_P_ho / q_P_ha)

    rho_min = 0.0001
    rho_max = 0.03
    n_rho = 30
    rho_grid = np.linspace(rho_min, rho_max, n_rho)

    ch_om_ho_list = []
    for i in range(n_rho):
        rho_i = rho_grid[i]
        if rho_i > 0:
            c_i = get_cbar(rho_i, distmat)
            sigdm_ho_i = sigma(c_i)
            om_ho_i = r.T @ sigdm_ho_i @ r
        else:
            om_ho_i = np.eye(q)
        ch_om_ho_list.append(cholesky_upper(om_ho_i))

    pvalue_vec = np.zeros(n_rho)
    cvalue_mat = np.zeros((n_rho, 3))
    sz_vec = np.array([0.01, 0.05, 0.10])

    for ir in range(n_rho):
        ch_om_ho_ir = ch_om_ho_list[ir]
        y_ho = ch_om_ho_ir.T @ emat
        y_ho_ho = ch_omi_ho @ y_ho
        y_ho_ha = ch_omi_ha @ y_ho
        q_ho_ho = np.sum(y_ho_ho**2, axis=0)
        q_ho_ha = np.sum(y_ho_ha**2, axis=0)
        lr_ho = q_ho_ho / q_ho_ha
        cvalue_mat[ir, :] = np.quantile(lr_ho, 1 - sz_vec)
        pvalue_vec[ir] = np.mean(lr_ho > LR)

    cvalue = cvalue_mat.max(axis=0)
    pvalue = float(pvalue_vec.max())

    return TestResult(
        test_type=test_type,
        LR=LR,
        pvalue=pvalue,
        cv=cvalue,
        ha_param=ha_parm,
        pvalue_mat=pvalue_vec,
        cvalue_mat=cvalue_mat,
        rho_grid=rho_grid,
        backend="fallback",
        tied=tied,
        fallback_reason=fallback_reason,
    )


def _fallback_spur_i1(
    test_type: str,
    distmat: np.ndarray,
    emat: np.ndarray,
    r: np.ndarray,
    P: np.ndarray,
    sigdm_bm: np.ndarray,
    sigma: Callable[[float], np.ndarray],
    projection: np.ndarray | None,
    tied: bool,
    fallback_reason: str | None,
) -> TestResult:
    om_ho = r.T @ sigdm_bm @ r

    _validate_i1_shapes(distmat, r, projection)
    ha_parm = (
        get_ha_param_i1(om_ho, distmat, r, emat)
        if projection is None
        else get_ha_param_i1_residual(om_ho, distmat, r, emat, projection)
    )
    sigdm_ha = sigma(ha_parm)
    om_ha = r.T @ sigdm_ha @ r

    ch_om_ho = cholesky_upper(om_ho)
    omi_ho = np.linalg.inv(om_ho)
    omi_ha = np.linalg.inv(om_ha)
    ch_omi_ho = cholesky_upper(omi_ho)
    ch_omi_ha = cholesky_upper(omi_ha)

    y_ho = ch_om_ho.T @ emat
    y_ho_ho = ch_omi_ho @ y_ho
    y_ho_ha = ch_omi_ha @ y_ho
    q_ho_ho = np.sum(y_ho_ho**2, axis=0)
    q_ho_ha = np.sum(y_ho_ha**2, axis=0)
    lr_ho = q_ho_ho / q_ho_ha

    sz_vec = np.array([0.01, 0.05, 0.10])
    cv_vec = np.quantile(lr_ho, 1 - sz_vec)

    LR = float((P.T @ omi_ho @ P) / (P.T @ omi_ha @ P))
    pvalue = float(np.mean(lr_ho > LR))

    return TestResult(
        test_type=test_type,
        LR=LR,
        pvalue=pvalue,
        cv=cv_vec,
        ha_param=ha_parm,
        backend="fallback",
        tied=tied,
        fallback_reason=fallback_reason,
    )


def spurtest_i1(
    var: str,
    data: pd.DataFrame,
    *,
    lon: str | None = None,
    lat: str | None = None,
    coords_euclidean: Sequence[str] | None = None,
    q: int = 15,
    nrep: int = 100000,
    seed: int = 42,
    backend: str | None = None,
    nthreads: int | None = None,
) -> TestResult:
    """Conduct the spatial I(1) unit-root test for a single variable.

    Use this function when you want to test one variable directly, without using
    the formula-based wrapper. Provide either longitude/latitude columns or
    Euclidean coordinate columns.

    Args:
        var: Name of the variable to test.
        data: DataFrame containing the variable and coordinate columns.
        lon: Name of the longitude column. Use together with `lat`.
        lat: Name of the latitude column. Use together with `lon`.
        coords_euclidean: Names of Euclidean coordinate columns. Use instead of
            `lon` and `lat`.
        q: Number of low-frequency weights used in the test statistic.
        nrep: Number of Monte Carlo draws used to simulate the null distribution.
        seed: Random seed used to generate the simulation draws.

    Returns:
        A `TestResult` containing the LR statistic, p-value, critical values,
        and the calibrated local-alternative parameter.

    Raises:
        ValueError: If `data` is not a DataFrame, `var` is missing or contains
            non-finite values, the coordinate specification is invalid, `q` is out
            of range, or `nrep` is less than 1.

    Example:
        >>> from spur import load_chetty_data, standardize, spurtest_i1
        >>> df = load_chetty_data()
        >>> df = df[["am", "lat", "lon"]].dropna()
        >>> df = standardize(df, ["am"])
        >>> result = spurtest_i1("am", df, lon="lon", lat="lat", q=10, nrep=500, seed=42)
        >>> print(result.summary())
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("`data` must be a pandas DataFrame.")
    if not isinstance(var, str) or not var:
        raise ValueError("`var` must be a non-empty column name.")
    if var not in data.columns:
        raise ValueError(f"Variable '{var}' not found in data.")

    Y = data[var].to_numpy(dtype=float)
    if not np.all(np.isfinite(Y)):
        raise ValueError(
            f"Variable '{var}' contains NaN or inf values. All values must be finite."
        )

    coord_info = resolve_spur_coords(
        data=data,
        use_rows=np.ones(len(data), dtype=bool),
        lon=lon,
        lat=lat,
        coords_euclidean=coords_euclidean,
    )

    return _run_spur_test(
        "i1",
        coord_info["coords"],
        coord_info["latlong"],
        Y,
        q,
        nrep,
        seed,
        resolve_backend(backend),
        validate_nthreads(nthreads),
    )


def spurtest_i0(
    var: str,
    data: pd.DataFrame,
    *,
    lon: str | None = None,
    lat: str | None = None,
    coords_euclidean: Sequence[str] | None = None,
    q: int = 15,
    nrep: int = 100000,
    seed: int = 42,
    backend: str | None = None,
    nthreads: int | None = None,
) -> TestResult:
    """Conduct the spatial I(0) test for a single variable.

    Use this function when you want to test one variable directly, without using
    the formula-based wrapper. Provide either longitude/latitude columns or
    Euclidean coordinate columns.

    Args:
        var: Name of the variable to test.
        data: DataFrame containing the variable and coordinate columns.
        lon: Name of the longitude column. Use together with `lat`.
        lat: Name of the latitude column. Use together with `lon`.
        coords_euclidean: Names of Euclidean coordinate columns. Use instead of
            `lon` and `lat`.
        q: Number of low-frequency weights used in the test statistic.
        nrep: Number of Monte Carlo draws used to simulate the null distribution.
        seed: Random seed used to generate the simulation draws.

    Returns:
        A `TestResult` containing the LR statistic, p-value, critical values,
        and the calibrated local-alternative parameter.

    Raises:
        ValueError: If `data` is not a DataFrame, `var` is missing or contains
            non-finite values, the coordinate specification is invalid, `q` is out
            of range, or `nrep` is less than 1.

    Example:
        >>> from spur import load_chetty_data, standardize, spurtest_i0
        >>> df = load_chetty_data()
        >>> df = df[["am", "lat", "lon"]].dropna()
        >>> df = standardize(df, ["am"])
        >>> result = spurtest_i0("am", df, lon="lon", lat="lat", q=10, nrep=500, seed=42)
        >>> print(result.summary())
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("`data` must be a pandas DataFrame.")
    if not isinstance(var, str) or not var:
        raise ValueError("`var` must be a non-empty column name.")
    if var not in data.columns:
        raise ValueError(f"Variable '{var}' not found in data.")

    Y = data[var].to_numpy(dtype=float)
    if not np.all(np.isfinite(Y)):
        raise ValueError(
            f"Variable '{var}' contains NaN or inf values. All values must be finite."
        )

    coord_info = resolve_spur_coords(
        data=data,
        use_rows=np.ones(len(data), dtype=bool),
        lon=lon,
        lat=lat,
        coords_euclidean=coords_euclidean,
    )

    return _run_spur_test(
        "i0",
        coord_info["coords"],
        coord_info["latlong"],
        Y,
        q,
        nrep,
        seed,
        resolve_backend(backend),
        validate_nthreads(nthreads),
    )


def spurtest_i1resid(
    formula: str,
    data: pd.DataFrame,
    *,
    lon: str | None = None,
    lat: str | None = None,
    coords_euclidean: Sequence[str] | None = None,
    q: int = 15,
    nrep: int = 100000,
    seed: int = 42,
    backend: str | None = None,
    nthreads: int | None = None,
) -> TestResult:
    """Conduct the spatial I(1) residual test for a regression formula.

    Use this function when you want to test the residual dependence implied by a
    formula such as `y ~ x1 + x2`. Provide either longitude/latitude columns or
    Euclidean coordinate columns.

    Args:
        formula: Two-sided regression formula identifying the dependent variable
            and covariates.
        data: DataFrame containing the variables and coordinate columns.
        lon: Name of the longitude column. Use together with `lat`.
        lat: Name of the latitude column. Use together with `lon`.
        coords_euclidean: Names of Euclidean coordinate columns. Use instead of
            `lon` and `lat`.
        q: Number of low-frequency weights used in the test statistic.
        nrep: Number of Monte Carlo draws used to simulate the null distribution.
        seed: Random seed used to generate the simulation draws.

    Returns:
        A `TestResult` containing the LR statistic, p-value, critical values,
        and the calibrated local-alternative parameter.

    Raises:
        ValueError: If `data` is not a DataFrame, `formula` is invalid, the
            coordinate specification is invalid, the regressor matrix is
            rank-deficient, `q` is out of range, or `nrep` is less than 1.

    Example:
        >>> from spur import load_chetty_data, standardize, spurtest_i1resid
        >>> df = load_chetty_data()
        >>> df = df[["am", "fracblack", "lat", "lon"]].dropna()
        >>> df = standardize(df, ["am", "fracblack"])
        >>> result = spurtest_i1resid(
        ...     "am ~ fracblack",
        ...     df,
        ...     lon="lon",
        ...     lat="lat",
        ...     q=10,
        ...     nrep=500,
        ...     seed=42,
        ... )
        >>> print(result.summary())
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("`data` must be a pandas DataFrame.")

    parsed = parse_residual_formula(formula, data, "spurtest_i1resid()")
    coord_info = resolve_spur_coords(
        data=data,
        use_rows=parsed["use_rows"],
        lon=lon,
        lat=lat,
        coords_euclidean=coords_euclidean,
    )

    Y = parsed["Y"].reshape(-1)
    return _run_spur_test(
        "i1resid",
        coord_info["coords"],
        coord_info["latlong"],
        Y,
        q,
        nrep,
        seed,
        resolve_backend(backend),
        validate_nthreads(nthreads),
        x_in=parsed["X_in"],
    )


def spurtest_i0resid(
    formula: str,
    data: pd.DataFrame,
    *,
    lon: str | None = None,
    lat: str | None = None,
    coords_euclidean: Sequence[str] | None = None,
    q: int = 15,
    nrep: int = 100000,
    seed: int = 42,
    backend: str | None = None,
    nthreads: int | None = None,
) -> TestResult:
    """Conduct the spatial I(0) residual test for a regression formula.

    Use this function when you want to test the residual dependence implied by a
    formula such as `y ~ x1 + x2`. Provide either longitude/latitude columns or
    Euclidean coordinate columns.

    Args:
        formula: Two-sided regression formula identifying the dependent variable
            and covariates.
        data: DataFrame containing the variables and coordinate columns.
        lon: Name of the longitude column. Use together with `lat`.
        lat: Name of the latitude column. Use together with `lon`.
        coords_euclidean: Names of Euclidean coordinate columns. Use instead of
            `lon` and `lat`.
        q: Number of low-frequency weights used in the test statistic.
        nrep: Number of Monte Carlo draws used to simulate the null distribution.
        seed: Random seed used to generate the simulation draws.

    Returns:
        A `TestResult` containing the LR statistic, p-value, critical values,
        and the calibrated local-alternative parameter.

    Raises:
        ValueError: If `data` is not a DataFrame, `formula` is invalid, the
            coordinate specification is invalid, the regressor matrix is
            rank-deficient, `q` is out of range, or `nrep` is less than 1.

    Example:
        >>> from spur import load_chetty_data, standardize, spurtest_i0resid
        >>> df = load_chetty_data()
        >>> df = df[["am", "fracblack", "lat", "lon"]].dropna()
        >>> df = standardize(df, ["am", "fracblack"])
        >>> result = spurtest_i0resid(
        ...     "am ~ fracblack",
        ...     df,
        ...     lon="lon",
        ...     lat="lat",
        ...     q=10,
        ...     nrep=500,
        ...     seed=42,
        ... )
        >>> print(result.summary())
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("`data` must be a pandas DataFrame.")

    parsed = parse_residual_formula(formula, data, "spurtest_i0resid()")
    coord_info = resolve_spur_coords(
        data=data,
        use_rows=parsed["use_rows"],
        lon=lon,
        lat=lat,
        coords_euclidean=coords_euclidean,
    )

    Y = parsed["Y"].reshape(-1)
    return _run_spur_test(
        "i0resid",
        coord_info["coords"],
        coord_info["latlong"],
        Y,
        q,
        nrep,
        seed,
        resolve_backend(backend),
        validate_nthreads(nthreads),
        x_in=parsed["X_in"],
    )


def spurtest(
    formula: str,
    data: pd.DataFrame,
    *,
    test: Literal["i1", "i0", "i1resid", "i0resid"],
    lon: str | None = None,
    lat: str | None = None,
    coords_euclidean: Sequence[str] | None = None,
    q: int = 15,
    nrep: int = 100000,
    seed: int = 42,
    backend: str | None = None,
    nthreads: int | None = None,
) -> TestResult:
    """Run one of the four public SPUR diagnostic tests.

    Use this wrapper when you want one entrypoint for all test types. For
    single-variable tests, pass either a bare variable name like `"am"` or
    `"am ~ 1"`. For residual tests, pass a two-sided formula such as
    `"am ~ fracblack"`.

    Args:
        formula: Variable string or formula, depending on the selected test.
        data: DataFrame containing the variables and coordinate columns.
        test: Which SPUR test to run. Must be one of `"i0"`, `"i1"`,
            `"i0resid"`, or `"i1resid"`.
        lon: Name of the longitude column. Use together with `lat`.
        lat: Name of the latitude column. Use together with `lon`.
        coords_euclidean: Names of Euclidean coordinate columns. Use instead of
            `lon` and `lat`.
        q: Number of low-frequency weights used in the test statistic.
        nrep: Number of Monte Carlo draws used to simulate the null distribution.
        seed: Random seed used to generate the simulation draws.

    Returns:
        A `TestResult` for the selected test.

    Raises:
        ValueError: If `test` is invalid or if the selected test receives invalid
            variables, formula inputs, coordinates, or simulation settings.

    Example:
        >>> from spur import load_chetty_data, standardize, spurtest
        >>> df = load_chetty_data()
        >>> df = df[["am", "lat", "lon"]].dropna()
        >>> df = standardize(df, ["am"])
        >>> result = spurtest("am", df, test="i1", lon="lon", lat="lat", q=10, nrep=500, seed=42)
        >>> print(result.summary())
    """
    if test == "i0":
        parsed = parse_single_var_formula(formula, data, "spurtest()")
        return spurtest_i0(
            parsed["var"],
            data,
            lon=lon,
            lat=lat,
            coords_euclidean=coords_euclidean,
            q=q,
            nrep=nrep,
            seed=seed,
            backend=backend,
            nthreads=nthreads,
        )
    if test == "i1":
        parsed = parse_single_var_formula(formula, data, "spurtest()")
        return spurtest_i1(
            parsed["var"],
            data,
            lon=lon,
            lat=lat,
            coords_euclidean=coords_euclidean,
            q=q,
            nrep=nrep,
            seed=seed,
            backend=backend,
            nthreads=nthreads,
        )
    if test == "i0resid":
        return spurtest_i0resid(
            formula,
            data,
            lon=lon,
            lat=lat,
            coords_euclidean=coords_euclidean,
            q=q,
            nrep=nrep,
            seed=seed,
            backend=backend,
            nthreads=nthreads,
        )
    if test == "i1resid":
        return spurtest_i1resid(
            formula,
            data,
            lon=lon,
            lat=lat,
            coords_euclidean=coords_euclidean,
            q=q,
            nrep=nrep,
            seed=seed,
            backend=backend,
            nthreads=nthreads,
        )
    raise ValueError("`test` must be one of 'i0', 'i1', 'i0resid', 'i1resid'.")


def spurtransform(
    formula: str,
    data: pd.DataFrame,
    *,
    lon: str | None = None,
    lat: str | None = None,
    coords_euclidean: Sequence[str] | None = None,
    prefix: str = "h_",
    transformation: str = "lbmgls",
    radius: Optional[float] = None,
    clustvar: Optional[str] = None,
    backend: str | None = None,
    nthreads: int | None = None,
) -> pd.DataFrame:
    """Apply a SPUR transformation to the variables referenced in a formula.

    Use this function when you want transformed versions of the variables in a
    model formula. The function builds one transformation matrix and applies it to
    each referenced variable, adding new columns with the chosen prefix.

    Args:
        formula: Formula identifying the variables to transform.
        data: DataFrame containing the variables and any required coordinate or
            cluster columns.
        lon: Name of the longitude column. Use together with `lat`.
        lat: Name of the latitude column. Use together with `lon`.
        coords_euclidean: Names of Euclidean coordinate columns. Use instead of
            `lon` and `lat`.
        prefix: Prefix added to each transformed variable name.
        transformation: Transformation method. Must be one of `"nn"`, `"iso"`,
            `"lbmgls"`, or `"cluster"`.
        radius: Radius used by the isotropic transformation. Required when
            `transformation="iso"`.
        clustvar: Cluster label column used by the cluster transformation.
            Required when `transformation="cluster"`.

    Returns:
        A copy of `data` with transformed columns added.

    Raises:
        ValueError: If the formula is invalid, variables are missing, coordinates
            are invalid, or the transformation name is unknown.
        AssertionError: If a required `radius` or `clustvar` is missing, or if the
            cluster column contains missing values.

    Example:
        >>> from spur import load_chetty_data, standardize, spurtransform
        >>> df = load_chetty_data()
        >>> df = df[["am", "fracblack", "lat", "lon"]].dropna()
        >>> df = standardize(df, ["am", "fracblack"])
        >>> df_out = spurtransform(
        ...     "am ~ fracblack",
        ...     df,
        ...     lon="lon",
        ...     lat="lat",
        ...     transformation="lbmgls",
        ... )
        >>> df_out[["h_am", "h_fracblack"]].head()
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("`data` must be a pandas DataFrame.")
    y, x = evaluate_formula(formula, data, allow_no_response=True)
    variables = pd.concat([y, x.drop(columns="Intercept", errors="ignore")], axis=1)
    variables = variables.loc[:, ~variables.columns.duplicated()]
    if variables.shape[1] == 0:
        raise ValueError("No variables found in `formula`.")
    use_rows = np.isin(np.arange(len(data)), variables.index)
    sample = data.iloc[np.flatnonzero(use_rows)]
    values = variables.to_numpy(dtype=np.float64)
    resolved_backend = resolve_backend(backend)
    resolved_nthreads = validate_nthreads(nthreads)

    if transformation not in ("cluster", "nn", "iso", "lbmgls"):
        raise ValueError(
            f"Unknown transformation: {transformation}. "
            "Use 'nn', 'iso', 'lbmgls', or 'cluster'."
        )

    n = len(sample)
    rust_mod = _require_rust_if_requested(resolved_backend)

    cluster_codes: np.ndarray | None = None
    cluster: np.ndarray | None = None
    coords_raw: np.ndarray | None = None
    latlon = False
    if transformation == "cluster":
        assert clustvar is not None, (
            "clustvar must be specified for transformation='cluster'"
        )
        assert clustvar in sample.columns, (
            f"Cluster variable '{clustvar}' not found in DataFrame"
        )
        assert not sample[clustvar].isna().any(), (
            f"Cluster column '{clustvar}' contains missing values."
        )
        cluster = sample[clustvar].to_numpy()
        codes, _ = pd.factorize(cluster, sort=False)
        cluster_codes = np.ascontiguousarray(codes, dtype=np.int64)
    else:
        coord_info = resolve_spur_coords(
            data=data,
            use_rows=use_rows,
            lon=lon,
            lat=lat,
            coords_euclidean=coords_euclidean,
        )
        coords_raw = coord_info["coords"]
        latlon = coord_info["latlong"]
        if coords_raw.shape[1] != 2:
            raise ValueError(
                f"Distance-based transformations require exactly 2 coordinate columns, got {coords_raw.shape[1]}"
            )
        if transformation == "iso":
            assert radius is not None, (
                "radius must be specified for transformation='iso'"
            )

    if rust_mod is not None and (n >= 2 or resolved_backend == "rust"):
        radius_arg = None
        if transformation == "iso" and radius is not None:
            radius_arg = (
                float(radius) / EARTH_CIRCUMFERENCE_M if latlon else float(radius)
            )
        coords_rust = None if coords_raw is None else _rust_coords(coords_raw, latlon)

        rust_res = rust_mod.run_spur_transform(
            transformation,
            n,
            y=np.ascontiguousarray(values),
            coords=coords_rust,
            latlong=bool(latlon),
            radius=radius_arg,
            strict_lt=True,
            exclude_zero_distances=transformation == "iso",
            cluster_id=cluster_codes,
            return_matrix=False,
            demean_lbmgls=False,
            nthreads=resolved_nthreads,
        )
        transformed = np.asarray(rust_res["transformed"], dtype=float)
    elif transformation == "cluster":
        assert cluster is not None
        transformed = cluster_matrix(cluster) @ values
    else:
        assert coords_raw is not None
        if transformation == "nn":
            M = nn_matrix(coords_raw, latlon=latlon)
        elif transformation == "iso":
            assert radius is not None
            M = iso_matrix(coords_raw, radius, latlon=latlon)
        else:
            M = lbmgls_matrix(coords_raw, latlon=latlon)

        transformed = M @ values

    df = data.copy()
    for var, column in zip(variables.columns, transformed.T, strict=True):
        out = np.full(len(data), np.nan)
        out[use_rows] = column
        df[f"{prefix}{var}"] = out

    return df


def spurhalflife(
    var: str,
    data: pd.DataFrame,
    *,
    lon: str | None = None,
    lat: str | None = None,
    coords_euclidean: Sequence[str] | None = None,
    q: int = 15,
    nrep: int = 100000,
    level: float = 95,
    normdist: bool = False,
    seed: int = 42,
    backend: str | None = None,
    nthreads: int | None = None,
) -> HalfLifeResult:
    """
    Compute confidence interval for spatial half-life.

    The half-life is the distance at which spatial correlation = 1/2.

    Parameters
    ----------
    data : DataFrame
        Input data
    var : str
        Variable to analyze
    lon, lat : str, optional
        Geographic coordinate column names
    coords_euclidean : sequence of str, optional
        Euclidean coordinate column names
    q : int, default 15
        Number of low-frequency weights
    nrep : int, default 100000
        Monte Carlo draws
    level : float, default 95
        Confidence level in percent
    normdist : bool, default False
        If True, return CI in fractions of max pairwise distance.
        If False, return in meters (if latlon) or coordinate units.
    seed : int, optional
        Random seed

    Returns
    -------
    HalfLifeResult
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("`data` must be a pandas DataFrame.")
    if not isinstance(var, str) or not var:
        raise ValueError("`var` must be a non-empty column name.")
    if var not in data.columns:
        raise ValueError(f"Variable '{var}' not found in data.")
    if not (0 < level < 100):
        raise ValueError(f"level={level} must be strictly between 0 and 100.")
    if q < 2:
        raise ValueError(f"q={q} must be >= 2 for informative SPUR inference.")
    if nrep < 1:
        raise ValueError(f"nrep={nrep} must be >= 1.")

    Y = data[var].to_numpy(dtype=float)
    if not np.all(np.isfinite(Y)):
        raise ValueError(
            f"Variable '{var}' contains NaN or inf values. All values must be finite."
        )

    coord_info = resolve_spur_coords(
        data=data,
        use_rows=np.ones(len(data), dtype=bool),
        lon=lon,
        lat=lat,
        coords_euclidean=coords_euclidean,
    )
    resolved_backend = resolve_backend(backend)
    resolved_nthreads = validate_nthreads(nthreads)
    coords = coord_info["coords"]
    latlon = coord_info["latlong"]
    n = coords.shape[0]
    if q >= n:
        raise ValueError(f"q={q} must be less than n={n}. Use q <= {n - 1}.")

    Y = validate_y_variation(Y)
    emat = _draw_emat(seed, q, nrep)

    tied_from_rust = False
    rust_mod = _require_rust_if_requested(resolved_backend)
    if rust_mod is not None:
        rust_res = rust_mod.run_spur_halflife(
            _rust_coords(coords, latlon),
            np.ascontiguousarray(Y, dtype=np.float64),
            np.ascontiguousarray(emat, dtype=np.float64),
            int(q),
            int(nrep),
            level=float(level / 100.0),
            latlong=bool(latlon),
            nthreads=resolved_nthreads,
        )
        tied_from_rust = bool(rust_res["tied"])
        if not (tied_from_rust and resolved_backend == "auto"):
            max_dist = float(rust_res["max_dist"])
            if latlon:
                max_dist *= EARTH_CIRCUMFERENCE_M
            ci_l = float(rust_res["ci_lower"])
            ci_u = float(rust_res["ci_upper"])
            if ci_u >= 100.0:
                ci_u = float(np.inf)
            if not normdist:
                ci_l = ci_l * max_dist if not np.isnan(ci_l) else ci_l
                ci_u = ci_u * max_dist if not np.isinf(ci_u) else ci_u
            return HalfLifeResult(
                ci_lower=ci_l,
                ci_upper=ci_u,
                max_dist=max_dist,
                level=level,
                normdist=normdist,
                pv_mat=np.asarray(rust_res["pv_mat"], dtype=float).ravel(),
                hl_grid=np.asarray(rust_res["hl_grid_ho"], dtype=float).ravel(),
                backend="rust",
                tied=tied_from_rust,
                distance_units="meters" if latlon else "coordinate units",
            )

    distmat_raw = get_distance_matrix(coords, latlon=latlon)

    max_dist = float(distmat_raw.max())
    if not np.isfinite(max_dist) or max_dist <= 0.0:
        raise ValueError(
            "All coordinates are identical — half-life normalization "
            "requires distinct locations."
        )
    distmat = distmat_raw / max_dist

    ci_l, ci_u = spur_persistence(Y, distmat, emat, level / 100)

    if ci_u >= 100:
        ci_u = np.inf

    if not normdist:
        ci_l = ci_l * max_dist if not np.isnan(ci_l) else ci_l
        ci_u = ci_u * max_dist if not np.isinf(ci_u) else ci_u

    return HalfLifeResult(
        ci_lower=ci_l,
        ci_upper=ci_u,
        max_dist=max_dist,
        level=level,
        normdist=normdist,
        backend="fallback",
        tied=tied_from_rust,
        distance_units="meters" if latlon else "coordinate units",
        fallback_reason=_fallback_reason(resolved_backend, tied_from_rust),
    )


def spur(
    formula: str,
    data: pd.DataFrame,
    *,
    lon: str | None = None,
    lat: str | None = None,
    coords_euclidean: list[str] | tuple[str, ...] | None = None,
    q: int = 15,
    nrep: int = 100000,
    seed: int = 42,
    avc: float = 0.03,
    uncond: bool = False,
    cvs: bool = False,
    backend: str | None = None,
    nthreads: int | None = None,
) -> PipelineResult:
    """Run the full SPUR pipeline and return all major outputs.

    Args:
        formula: Two-sided regression formula for the model of interest.
        data: DataFrame containing the model variables and coordinate columns.
        lon: Name of the longitude column. Use together with `lat`.
        lat: Name of the latitude column. Use together with `lon`.
        coords_euclidean: Names of Euclidean coordinate columns. Use instead of
            `lon` and `lat`.
        q: Number of low-frequency weights used in the SPUR tests.
        nrep: Number of Monte Carlo draws used in the SPUR tests.
        seed: Random seed used to generate the SPUR simulation draws.
        avc: SCPC variance-covariance tuning parameter.
        uncond: Passed through to `scpc()`.
        cvs: Passed through to `scpc()`.

    Returns:
        A `PipelineResult` containing the test results and both fitted branches.

    Raises:
        ValueError: If `formula` is not two-sided or does not contain a dependent
            variable on the left-hand side.
        ValueError: Propagated from the SPUR test and transformation steps if the
            inputs are invalid.

    Example:
        >>> from spur import load_chetty_data, standardize, spur
        >>> df = load_chetty_data()
        >>> df = df[["am", "fracblack", "lat", "lon"]].dropna()
        >>> df = standardize(df, ["am", "fracblack"])
        >>> result = spur("am ~ fracblack", df, lon="lon", lat="lat", q=10, nrep=500, seed=42)
        >>> print(result.summary())
    """
    if not isinstance(formula, str) or "~" not in formula:
        raise ValueError("`formula` must be two-sided, e.g. `y ~ x1 + x2`.")

    if not all(part.strip() for part in formula.split("~", 1)):
        raise ValueError("`formula` must have both a response and a right-hand side.")
    if not isinstance(data, pd.DataFrame):
        raise ValueError("`data` must be a pandas DataFrame.")

    backend = resolve_backend(backend)
    nthreads = validate_nthreads(nthreads)
    rust_mod = _require_rust_if_requested(backend)
    scpc_options: dict[str, Any] = {}
    scpc_parameters = signature(scpc).parameters
    if "backend" in scpc_parameters:
        scpc_options["backend"] = backend
    elif backend == "rust":
        raise RuntimeError(
            "Rust execution in spur() requires a compatible SCPC package."
        )
    if nthreads is not None:
        if "nthreads" in scpc_parameters:
            scpc_options["nthreads"] = nthreads
        elif "backend" in scpc_parameters and backend != "fallback":
            raise RuntimeError(
                "Selecting a thread count for spur() requires a compatible SCPC package."
            )

    if not data.index.is_unique:
        data = data.reset_index(drop=True)
    try:
        fitted = smf.ols(formula, data=data)
    except MissingDataError as error:
        raise ValueError(
            "The evaluated regression design and response must be finite."
        ) from error
    sample = data.loc[fitted.data.row_labels].copy()
    y = np.asarray(fitted.endog, dtype=float)
    x = np.asarray(fitted.exog, dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("The evaluated regression design and response must be finite.")
    design_names = fitted.exog_names
    intercept = "Intercept" in fitted.data.design_info.term_name_slices
    design_basis = orthonormal_design_basis(x)
    if not intercept:
        constant = np.ones(len(sample))
        if np.linalg.norm(constant - design_basis @ (design_basis.T @ constant)) <= (
            1e-10 * math.sqrt(len(sample))
        ):
            raise ValueError(
                "The intercept-free design spans a constant, so the transformed "
                "regression would be rank-deficient. Use an explicit intercept."
            )
    _validate_test_dimensions(q, nrep, len(sample), design_basis)
    levels_model = fitted.fit()

    coord_info = resolve_spur_coords(
        data=sample,
        use_rows=np.ones(len(sample), dtype=bool),
        lon=lon,
        lat=lat,
        coords_euclidean=coords_euclidean,
    )
    coords, latlong = coord_info["coords"], coord_info["latlong"]
    emat = _draw_emat(seed, q, nrep)

    transform_idx = [
        i
        for i, name in enumerate(design_names)
        if not (intercept and name == "Intercept")
    ]
    transform_input = np.column_stack((y, x[:, transform_idx]))
    test_types = ("i0", "i1", "i0resid", "i1resid")

    native = None
    if rust_mod is not None:
        native = rust_mod.run_spur_pipeline(
            _rust_coords(coords, latlong),
            np.ascontiguousarray(y, dtype=np.float64),
            np.ascontiguousarray(x, dtype=np.float64),
            emat,
            int(q),
            int(nrep),
            np.ascontiguousarray(transform_input, dtype=np.float64),
            latlong=bool(latlong),
            demean_lbmgls=False,
            nthreads=nthreads,
        )

    distmat = None
    stage_backends: dict[str, str] = {}
    fallback_reasons: dict[str, str] = {}
    tests: dict[str, TestResult] = {}
    for test_type in test_types:
        stage = None if native is None else native[test_type]
        tied = stage is not None and bool(stage["tied"])
        if stage is not None and (backend == "rust" or not tied):
            result = _test_result_from_rust(
                test_type, stage, is_i0=test_type.startswith("i0")
            )
        else:
            if distmat is None:
                distmat = normalized_distmat(coords, latlon=latlong)
            result = _fallback_spur_test(
                test_type,
                y,
                distmat,
                emat,
                design_basis if test_type.endswith("resid") else None,
                tied=tied,
                fallback_reason=_fallback_reason(backend, tied),
            )
        tests[test_type] = result
        stage_backends[test_type] = result.backend
        if result.fallback_reason is not None:
            fallback_reasons[test_type] = result.fallback_reason

    if native is not None:
        transformed = np.asarray(native["transformed_vars"], dtype=float)
        stage_backends["transform"] = "rust"
    else:
        transformed = lbmgls_matrix(coords, latlon=latlong) @ transform_input
        stage_backends["transform"] = "fallback"
        reason = _fallback_reason(backend)
        if reason is not None:
            fallback_reasons["transform"] = reason

    transformed_data = sample.copy()
    names = [fitted.endog_names, *(design_names[i] for i in transform_idx)]
    transformed_columns = []
    transformed_terms = []
    for name, col in zip(names, transformed.T, strict=True):
        column = f"h_{name}"
        while column in transformed_data.columns:
            column = f"h_{column}"
        transformed_data[column] = col
        transformed_columns.append(column)
        transformed_terms.append(column if column.isidentifier() else f"Q({column!r})")
    rhs = transformed_terms[1:] or ["1"]
    if not intercept:
        rhs = ["0", *transformed_terms[1:]]
    transformed_formula = f"{transformed_terms[0]} ~ {' + '.join(rhs)}"
    # Formula evaluation needs only the new variables. In particular a column
    # named Q in the input must not shadow Patsy's quoting helper.
    transformed_spec = smf.ols(
        transformed_formula, data=transformed_data[transformed_columns]
    )
    orthonormal_design_basis(np.asarray(transformed_spec.exog, dtype=float))
    transformed_model = transformed_spec.fit()

    fits: dict[str, RegressionResult] = {}
    for stage, model, frame in (
        ("levels_scpc", levels_model, sample),
        ("transformed_scpc", transformed_model, transformed_data),
    ):
        result = scpc(
            model,
            frame,
            lon=lon,
            lat=lat,
            coords_euclidean=coords_euclidean,
            avc=avc,
            uncond=uncond,
            cvs=cvs,
            **scpc_options,
        )
        fits[stage] = RegressionResult(
            model=model,
            scpc=result,
            coefficient_labels=tuple(design_names)
            if stage == "transformed_scpc"
            else None,
        )
        stage_backends[stage] = getattr(result, "backend", "fallback")
        reason = getattr(result, "fallback_reason", None)
        if backend == "auto" and "backend" not in scpc_parameters:
            reason = "scpc_backend_api_unavailable"
        if reason is not None:
            fallback_reasons[stage] = reason

    return PipelineResult(
        tests=Tests(tests["i0"], tests["i1"], tests["i0resid"], tests["i1resid"]),
        fits=Fits(levels=fits["levels_scpc"], transformed=fits["transformed_scpc"]),
        backend="rust"
        if all(x == "rust" for x in stage_backends.values())
        else "fallback",
        tied=any(result.tied for result in tests.values())
        or any(bool(getattr(fit.scpc, "tied", False)) for fit in fits.values()),
        stage_backends=stage_backends,
        fallback_reasons=fallback_reasons,
    )
