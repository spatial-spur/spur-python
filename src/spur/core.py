from __future__ import annotations

from collections.abc import Callable, Sequence
import importlib
from dataclasses import dataclass
import os
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from patsy.eval import EvalEnvironment
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
    EARTH_CIRCUMFERENCE_M,
    get_cbar,
    get_distance_matrix,
    normalized_distmat,
    resolve_spur_coords,
)
from .utils.formula import (
    evaluate_formula,
    intercept_columns,
    parse_residual_formula,
    parse_single_var_formula,
)
from .utils.inference import (
    get_ha_param_i0,
    get_ha_param_i1,
    get_ha_param_i1_residual,
    monte_carlo_cutoffs,
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
    lbmgls_matrix_from_distances,
    nn_matrix,
    orthonormal_design_basis,
    validate_and_project_residual_y,
    validate_and_project_y,
    validate_constant_span,
    validate_y_variation,
)

VALID_BACKENDS = {"auto", "rust", "fallback"}
RUST_API_VERSION = 7
SPUR_RUST_API_VERSION = 5
RUST_MIN_SPEC_VERSION = (0, 3, 7)


def _different_covariance_shapes(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.max(np.abs(a / np.trace(a) - b / np.trace(b))) > 1e-12)


def _validate_i1_shapes(
    distmat: np.ndarray, omega: Callable[[float], np.ndarray]
) -> None:
    lo, mid, hi = (omega(get_cbar(rho, distmat)) for rho in (0.01, 0.5, 0.95))
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
    """Import the Rust extension if its ABIs and methods version are compatible."""
    try:
        rust_mod = importlib.import_module("spatial_spur_rust")
    except ImportError:
        return None
    if getattr(rust_mod, "API_VERSION", None) != RUST_API_VERSION:
        return None
    if getattr(rust_mod, "SPUR_API_VERSION", None) != SPUR_RUST_API_VERSION:
        return None
    build_info = getattr(rust_mod, "build_info", None)
    if not callable(build_info):
        return None
    try:
        version = build_info()["spec_version"]
        spec = tuple(int(part) for part in version.split("."))
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    if len(spec) != 3 or spec < RUST_MIN_SPEC_VERSION:
        return None
    return rust_mod


def _require_rust_if_requested(resolved_backend: str) -> Any | None:
    """Return the Rust module when enabled, raising if `backend='rust'` cannot load it."""
    if resolved_backend == "fallback":
        return None
    rust_mod = load_rust_backend()
    if rust_mod is None and resolved_backend == "rust":
        raise RuntimeError(
            "Rust backend requested, but a compatible `spatial_spur_rust` is unavailable. "
            f"Build the matching development adapter with SPUR API {SPUR_RUST_API_VERSION} "
            f"and SCPC API {RUST_API_VERSION}, with methods spec 0.3.7 or newer."
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
) -> TestResult | None:
    """Execute one SPUR diagnostic test with a compatible Rust build."""
    rust_mod = _require_rust_if_requested(resolved_backend)
    if rust_mod is None:
        return None

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

    return _test_result_from_rust(test_type, rust_res, is_i0=is_i0)


def _fallback_reason(backend: str) -> str | None:
    return "missing_or_incompatible_native_extension" if backend == "auto" else None


def _validate_test_dimensions(q: int, nrep: int, n: int, n_regressors: int = 0) -> None:
    if q >= n:
        raise ValueError(f"q={q} must be less than n={n}. Use q <= {n - 1}.")
    if q < 2:
        raise ValueError(f"q={q} must be >= 2 for informative SPUR inference.")
    if nrep < 1:
        raise ValueError(f"nrep={nrep} must be >= 1.")
    if q > n - n_regressors:
        raise ValueError(f"q={q} exceeds the residual dimension {n - n_regressors}.")


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
    _validate_test_dimensions(q, nrep, len(y), 0 if x_in is None else x_in.shape[1])
    design_basis = None
    if x_in is not None and (x_in.shape[1] == 0 or not np.all(x_in[:, 0] == 1.0)):
        # Patsy places a formula intercept first. Check other designs before dispatch.
        design_basis = orthonormal_design_basis(x_in)
        validate_constant_span(design_basis)
    emat = _draw_emat(seed, 2 * q, nrep)
    rust_res = _try_rust_spur_test(
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
    if x_in is not None and design_basis is None:
        design_basis = orthonormal_design_basis(x_in)
    distmat = normalized_distmat(coords, latlon=latlong)
    center_response = x_in is not None and bool(np.any(np.all(x_in == x_in[0], axis=0)))
    basis = _prepare_fallback_spur_basis(
        y, distmat, q, design_basis, center_response=center_response
    )
    return _fallback_spur_test(
        test_type,
        distmat,
        emat,
        basis,
        tied=False,
        fallback_reason=_fallback_reason(backend),
    )


@dataclass(slots=True)
class _FallbackSpurBasis:
    r: np.ndarray
    response: np.ndarray
    brownian: np.ndarray
    omega: Callable[[float], np.ndarray]
    projection: np.ndarray | None


def _prepare_fallback_spur_basis(
    y: np.ndarray,
    distmat: np.ndarray,
    q: int,
    design_basis: np.ndarray | None,
    *,
    center_response: bool = False,
) -> _FallbackSpurBasis:
    projection = (
        None if design_basis is None else np.eye(len(y)) - design_basis @ design_basis.T
    )
    if projection is None:
        sigma_bm = demean_matrix(get_sigma_lbm(distmat))
    else:
        sigma_bm = projection @ (-0.5 * distmat) @ projection
    r = get_r(sigma_bm, q)
    response = (
        validate_and_project_y(y, r)
        if design_basis is None
        else validate_and_project_residual_y(
            y, r, design_basis, center_response=center_response
        )
    )

    def omega(c: float) -> np.ndarray:
        if np.isposinf(c):
            kernel = (distmat == 0.0).astype(float)
            sigma = (
                demean_matrix(kernel)
                if projection is None
                else projection @ kernel @ projection
            )
            return r.T @ sigma @ r
        sigma = (
            get_sigma_dm(distmat, c)
            if projection is None
            else get_sigma_residual(distmat, c, projection)
        )
        return r.T @ sigma @ r

    return _FallbackSpurBasis(r, response, r.T @ sigma_bm @ r, omega, projection)


def _fallback_spur_test(
    test_type: str,
    distmat: np.ndarray,
    emat: np.ndarray,
    basis: _FallbackSpurBasis,
    *,
    tied: bool,
    fallback_reason: str | None,
) -> TestResult:
    run = _fallback_spur_i0 if test_type.startswith("i0") else _fallback_spur_i1
    return run(test_type, distmat, emat, basis, tied, fallback_reason)


def _fallback_spur_i0(
    test_type: str,
    distmat: np.ndarray,
    emat: np.ndarray,
    basis: _FallbackSpurBasis,
    tied: bool,
    fallback_reason: str | None,
) -> TestResult:
    emat, tuning = np.split(emat, 2, axis=0)
    P = basis.response
    om_rho = basis.omega(get_cbar(0.001, distmat))
    om_bm = basis.brownian
    if not _different_covariance_shapes(om_rho, om_bm):
        raise ValueError(
            "SPUR experiment is uninformative: covariance shapes are proportional."
        )

    om_i0 = om_rho
    om_ho = om_rho
    ha_parm = get_ha_param_i0(om_ho, om_i0, om_bm, tuning)
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
    floor = np.mean(distmat[np.tril_indices(len(distmat), -1)] == 0.0)
    rho_grid = np.r_[floor, np.linspace(rho_min, rho_max, n_rho)]
    n_rho = rho_grid.size

    pvalue_vec = np.zeros(n_rho)
    cvalue_mat = np.zeros((n_rho, 3))

    for ir, rho in enumerate(rho_grid):
        ch_om_ho_ir = cholesky_upper(
            basis.omega(np.inf if ir == 0 else get_cbar(rho, distmat))
        )
        y_ho = ch_om_ho_ir.T @ emat
        y_ho_ho = ch_omi_ho @ y_ho
        y_ho_ha = ch_omi_ha @ y_ho
        q_ho_ho = np.sum(y_ho_ho**2, axis=0)
        q_ho_ha = np.sum(y_ho_ha**2, axis=0)
        lr_ho = q_ho_ho / q_ho_ha
        cvalue_mat[ir, :] = monte_carlo_cutoffs(lr_ho)
        pvalue_vec[ir] = (1 + np.count_nonzero(lr_ho >= LR)) / (lr_ho.size + 1)

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
    basis: _FallbackSpurBasis,
    tied: bool,
    fallback_reason: str | None,
) -> TestResult:
    emat, tuning = np.split(emat, 2, axis=0)
    P = basis.response
    om_ho = basis.brownian

    _validate_i1_shapes(distmat, basis.omega)
    ha_parm = (
        get_ha_param_i1(om_ho, distmat, basis.r, tuning)
        if basis.projection is None
        else get_ha_param_i1_residual(om_ho, distmat, basis.r, tuning, basis.projection)
    )
    om_ha = basis.omega(ha_parm)

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

    cv_vec = monte_carlo_cutoffs(lr_ho)

    LR = float((P.T @ omi_ho @ P) / (P.T @ omi_ha @ P))
    pvalue = float((1 + np.count_nonzero(lr_ho >= LR)) / (lr_ho.size + 1))

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


def _run_residual_formula_test(
    test_type: str,
    formula: str,
    data: pd.DataFrame,
    eval_env: EvalEnvironment,
    lon: str | None,
    lat: str | None,
    coords_euclidean: Sequence[str] | None,
    q: int,
    nrep: int,
    seed: int,
    backend: str | None,
    nthreads: int | None,
) -> TestResult:
    if not isinstance(data, pd.DataFrame):
        raise ValueError("`data` must be a pandas DataFrame.")
    parsed = parse_residual_formula(
        formula, data, f"spurtest_{test_type}()", eval_env=eval_env
    )
    coord_info = resolve_spur_coords(
        data=data,
        use_rows=parsed["use_rows"],
        lon=lon,
        lat=lat,
        coords_euclidean=coords_euclidean,
    )
    return _run_spur_test(
        test_type,
        coord_info["coords"],
        coord_info["latlong"],
        parsed["Y"].reshape(-1),
        q,
        nrep,
        seed,
        resolve_backend(backend),
        validate_nthreads(nthreads),
        x_in=parsed["X_in"],
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
    Euclidean coordinate columns. The evaluated design must span a constant,
    for example through an intercept or a full set of group indicators.

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
    return _run_residual_formula_test(
        "i1resid",
        formula,
        data,
        EvalEnvironment.capture(1),
        lon,
        lat,
        coords_euclidean,
        q,
        nrep,
        seed,
        backend,
        nthreads,
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
    Euclidean coordinate columns. The evaluated design must span a constant,
    for example through an intercept or a full set of group indicators.

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
    return _run_residual_formula_test(
        "i0resid",
        formula,
        data,
        EvalEnvironment.capture(1),
        lon,
        lat,
        coords_euclidean,
        q,
        nrep,
        seed,
        backend,
        nthreads,
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
    if test in {"i0resid", "i1resid"}:
        return _run_residual_formula_test(
            test,
            formula,
            data,
            EvalEnvironment.capture(1),
            lon,
            lat,
            coords_euclidean,
            q,
            nrep,
            seed,
            backend,
            nthreads,
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
    model formula. The function applies the same spatial transformation to each
    referenced variable and adds columns with the chosen prefix.

    Nearest-neighbor differencing averages the observations tied at the
    smallest positive distance. It requires at least two distinct locations.
    Isotropic differencing averages neighbors at positive distances up to and
    including the radius. Observations with no such neighbors transform to zero.
    LBM-GLS centers each transformed column.

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
        radius: Radius used by the isotropic transformation in meters for
            longitude and latitude or in the units of Euclidean coordinates.
            Required when `transformation="iso"`.
        clustvar: Cluster label column used by the cluster transformation.
            Required when `transformation="cluster"`.

    Returns:
        A copy of `data` with transformed columns added.

    Raises:
        ValueError: If the formula is invalid, variables are missing, coordinates
            or radius are invalid, a required `radius` or `clustvar` is missing,
            the cluster column contains missing values, a spatial transformation
            has fewer than 2 observations, or the transformation name is unknown.

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
    y, x = evaluate_formula(
        formula, data, eval_env=EvalEnvironment.capture(1), allow_no_response=True
    )
    intercept_slice = intercept_columns(getattr(x, "design_info"))
    if intercept_slice is not None:
        x = x.drop(columns=x.columns[intercept_slice])
    variables = pd.concat([y, x], axis=1)
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
        if clustvar is None:
            raise ValueError("clustvar must be specified for transformation='cluster'")
        if clustvar not in sample.columns:
            raise ValueError(f"Cluster variable '{clustvar}' not found in DataFrame")
        if sample[clustvar].isna().any():
            raise ValueError(f"Cluster column '{clustvar}' contains missing values.")
        cluster = sample[clustvar].to_numpy()
        if rust_mod is not None:
            codes, _ = pd.factorize(cluster, sort=False)
            cluster_codes = np.ascontiguousarray(codes, dtype=np.int64)
    else:
        if n < 2:
            raise ValueError(
                f"Spatial transformation '{transformation}' requires at least 2 observations."
            )
        coord_info = resolve_spur_coords(
            data=data,
            use_rows=use_rows,
            lon=lon,
            lat=lat,
            coords_euclidean=coords_euclidean,
        )
        coords_raw = coord_info["coords"]
        latlon = coord_info["latlong"]
        if transformation == "iso":
            if radius is None:
                raise ValueError("radius must be specified for transformation='iso'")
            if isinstance(radius, bool):
                raise ValueError("radius must be a finite positive number.")
            try:
                radius_val = float(radius)
            except (TypeError, ValueError) as exc:
                raise ValueError("radius must be a finite positive number.") from exc
            if not np.isfinite(radius_val) or radius_val <= 0.0:
                raise ValueError("radius must be finite and strictly positive.")
            radius = radius_val

    if rust_mod is not None:
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
            cluster_id=cluster_codes,
            nthreads=resolved_nthreads,
        )
        transformed = np.asarray(rust_res["transformed"], dtype=float)
    elif transformation == "cluster":
        if cluster is None:
            raise ValueError("clustvar must be specified for transformation='cluster'")
        transformed = cluster_matrix(cluster) @ values
    else:
        if coords_raw is None:
            raise ValueError("Coordinates must be resolved for spatial transformation.")
        if transformation == "nn":
            M = nn_matrix(coords_raw, latlon=latlon)
        elif transformation == "iso":
            if radius is None:
                raise ValueError("radius must be specified for transformation='iso'")
            M = iso_matrix(coords_raw, radius, latlon=latlon)
        else:
            M = lbmgls_matrix(coords_raw, latlon=latlon)

        transformed = M @ values
        if transformation == "lbmgls":
            transformed -= transformed.mean(axis=0)

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
    Invert spatial half-life tests by refining changes in acceptance.

    The half-life is the distance at which spatial correlation equals one half.
    The result contains detected components and their hull. Narrow components
    between search probes can escape detection.

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
        max_dist = float(rust_res["max_dist"])
        if latlon:
            max_dist *= EARTH_CIRCUMFERENCE_M
        components = np.asarray(rust_res["components"], dtype=float).reshape(-1, 2)
        pvalues = np.asarray(rust_res["pv_mat"], dtype=float).ravel()
        h_grid = np.asarray(rust_res["hl_grid_ho"], dtype=float).ravel()
        tied = bool(rust_res["tied"])
    else:
        distmat_raw = get_distance_matrix(coords, latlon=latlon)
        max_dist = float(distmat_raw.max())
        if not np.isfinite(max_dist) or max_dist <= 0.0:
            raise ValueError(
                "Half-life inference requires at least two distinct locations."
            )
        components, pvalues, h_grid = spur_persistence(
            Y, distmat_raw / max_dist, emat, level / 100
        )
        tied = False

    if not normdist:
        components = components * max_dist
    return HalfLifeResult(
        ci_lower=float(components[0, 0]) if components.size else np.nan,
        ci_upper=float(components[-1, 1]) if components.size else np.nan,
        max_dist=max_dist,
        level=level,
        normdist=normdist,
        pv_mat=pvalues,
        hl_grid=h_grid,
        components=components,
        backend="rust" if rust_mod is not None else "fallback",
        tied=tied,
        distance_units="meters" if latlon else "coordinate units",
        fallback_reason=_fallback_reason(resolved_backend)
        if rust_mod is None
        else None,
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
        formula: Two-sided regression formula with an intercept.
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
    if not data.index.is_unique:
        data = data.reset_index(drop=True)
    try:
        fitted = smf.ols(formula, data=data, eval_env=EvalEnvironment.capture(1))
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
    intercept_slice = intercept_columns(fitted.data)
    if intercept_slice is None:
        raise ValueError("spur() requires a formula with an intercept.")
    design_basis = orthonormal_design_basis(x) if rust_mod is None else None
    _validate_test_dimensions(q, nrep, len(sample), x.shape[1])

    coord_info = resolve_spur_coords(
        data=sample,
        use_rows=np.ones(len(sample), dtype=bool),
        lon=lon,
        lat=lat,
        coords_euclidean=coords_euclidean,
    )
    coords, latlong = coord_info["coords"], coord_info["latlong"]
    emat = _draw_emat(seed, 2 * q, nrep)

    transform_idx = list(range(x.shape[1]))
    del transform_idx[intercept_slice]
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
            nthreads=nthreads,
        )

    use_native = native is not None
    distmat = None if use_native else normalized_distmat(coords, latlon=latlong)
    if not use_native:
        assert distmat is not None
        if design_basis is None:
            design_basis = orthonormal_design_basis(x)
        levels_basis = _prepare_fallback_spur_basis(y, distmat, q, None)
        residual_basis = _prepare_fallback_spur_basis(
            y, distmat, q, design_basis, center_response=True
        )
    levels_model = fitted.fit()
    fallback_reason = _fallback_reason(backend)
    stage_backends: dict[str, str] = {}
    fallback_reasons: dict[str, str] = {}
    tests: dict[str, TestResult] = {}
    for test_type in test_types:
        stage = None if native is None else native[test_type]
        tied = stage is not None and bool(stage["tied"])
        if use_native:
            assert stage is not None
            result = _test_result_from_rust(
                test_type, stage, is_i0=test_type.startswith("i0")
            )
        else:
            assert distmat is not None
            result = _fallback_spur_test(
                test_type,
                distmat,
                emat,
                residual_basis if test_type.endswith("resid") else levels_basis,
                tied=tied,
                fallback_reason=fallback_reason,
            )
        tests[test_type] = result
        stage_backends[test_type] = result.backend
        if result.fallback_reason is not None:
            fallback_reasons[test_type] = result.fallback_reason

    if use_native:
        assert native is not None
        transformed = np.asarray(native["transformed_vars"], dtype=float)
        stage_backends["transform"] = "rust"
    else:
        assert distmat is not None
        transformed = lbmgls_matrix_from_distances(distmat) @ transform_input
        transformed -= transformed.mean(axis=0)
        stage_backends["transform"] = "fallback"
        if fallback_reason is not None:
            fallback_reasons["transform"] = fallback_reason

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
            backend=backend,
            nthreads=nthreads,
        )
        fits[stage] = RegressionResult(
            model=model,
            scpc=result,
            coefficient_labels=tuple(design_names)
            if stage == "transformed_scpc"
            else None,
        )
        stage_backends[stage] = result.backend
        if result.fallback_reason is not None:
            fallback_reasons[stage] = result.fallback_reason

    return PipelineResult(
        tests=Tests(tests["i0"], tests["i1"], tests["i0resid"], tests["i1resid"]),
        fits=Fits(levels=fits["levels_scpc"], transformed=fits["transformed_scpc"]),
        backend="rust"
        if all(x == "rust" for x in stage_backends.values())
        else "fallback",
        tied=any(result.tied for result in tests.values())
        or any(fit.scpc.tied for fit in fits.values()),
        stage_backends=stage_backends,
        fallback_reasons=fallback_reasons,
    )
