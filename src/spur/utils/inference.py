from collections.abc import Callable

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

from .dist import get_cbar
from .matrix import (
    get_r,
    validate_and_project_y,
    demean_matrix,
    get_sigma_dm,
)


def get_pow_qf(om0: np.ndarray, om1: np.ndarray, e: np.ndarray) -> float:
    """
    Compute power of test via quadratic forms.

    Parameters
    ----------
    om0 : ndarray (q, q)
        Null covariance matrix
    om1 : ndarray (q, q)
        Alternative covariance matrix
    e : ndarray (q, nrep)
        Random Monte Carlo draws

    Returns
    -------
    float
        Power (probability of rejecting H0 at 5% level)
    """
    ch_om0 = np.linalg.cholesky(om0)
    ch_om1 = np.linalg.cholesky(om1)
    ho = solve_triangular(ch_om1, ch_om0, lower=True)
    ha = solve_triangular(ch_om0, ch_om1, lower=True)

    qe = np.sum(e**2, axis=0)
    ya_o = ho @ e
    yo_a = ha @ e
    qa_o = np.sum(ya_o**2, axis=0)
    qo_a = np.sum(yo_a**2, axis=0)

    lr_o = qe / qa_o
    lr_a = qo_a / qe

    cv = np.quantile(lr_o, 0.95)
    pow_ = np.mean(lr_a > cv)

    return pow_


def fit_spur_i1(
    om_ho: np.ndarray,
    distmat: np.ndarray,
    e: np.ndarray,
    covariance: Callable[[float], np.ndarray],
) -> tuple[float, np.ndarray]:
    """Tune the I(1) alternative on the independent draws."""
    floor = np.count_nonzero(distmat == 0.0) - len(distmat)
    floor /= len(distmat) * (len(distmat) - 1)
    min_distance = float(np.min(distmat, where=distmat > 0, initial=np.inf))
    lower, middle, upper = (
        get_cbar(floor + (1 - floor) * rho, distmat) for rho in (0.95, 0.5, 0.01)
    )
    low_cov, mid_cov, high_cov = (covariance(c) for c in (lower, middle, upper))
    null_shape = om_ho / np.trace(om_ho)
    if all(
        np.max(np.abs(omega / np.trace(omega) - null_shape)) <= 1e-12
        for omega in (low_cov, mid_cov, high_cov)
    ):
        raise ValueError(
            "SPUR experiment is uninformative: covariance shapes are proportional."
        )

    power = get_pow_qf(om_ho, low_cov, e)
    for _ in range(199):
        if power <= 0.5:
            break
        lower *= 0.5
        power = get_pow_qf(om_ho, covariance(lower), e)
    if power > 0.5:
        raise RuntimeError("SPUR I(1) step-1 bracketing did not converge.")

    omega = high_cov
    power = get_pow_qf(om_ho, omega, e)
    rate = upper
    best_rate, best_cov, best_power = rate, omega, power
    for _ in range(199):
        if power >= 0.5:
            break
        if np.exp(-upper * min_distance) == 0:
            return best_rate, best_cov
        upper *= 2
        rate = upper
        omega = covariance(rate)
        power = get_pow_qf(om_ho, omega, e)
        if power > best_power:
            best_rate, best_cov, best_power = rate, omega, power
    if power < 0.5:
        return best_rate, best_cov

    for _ in range(20):
        if abs(power - 0.5) <= 0.01:
            break
        rate = (lower + upper) / 2
        omega = covariance(rate)
        power = get_pow_qf(om_ho, omega, e)
        if power > 0.5:
            upper = rate
        else:
            lower = rate
    return rate, omega


def precision_factor(lower: np.ndarray) -> np.ndarray:
    """Return the triangular factor used in the covariance quadratic form."""
    return solve_triangular(lower, np.eye(len(lower)), lower=True)


def get_ha_param_i1(
    om_ho: np.ndarray, distmat: np.ndarray, R: np.ndarray, e: np.ndarray
) -> float:
    """Return the I(1) alternative decay for a levels basis."""
    return fit_spur_i1(om_ho, distmat, e, lambda c: R.T @ get_sigma_dm(distmat, c) @ R)[
        0
    ]


def get_ha_param_i0(
    om_ho: np.ndarray, om_i0: np.ndarray, om_bm: np.ndarray, e: np.ndarray
) -> float:
    """
    Find alternative hypothesis parameter g that yields ~50% power for I(0) test.
    """
    pow_ = 1.0
    gtry = 1.0
    _MAX_BRACKET = 200

    _iter = 0
    while pow_ > 0.5:
        g = gtry
        pow_ = get_pow_qf(om_ho, om_i0 + g * om_bm, e)
        gtry = g / 2
        _iter += 1
        if _iter >= _MAX_BRACKET and pow_ > 0.5:
            raise RuntimeError(
                f"get_ha_param_i0: step-1 bracketing did not converge in {_MAX_BRACKET} iterations."
            )
    g1 = g

    pow_ = 0.0
    gtry = 30.0
    best_param, best_power = g, -1.0
    _iter = 0
    while pow_ < 0.5:
        g = gtry
        pow_ = get_pow_qf(om_ho, om_i0 + g * om_bm, e)
        gtry = g * 2
        if pow_ > best_power:
            best_param, best_power = g, pow_
        _iter += 1
        if _iter >= _MAX_BRACKET and pow_ < 0.5:
            return best_param
    g2 = g

    ii = 1
    # https://github.com/pdavidboll/SPUR/blob/main/mata/get_ha_parm_I0.mata#L33
    while abs(pow_ - 0.5) > 0.01:
        g = (g1 + g2) / 2
        pow_ = get_pow_qf(om_ho, om_i0 + g * om_bm, e)
        if pow_ > 0.5:
            g2 = g
        else:
            g1 = g
        ii += 1
        if ii > 20:
            break

    return g


def get_ha_param_i1_residual(
    om_ho: np.ndarray, distmat: np.ndarray, R: np.ndarray, e: np.ndarray, M: np.ndarray
) -> float:
    """Return the I(1) alternative decay for a residual basis."""
    projected = M @ R
    return fit_spur_i1(
        om_ho, distmat, e, lambda c: projected.T @ np.exp(-c * distmat) @ projected
    )[0]


def monte_carlo_cutoffs(draws: np.ndarray) -> np.ndarray:
    """Critical values for rank tests at 1%, 5%, and 10%."""
    ranks = (draws.size + 1) // np.array([100, 20, 10])
    active = ranks > 0
    cutoffs = np.full(3, np.inf)
    indices = draws.size - ranks[active]
    if indices.size:
        cutoffs[active] = np.partition(draws, indices)[indices]
    return cutoffs


def _half_life_mixture(min_distance: float) -> tuple[np.ndarray, np.ndarray]:
    nodes = np.array(
        [
            -0.906179845938664,
            -0.5384693101056831,
            0.0,
            0.5384693101056831,
            0.906179845938664,
        ]
    )
    weights = np.array(
        [
            0.2369268850561891,
            0.4786286704993665,
            128 / 225,
            0.4786286704993665,
            0.2369268850561891,
        ]
    )
    near_zero = max(min_distance / 32, np.finfo(float).tiny)
    exponent = -9
    while 2.0**exponent > near_zero:
        exponent -= 1
    edges = np.r_[0, 2.0 ** np.arange(exponent, 1)]
    width = np.diff(edges)[:, None] / 2
    points = edges[:-1, None] + width * (1 + nodes)
    return points.ravel(), (width * weights).ravel()


def _compact_to_half_life(t: float) -> float:
    return np.inf if t == 1 else t / (1 - t)


def _half_life_search_grid(min_distance: float) -> np.ndarray:
    short = []
    h = 1 / 127
    near_zero = max(min_distance / 32, np.finfo(float).tiny)
    while h > near_zero:
        h *= 0.8408964152537145
        short.append(h / (1 + h))
    return np.r_[0, short[::-1], np.linspace(1 / 128, 1, 128)]


def _invert_half_life(grid, accepted, accepts) -> np.ndarray:
    """Bracket each detected transition and round the components outward."""
    intervals = []
    start = 0.0
    for i in np.flatnonzero(accepted[:-1] != accepted[1:]):
        left, right = grid[i : i + 2]
        left_accepted = accepted[i]
        tolerance = min(1e-7, 1e-5 * (right - left))
        while right - left > tolerance:
            mid = (left + right) / 2
            if accepts(mid) == left_accepted:
                left = mid
            else:
                right = mid
        if left_accepted:
            intervals.append(
                (_compact_to_half_life(start), _compact_to_half_life(right))
            )
        else:
            start = left
    if accepted[-1]:
        intervals.append((_compact_to_half_life(start), np.inf))
    return np.asarray(intervals, dtype=float).reshape(-1, 2)


def spur_persistence(
    y: np.ndarray, distmat: np.ndarray, emat: np.ndarray, level: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numerically invert the half-life rank tests in sample-diameter units."""
    q, nrep = emat.shape
    brownian = demean_matrix(-0.5 * distmat)
    basis = get_r(brownian, q)
    projected = basis - basis.mean(axis=0)
    observed = validate_and_project_y(y, basis)

    def covariance(h: float) -> np.ndarray:
        if h == 0:
            kernel = distmat == 0
        elif np.isinf(h):
            kernel = -0.5 * distmat
        else:
            rate = float(np.log(2)) / float(h)
            if rate < 0.01:
                kernel = 0.5 * np.expm1(-rate * distmat) / rate
            else:
                with np.errstate(over="ignore"):
                    kernel = np.exp(-np.log(2) * (distmat / h))
        omega = projected.T @ (kernel @ projected)
        return 0.5 * (omega + omega.T)

    def log_density(upper, values, log_det):
        return log_det - 0.5 * q * np.log(np.sum((upper @ values) ** 2, axis=0))

    min_distance = float(np.min(distmat, where=distmat > 0, initial=np.inf))
    grid = _half_life_search_grid(min_distance)
    h_grid = np.array([_compact_to_half_life(t) for t in grid])
    null_covariances = [covariance(h) for h in h_grid]
    mixture_h, mixture_weights = _half_life_mixture(min_distance)
    alternatives = [covariance(h) for h in mixture_h]
    trace_scaled = null_covariances[0] / np.trace(null_covariances[0])
    if all(
        np.max(np.abs(m / np.trace(m) - trace_scaled)) <= 1e-12
        for m in [*null_covariances[1:], *alternatives]
    ):
        raise ValueError(
            "Half-life experiment is uninformative: covariance shapes are proportional."
        )

    factors = [precision_factor(np.linalg.cholesky(omega)) for omega in alternatives]
    log_weights = [
        np.log(w) + np.log(np.diag(upper)).sum()
        for w, upper in zip(mixture_weights, factors)
    ]
    observed_mix = logsumexp(
        [
            log_density(upper, observed, log_weight)
            for upper, log_weight in zip(factors, log_weights)
        ]
    )
    log_null_draw_norm = 0.5 * q * np.log(np.sum(emat**2, axis=0))

    def exceedances(omega):
        null = np.linalg.cholesky(omega)
        simulated = null @ emat
        observed_score = observed_mix + 0.5 * q * np.log(
            np.sum(solve_triangular(null, observed, lower=True) ** 2)
        )
        simulated_score = np.full(nrep, -np.inf)
        for upper, log_weight in zip(factors, log_weights):
            np.logaddexp(
                simulated_score,
                log_density(upper, simulated, log_weight),
                out=simulated_score,
            )
        simulated_score += log_null_draw_norm
        return np.count_nonzero(simulated_score >= observed_score)

    counts = np.array([exceedances(omega) for omega in null_covariances])
    accepted = (nrep - counts) / (nrep + 1) < level
    components = _invert_half_life(
        grid,
        accepted,
        lambda t: (nrep - exceedances(covariance(_compact_to_half_life(t))))
        / (nrep + 1)
        < level,
    )
    return components, (1 + counts) / (nrep + 1), h_grid
