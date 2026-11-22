import numpy as np
from scipy.special import logsumexp

from .dist import get_cbar
from .matrix import (
    cholesky_upper,
    get_r,
    validate_and_project_y,
    get_sigma_dm,
    get_sigma_residual,
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
    om0i = np.linalg.inv(om0)
    om1i = np.linalg.inv(om1)

    ch_om0 = cholesky_upper(om0)
    ch_om1 = cholesky_upper(om1)
    ch_om0i = cholesky_upper(om0i)
    ch_om1i = cholesky_upper(om1i)

    ho = ch_om1i @ ch_om0.T
    ha = ch_om0i @ ch_om1.T

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


def get_ha_param_i1(
    om_ho: np.ndarray, distmat: np.ndarray, R: np.ndarray, e: np.ndarray
) -> float:
    """
    Find alternative hypothesis parameter c that yields ~50% power for I(1) test.
    """
    pow50 = 0.5
    pow_ = 1.0
    ctry = get_cbar(0.95, distmat)
    _MAX_BRACKET = 200

    _iter = 0
    while pow_ > pow50:
        c = ctry
        sigdm_c = get_sigma_dm(distmat, c)
        om_c = R.T @ sigdm_c @ R
        pow_ = get_pow_qf(om_ho, om_c, e)
        ctry = ctry / 2
        _iter += 1
        if _iter >= _MAX_BRACKET and pow_ > pow50:
            raise RuntimeError(
                f"get_ha_param_i1: step-1 bracketing did not converge in {_MAX_BRACKET} iterations. "
                "Power may be non-monotone for this dataset."
            )

    c1 = c

    pow_ = 0.0
    ctry = get_cbar(0.01, distmat)
    _iter = 0
    while pow_ < pow50:
        c = ctry
        sigdm_c = get_sigma_dm(distmat, c)
        om_c = R.T @ sigdm_c @ R
        pow_ = get_pow_qf(om_ho, om_c, e)
        ctry = 2 * ctry
        _iter += 1
        if _iter >= _MAX_BRACKET and pow_ < pow50:
            raise RuntimeError(
                f"get_ha_param_i1: step-2 bracketing did not converge in {_MAX_BRACKET} iterations. "
                "Power may be non-monotone for this dataset."
            )

    c2 = c

    ii = 0
    while abs(pow_ - pow50) > 0.01:
        c = (c1 + c2) / 2
        sigdm_c = get_sigma_dm(distmat, c)
        om_c = R.T @ sigdm_c @ R
        pow_ = get_pow_qf(om_ho, om_c, e)

        if pow_ > pow50:
            c2 = c
        else:
            c1 = c

        ii += 1
        if ii > 20:
            break

    return c


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
    _iter = 0
    while pow_ < 0.5:
        g = gtry
        pow_ = get_pow_qf(om_ho, om_i0 + g * om_bm, e)
        gtry = g * 2
        _iter += 1
        if _iter >= _MAX_BRACKET and pow_ < 0.5:
            raise RuntimeError(
                f"get_ha_param_i0: step-2 bracketing did not converge in {_MAX_BRACKET} iterations."
            )
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
    """
    Find alternative parameter c yielding ~50% power for I(1) residual test.
    Same structure as get_ha_param_i1 but uses get_sigma_residual.
    """
    pow50 = 0.5
    pow_ = 1.0
    ctry = get_cbar(0.95, distmat)
    _MAX_BRACKET = 200

    _iter = 0
    while pow_ > pow50:
        c = ctry
        sigdm_c = get_sigma_residual(distmat, c, M)
        om_c = R.T @ sigdm_c @ R
        pow_ = get_pow_qf(om_ho, om_c, e)
        ctry = ctry / 2
        _iter += 1
        if _iter >= _MAX_BRACKET and pow_ > pow50:
            raise RuntimeError(
                f"get_ha_param_i1_residual: step-1 bracketing did not converge in {_MAX_BRACKET} iterations."
            )
    c1 = c

    pow_ = 0.0
    ctry = get_cbar(0.01, distmat)
    _iter = 0
    while pow_ < pow50:
        c = ctry
        sigdm_c = get_sigma_residual(distmat, c, M)
        om_c = R.T @ sigdm_c @ R
        pow_ = get_pow_qf(om_ho, om_c, e)
        ctry = 2 * ctry
        _iter += 1
        if _iter >= _MAX_BRACKET and pow_ < pow50:
            raise RuntimeError(
                f"get_ha_param_i1_residual: step-2 bracketing did not converge in {_MAX_BRACKET} iterations."
            )
    c2 = c

    ii = 0
    while abs(pow_ - pow50) > 0.01:
        c = (c1 + c2) / 2
        sigdm_c = get_sigma_residual(distmat, c, M)
        om_c = R.T @ sigdm_c @ R
        pow_ = get_pow_qf(om_ho, om_c, e)
        if pow_ > pow50:
            c2 = c
        else:
            c1 = c
        ii += 1
        if ii > 20:
            break

    return c


def _half_life_exceedance_counts(
    Y: np.ndarray,
    distmat: np.ndarray,
    emat: np.ndarray,
    c_grid_ho: np.ndarray,
    c_grid_ha: np.ndarray,
) -> np.ndarray:
    """
    Count strict LR exceedances over c_grid_ho using the averaged Ha likelihood.

    Returns
    -------
    ndarray (n_c_ho,)
        Number of simulated LR values above the observed LR at each c in c_grid_ho
    """
    q = emat.shape[0]
    n_c_ho = len(c_grid_ho)
    n_c_ha = len(c_grid_ha)

    rho_bm = 0.999
    c_bm = get_cbar(rho_bm, distmat)
    sigdm_bm = get_sigma_dm(distmat, c_bm)

    R = get_r(sigdm_bm, q)

    reference_shape = None
    informative = False
    ch_om_list = []
    ch_omi_list = []
    log_det_list = np.zeros(n_c_ho)

    for i in range(n_c_ho):
        c = c_grid_ho[i]
        sigdm = get_sigma_dm(distmat, c)
        om = R.T @ sigdm @ R
        shape = om / np.trace(om)
        if reference_shape is None:
            reference_shape = shape
        elif np.max(np.abs(shape - reference_shape)) > 1e-12:
            informative = True
        ch_om_list.append(cholesky_upper(om))
        omi = np.linalg.inv(om)
        ch_omi = cholesky_upper(omi)
        ch_omi_list.append(ch_omi)
        log_det_list[i] = np.sum(np.log(np.diag(ch_omi)))

    ch_omi_ha_list = []
    log_det_ha_list = np.zeros(n_c_ha)

    for i in range(n_c_ha):
        c = c_grid_ha[i]
        sigdm = get_sigma_dm(distmat, c)
        om = R.T @ sigdm @ R
        shape = om / np.trace(om)
        if np.max(np.abs(shape - reference_shape)) > 1e-12:
            informative = True
        omi = np.linalg.inv(om)
        ch_omi = cholesky_upper(omi)
        ch_omi_ha_list.append(ch_omi)
        log_det_ha_list[i] = np.sum(np.log(np.diag(ch_omi)))

    if not informative:
        raise ValueError(
            "Half-life experiment is uninformative: covariance shapes are proportional."
        )
    X = validate_and_project_y(Y, R)
    counts = np.empty(n_c_ho, dtype=np.intp)
    half_q = 0.5 * q
    log_n_c_ha = np.log(n_c_ha)

    for i in range(n_c_ho):
        ch_null = ch_om_list[i]
        e = ch_null.T @ emat

        ch_omi = ch_omi_list[i]
        log_det = log_det_list[i]
        xc = ch_omi @ e
        Xc = ch_omi @ X
        with np.errstate(divide="ignore", invalid="ignore"):
            log_den_ho = log_det - half_q * np.log(np.sum(xc**2, axis=0))
            log_den_ho_X = log_det - half_q * np.log(np.sum(Xc**2))

        log_den_ha_mat = np.empty((emat.shape[1], n_c_ha))
        log_den_ha_mat_X = np.empty(n_c_ha)

        for j in range(n_c_ha):
            ch_omi_j = ch_omi_ha_list[j]
            log_det_j = log_det_ha_list[j]
            xc_j = ch_omi_j @ e
            Xc_j = ch_omi_j @ X
            with np.errstate(divide="ignore", invalid="ignore"):
                log_den_ha_mat[:, j] = log_det_j - half_q * np.log(
                    np.sum(xc_j**2, axis=0)
                )
                log_den_ha_mat_X[j] = log_det_j - half_q * np.log(np.sum(Xc_j**2))

        with np.errstate(divide="ignore", invalid="ignore"):
            log_lr = logsumexp(log_den_ha_mat, axis=1) - log_n_c_ha - log_den_ho
            log_lr_X = logsumexp(log_den_ha_mat_X) - log_n_c_ha - log_den_ho_X

        counts[i] = np.count_nonzero(log_lr > log_lr_X)

    return counts


def spur_persistence(
    Y: np.ndarray, distmat: np.ndarray, emat: np.ndarray, level: float
) -> tuple:
    """
    Report the smallest and largest accepted points on the half-life grid.

    Parameters
    ----------
    Y : ndarray (n,)
        Variable (centered)
    distmat : ndarray (n, n)
        Normalized distance matrix
    emat : ndarray (q, nrep)
        Monte Carlo draws
    level : float
        Confidence level in [0, 1]

    Returns
    -------
    tuple (ci_lower, ci_upper)
        Bounds on half-life (normalized; i.e., fraction of max distance)
    """
    n_hl = 100
    hl_grid_ho = np.concatenate(
        [np.linspace(0.001, 1, n_hl), np.linspace(1.01, 3, 30), [100.0]]
    )

    n_hl_ha = 50
    hl_grid_ha = np.linspace(0.001, 1.0, n_hl_ha)

    c_grid_ho = -np.log(0.5) / hl_grid_ho
    c_grid_ha = -np.log(0.5) / hl_grid_ha

    counts = _half_life_exceedance_counts(Y, distmat, emat, c_grid_ho, c_grid_ha)
    nrep = emat.shape[1]
    # p > 1 - level, evaluated on the integer Monte Carlo grid without
    # subtracting the floating-point confidence level from one.
    ii = (nrep - counts) / nrep < level
    if not ii.any():
        return np.nan, np.nan

    hl_ci = hl_grid_ho[ii]
    return float(hl_ci.min()), float(hl_ci.max())
