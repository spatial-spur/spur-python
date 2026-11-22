import numpy as np
from .dist import (
    EARTH_CIRCUMFERENCE_M,
    get_distance_matrix,
    great_circle_fraction_matrix,
    normalized_distmat,
)


def demean_matrix(mat: np.ndarray) -> np.ndarray:
    """
    Double-demean a matrix (row then column demeaning).

    Parameters
    ----------
    mat : ndarray of shape (n, n)
        Input matrix

    Returns
    -------
    ndarray of shape (n, n)
        Demeaned matrix
    """
    mat = mat - mat.mean(axis=1, keepdims=True)
    mat = mat - mat.mean(axis=0, keepdims=True)
    return mat


def get_r(sigma: np.ndarray, qmax: int) -> np.ndarray:
    """
    Get the top qmax eigenvectors of covariance matrix (sorted by eigenvalue desc).

    Canonicalizes each eigenvector's sign so its largest-magnitude entry is
    non-negative (earliest index breaks numerical ties), and verifies that the
    retained basis has positive covariance rank.

    Parameters
    ----------
    sigma : ndarray (n, n)
        Symmetric covariance matrix
    qmax : int
        Number of top eigenvectors to return

    Returns
    -------
    ndarray (n, qmax)
        Matrix of top eigenvectors
    """
    sigma_arr = np.asarray(sigma, dtype=float)
    if sigma_arr.ndim != 2 or sigma_arr.shape[0] != sigma_arr.shape[1]:
        raise ValueError("`sigma` must be a square 2D matrix.")
    n = sigma_arr.shape[0]
    if qmax < 1 or qmax > n:
        raise ValueError(f"q={qmax} must satisfy 1 <= q <= n={n}.")

    sigma_sym = 0.5 * (sigma_arr + sigma_arr.T)
    eigenvalues, eigenvectors = np.linalg.eigh(sigma_sym)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    lead = float(eigenvalues[0])
    tail = float(eigenvalues[qmax - 1])
    if (
        not np.isfinite(lead)
        or lead <= 0.0
        or not np.isfinite(tail)
        or tail <= 1e-12 * lead
    ):
        raise ValueError(
            "SPUR covariance basis has insufficient positive rank for requested q."
        )

    R = eigenvectors[:, :qmax].copy()
    for j in range(R.shape[1]):
        col = R[:, j]
        magnitude = np.abs(col)
        maximum = np.max(magnitude)
        tie_tolerance = 1e-12 * maximum
        first = np.flatnonzero(maximum - magnitude <= tie_tolerance)[0]
        if col[first] < 0.0:
            col *= -1.0
    return R


def orthonormal_design_basis(x_in: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Construct an orthonormal basis Q_x for the column space of X_in.

    Column-normalizes each regressor column safely by its L2 norm first so that
    both rank detection and residual projection M = I - Q_x @ Q_x.T are invariant
    to arbitrary positive column rescaling, then applies twice-reorthogonalized
    modified Gram-Schmidt.
    """
    x_arr = np.asarray(x_in, dtype=float)
    if x_arr.ndim != 2:
        raise ValueError("Regressor matrix X must be 2-dimensional.")
    n, k_x = x_arr.shape
    if k_x < 1 or k_x >= n:
        raise ValueError(
            f"Invalid residual test design matrix dimensions ({n}, {k_x}): "
            "must satisfy 1 <= k_x < n."
        )
    if not np.all(np.isfinite(x_arr)):
        raise ValueError(
            "Regressor matrix X contains NaN or inf values. All values must be finite."
        )

    q_mat = np.empty((n, k_x), dtype=float)
    for j in range(k_x):
        col = x_arr[:, j]
        max_abs = float(np.max(np.abs(col)))
        if not np.isfinite(max_abs) or max_abs == 0.0:
            raise ValueError(
                "Regressor matrix X is rank-deficient (collinear regressors or dummy trap). "
                "Check for zero columns, perfectly collinear columns, or a full set of category dummies."
            )
        scaled = col / max_abs
        scaled_norm = float(np.linalg.norm(scaled))
        if not np.isfinite(scaled_norm) or scaled_norm == 0.0:
            raise ValueError(
                "Regressor matrix X is rank-deficient (collinear regressors or dummy trap). "
                "Check for zero columns, perfectly collinear columns, or a full set of category dummies."
            )
        q_mat[:, j] = scaled / scaled_norm

    for j in range(k_x):
        col = q_mat[:, j]
        for _ in range(2):
            for prev in range(j):
                prev_col = q_mat[:, prev]
                col -= float(np.dot(prev_col, col)) * prev_col
        norm = float(np.linalg.norm(col))
        if not np.isfinite(norm) or norm <= tol:
            raise ValueError(
                "Regressor matrix X is rank-deficient (collinear regressors or dummy trap). "
                "Check for perfectly collinear columns or a full set of category dummies."
            )
        col /= norm

    return q_mat


def validate_y_variation(y: np.ndarray) -> np.ndarray:
    """Validate response vector is finite and non-constant, returning demeaned values."""
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if y_arr.size == 0 or not np.all(np.isfinite(y_arr)):
        raise ValueError(
            "Response Y contains NaN or inf values. All values must be finite."
        )
    if float(np.ptp(y_arr)) == 0.0:
        raise ValueError(
            "Response Y has zero variation; SPUR inference requires non-constant Y."
        )
    y_dm = y_arr - float(np.mean(y_arr))
    max_d = float(np.max(np.abs(y_dm)))
    if not np.isfinite(max_d) or max_d == 0.0:
        raise ValueError(
            "Response Y is effectively constant; SPUR inference requires variation in Y."
        )
    return y_dm


def validate_and_project_y(y: np.ndarray, basis_r: np.ndarray) -> np.ndarray:
    """Validate response variation and project demeaned Y onto SPUR basis R."""
    y_dm = validate_y_variation(y)
    max_d = float(np.max(np.abs(y_dm)))
    y_dm = y_dm / max_d
    y_dm_norm = float(np.linalg.norm(y_dm))
    p_vec = basis_r.T @ y_dm
    p_max = float(np.max(np.abs(p_vec)))
    if not np.isfinite(p_max) or p_max == 0.0:
        raise ValueError(
            "Projected SPUR vector R'Y has zero norm; response has no variation in the SPUR basis."
        )
    p_norm = p_max * float(np.linalg.norm(p_vec / p_max))
    if not np.isfinite(p_norm) or p_norm <= 1e-12 * y_dm_norm:
        raise ValueError(
            "Projected SPUR vector R'Y has zero norm; response has negligible variation in the SPUR basis."
        )
    return p_vec


def validate_constant_span(design_basis: np.ndarray) -> None:
    """Require the nuisance space used by the residual LBM approximation."""
    n = design_basis.shape[0]
    constant = np.full(n, 1.0 / np.sqrt(n))
    for _ in range(2):
        constant -= design_basis @ (design_basis.T @ constant)
    # Squared norm bounds the fraction of the constant covariance left by projection.
    if float(constant @ constant) > np.finfo(float).eps:
        raise ValueError("SPUR residual tests require a design that spans a constant.")


def validate_and_project_residual_y(
    y: np.ndarray, basis_r: np.ndarray, design_basis: np.ndarray
) -> np.ndarray:
    """Residualize finite non-constant Y against Q_x and project onto SPUR basis R."""
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if y_arr.size == 0 or not np.all(np.isfinite(y_arr)):
        raise ValueError(
            "Response Y contains NaN or inf values. All values must be finite."
        )
    if np.all(y_arr == y_arr[0]):
        raise ValueError(
            "Response Y has zero variation; SPUR inference requires non-constant Y."
        )

    # The nuisance space spans a constant, so remove it before scaling.
    # Scaling first could otherwise erase useful variation near a large offset.
    with np.errstate(over="ignore"):
        centered = y_arr - y_arr[0]
    if not np.isfinite(centered).all():
        scaled = y_arr / np.max(np.abs(y_arr))
        centered = scaled - scaled[0]

    col_scaled = centered / np.max(np.abs(centered))
    col_norm = float(np.linalg.norm(col_scaled))
    u = col_scaled - design_basis @ (design_basis.T @ col_scaled)
    u = u - design_basis @ (design_basis.T @ u)
    u_max = float(np.max(np.abs(u)))
    u_norm = float(np.linalg.norm(u))
    if (
        not np.isfinite(u_max)
        or u_max == 0.0
        or not np.isfinite(u_norm)
        or u_norm <= 1e-12 * col_norm
    ):
        raise ValueError(
            "Residualized response Y has zero variation; SPUR inference requires variation orthogonal to X."
        )
    u_scaled = u / u_max
    u_scaled_norm = u_norm / u_max
    p_vec = basis_r.T @ u_scaled
    p_max = float(np.max(np.abs(p_vec)))
    if not np.isfinite(p_max) or p_max == 0.0:
        raise ValueError(
            "Projected SPUR vector R'Y has zero norm; response has no variation in the SPUR basis."
        )
    p_norm = p_max * float(np.linalg.norm(p_vec / p_max))
    if not np.isfinite(p_norm) or p_norm <= 1e-12 * u_scaled_norm:
        raise ValueError(
            "Projected SPUR vector R'Y has zero norm; response has negligible variation in the SPUR basis."
        )
    return p_vec


def cholesky_upper(M: np.ndarray) -> np.ndarray:
    """
    Get upper triangular Cholesky factor.

    NumPy returns a lower-triangular factor, while R `chol()` already returns
    the upper-triangular factor used in the SPUR translations. This helper
    bridges that library convention difference in Python.
    """
    L = np.linalg.cholesky(M)
    return L.T


def get_sigma_lbm(distmat: np.ndarray) -> np.ndarray:
    """
    Compute the Lévy Brownian Motion (LBM) covariance matrix.

    Uses the first observation as the origin. The covariance between
    locations i and j is: 0.5 * (d(i,0) + d(j,0) - d(i,j))

    Parameters
    ----------
    distmat : ndarray of shape (n, n)
        Normalized distance matrix (max distance = 1)

    Returns
    -------
    ndarray of shape (n, n)
        LBM covariance matrix
    """
    sigma_lbm = 0.5 * (distmat[:, 0:1] + distmat[0:1, :] - distmat)
    return sigma_lbm


def get_sigma_dm(distmat: np.ndarray, c: float) -> np.ndarray:
    """
    Compute demeaned exponential covariance matrix.

    sigma(i,j) = exp(-c * distmat(i,j)), then double-demean.
    """
    return demean_matrix(np.expm1(-c * distmat))


def get_sigma_residual(distmat: np.ndarray, c: float, M: np.ndarray) -> np.ndarray:
    """
    Compute residualized covariance matrix: M @ exp(-c * distmat) @ M.T

    Used for I(1) and I(0) residual tests. The projection matrix M already
    accounts for the regressors, so we project the spatial covariance.
    """
    # Project the exponential deviation without first subtracting order-one
    # matrices. Restore the constant term for general projection matrices.
    constant = M.sum(axis=1)
    return M @ np.expm1(-c * distmat) @ M.T + np.outer(constant, constant)


def nn_matrix(coords: np.ndarray, latlon: bool = True) -> np.ndarray:
    """
    Compute nearest-neighbor transformation matrix.

    Difference each observation from the average at the nearest distinct
    location or locations. Observations at zero distance are excluded.

    Parameters
    ----------
    coords : ndarray of shape (n, 2)
        Coordinates array
    latlon : bool, default True
        If True, use Haversine distance; if False, use Euclidean

    Returns
    -------
    ndarray of shape (n, n)
        Transformation matrix M = I - W where W is the normalized
        nearest-neighbor weight matrix
    """
    n = coords.shape[0]
    distmat = get_distance_matrix(coords, latlon=latlon)

    distmat[distmat == 0.0] = np.inf
    min_dist = np.min(distmat, axis=1, keepdims=True)
    if not np.isfinite(min_dist).all():
        raise ValueError(
            "Nearest-neighbor transform requires at least two distinct locations."
        )
    NN = (distmat == min_dist).astype(float)
    row_sums = NN.sum(axis=1, keepdims=True)
    NN = NN / row_sums
    M = np.eye(n) - NN

    return M


def iso_matrix(coords: np.ndarray, radius: float, latlon: bool = True) -> np.ndarray:
    """
    Compute isotropic transformation matrix.

    For each observation, identifies all distinct-location neighbors within
    the specified inclusive radius (`0 < d <= radius`) and creates a
    row-normalized weight matrix. The transformation matrix is I - W, which
    differences each observation from the average of its neighbors.

    Parameters
    ----------
    coords : ndarray of shape (n, 2)
        Coordinates array
    radius : float
        Radius threshold for neighbor inclusion. Units depend on latlon:
        - If latlon=True: radius in meters
        - If latlon=False: radius in same units as coordinates
    latlon : bool, default True
        If True, use Haversine distance; if False, use Euclidean

    Returns
    -------
    ndarray of shape (n, n)
        Transformation matrix M = I - W where W is the normalized
        isotropic weight matrix

    Notes
    -----
    Observations with no neighbors within the radius have their transformed
    value set to 0 (matching Stata's behavior). The entire row of M is set
    to zeros for these observations.
    """
    n = coords.shape[0]
    if latlon:
        fraction = great_circle_fraction_matrix(coords)
        neighbors = (fraction <= float(radius) / EARTH_CIRCUMFERENCE_M) & (
            fraction > 0.0
        )
    else:
        distmat = get_distance_matrix(coords, latlon=False)
        neighbors = (distmat <= radius) & (distmat > 0.0)
    row_sums = neighbors.sum(axis=1, keepdims=True).astype(float)
    no_neighbors = (row_sums == 0).flatten()
    row_sums[row_sums == 0] = 1
    W = neighbors.astype(float) / row_sums
    M = np.eye(n) - W
    M[no_neighbors, :] = 0

    return M


def lbmgls_matrix(coords: np.ndarray, latlon: bool = True) -> np.ndarray:
    """
    Compute the LBM-GLS transformation matrix.

    This is the default transformation recommended by Muller & Watson (2024).
    It uses GLS based on the covariance matrix of a Lévy-Brownian motion.

    Parameters
    ----------
    coords : ndarray of shape (n, 2)
        Coordinates array
    latlon : bool, default True
        If True, use Haversine distance; if False, use Euclidean

    Returns
    -------
    ndarray of shape (n, n)
        LBM-GLS transformation matrix

    Notes
    -----
    Algorithm:
    1. Compute normalized distance matrix (max = 1)
    2. Compute LBM covariance matrix
    3. Double-demean the covariance matrix
    4. Eigendecomposition
    5. GLS transform: V @ diag(1/sqrt(eigenvalues)) @ V'
    """
    return lbmgls_matrix_from_distances(normalized_distmat(coords, latlon=latlon))


def lbmgls_matrix_from_distances(distmat: np.ndarray) -> np.ndarray:
    """Build LBM-GLS weights from an already normalized distance matrix."""
    small = 1e-10
    sigma_lbm = get_sigma_lbm(distmat)
    sigma_lbm_dm = demean_matrix(sigma_lbm)

    eigenvalues, eigenvectors = np.linalg.eigh(sigma_lbm_dm)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    mask = eigenvalues > small
    eigenvalues = eigenvalues[mask]
    eigenvectors = eigenvectors[:, mask]
    dsi = 1.0 / np.sqrt(eigenvalues)
    return (eigenvectors * dsi) @ eigenvectors.T


def cluster_matrix(cluster: np.ndarray) -> np.ndarray:
    """
    Compute cluster demeaning transformation matrix.

    This is equivalent to within-cluster demeaning (like fixed effects).
    Each observation is differenced from its cluster mean.

    Parameters
    ----------
    cluster : ndarray of shape (n,)
        Cluster identifiers for each observation

    Returns
    -------
    ndarray of shape (n, n)
        Cluster transformation matrix
    """
    n = len(cluster)
    clust_mat = (cluster.reshape(-1, 1) == cluster.reshape(1, -1)).astype(float)
    clust_mat = clust_mat / clust_mat.sum(axis=1, keepdims=True)
    M = np.eye(n) - clust_mat

    return M
