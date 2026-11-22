from __future__ import annotations
from collections.abc import Sequence
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


EARTH_RADIUS_M = 6_371_000.009
EARTH_CIRCUMFERENCE_M = 2.0 * np.pi * EARTH_RADIUS_M


def resolve_spur_coords(
    data: pd.DataFrame,
    use_rows: np.ndarray,
    lon: str | None = None,
    lat: str | None = None,
    coords_euclidean: Sequence[str] | None = None,
) -> dict:
    """Resolve the public coordinate inputs to SPUR's internal coordinate payload.

    This is the bridge between the harmonized public API (`lon`/`lat` or
    `coords_euclidean`) and the current SPUR numerical code.
    """
    use_geodesic = lon is not None or lat is not None
    use_euclidean = coords_euclidean is not None

    if use_geodesic and use_euclidean:
        raise ValueError("Specify either `lon`/`lat` or `coords_euclidean`, not both.")
    if not use_geodesic and not use_euclidean:
        raise ValueError("Specify coordinates via `lon`/`lat` or `coords_euclidean`.")

    if use_geodesic:
        if lon is None or lat is None:
            raise ValueError(
                "Both `lon` and `lat` must be specified for geodesic coordinates."
            )
        missing = [name for name in (lon, lat) if name not in data.columns]
        if missing:
            raise ValueError(
                f"Coordinate variables not found in data: {', '.join(missing)}"
            )

        coords_lon_lat = data.loc[use_rows, [lon, lat]]
        if not all(np.issubdtype(dtype, np.number) for dtype in coords_lon_lat.dtypes):
            raise ValueError("`lon` and `lat` must reference numeric columns.")
        arr = coords_lon_lat.to_numpy(dtype=float)
        if not np.isfinite(arr).all():
            raise ValueError("Geodesic coordinates must be finite.")
        if not ((coords_lon_lat[lon] >= -180) & (coords_lon_lat[lon] <= 180)).all():
            raise ValueError("Longitude values must be in [-180, 180].")
        if not ((coords_lon_lat[lat] >= -90) & (coords_lon_lat[lat] <= 90)).all():
            raise ValueError("Latitude values must be in [-90, 90].")
        return {
            "coords": data.loc[use_rows, [lat, lon]].to_numpy(dtype=float),
            "latlong": True,
        }

    if (
        isinstance(coords_euclidean, str)
        or coords_euclidean is None
        or len(coords_euclidean) < 1
    ):
        raise ValueError(
            "`coords_euclidean` must be a sequence with at least one column name."
        )

    missing = [name for name in coords_euclidean if name not in data.columns]
    if missing:
        raise ValueError(
            f"Coordinate variables not found in data: {', '.join(missing)}"
        )

    coords = data.loc[use_rows, list(coords_euclidean)]
    if not all(np.issubdtype(dtype, np.number) for dtype in coords.dtypes):
        raise ValueError("`coords_euclidean` columns must be numeric.")
    arr = coords.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("Euclidean coordinates must be finite.")

    return {"coords": arr, "latlong": False}


def haversine_distance(
    lat1: ArrayLike, lon1: ArrayLike, lat2: ArrayLike, lon2: ArrayLike
) -> np.ndarray:
    """
    Compute great-circle distance between points using Haversine formula.

    Parameters
    ----------
    lat1, lon1 : array-like
        Latitude and longitude of first point(s) in degrees; accepts scalars or arrays.
    lat2, lon2 : array-like
        Latitude and longitude of second point(s) in degrees; accepts scalars or arrays.

    Returns
    -------
    ndarray
        Distance in meters
    """
    return EARTH_CIRCUMFERENCE_M * great_circle_fraction(lat1, lon1, lat2, lon2)


def great_circle_fraction(
    lat1: ArrayLike, lon1: ArrayLike, lat2: ArrayLike, lon2: ArrayLike
) -> np.ndarray:
    """Compute great-circle distance as a fraction of Earth circumference."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    lam1, lam2 = np.radians(lon1), np.radians(lon2)
    lat_sin = np.sin((phi2 - phi1) * 0.5)
    lon_sin = np.sin((lam2 - lam1) * 0.5)
    a = lat_sin * lat_sin + np.cos(phi1) * np.cos(phi2) * lon_sin * lon_sin
    # The shared core uses the reciprocal of pi for the inclusive ISO cutoff.
    return np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))) * (1.0 / np.pi)


def great_circle_fraction_matrix(coords: np.ndarray) -> np.ndarray:
    """Compute pairwise great-circle fractions for latitude and longitude."""
    lat, lon = coords.T
    return great_circle_fraction(lat[:, None], lon[:, None], lat, lon)


def get_distance_matrix(coords: np.ndarray, latlon: bool = True) -> np.ndarray:
    """
    Compute pairwise distance matrix between all observations.

    Parameters
    ----------
    coords : ndarray of shape (n, 2)
        Coordinates array. If latlon=True, columns are [latitude, longitude].
        If latlon=False, columns are [x, y].
    latlon : bool, default True
        If True, use Haversine formula for great-circle distance (in meters).
        If False, use Euclidean distance.

    Returns
    -------
    ndarray of shape (n, n)
        Symmetric distance matrix where entry (i,j) is distance between
        observations i and j.
    """
    n = coords.shape[0]

    if latlon:
        if n > 10_000:
            raise ValueError(
                f"n={n} exceeds the safe limit for full n×n haversine distance computation "
                f"(limit: 10,000). At this size the matrix requires >{n**2 * 8 // 1_000_000} MB "
                "and will likely OOM. Use a blocked/sparse implementation for large datasets."
            )
        lat, lon = coords.T
        distmat = haversine_distance(lat[:, None], lon[:, None], lat, lon)
    else:
        from scipy.spatial.distance import cdist

        # Squaring distances is faster at ordinary scales. When coordinates
        # are large or adjacencies may be subnormal after squaring, use hypot.
        coords = np.asarray(coords, dtype=float)
        mag = np.abs(coords)
        finfo = np.finfo(float)
        large = np.sqrt(finfo.max / coords.shape[1]) / 2
        small = 2 * np.sqrt(finfo.tiny) / finfo.eps
        if np.any(mag > large) or np.any((mag > 0) & (mag < small)):
            distmat = np.zeros((n, n), dtype=float)
            for axis in coords.T:
                for start in range(0, n, 256):
                    stop = start + 256
                    block = distmat[start:stop]
                    differences = axis[start:stop, None] - axis[None, :]
                    np.hypot(block, differences, out=block)
        else:
            distmat = cdist(coords, coords, metric="euclidean")

    return distmat


def normalized_distmat(coords: np.ndarray, latlon: bool = True) -> np.ndarray:
    """Get distance matrix normalized so max distance = 1."""
    distmat = get_distance_matrix(coords, latlon=latlon)
    max_dist = float(distmat.max())
    if not np.isfinite(max_dist) or max_dist <= 0.0:
        raise ValueError(
            "All coordinates are identical; distance normalization "
            "requires distinct locations."
        )
    return distmat / max_dist


def lvech(S: np.ndarray) -> np.ndarray:
    """Extract lower triangular part (below diagonal) as vector."""
    n = S.shape[0]
    i, j = np.tril_indices(n, k=-1)
    return S[i, j]


def get_cbar(rhobar: float, distmat: np.ndarray) -> float:
    """
    Bisection method to find c such that mean(exp(-c*d)) = rhobar.

    Parameters
    ----------
    rhobar : float
        Target average correlation
    distmat : ndarray
        Distance matrix

    Returns
    -------
    float
        c value
    """
    vd = lvech(distmat)

    c0 = 10.0
    c1 = 10.0

    i1 = False
    jj = 0
    while not i1:
        v = np.mean(np.exp(-c0 * vd))
        i1 = v > rhobar
        if not i1:
            c1 = c0
            c0 = c0 / 2
            jj += 1
        if jj > 500:
            raise ValueError("rhobar too large")

    i1 = False
    jj = 0
    while not i1:
        v = np.mean(np.exp(-c1 * vd))
        i1 = v < rhobar
        if not i1:
            c0 = c1
            c1 = 2 * c1
            jj += 1
        if c1 > 10000:
            i1 = True
        if jj > 500:
            raise ValueError("rhobar too small")

    while (c1 - c0) > 0.001:
        cm = np.sqrt(c0 * c1)
        v = np.mean(np.exp(-cm * vd))
        if v < rhobar:
            c1 = cm
        else:
            c0 = cm

    return np.sqrt(c0 * c1)
