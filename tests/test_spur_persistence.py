import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spur import load_chetty_data, spurhalflife, standardize
from spur.core import load_rust_backend
from spur.utils.dist import get_distance_matrix
from spur.utils.inference import spur_persistence
from tests.utils import (
    STATA,
    ensure_spur_stata_installed,
    execute_stata_command,
    stata_path,
)

HALFLIFE_ATOL = 1e-2
NREP = 100000


@pytest.fixture
def chetty_df() -> pd.DataFrame:
    df = load_chetty_data()
    df = df[~df["state"].isin(["AK", "HI"])][["am", "lat", "lon", "state"]]
    df = df.dropna(subset=["am", "lat", "lon"]).reset_index(drop=True)
    return standardize(df, ["am"])


def run_stata_spur_persistence(tmp_path: Path, df: pd.DataFrame) -> tuple[float, float]:
    stata_root = ensure_spur_stata_installed()
    plus = stata_root / "plus"
    personal = stata_root / "personal"
    input_csv = tmp_path / "spatial_persistence_input.csv"
    output_csv = tmp_path / "spatial_persistence_output.csv"

    df.to_csv(input_csv, index=False)

    script = textwrap.dedent(
        f"""
        clear all
        set more off

        sysdir set PLUS "{stata_path(plus)}"
        sysdir set PERSONAL "{stata_path(personal)}"

        import delimited using "{stata_path(input_csv)}", clear asdouble
        rename lat s_1
        rename lon s_2
        set seed 42

        spurhalflife am, q(15) nrep({NREP}) level(95) latlong normdist

        scalar ci_l = r(ci_l)
        scalar ci_u = r(ci_u)

        clear
        set obs 1
        gen double ci_lower = ci_l
        gen double ci_upper = ci_u

        export delimited using "{stata_path(output_csv)}", replace
        """
    )
    execute_stata_command(script, tmp_path)
    row = pd.read_csv(output_csv).iloc[0]
    return float(row["ci_lower"]), float(row["ci_upper"])


def test_spur_persistence_returns_ordered_interval(chetty_df: pd.DataFrame) -> None:
    coords = chetty_df[["lat", "lon"]].to_numpy()
    y = chetty_df["am"].to_numpy()
    distmat = get_distance_matrix(coords, latlon=True)
    distmat = distmat / distmat.max()
    emat = np.random.default_rng(42).standard_normal((15, 200))

    components, pvalues, points = spur_persistence(y - y.mean(), distmat, emat, 0.95)
    ci_lower, ci_upper = components[0, 0], components[-1, 1]
    assert points[0] == 0 and np.isinf(points[-1])
    assert np.all(pvalues >= 1 / 201)

    assert np.isfinite(ci_lower)
    assert 0.0 < ci_lower <= ci_upper


@pytest.mark.skipif(STATA is None, reason="stata-mp not installed")
def test_spur_persistence_matches_stata(
    tmp_path: Path,
    chetty_df: pd.DataFrame,
) -> None:
    coords = chetty_df[["lat", "lon"]].to_numpy()
    y = chetty_df["am"].to_numpy()
    distmat = get_distance_matrix(coords, latlon=True)
    distmat = distmat / distmat.max()
    emat = np.random.default_rng(42).standard_normal((15, NREP))

    components, _, _ = spur_persistence(y - y.mean(), distmat, emat, 0.95)
    py_lower, py_upper = components[0, 0], components[-1, 1]
    st_lower, st_upper = run_stata_spur_persistence(tmp_path, chetty_df)

    assert py_lower == pytest.approx(st_lower, abs=HALFLIFE_ATOL)
    if np.isnan(st_upper):
        assert np.isinf(py_upper)
    else:
        assert py_upper == pytest.approx(st_upper, abs=HALFLIFE_ATOL)


def test_half_life_continuous_inversion_against_three_site_reference(monkeypatch):
    import spur
    import spur.core as spur_core

    # Stratified angles serve as numerical quadrature for the two-dimensional
    # null, whose exact population endpoint is available in the methods spec.
    nrep = 32767
    angles = (np.arange(nrep) + 0.5) * np.pi / (2 * nrep)
    draws = np.array([np.cos(angles), np.sin(angles)])
    monkeypatch.setattr(spur_core, "_draw_emat", lambda *_: draws)
    data = pd.DataFrame({"s": [0.0, 1.0, 2.0], "y": [1.0, -6.0, 0.0]})
    backends = ["fallback"]
    if spur_core.load_rust_backend() is not None:
        backends.append("rust")
    for backend in backends:
        result = spur.spurhalflife(
            "y",
            data,
            coords_euclidean=["s"],
            q=2,
            nrep=nrep,
            normdist=True,
            backend=backend,
        )
        assert result.ci_lower == 0.0
        assert result.ci_upper == pytest.approx(7.4073897366, abs=0.25)
        assert result.components.shape == (1, 2)
        assert result.pv_mat is not None
        assert result.pv_mat[-1] < 0.05


@pytest.mark.parametrize("backend", ["fallback", "rust"])
def test_half_life_detects_a_short_disconnected_component(backend: str) -> None:
    if backend == "rust" and load_rust_backend() is None:
        pytest.skip("compatible native extension unavailable")
    data = pd.DataFrame(
        {
            "s": [0.0, 2e-6, 2.845e-6, 0.2748, 0.5083, 0.5709, 0.646, 1.0],
            "y": [
                -0.780006,
                -0.780004,
                -0.780003,
                -0.197270,
                0.238765,
                0.370591,
                0.546977,
                1.380950,
            ],
        }
    )
    result = spurhalflife(
        "y",
        data,
        coords_euclidean=["s"],
        q=3,
        nrep=8000,
        seed=2945,
        normdist=True,
        backend=backend,
        nthreads=1,
    )
    assert result.hl_grid is not None and result.pv_mat is not None
    old_first = np.searchsorted(result.hl_grid, 1 / 127)
    assert result.pv_mat[0] < 0.05 and result.pv_mat[old_first] < 0.05
    assert result.components.shape == (2, 2)
    assert 0 < result.components[0, 0] < 6e-7 < result.components[0, 1] < 4e-6
    assert result.components[1, 0] > 0.1 and np.isposinf(result.components[1, 1])


def test_spur_persistence_high_q_avoids_overflow(monkeypatch) -> None:
    import spur
    import spur.core as spur_core

    rng = np.random.default_rng(2026)
    n, q, nrep = 70, 40, 100
    s1 = rng.uniform(0.0, 1.0, size=n)
    s2 = rng.uniform(0.0, 1.0, size=n)
    df = pd.DataFrame(
        {
            "y": np.sin(3.0 * s1)
            + 0.4 * np.cos(2.0 * s2)
            + 0.1 * rng.standard_normal(n),
            "s1": s1,
            "s2": s2,
        }
    )
    emat = rng.standard_normal((q, nrep))

    monkeypatch.setattr(spur_core, "_draw_emat", lambda _s, _q, _n, e=emat: e)
    fb = spur.spurhalflife(
        "y",
        df,
        coords_euclidean=("s1", "s2"),
        q=q,
        nrep=nrep,
        level=95,
        normdist=False,
        backend="fallback",
    )
    assert np.isfinite(fb.ci_lower)
    assert 0.0 < fb.ci_lower <= fb.ci_upper
    assert fb.backend == "fallback"

    if spur_core.load_rust_backend() is None:
        return
    ru = spur.spurhalflife(
        "y",
        df,
        coords_euclidean=("s1", "s2"),
        q=q,
        nrep=nrep,
        level=95,
        normdist=False,
        backend="rust",
        nthreads=1,
    )
    assert ru.backend == "rust"
    assert ru.ci_lower == pytest.approx(fb.ci_lower, rel=1e-12, abs=1e-12)
    if np.isinf(fb.ci_upper):
        assert np.isinf(ru.ci_upper)
    else:
        assert ru.ci_upper == pytest.approx(fb.ci_upper, rel=1e-12, abs=1e-12)
    assert ru.pv_mat is not None
    assert ru.hl_grid is not None and len(ru.pv_mat) == len(ru.hl_grid) >= 129
    assert np.all(np.isfinite(ru.pv_mat))
