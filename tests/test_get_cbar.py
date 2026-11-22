import numpy as np
import pytest

from spur.utils.dist import get_cbar, lvech


def test_get_cbar_matches_target_average_correlation() -> None:
    distmat = np.array(
        [
            [0.0, 0.2, 0.8],
            [0.2, 0.0, 0.6],
            [0.8, 0.6, 0.0],
        ]
    )
    rhobar = 0.5

    c = get_cbar(rhobar, distmat)
    avg_corr = np.mean(np.exp(-c * lvech(distmat)))

    assert c > 0
    assert avg_corr == pytest.approx(rhobar, abs=1e-3)


@pytest.mark.parametrize("rho", [0.03, 0.001, 0.0001])
def test_get_cbar_brackets_a_root_at_tightly_clustered_sites(rho: float) -> None:
    sites = np.r_[1e-8 * np.arange(10), 1.0]
    distances = np.abs(sites[:, None] - sites)
    decay = get_cbar(rho, distances)
    assert decay > 10_240
    assert np.mean(np.exp(-decay * lvech(distances))) == pytest.approx(rho, rel=1e-10)


@pytest.mark.parametrize("rho", [0.3, 1 / 3])
def test_get_cbar_rejects_unattainable_correlation(rho: float) -> None:
    sites = np.array([0.0, 0.0, 1.0])
    distances = np.abs(sites[:, None] - sites)
    with pytest.raises(ValueError, match="co-located-pair fraction"):
        get_cbar(rho, distances)
