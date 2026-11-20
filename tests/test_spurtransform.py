import textwrap
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from spur import spurtransform
from spur.utils.matrix import transform
from tests.config import ATOL, PARITY_ATOL, PARITY_RTOL
from tests.utils import (
    STATA,
    ensure_spur_stata_installed,
    execute_stata_command,
    stata_path,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "lat": rng.uniform(45, 55, 10),
            "lon": rng.uniform(5, 15, 10),
            "y": rng.standard_normal(10),
            "x": rng.standard_normal(10),
        }
    )


def run_stata_spurtransform(tmp_path: Path, df: pd.DataFrame) -> pd.DataFrame:
    stata_root = ensure_spur_stata_installed()
    plus = stata_root / "plus"
    personal = stata_root / "personal"
    input_csv = tmp_path / "spurtransform_input.csv"
    output_csv = tmp_path / "spurtransform_output.csv"

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

        spurtransform y x, prefix(d_) transformation(lbmgls) latlong

        export delimited using "{stata_path(output_csv)}", replace
        """
    )
    execute_stata_command(script, tmp_path)

    return pd.read_csv(output_csv)


def test_spurtransform_adds_columns_without_mutating_inputs(
    sample_df: pd.DataFrame,
) -> None:
    y_before = sample_df["y"].copy()

    out = spurtransform(
        "y ~ x",
        sample_df,
        lon="lon",
        lat="lat",
        transformation="nn",
        prefix="nn_",
    )

    assert "nn_y" in out.columns
    assert "nn_x" in out.columns
    assert len(out) == len(sample_df)
    npt.assert_array_equal(sample_df["y"].to_numpy(), y_before.to_numpy())


def test_spurtransform_supports_literal_names_and_empty_right_side():
    df = pd.DataFrame({"outcome level": [2.0, 5.0, 9.0], "site": [0.0, 1.0, 5.0]})
    result = spurtransform(
        "outcome level ~ ",
        df,
        transformation="nn",
        coords_euclidean=["site"],
        backend="fallback",
    )
    npt.assert_array_equal(result["h_outcome level"], [-3.0, 3.0, 4.0])


def test_spurtransform_preserves_literal_intercept_and_captures_helpers():
    df = pd.DataFrame(
        {"y": [2.0, 5.0, 9.0], "Intercept": [1.0, 3.0, 4.0], "site": [0.0, 1.0, 5.0]}
    )

    def square(a):
        return a**2

    result = spurtransform(
        "square(y) ~ 0 + Intercept",
        df,
        transformation="nn",
        coords_euclidean=["site"],
        backend="fallback",
    )
    npt.assert_array_equal(result["h_square(y)"], [-21.0, 21.0, 56.0])
    npt.assert_array_equal(result["h_Intercept"], [-2.0, 2.0, 1.0])
    with pytest.raises(AssertionError, match="Variable not found"):
        spurtransform(
            "y ~ allow_no_response", df, coords_euclidean=["site"], backend="fallback"
        )


def test_spurtransform_rejects_missing_coordinates() -> None:
    df = pd.DataFrame(
        {
            "lat": [45.0, np.nan, 47.0],
            "lon": [10.0, 11.0, 12.0],
            "y": [1.0, 2.0, 3.0],
        }
    )

    with pytest.raises(AssertionError, match="finite"):
        spurtransform("y ~ 1", df, lon="lon", lat="lat", transformation="nn")


def test_spurtransform_rejects_missing_variable(sample_df: pd.DataFrame) -> None:
    with pytest.raises(AssertionError, match="not found"):
        spurtransform(
            "missing ~ 1", sample_df, lon="lon", lat="lat", transformation="nn"
        )


def test_spurtransform_evaluates_terms_on_one_complete_case_sample() -> None:
    df = pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0], "x": [1.0, np.nan, 3.0, 5.0], "g": ["a"] * 4},
        index=[10, 10, 20, 10],
    )
    out = spurtransform(
        "np.log(y) ~ I(x ** 2)", df, transformation="cluster", clustvar="g"
    )
    y = np.log([1.0, 3.0, 4.0])
    x2 = np.array([1.0, 9.0, 25.0])
    assert out.index.equals(df.index)
    npt.assert_allclose(out["h_np.log(y)"], np.insert(y - np.mean(y), 1, np.nan))
    npt.assert_allclose(out["h_I(x ** 2)"], np.insert(x2 - np.mean(x2), 1, np.nan))


def test_spurtransform_matches_direct_transform(
    sample_df: pd.DataFrame,
) -> None:
    out = spurtransform(
        "y ~ 1",
        sample_df,
        lon="lon",
        lat="lat",
        transformation="nn",
        prefix="nn_",
    )
    coords = sample_df[["lat", "lon"]].to_numpy()
    direct = transform(sample_df["y"].to_numpy(), coords, method="nn", latlon=True)

    npt.assert_allclose(out["nn_y"].to_numpy(), direct, atol=1e-14, rtol=0.0)


def test_spurtransform_cluster_demeans_within_groups() -> None:
    df = pd.DataFrame(
        {
            "lat": [0.0] * 6,
            "lon": [0.0] * 6,
            "y": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
            "grp": [1, 1, 1, 2, 2, 2],
        }
    )

    out = spurtransform(
        "y ~ 1",
        df,
        transformation="cluster",
        clustvar="grp",
        prefix="cl_",
    )

    for group in [1, 2]:
        group_result = out.loc[df["grp"] == group, "cl_y"]
        assert group_result.sum() == pytest.approx(0.0, abs=ATOL)


def test_spurtransform_rejects_nullable_string_clusters_with_missing() -> None:
    df = pd.DataFrame(
        {
            "lat": [0.0] * 6,
            "lon": [0.0] * 6,
            "y": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
            "province": pd.array(
                ["Zurich", "Zurich", pd.NA, "Bern", "Bern", "Bern"],
                dtype="string",
            ),
        }
    )

    with pytest.raises(AssertionError, match="missing"):
        spurtransform(
            "y ~ 1",
            df,
            transformation="cluster",
            clustvar="province",
        )


def test_spurtransform_accepts_string_cluster_labels() -> None:
    df = pd.DataFrame(
        {
            "lat": [0.0] * 6,
            "lon": [0.0] * 6,
            "y": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
            "province": pd.array(
                ["Zurich", "Zurich", "Zurich", "Bern", "Bern", "Bern"],
                dtype="string",
            ),
        }
    )

    out = spurtransform(
        "y ~ 1",
        df,
        transformation="cluster",
        clustvar="province",
        prefix="cl_",
    )

    for group in ["Zurich", "Bern"]:
        group_result = out.loc[df["province"] == group, "cl_y"]
        assert group_result.sum() == pytest.approx(0.0, abs=ATOL)


def test_spurtransform_cluster_works_without_coordinate_columns_present() -> None:
    df = pd.DataFrame(
        {
            "y": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
            "grp": ["A", "A", "A", "B", "B", "B"],
        }
    )

    out = spurtransform("y ~ 1", df, transformation="cluster", clustvar="grp")

    for group in ["A", "B"]:
        assert out.loc[df["grp"] == group, "h_y"].sum() == pytest.approx(0.0, abs=ATOL)


def test_spurtransform_accepts_euclidean_coordinates() -> None:
    df = pd.DataFrame(
        {
            "xcoord": [0.0, 1.0, 2.0, 3.0],
            "ycoord": [0.0, 1.0, 0.0, 1.0],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )

    out = spurtransform(
        "y ~ 1",
        df,
        coords_euclidean=["xcoord", "ycoord"],
        transformation="nn",
        prefix="e_",
    )

    assert "e_y" in out.columns
    assert len(out) == len(df)


@pytest.mark.skipif(STATA is None, reason="stata-mp not installed")
def test_spurtransform_matches_stata(
    tmp_path: Path,
    sample_df: pd.DataFrame,
) -> None:
    py_value = spurtransform(
        "y ~ x",
        sample_df,
        lon="lon",
        lat="lat",
        transformation="lbmgls",
        prefix="d_",
    )
    st_value = run_stata_spurtransform(tmp_path, sample_df)

    npt.assert_allclose(
        py_value["d_y"].to_numpy(),
        st_value["d_y"].to_numpy(),
        atol=PARITY_ATOL,
        rtol=PARITY_RTOL,
    )
    npt.assert_allclose(
        py_value["d_x"].to_numpy(),
        st_value["d_x"].to_numpy(),
        atol=PARITY_ATOL,
        rtol=PARITY_RTOL,
    )
