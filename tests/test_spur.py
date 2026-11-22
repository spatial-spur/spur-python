from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import spur.core as pipeline
from spur import (
    Fits,
    PipelineResult,
    RegressionResult,
    TestResult,
    Tests,
    spur,
)


@pytest.mark.parametrize("engine", ["patsy", "formulaic"])
def test_spur_uses_intercept_from_real_formula_engine(monkeypatch, engine):
    import statsmodels.formula

    if engine == "formulaic":
        pytest.importorskip("formulaic")
    options = getattr(statsmodels.formula, "options", None)
    if options is not None:
        monkeypatch.setattr(options, "formula_engine", engine)
    elif engine != "patsy":
        pytest.skip("statsmodels 0.15 or later is required for Formulaic")
    from tests.config import BENCH_200

    df = pd.read_csv(BENCH_200).iloc[:120]
    result = spur(
        "y ~ x1",
        df,
        lon="s_1",
        lat="s_2",
        q=8,
        nrep=400,
        seed=42,
        backend="fallback",
    )
    assert result.fits.levels.model.model.k_constant == 1
    assert result.fits.transformed.model.model.exog_names == ["Intercept", "h_x1"]


def test_real_model_specs_distinguish_intercept_from_named_regressor():
    from patsy.highlevel import dmatrix
    from spur.utils.formula import evaluate_formula, intercept_columns

    df = pd.DataFrame({"y": [1, 2, 4], "Intercept": [2, 3, 8], "x": [0, 1, 2]})
    with pytest.raises(ValueError, match="Invalid formula"):
        evaluate_formula("y ~ (", df, allow_no_response=True)
    formulaic = pytest.importorskip("formulaic")
    for rhs, expected in [("x", slice(0, 1)), ("0 + Intercept", None)]:
        patsy_spec = dmatrix(rhs, df).design_info
        formulaic_spec = formulaic.model_matrix(rhs, df).model_spec
        assert intercept_columns(patsy_spec) == expected
        assert intercept_columns(formulaic_spec) == expected


def test_spur_evaluates_formula_once_on_matched_sample(monkeypatch) -> None:
    rng = np.random.default_rng(441)
    n = 30
    df = pd.DataFrame(
        {
            "sx": rng.uniform(size=n),
            "sy": rng.uniform(size=n),
            "y": np.exp(rng.normal(size=n)),
            "x": np.exp(rng.normal(size=n)),
            "Q": np.arange(n),
        },
        index=np.repeat(np.arange(n // 2), 2),
    )
    # Missing coordinates on a row omitted by the formula do not enter any stage.
    df.iloc[4, [0, 3]] = np.nan
    keep = np.arange(n) != 4
    draws_seen = []
    tests_seen = {}

    prep_calls = []
    prepare = pipeline._prepare_fallback_spur_basis

    def record_preparation(response, distances, q, design_basis, **kwargs):
        basis = prepare(response, distances, q, design_basis, **kwargs)
        prep_calls.append((response, distances, basis))
        return basis

    def fake_test(test_type, distances, draws, basis, **kwargs):
        tests_seen[test_type] = basis
        draws_seen.append(draws)
        return TestResult(test_type, 1.0, 1.0, np.array([1.0, 2.0, 3.0]), 0.01)

    monkeypatch.setattr(pipeline, "_prepare_fallback_spur_basis", record_preparation)
    monkeypatch.setattr(pipeline, "_fallback_spur_test", fake_test)

    scpc_calls = []

    # Wrappers can forward dispatch options through **kwargs.
    def fake_scpc(model, data, **kwargs):
        scpc_calls.append((model, data, kwargs))
        return SimpleNamespace(backend="fallback", fallback_reason=None, tied=False)

    monkeypatch.setattr(pipeline, "scpc", fake_scpc)
    from numpy import log

    result = spur(
        "log(y) ~ log(x)",
        df,
        coords_euclidean=("sx", "sy"),
        q=3,
        nrep=20,
        seed=42,
        backend="fallback",
    )

    assert isinstance(result, PipelineResult)
    assert set(tests_seen) == {"i0", "i1", "i0resid", "i1resid"}
    assert all(draws is draws_seen[0] for draws in draws_seen)
    expected_y = log(df["y"].to_numpy()[keep])
    expected_x = log(df["x"].to_numpy()[keep])
    assert len(prep_calls) == 2
    assert prep_calls[0][1] is prep_calls[1][1]
    assert tests_seen["i0"] is tests_seen["i1"] is prep_calls[0][2]
    assert tests_seen["i0resid"] is tests_seen["i1resid"] is prep_calls[1][2]
    for response, distances, _ in prep_calls:
        np.testing.assert_allclose(response, expected_y)
        assert distances.shape == (n - 1, n - 1)
    h = pipeline.lbmgls_matrix(df[["sx", "sy"]].to_numpy()[keep], latlon=False)
    assert result.fits.levels.model.nobs == result.fits.transformed.model.nobs == n - 1
    np.testing.assert_allclose(
        result.fits.transformed.model.model.endog, h @ expected_y
    )
    np.testing.assert_allclose(
        result.fits.transformed.model.model.exog[:, 1], h @ expected_x
    )
    assert len(scpc_calls) == 2
    for _, frame, kwargs in scpc_calls:
        assert len(frame) == n - 1
        assert kwargs["coords_euclidean"] == ("sx", "sy")
        assert kwargs["backend"] == "fallback"
    assert set(result.stage_backends.values()) == {"fallback"}
    assert result.fallback_reasons == {}


def test_spur_preserves_simple_transformed_formula_and_records_stage_backends(
    monkeypatch,
) -> None:
    rng = np.random.default_rng(600)
    df = pd.DataFrame(rng.normal(size=(12, 4)), columns=["sx", "sy", "y", "x"])
    native_output = {
        "lr": [1.0],
        "pvalue": [0.5],
        "cvalue": [1.0, 2.0, 3.0],
        "ha_param": 0.1,
        "pvalue_mat": [0.5],
        "cvalue_mat": [[1.0, 2.0, 3.0]],
        "rho_grid": [0.001],
        "tied": False,
    }

    def fake_pipeline(*args, **kwargs):
        return {
            "i0": native_output,
            "i1": native_output,
            "i0resid": dict(native_output, tied=True),
            "i1resid": dict(native_output, tied=True),
            "transformed_vars": args[-1],
        }

    monkeypatch.setattr(
        pipeline,
        "load_rust_backend",
        lambda: SimpleNamespace(run_spur_pipeline=fake_pipeline),
    )

    def fake_scpc(model, data, **kwargs):
        assert kwargs["backend"] == "auto"
        assert kwargs["nthreads"] == 1
        return SimpleNamespace(backend="rust", fallback_reason=None, tied=False)

    monkeypatch.setattr(pipeline, "scpc", fake_scpc)
    result = spur(
        "y ~ x",
        df,
        coords_euclidean=("sx", "sy"),
        q=2,
        nrep=10,
        backend="auto",
        nthreads=1,
    )

    assert result.fits.levels.model.model.formula == "y ~ x"
    assert result.fits.transformed.model.model.formula == "h_y ~ h_x"
    assert result.backend == "rust"
    assert result.tied
    assert result.stage_backends == {
        "i0": "rust",
        "i1": "rust",
        "i0resid": "rust",
        "i1resid": "rust",
        "transform": "rust",
        "levels_scpc": "rust",
        "transformed_scpc": "rust",
    }
    assert not result.fallback_reasons
    assert not result.tests.i0.tied and result.tests.i0resid.tied


def test_pipeline_requires_an_intercept_term():
    rng = np.random.default_rng(95)
    df = pd.DataFrame(rng.normal(size=(20, 3)), columns=["y", "Intercept", "sx"])
    with pytest.raises(ValueError, match="requires a formula with an intercept"):
        spur(
            "y ~ 0 + Intercept",
            df,
            coords_euclidean=["sx"],
            q=8,
            nrep=512,
            backend="fallback",
        )


@pytest.mark.parametrize("test_type", ["i0resid", "i1resid"])
@pytest.mark.parametrize("wrapper", [False, True])
def test_residual_formula_uses_callers_namespace(monkeypatch, test_type, wrapper):
    df = pd.DataFrame(
        {"y": [1.0, 3.0, 5.0], "x": [2.0, 3.0, 4.0], "sx": [0.0, 1.0, 2.0]}
    )

    def shift(a):
        return a + 4

    def check_design(kind, coords, latlong, y, *args, x_in):
        assert kind == test_type
        np.testing.assert_array_equal(y, [5.0, 7.0, 9.0])
        np.testing.assert_array_equal(x_in, [[1.0, 6.0], [1.0, 7.0], [1.0, 8.0]])
        return TestResult(test_type, 1.0, 0.5, np.array([1.0, 2.0, 3.0]), 0.01)

    monkeypatch.setattr(pipeline, "_run_spur_test", check_design)
    if wrapper:
        pipeline.spurtest(
            "shift(y) ~ shift(x)", df, coords_euclidean=["sx"], test=test_type
        )
    else:
        getattr(pipeline, f"spurtest_{test_type}")(
            "shift(y) ~ shift(x)", df, coords_euclidean=["sx"]
        )


@pytest.mark.parametrize(
    ("formula", "response", "message"),
    [
        ("y ~ x", [1.0, 2.0, 3.0, np.inf], "must be finite"),
        ("y ~ 0 + C(g)", [1.0, 2.0, 3.0, 4.0], "requires a formula with an intercept"),
    ],
)
def test_spur_validates_design_before_ols_fit(monkeypatch, formula, response, message):
    df = pd.DataFrame({"y": response, "x": [1, 2, 5, 3], "g": ["a", "b", "b", "a"]})

    def unexpected_fit(*args, **kwargs):
        pytest.fail("spur() fitted OLS before validating the design")

    monkeypatch.setattr("statsmodels.regression.linear_model.OLS.fit", unexpected_fit)
    with pytest.raises(ValueError, match=message):
        spur(formula, df, q=2, backend="fallback")


def test_pipeline_result_summary_formats_comparison_table() -> None:
    levels_model = SimpleNamespace(
        params=pd.Series([1.0, 0.5], index=["Intercept", "x"]),
        nobs=100.0,
        rsquared=0.2500,
        rsquared_adj=0.2300,
        model=SimpleNamespace(endog_names="mobility"),
    )
    transformed_model = SimpleNamespace(
        params=pd.Series([0.9, 0.4, 0.2], index=["Intercept", "h_x", "h_z"]),
        nobs=100.0,
        rsquared=0.2100,
        rsquared_adj=0.1900,
        model=SimpleNamespace(endog_names="h_mobility"),
    )

    result = PipelineResult(
        tests=Tests(
            i0=TestResult("i0", 1.1000, 0.2100, np.array([1.0, 2.0, 3.0]), 0.0100),
            i1=TestResult("i1", 1.2000, 0.1100, np.array([1.0, 2.0, 3.0]), 0.0200),
            i0resid=TestResult(
                "i0resid", 1.3000, 0.3100, np.array([1.0, 2.0, 3.0]), 0.0300
            ),
            i1resid=TestResult(
                "i1resid", 1.4000, 0.4100, np.array([1.0, 2.0, 3.0]), 0.0400
            ),
        ),
        fits=Fits(
            levels=RegressionResult(
                model=levels_model,
                scpc=SimpleNamespace(
                    scpcstats=np.array(
                        [
                            [1.0000, 0.2000, 5.0000, 0.0100, 0.6000, 1.4000],
                            [0.5000, 0.1000, 5.0000, 0.0200, 0.3000, 0.7000],
                        ]
                    ),
                    q=15,
                    cv=2.1034,
                    avc=0.0300,
                ),
            ),
            transformed=RegressionResult(
                model=transformed_model,
                scpc=SimpleNamespace(
                    scpcstats=np.array(
                        [
                            [0.9000, 0.1500, 6.0000, 0.0100, 0.6000, 1.2000],
                            [0.4000, 0.0500, 8.0000, 0.0200, 0.3000, 0.5000],
                            [0.2000, 0.0300, 6.7000, 0.0300, 0.1400, 0.2600],
                        ]
                    ),
                    q=12,
                    cv=2.0010,
                    avc=0.0300,
                ),
            ),
        ),
    )

    text = result.summary()
    lines = text.splitlines()
    diagnostics_header = next(
        line for line in lines if line.lstrip().startswith("Test")
    )
    regression_header = next(
        line for line in lines if line.lstrip().startswith("Coefficient")
    )
    regression_title_index = lines.index("Regression results".center(len(lines[0])))

    assert lines[0] == lines[1]
    assert set(lines[0]) == {"-"}
    assert lines[regression_title_index - 1] == ""
    assert lines[regression_title_index - 2] == lines[0]
    assert lines[regression_title_index - 3] == lines[0]
    assert lines[regression_title_index + 1] == lines[0]
    assert lines[regression_title_index + 2] == lines[0]
    assert lines[-4] == lines[-3] == lines[0]
    assert "The transformed intercept reflects LBM-GLS centering." in text
    assert "SPUR Pipeline Results" not in text
    assert "mobility" in text
    assert "SPUR Diagnostics" in text
    assert "Regression results" in text
    assert text.index("SPUR Diagnostics") < text.index("Regression results")
    assert text.index("Regression results") < text.index("Coefficient")
    assert diagnostics_header.index("Test") == regression_header.index("Coefficient")
    assert diagnostics_header.index("LR") == regression_header.index("Levels")
    assert diagnostics_header.index("p-value") == regression_header.index("Transformed")
    assert "Levels" in text
    assert "Transformed" in text
    assert "Intercept" in text
    assert "x" in text
    assert "z" in text
    assert "h_x" not in text
    assert "h_z" not in text
    assert "(0.2000)" in text
    assert "(0.0500)" in text
    assert "Model statistics" not in text
    assert "ha_param" not in text
    assert "R-squared" in text
    assert "Adj. R-squared" in text
    assert "SCPC q" in text
    assert "SCPC cv" in text
    assert "SCPC avc" in text
    assert "i0" in text
    assert "i1" in text
    assert "i0resid" in text
    assert "i1resid" in text
