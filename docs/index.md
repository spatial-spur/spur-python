# spur-python

`spur-python` implements the SPUR workflow for diagnosing and correcting
spatial unit roots in cross-sectional regressions. It covers the diagnostic and
transformation stage of the workflow. For standalone SCPC inference on fitted
models, see `scpc-python`.

## Installation

```bash
uv pip install spur-python
```

## Example: Chetty Dataset

This walkthrough follows the practitioner guide in Becker, Boll, and Voth
(2026). The branch decision is based on the dependent-variable `I(0)` and
`I(1)` tests, using a 10% significance level.

## 1. Prepare the sample

Construct the estimation sample and retain the variables used in the
diagnostics and regression.

```python
from spur import load_chetty_data

df = load_chetty_data()

df = df[~df["state"].isin(["AK", "HI"])][
    ["am", "gini", "fracblack", "lat", "lon", "state"]
].copy()

df = df.dropna(subset=["am", "gini", "fracblack", "lat", "lon"])
```

## 2. Test the dependent variable against the I(0) alternative

The first diagnostic tests the dependent variable under the spatial `I(0)`
null.

```python
from spur import spurtest_i0

i0 = spurtest_i0("am", df, lon="lon", lat="lat")

print(i0.summary())
```

## 3. Test the dependent variable against the I(1) alternative

The second diagnostic tests the same variable under the spatial `I(1)` null.

```python
from spur import spurtest_i1

i1 = spurtest_i1("am", df, lon="lon", lat="lat")

print(i1.summary())
```

## 4. Apply the decision rule

Using a 10% significance threshold:

- If you do **not** reject `I(0)` and you **do** reject `I(1)`, proceed in
  levels.
- In every other case, treat the specification as requiring spatial
  differencing and transform the dependent and independent variables together.

## 5. Levels branch

Use this branch only when the decision rule implies that the dependent variable
is consistent with `I(0)` and inconsistent with `I(1)`.

```python
import statsmodels.formula.api as smf
from scpc import scpc

fit_levels = smf.ols("am ~ gini + fracblack", data=df).fit()
scpc_levels = scpc(fit_levels, df, lon="lon", lat="lat")

print(scpc_levels.summary())
```

In this branch there is no SPUR transformation step; the regression is estimated
in levels and SCPC is used for inference.

## 6. Transformed branch

In every other case, transform the dependent and independent variables
together, re-estimate the regression on the transformed data, and use SCPC for
inference there.

```python
import statsmodels.formula.api as smf
from scpc import scpc
from spur import spurtransform

transformed = spurtransform(
    "am ~ gini + fracblack",
    df,
    lon="lon",
    lat="lat",
    transformation="lbmgls",
)

fit_transformed = smf.ols(
    "h_am ~ h_gini + h_fracblack",
    data=transformed,
).fit()

scpc_transformed = scpc(
    fit_transformed,
    transformed,
    lon="lon",
    lat="lat",
)

print(scpc_transformed.summary())
```

The default empirical branch is the `lbmgls` transformation.

## 7. Residual diagnostics

`spur-python` also provides residual-based `I(0)` and `I(1)` tests:

```python
from spur import spurtest_i0resid, spurtest_i1resid

i0resid = spurtest_i0resid(
    "am ~ gini + fracblack",
    df,
    lon="lon",
    lat="lat",
)

i1resid = spurtest_i1resid(
    "am ~ gini + fracblack",
    df,
    lon="lon",
    lat="lat",
)
```

## 8. Packaged shortcut

If you want the package’s default end-to-end implementation rather than the
manual branch logic, use `spur()`.

```python
import spur

pipeline = spur(
    "am ~ gini + fracblack",
    df,
    lon="lon",
    lat="lat",
)

print(pipeline.summary())
```

`spur()` returns all four SPUR diagnostics together with the levels and transformed fits.

## Next Step

See [Reference](reference.md) for the full public API, parameter meanings, and
return objects.
