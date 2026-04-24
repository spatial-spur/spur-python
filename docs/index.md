# spur-python

`spur-python` implements the SPUR workflow for diagnosing and correcting
spatial unit roots in cross-sectional regressions. It covers the diagnostic and
transformation stage of the workflow. For standalone SCPC inference on fitted
models, see `scpc-python`.

## Installation

`spur-python` can be installed from PyPI; we recommend installing into a virtual environment using `uv`:

```bash
uv pip install spur-python
```

## Example: Chetty Dataset

In this example, we walk you through the workflow we recommend with the packages step-by-step. We also provide a one-stop [pipeline wrapper](#pipeline-wrapper) implementing the entire workflow in one step.

### Data preparation

For illustration, we load the Chetty dataset we ship as part of the package. Of course,
the analysis in principle follow the same logic on any other dataset. In this specific case, 
we first omit the non-contiguous US states. We also drop rows with missing values.

```python
from spur import load_chetty_data

df = load_chetty_data()

df = df[~df["state"].isin(["AK", "HI"])][
    ["am", "gini", "fracblack", "lat", "lon", "state"]
].copy()

df = df.dropna(subset=["am", "gini", "fracblack", "lat", "lon"])
```

### Testing for a spatial unit root

Based on MW 2024, we suggest first testing for a spatial unit root setting using the `I(0)` and `I(1)` tests on the dependent variable.

One way to do this is to use the `spurtest_i0()` and `spurtest_i1()` functions directly:

```python
from spur import spurtest_i0, spurtest_i1

# am is the dependent variable
i0 = spurtest_i0("am", df, lon="lon", lat="lat")
i1 = spurtest_i1("am", df, lon="lon", lat="lat")

print(i0.summary())
print(i1.summary())
```

### Interpreting the test statistics

Using a 10% significance threshold, we suggest interpreting the results with the following heuristic:

- If you do **not** reject `I(0)` and you **do** reject `I(1)`, there is **likely no spatial unit root** and you can proceed in levels
- every other case implies a **possible spatial unit root** - in that case, we suggest transforming all dependent and independent variables before running regressions

We suggest always applying SCPC inference.

### Case 1: likely no spatial unit root

If the heuristic implies your scenario is unlikely to be a spatial unit root, we suggest proceeding in levels but applying SCPC inference:

```python
import statsmodels.formula.api as smf
from scpc import scpc

fit_levels = smf.ols("am ~ gini + fracblack", data=df).fit()
scpc_levels = scpc(fit_levels, df, lon="lon", lat="lat")

print(scpc_levels.summary())
```


### Case 2: likely spatial unit root

If you do have a likely spatial unit root according to the heuristic above, we suggest applying the transformation and running the regression on transformed variables with SCPC inference:

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

### Pipeline wrapper

As a shortcut to implementing all of those steps individually, we also provide a `spur()` wrapper that implements the entire pipeline in one step. It simply runs all tests and returns all results.

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

### Residual tests

We also provide tests for spatial unit roots in regression residuals
rather than the dependent variable itself:

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

## Next Step

See [Reference](reference.md) for the full public API, options, explanations of parameters, and
return objects.
