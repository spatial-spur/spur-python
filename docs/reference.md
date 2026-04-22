# Reference

This page documents the public `spur-python` API.

## Overview

| Function | Description |
|---|---|
| `spur()` | Run the full SPUR workflow and return diagnostics plus levels and transformed branches |
| `spurtest()` | Wrapper to run one of the four SPUR diagnostic tests |
| `spurtest_i0()` | Variable-level `I(0)` test |
| `spurtest_i1()` | Variable-level `I(1)` test |
| `spurtest_i0resid()` | Residual-based `I(0)` test for a regression formula |
| `spurtest_i1resid()` | Residual-based `I(1)` test for a regression formula |
| `spurtransform()` | Transform all variables referenced in a formula |
| `spurhalflife()` | Estimate a spatial half-life confidence interval |
| `load_chetty_data()` | Load the packaged Chetty dataset |
| `standardize()` | Standardize selected columns in a DataFrame |

## Conventions

### Coordinates

Functions that require spatial coordinates accept exactly one of:

- `lon` and `lat` for geographic coordinates in degrees
- `coords_euclidean` for planar coordinates

Do not pass both specifications in the same call.

### Simulation Controls

The SPUR tests and half-life interval are simulation-based.

- `q`: number of low-frequency weights used in the statistic; must satisfy
  `1 <= q < n`
- `nrep`: number of Monte Carlo draws used to simulate the reference
  distribution
- `seed`: random seed used for reproducibility

### Formula Inputs

- Variable-level diagnostics accept either a bare variable name such as `"am"`
  or a one-variable formula such as `"am ~ 1"`
- Residual diagnostics, transformation, and the full pipeline accept a
  two-sided formula such as `"am ~ gini + fracblack"`

## Full Workflow

### `spur()`

Run the full SPUR workflow.

**Signature**

```python
spur(
    formula,
    data,
    *,
    lon=None,
    lat=None,
    coords_euclidean=None,
    q=15,
    nrep=100000,
    seed=42,
    avc=0.03,
    uncond=False,
    cvs=False,
)
```

**Parameters**

- `formula` (`str`): two-sided regression formula. The left-hand side is the
  dependent variable. The right-hand side defines the regressors used in both
  the levels and transformed branches.
- `data` (`pandas.DataFrame`): input data containing the variables referenced in
  `formula` and the coordinate columns.
- `lon`, `lat` (`str | None`): longitude and latitude column names for
  geographic distance calculations.
- `coords_euclidean` (`Sequence[str] | None`): Euclidean coordinate column
  names. Use instead of `lon` and `lat`.
- `q`, `nrep`, `seed`: SPUR diagnostic simulation controls.
- `avc` (`float`): upper bound on average pairwise correlation passed through to
  `scpc()`.
- `uncond` (`bool`): passed through to `scpc()`. If `True`, use the
  unconditional SCPC branch.
- `cvs` (`bool`): passed through to `scpc()`. If `True`, store additional
  critical values in the SCPC result.

**Returns**

- `PipelineResult`: full pipeline output.
  - `tests`: `Tests` object containing `i0`, `i1`, `i0resid`, and `i1resid`
  - `fits`: `Fits` object containing `levels` and `transformed`
  - each fit is a `RegressionResult` with `model` and `scpc`

## Diagnostics

### `spurtest()`

Dispatch to one of the four public SPUR tests.

**Signature**

```python
spurtest(
    formula,
    data,
    *,
    test,
    lon=None,
    lat=None,
    coords_euclidean=None,
    q=15,
    nrep=100000,
    seed=42,
)
```

**Parameters**

- `formula` (`str`): variable name or formula, depending on `test`.
  - for `test="i0"` and `test="i1"`: `"am"` or `"am ~ 1"`
  - for `test="i0resid"` and `test="i1resid"`: two-sided regression formula
- `data` (`pandas.DataFrame`): input data.
- `test` (`Literal["i0", "i1", "i0resid", "i1resid"]`): diagnostic to run.
- `lon`, `lat`, `coords_euclidean`, `q`, `nrep`, `seed`: as above.

**Returns**

- `TestResult`

### `spurtest_i0()`

Variable-level `I(0)` test.

**Signature**

```python
spurtest_i0(
    var,
    data,
    *,
    lon=None,
    lat=None,
    coords_euclidean=None,
    q=15,
    nrep=100000,
    seed=42,
)
```

**Parameters**

- `var` (`str`): variable to test.
- `data` (`pandas.DataFrame`): input data containing `var` and the coordinate
  columns.
- `lon`, `lat`, `coords_euclidean`, `q`, `nrep`, `seed`: as above.

**Returns**

- `TestResult`
  - `test_type`
  - `LR`
  - `pvalue`
  - `cv`
  - `ha_param`

### `spurtest_i1()`

Variable-level `I(1)` test.

**Signature**

```python
spurtest_i1(
    var,
    data,
    *,
    lon=None,
    lat=None,
    coords_euclidean=None,
    q=15,
    nrep=100000,
    seed=42,
)
```

**Parameters**

- `var` (`str`): variable to test.
- `data` (`pandas.DataFrame`): input data containing `var` and the coordinate
  columns.
- `lon`, `lat`, `coords_euclidean`, `q`, `nrep`, `seed`: as above.

**Returns**

- `TestResult`

### `spurtest_i0resid()`

Residual-based `I(0)` test for a regression formula.

**Signature**

```python
spurtest_i0resid(
    formula,
    data,
    *,
    lon=None,
    lat=None,
    coords_euclidean=None,
    q=15,
    nrep=100000,
    seed=42,
)
```

**Parameters**

- `formula` (`str`): two-sided regression formula defining the residual process
  to be tested.
- `data` (`pandas.DataFrame`): input data.
- `lon`, `lat`, `coords_euclidean`, `q`, `nrep`, `seed`: as above.

**Returns**

- `TestResult`

### `spurtest_i1resid()`

Residual-based `I(1)` test for a regression formula.

**Signature**

```python
spurtest_i1resid(
    formula,
    data,
    *,
    lon=None,
    lat=None,
    coords_euclidean=None,
    q=15,
    nrep=100000,
    seed=42,
)
```

**Parameters**

- `formula` (`str`): two-sided regression formula defining the residual process
  to be tested.
- `data` (`pandas.DataFrame`): input data.
- `lon`, `lat`, `coords_euclidean`, `q`, `nrep`, `seed`: as above.

**Returns**

- `TestResult`

## Transformation

### `spurtransform()`

Transform all variables referenced in a formula and append the transformed
columns to a copy of the input data.

**Signature**

```python
spurtransform(
    formula,
    data,
    *,
    lon=None,
    lat=None,
    coords_euclidean=None,
    prefix="h_",
    transformation="lbmgls",
    radius=None,
    clustvar=None,
)
```

**Parameters**

- `formula` (`str`): formula identifying the variables to transform. Every
  variable referenced in the formula is transformed.
- `data` (`pandas.DataFrame`): input data.
- `lon`, `lat`, `coords_euclidean`: coordinate specification for
  distance-based transformations.
- `prefix` (`str`): prefix applied to transformed variable names. Default:
  `h_`.
- `transformation` (`str`): transformation mode.
  - `"lbmgls"`: GLS-style transformation; default branch used by `spur()`
  - `"nn"`: nearest-neighbor differencing
  - `"iso"`: radius-based differencing
  - `"cluster"`: within-cluster demeaning
- `radius` (`float | None`): radius used by the isotropic transformation;
  required when `transformation="iso"`.
- `clustvar` (`str | None`): cluster label column; required when
  `transformation="cluster"`.

**Returns**

- `pandas.DataFrame`: copy of the input data with transformed columns appended

## Persistence

### `spurhalflife()`

Estimate a confidence interval for the spatial half-life of a variable.

**Signature**

```python
spurhalflife(
    var,
    data,
    *,
    lon=None,
    lat=None,
    coords_euclidean=None,
    q=15,
    nrep=100000,
    level=95,
    normdist=False,
    seed=42,
)
```

**Parameters**

- `var` (`str`): variable to analyze.
- `data` (`pandas.DataFrame`): input data containing `var` and the coordinate
  columns.
- `lon`, `lat`, `coords_euclidean`, `q`, `nrep`, `seed`: as above.
- `level` (`float`): confidence level in percent.
- `normdist` (`bool`): if `True`, report the interval as a fraction of the
  maximum pairwise distance; otherwise report it in distance units.

**Returns**

- `HalfLifeResult`
  - `ci_lower`
  - `ci_upper`
  - `max_dist`
  - `level`
  - `normdist`

## Helpers

### `load_chetty_data()`

Load the packaged Chetty dataset.

**Signature**

```python
load_chetty_data()
```

**Returns**

- `pandas.DataFrame`

### `standardize()`

Standardize selected columns and return a new DataFrame.

**Signature**

```python
standardize(
    df,
    vars,
    appendix=None,
)
```

**Parameters**

- `df` (`pandas.DataFrame`): input data.
- `vars` (`list[str]`): columns to standardize.
- `appendix` (`str | None`): suffix applied to the standardized columns. If
  `None`, overwrite the original columns.

**Returns**

- `pandas.DataFrame`: copy of the input data with standardized columns

## Return Objects

### `TestResult`

Returned by `spurtest()`, `spurtest_i0()`, `spurtest_i1()`,
`spurtest_i0resid()`, and `spurtest_i1resid()`.

**Fields**

- `test_type`
- `LR`
- `pvalue`
- `cv`
- `ha_param`

**Methods**

- `summary()`

### `HalfLifeResult`

Returned by `spurhalflife()`.

**Fields**

- `ci_lower`
- `ci_upper`
- `max_dist`
- `level`
- `normdist`

**Methods**

- `summary()`

### `PipelineResult`

Returned by `spur()`.

**Fields**

- `tests`
- `fits`

`tests` contains:

- `i0`
- `i1`
- `i0resid`
- `i1resid`

`fits` contains:

- `levels`
- `transformed`

Each fit is a `RegressionResult` with:

- `model`
- `scpc`

**Methods**

- `summary()`
