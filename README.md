![Tests](https://github.com/spatial-spur/spur-python/actions/workflows/test.yaml/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.11+-blue)

<p align="center">
  <img src="assets/logo.png" alt="SPUR logo">
</p>

# spur-python

`spur-python` implements diagnostics and correction methods for spatial unit roots developed by Müller & Watson 2024 in Python.

**Citation:** If you use this package, please cite Becker, Boll and Voth 2026 and Müller & Watson 2024. See [CITATION.bib](CITATION.bib) for the BibTeX entries.

If you encounter any issues or have any questions, please open an issue on GitHub or contact the authors.


## Installation

You can install the package with:

```bash
uv pip install spur-python
```

## Example Usage

We expose both the individual test functions and convenience wrappers running the entire pipeline. 

To run the full pipeline, use the `spur()` wrapper:

```python
import spur
from spur import load_chetty_data, standardize

#  --- data processing ---
df = load_chetty_data()

df = df[~df.state.isin(["AK", "HI"])][["am", "fracblack", "lat", "lon"]]
df = df.dropna(subset=["am", "fracblack", "lat", "lon"])
df = standardize(df, ["am", "fracblack"])

# --- spur pipeline ---
result = spur.spur(
    "am ~ fracblack",
    df,
    lon="lon",
    lat="lat",
    q=10,
    nrep=500,
    seed=42,
)
print(result.summary())
```

This prints both the `spur`-diagnostics:

```
--------------------------------------------
--------------------------------------------
              SPUR Diagnostics              
--------------------------------------------
Test              LR             p-value    
i0                     4.2961         0.0080
i1                     2.5240         0.4660
i0resid                3.3153         0.0700
i1resid              570.7543         0.2540
--------------------------------------------
--------------------------------------------
```

and the regression results with transformed and untransformed variables:

```text
             Regression results             
--------------------------------------------
--------------------------------------------
                              am            
                  --------------------------
Coefficient       Levels         Transformed
--------------------------------------------
Intercept             -0.0000        -0.0000
                     (0.1732)       (0.0789)
fracblack             -0.6009        -0.4240
                     (0.1187)       (0.0903)
--------------------------------------------
N                         693            693
R-squared              0.3611         0.1029
Adj. R-squared         0.3601         0.1016
SCPC q                      8              8
SCPC cv                2.6097         2.6097
SCPC avc               0.0300         0.0300
--------------------------------------------

The transformed intercept reflects LBM-GLS centering.
```

## Documentation

Please refer to [the package documentation](https://spatial-spur.github.io/spur-python/) for detailed information and other (R, Python, Stata) packages.

## References

Becker, Sascha O., P. David Boll and Hans-Joachim Voth "Testing and Correcting for Spatial Unit Roots in Regression Analysis", The Stata Journal 26(2): 177–202. [https://doi.org/10.1177/1536867X261449932](https://journals.sagepub.com/doi/10.1177/1536867X261449932).

Chetty, Raj, Nathaniel Hendren, Patrick Kline, Emmanuel Saez "Where is the land of Opportunity? The Geography of Intergenerational Mobility in the United States" , The Quarterly Journal of Economics 129(4) (2014), 1553–1623, https://doi.org/10.1093/qje/qju022

Müller, Ulrich K. and Mark W. Watson "Spatial Unit Roots and Spurious Regression", Econometrica 92(5) (2024), 1661–1695. doi:10.3982/ECTA21654.

SPUR inference requires at least two retained directions (`q >= 2`). Residual
tests also require the evaluated design to span a constant, `q` to fit within
the residual degrees of freedom, and the retained covariance to have positive
rank. Experiments with no projected outcome variation or no covariance-shape
information raise `ValueError`. The backends use the same convention for
orienting eigenvectors before applying seeded draws. Nearly tied eigenvalues
or loadings can affect Monte Carlo reproducibility across backends.

`spurhalflife()` returns the smallest and largest accepted points on the
Müller–Watson replication grid. The grid is expressed in fractions of the
sample diameter and its last two points are 3 and 100. An upper bound of
`inf` follows the replication convention when the point at 100 is accepted.
Inference applies to the evaluated grid points. Distance units are metres for
geographic coordinates and the supplied units for Euclidean coordinates.
Use `backend="auto"` (the default), `"rust"`, or `"fallback"` to select
execution. `SPATIAL_BACKEND` applies when you omit the argument.
Set `nthreads=1` for a single Rust worker. Linear algebra providers manage
their own internal threading. For process-based parallelism after native
threads have been used, start workers with
`spawn` or `forkserver`.

`spur()` evaluates the formula once and runs all diagnostics and both fits on
the same complete-case sample. It applies the LBM-GLS transform to the evaluated
outcome and each non-intercept design column. The result reports the actual
backend of each calculation in `stage_backends` and explains automatic changes
in `fallback_reasons`. With `backend="auto"`, a native eigenspace tie reruns
all four diagnostics and the transformation together in Python using the same
simulation draws. Each SCPC fit selects its own backend.

The combined workflow requires a formula intercept, as in `y ~ C(region)`.
Standalone residual tests also accept a full set of indicators, as in
`y ~ 0 + C(region)`, because those columns span a constant. The transformed
fit in `spur()` uses centered transformed variables, so its intercept reflects
that centering.

The residual tests and `spurtransform()` accept evaluated terms such as
`np.log(y) ~ x * C(region)`. `spurtransform()` names the appended columns using
the prefix and evaluated column names. Rows excluded by the formula have missing
values in those columns. `spurhalflife()` also reports `fallback_reason` when
automatic dispatch uses the Python implementation.

For manual IV estimation after applying a SPUR transformation, justify
instrument exclusion and first-stage relevance for the transformed moments.
Differencing combines observations across locations, so exclusion at each
original location alone is insufficient. See [Becker, Boll and Voth, §4.1](https://www.sobecker.de/SPUR_Stata_Journal_website.pdf#page=16).

This development branch requires the matching `scpc-python` checkout at version
`0.1.3.dev0` or newer. Place it alongside `spur-python` before running
`uv sync --locked` from `spur-python`. This setup supports the pure-Python backend.

For Rust, install the native adapter from the matching sibling `spatial-rust`
checkout. The adapter must provide SPUR API 4 and SCPC API 7.

```bash
uv pip install ../spatial-rust/crates/spatial-py
```
