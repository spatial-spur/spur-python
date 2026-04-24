![Tests](https://github.com/spatial-spur/spur-python/actions/workflows/test.yaml/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.11+-blue)

<p align="center">
  <img src="assets/logo.png" alt="SPUR logo">
</p>

# spur-python: A Python Package for Spatial Unit Roots

A Python implementation of the methods for diagnosing and correcting spatial unit roots developed by Muller and Watson (2024). This is a complete port of the Stata package [SPUR](https://github.com/pdavidboll/SPUR) (Becker, Boll and Voth 2025) — see the [forthcoming *Stata Journal* article](https://warwick.ac.uk/fac/soc/economics/research/workingpapers/2025/twerp_1541-_becker.pdf) for the practitioner's guide.


## Installation

The easiest way to get started is to install our coding-agent skills. 
Just point your agent at our `spur-skills` repository:

```bash
codex --dangerously-bypass-approvals-and-sandbox "Install spur-skills by following https://github.com/spatial-spur/spur-skills#install"
```

```bash
claude --dangerously-skip-permissions "Install spur-skills by following https://github.com/spatial-spur/spur-skills#install"
```

You can install just the package with:

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
```

## Citation

```bibtex
@Article{becker2025,
  author  = {Becker, Sascha O. and Boll, P. David and Voth, Hans-Joachim},
  title   = {Testing and Correcting for Spatial Unit Roots in Regression Analysis},
  journal = {Stata Journal},
  year    = {forthcoming}
}

@Article{muller2024,
  author  = {M{\"u}ller, Ulrich K. and Watson, Mark W.},
  title   = {Spatial Unit Roots and Spurious Regression},
  journal = {Econometrica},
  year    = {2024},
  volume  = {92},
  number  = {5},
  pages   = {1661--1695}
}
```

## References

- Muller, Ulrich K. and Mark W. Watson (2024). "Spatial Unit Roots and Spurious Regression." *Econometrica* 92(5), 1661-1695.
- Becker, Sascha O., P. David Boll, and Hans-Joachim Voth (2025). "Testing and Correcting for Spatial Unit Roots in Regression Analysis." *Stata Journal*, forthcoming. [[PDF]](https://warwick.ac.uk/fac/soc/economics/research/workingpapers/2025/twerp_1541-_becker.pdf)
- Chetty, Raj, Nathaniel Hendren, Patrick Kline, and Emmanuel Saez (2014). "Where is the Land of Opportunity? The Geography of Intergenerational Mobility in the United States." *QJE* 129(4).
