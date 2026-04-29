![Tests](https://github.com/spatial-spur/spur-python/actions/workflows/test.yaml/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.11+-blue)

<p align="center">
  <img src="assets/logo.png" alt="SPUR logo">
</p>

# spur-python

`spur-python` implements diagnostics and correction methods for spatial unit roots developed by Müller and Watson (2024) in Python.

**When using this code, please cite [Becker, Boll and Voth (2026)](https://pauldavidboll.com/SPUR_Stata_Journal_website.pdf):**

```bibtex
@Article{becker2026,
  author    = {Becker, Sascha O. and Boll, P. David and Voth, Hans-Joachim},
  title     = {Testing and Correcting for Spatial Unit Roots in Regression Analysis},
  journal   = {Stata Journal},
  year      = {forthcoming},
  note      = {Forthcoming}
}
```

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
```

## Documentation

Please refer to [the package documentation](https://spatial-spur.github.io/scpcR/) for detailed information and other (R, Python, Stata) packages.

## References

Becker, Sascha O., P. David Boll and Hans-Joachim Voth "Testing and Correcting for Spatial Unit Roots in Regression Analysis", Forthcoming at the Stata Journal.

Chetty, Raj, Nathaniel Hendren, Patrick Kline, Emmanuel Saez "Where is the land of Opportunity? The Geography of Intergenerational Mobility in the United States" , The Quarterly Journal of Economics 129(4) (2014), 1553–1623, https://doi.org/10.1093/qje/qju022

Müller, Ulrich K. and Mark W. Watson "Spatial Unit Roots and Spurious Regression", Econometrica 92(5) (2024), 1661–1695. https://www.princeton.edu/~umueller/SPUR.pdf.
