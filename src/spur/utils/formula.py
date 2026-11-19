from __future__ import annotations
import numpy as np
import pandas as pd
from patsy import PatsyError
from patsy.desc import ModelDesc
from patsy.highlevel import dmatrices, dmatrix


def parse_single_var_formula(formula: str, data: pd.DataFrame, fn_name: str) -> dict:
    """Parse the selector string used by the single-variable SPUR APIs.

    This exists to normalize the selector wrapper input (`"y"` or `"y ~ 1"`)
    before dispatching to the dedicated single-variable SPUR functions.
    """
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError(f"`formula` for {fn_name} must be a non-empty string.")

    raw = formula.strip()

    if "+" in raw:
        raise ValueError(
            f"`formula` for {fn_name} must reference exactly one variable."
        )

    if "~" not in raw:
        var = raw
    else:
        lhs, rhs = [part.strip() for part in raw.split("~", 1)]
        if not lhs or rhs != "1":
            raise ValueError(f"For {fn_name} use `y` or `y ~ 1`.")
        var = lhs

    assert var in data.columns, f"Variable '{var}' not found in data."
    return {"var": var, "use_rows": np.ones(len(data), dtype=bool)}


def parse_residual_formula(formula: str, data: pd.DataFrame, fn_name: str) -> dict:
    """Parse a residual-test formula into `Y`, `X_in`, and the estimation sample.

    This keeps the public residual APIs formula-based while preserving the
    existing matrix-based SPUR residual implementation.
    """
    if not isinstance(formula, str) or "~" not in formula:
        raise ValueError(
            f"`formula` for {fn_name} must be two-sided, e.g. `y ~ x1 + x2`."
        )

    y, x = evaluate_formula(formula, data)
    if y.shape[1] != 1 or not getattr(y, "design_info").terms[0].factors:
        raise ValueError(f"`formula` for {fn_name} must have one numeric response.")
    return {
        "Y": y.to_numpy(dtype=float),
        "X_in": x.to_numpy(dtype=float),
        "use_rows": np.isin(np.arange(len(data)), x.index),
    }


def evaluate_formula(
    formula: str, data: pd.DataFrame, *, allow_no_response: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate formula terms once and retain input row positions."""
    if not isinstance(formula, str) or "~" not in formula:
        raise ValueError("`formula` must contain `~`, e.g. `y ~ x1 + x2`.")

    # A fresh index preserves exact row positions with duplicate data labels.
    frame = data.reset_index(drop=True)
    try:
        desc = ModelDesc.from_formula(formula)
        if allow_no_response:
            desc.lhs_termlist = [term for term in desc.lhs_termlist if term.factors]
        if not desc.lhs_termlist and allow_no_response:
            x = dmatrix(desc, frame, return_type="dataframe")
            y = pd.DataFrame(index=x.index)
        else:
            y, x = dmatrices(desc, frame, return_type="dataframe")
    except PatsyError as error:
        if isinstance(error.__cause__, NameError):
            raise AssertionError(f"Variable not found in data: {error}") from error
        raise ValueError(f"Invalid formula: {error}") from error

    if len(x) == 0:
        raise ValueError("No observations with complete data for the formula.")
    if not np.isfinite(y.to_numpy()).all() or not np.isfinite(x.to_numpy()).all():
        raise ValueError("The evaluated design and response must be finite.")
    return y, x


def parse_transform_formula(formula: str, data: pd.DataFrame) -> list[str]:
    """Collect all variables referenced by `spurtransform()`.

    This lets the public transform API accept formulas while the existing
    transformation code continues to work with a flat variable list.
    """
    if not isinstance(formula, str) or "~" not in formula:
        raise ValueError("`formula` must be two-sided, e.g. `y ~ x1 + x2`.")

    lhs, rhs = [part.strip() for part in formula.split("~", 1)]
    vars = [lhs] if lhs and lhs != "1" else []

    if rhs not in ("", "1"):
        rhs_terms = [term.strip() for term in rhs.split("+")]
        if any(not term for term in rhs_terms):
            raise ValueError("`formula` is invalid.")
        vars.extend(rhs_terms)
    else:
        rhs_terms = []

    vars = list(dict.fromkeys(vars))
    assert vars, "No variables found in `formula`."
    missing = [name for name in vars if name not in data.columns]
    assert not missing, f"Variables not found in data: {', '.join(missing)}"

    return vars


def rewrite_formula_with_prefix(formula: str, prefix: str) -> str:
    """Rewrite `y ~ x1 + x2` as `h_y ~ h_x1 + h_x2` for transformed SPUR models."""
    lhs, rhs = [part.strip() for part in formula.split("~", 1)]

    lhs_out = f"{prefix}{lhs}" if lhs and lhs != "1" else ""
    rhs_terms = [] if rhs in ("", "1") else [term.strip() for term in rhs.split("+")]
    rhs_out = " + ".join(f"{prefix}{term}" for term in rhs_terms) if rhs_terms else "1"

    return f"{lhs_out} ~ {rhs_out}"
