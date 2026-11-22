from __future__ import annotations
from keyword import iskeyword

import numpy as np
import pandas as pd
from patsy import PatsyError
from patsy.desc import INTERCEPT, ModelDesc, Term
from patsy.eval import EvalEnvironment
from patsy.highlevel import dmatrices, dmatrix
from patsy.user_util import LookupFactor


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

    if var not in data.columns:
        raise ValueError(f"Variable '{var}' not found in data.")
    return {"var": var, "use_rows": np.ones(len(data), dtype=bool)}


def parse_residual_formula(
    formula: str,
    data: pd.DataFrame,
    fn_name: str,
    *,
    eval_env: EvalEnvironment | int = -1,
) -> dict:
    """Parse a residual-test formula into `Y`, `X_in`, and the estimation sample.

    This keeps the public residual APIs formula-based while preserving the
    existing matrix-based SPUR residual implementation.
    """
    if not isinstance(formula, str) or "~" not in formula:
        raise ValueError(
            f"`formula` for {fn_name} must be two-sided, e.g. `y ~ x1 + x2`."
        )

    y, x = evaluate_formula(formula, data, eval_env=eval_env)
    if y.shape[1] != 1 or not getattr(y, "design_info").terms[0].factors:
        raise ValueError(f"`formula` for {fn_name} must have one numeric response.")
    return {
        "Y": y.to_numpy(dtype=float),
        "X_in": x.to_numpy(dtype=float),
        "use_rows": np.isin(np.arange(len(data)), x.index),
    }


def evaluate_formula(
    formula: str,
    data: pd.DataFrame,
    *,
    eval_env: EvalEnvironment | int = -1,
    allow_no_response: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate formula terms once and retain input row positions."""
    if not isinstance(formula, str) or "~" not in formula:
        raise ValueError("`formula` must contain `~`, e.g. `y ~ x1 + x2`.")

    # A fresh index preserves exact row positions with duplicate data labels.
    frame = data.reset_index(drop=True)
    try:
        desc = (
            transform_formula_description(formula, frame.columns)
            if allow_no_response
            else (ModelDesc.from_formula(formula))
        )
        if allow_no_response:
            desc.lhs_termlist = [term for term in desc.lhs_termlist if term.factors]
        if not desc.lhs_termlist and allow_no_response:
            x = dmatrix(desc, frame, eval_env=eval_env, return_type="dataframe")
            y = pd.DataFrame(index=x.index)
        else:
            y, x = dmatrices(desc, frame, eval_env=eval_env, return_type="dataframe")
    except (PatsyError, SyntaxError) as error:
        if isinstance(error.__cause__, NameError):
            raise ValueError(f"Variable not found in data: {error}") from error
        raise ValueError(f"Invalid formula: {error}") from error

    if len(x) == 0:
        raise ValueError("No observations with complete data for the formula.")
    if not np.isfinite(y.to_numpy()).all() or not np.isfinite(x.to_numpy()).all():
        raise ValueError("The evaluated design and response must be finite.")
    return y, x


def transform_formula_description(formula: str, columns: pd.Index) -> ModelDesc:
    """Retain the original transform shorthand for literal column names."""
    lhs, rhs = (part.strip() for part in formula.split("~", 1))
    left = [] if lhs in ("", "1") else [lhs]
    right = [] if rhs in ("", "1") else [part.strip() for part in rhs.split("+")]
    names = left + right
    if (
        names
        and all(name in columns for name in names)
        and any(not name.isidentifier() or iskeyword(name) for name in names)
    ):
        return ModelDesc(
            [Term([LookupFactor(name)]) for name in left],
            [Term([]), *(Term([LookupFactor(name)]) for name in right)],
        )
    return ModelDesc.from_formula(formula if rhs else f"{lhs} ~ 1")


def intercept_columns(design_info_or_data) -> slice | None:
    """Locate the intercept term in a Patsy DesignInfo or Formulaic ModelSpec."""
    spec = design_info_or_data
    if not hasattr(spec, "term_slices"):
        spec = getattr(design_info_or_data, "model_spec", None)
        if spec is None:
            spec = design_info_or_data.design_info
    slices = spec.term_slices
    # Formulaic terms support their formula spelling as a lookup key.
    # Both lookups refer to terms, so a data column named Intercept stays a regressor.
    return slices.get(INTERCEPT) or slices.get("1")
