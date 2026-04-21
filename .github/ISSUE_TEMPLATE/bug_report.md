---
name: Bug report
about: Report a reproducible problem with spur-python
title: "[bug] "
labels: bug
---

## Summary

Describe the problem in 1-3 sentences.

## Minimal reproduction

Provide a copy-pasteable example if possible.

```python

```

## Expected behavior

What did you expect to happen?

## Actual behavior

What happened instead? Include the full traceback, warning, or numerical mismatch if relevant.

```text

```

## Environment

- OS:
- Python version:
- install method: `uv` / `pip` / editable install / git install
- spur-python version or commit:
- scpc-python version, if relevant:
- affected function or workflow: `spur` / `spurtest` / `spurtransform` / `spurhalflife` / `load_chetty_data` / docs / CI

## Additional context

Anything else that might help reproduce or explain the issue.

- Does this reproduce on synthetic data, Chetty data, or only your own data?
- Does this affect `lon`/`lat`, `coords_euclidean`, or both?
- If relevant, which mode or option is affected: `i0`, `i1`, `i0resid`, `i1resid`, `lbmgls`, `nn`, `iso`, `cluster`, `level`, `normdist`, or the formula workflow?
- Is the issue a crash, wrong result, parity mismatch, docs mismatch, install/import problem, or performance regression?
- If this is a numerical mismatch, do you see the same issue relative to `spuR` or `spur` in Stata?
