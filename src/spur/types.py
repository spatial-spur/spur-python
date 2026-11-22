from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass(slots=True)
class TestResult:
    """One SPUR test result."""

    __test__ = False

    test_type: str
    """Name of the test that was run."""
    LR: float
    """Observed value of the test statistic."""
    pvalue: float
    """P-value associated with the test statistic."""
    cv: np.ndarray
    """Reference critical values for the test."""
    ha_param: float
    """Estimated local-alternative tuning parameter."""
    pvalue_mat: np.ndarray | None = field(default=None, repr=False, compare=False)
    """Optional null-grid p-value vector/matrix when computed."""
    cvalue_mat: np.ndarray | None = field(default=None, repr=False, compare=False)
    """Optional null-grid critical-value matrix when computed."""
    rho_grid: np.ndarray | None = field(default=None, repr=False, compare=False)
    """Optional null-grid rho values when computed."""
    backend: str = field(default="fallback", repr=False, compare=False)
    """Execution backend used to compute this result."""
    tied: bool = field(default=False, repr=False, compare=False)
    """Whether the eigenspace cutoff exhibited numerical ties."""

    fallback_reason: str | None = field(default=None, repr=False, compare=False)
    """Reason an automatically selected native diagnostic used the fallback."""

    def summary(self) -> str:
        """Format test results for display."""
        stat_name = "LFUR" if self.test_type.startswith("i1") else "LFST"
        lines = [
            f"Spatial {self.test_type.upper()} Test Results",
            "-" * 45,
            f"Test Statistic ({stat_name}):  {self.LR:9.4f}",
            f"P-value:                {self.pvalue:9.4f}",
            f"CV 1%:                  {self.cv[0]:9.4f}",
            f"CV 5%:                  {self.cv[1]:9.4f}",
            f"CV 10%:                 {self.cv[2]:9.4f}",
            "-" * 45,
        ]
        return "\n".join(lines)


@dataclass
class HalfLifeResult:
    """Numerically inverted spatial half-life confidence set."""

    ci_lower: float
    """Lower endpoint of the confidence-set hull."""
    ci_upper: float
    """Upper endpoint of the confidence-set hull."""
    max_dist: float
    """Largest observed distance in the sample."""
    level: float
    """Confidence level used for the interval."""
    normdist: bool
    """Whether distances were normalized before reporting."""
    pv_mat: np.ndarray | None = field(default=None, repr=False, compare=False)
    """P-values at the initial search points."""
    hl_grid: np.ndarray | None = field(default=None, repr=False, compare=False)
    """Initial half-life search points in sample-diameter units."""
    backend: str = field(default="fallback", repr=False, compare=False)
    """Execution backend used to compute this result."""
    tied: bool = field(default=False, repr=False, compare=False)
    """Whether the native solver reported a tie inside the retained basis."""

    distance_units: str = "meters"
    """Units of max_dist and, when normdist is false, of the interval bounds."""
    fallback_reason: str | None = field(default=None, repr=False, compare=False)
    """Reason an automatically selected native calculation used the fallback."""
    components: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2)), repr=False, compare=False
    )
    """Detected confidence-set components in the reporting units."""

    def summary(self) -> str:
        """Format results for display."""
        units = "fractions of max distance" if self.normdist else self.distance_units
        lines = [
            f"Approximate spatial half-life {self.level:g}% confidence set ({units})",
            "-" * 45,
        ]
        if self.components.size:
            lines.extend(f"[{lo:.6g}, {hi:.6g}]" for lo, hi in self.components)
        else:
            lines.append("Empty confidence set")
        lines.extend(
            [
                f"Max distance in sample {self.max_dist:.6g} {self.distance_units}",
                "-" * 45,
            ]
        )
        return "\n".join(lines)


@dataclass(slots=True)
class Tests:
    """The four SPUR diagnostics returned by `spur()`."""

    __test__ = False

    i0: TestResult
    """I(0) test run on the dependent variable in levels."""
    i1: TestResult
    """I(1) test run on the dependent variable in levels."""
    i0resid: TestResult
    """I(0) test run on the regression residuals."""
    i1resid: TestResult
    """I(1) test run on the regression residuals."""


@dataclass(slots=True)
class RegressionResult:
    model: Any
    """Fitted regression model for this branch."""
    scpc: Any
    """SCPC output computed from that fitted model."""
    coefficient_labels: tuple[str, ...] | None = field(
        default=None, repr=False, compare=False
    )
    """Optional labels matching the original evaluated regression design."""


@dataclass(slots=True)
class Fits:
    """The two fitted regression branches returned by `spur()`."""

    levels: RegressionResult
    """Fit based on the original variables."""
    transformed: RegressionResult
    """Fit based on the transformed variables."""


@dataclass(slots=True)
class PipelineResult:
    """Full output of `spur()`."""

    tests: Tests
    """All diagnostic test results."""
    fits: Fits
    """Both fitted regression branches."""
    backend: str = field(default="fallback", repr=False, compare=False)
    """Rust if every stage used Rust, otherwise fallback."""
    tied: bool = field(default=False, repr=False, compare=False)
    """Whether any SPUR diagnostic or SCPC fit detected eigenspace ties."""

    stage_backends: dict[str, str] = field(default_factory=dict)
    """Actual backend of each diagnostic, the transform and each SCPC fit."""
    fallback_reasons: dict[str, str] = field(default_factory=dict)
    """Reasons automatic dispatch used a stage's fallback."""

    def summary(self) -> str:
        """Format the full SPUR pipeline result for display."""
        from .utils.summary import render_pipeline_summary

        return render_pipeline_summary(self)
