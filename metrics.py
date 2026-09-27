"""
Statistical metrics for the Section 6 empirical comparison.

VaR, CVaR, Sharpe/Sortino/Calmar ratios, moments, bootstrap
confidence intervals, and Kolmogorov-Smirnov distribution tests.
"""
from __future__ import annotations
import numpy as np
from scipy import stats


def moments(x: np.ndarray) -> dict[str, float]:
    """First four moments of a sample."""
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)),
        "skew": float(stats.skew(x)),
        "excess_kurt": float(stats.kurtosis(x)),
    }


def var_cvar(x: np.ndarray, alpha: float = 0.95) -> tuple[float, float]:
    """
    Value-at-risk and conditional value-at-risk at level alpha.

    Convention: x is a P&L sample (positive = gain). VaR and CVaR
    are reported as positive numbers representing the magnitude of
    loss in the left tail.
    """
    q = float(np.quantile(x, 1.0 - alpha))
    var = -q
    tail = x[x <= q]
    cvar = -float(tail.mean()) if len(tail) > 0 else var
    return var, cvar


def max_drawdown(paths: np.ndarray) -> np.ndarray:
    """
    Per-path maximum drawdown (as a positive fraction) of a P&L
    process. paths is (n_paths, n_steps + 1) cumulative P&L.
    """
    cumulative = np.cumsum(paths, axis=1) if paths.ndim == 2 else paths
    running_max = np.maximum.accumulate(cumulative, axis=1)
    drawdown = (cumulative - running_max)
    return -drawdown.min(axis=1)


def sharpe(x: np.ndarray, periods_per_year: float = 252.0) -> float:
    """Annualised Sharpe ratio from a return sample."""
    s = np.std(x, ddof=1)
    if s == 0:
        return float("nan")
    return float(np.mean(x) / s * np.sqrt(periods_per_year))


def sortino(x: np.ndarray, periods_per_year: float = 252.0) -> float:
    """Annualised Sortino ratio."""
    downside = x[x < 0]
    if len(downside) < 2:
        return float("nan")
    downside_std = np.std(downside, ddof=1)
    if downside_std == 0:
        return float("nan")
    return float(np.mean(x) / downside_std * np.sqrt(periods_per_year))


def calmar(x: np.ndarray, periods_per_year: float = 252.0) -> float:
    """Calmar ratio: annualised return / max drawdown."""
    cumulative = np.cumsum(x)
    running_max = np.maximum.accumulate(cumulative)
    dd = -(cumulative - running_max).min()
    if dd == 0:
        return float("nan")
    annualised_ret = np.mean(x) * periods_per_year
    return float(annualised_ret / dd)


def bootstrap_ci(
    x: np.ndarray,
    statistic,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """
    Percentile bootstrap CI for a statistic.

    Returns (point_estimate, lower, upper).
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    stat_boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stat_boot[i] = statistic(x[idx])
    point = statistic(x)
    lo = float(np.quantile(stat_boot, alpha / 2))
    hi = float(np.quantile(stat_boot, 1.0 - alpha / 2))
    return float(point), lo, hi


def ks_test(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Two-sample KS test statistic and p-value."""
    stat, p = stats.ks_2samp(x, y)
    return float(stat), float(p)


def summarise(x: np.ndarray) -> dict[str, float]:
    """Comprehensive summary of a P&L sample."""
    m = moments(x)
    var95, cvar95 = var_cvar(x, 0.95)
    return {
        **m,
        "var_95": var95,
        "cvar_95": cvar95,
        "sharpe": sharpe(x),
        "sortino": sortino(x),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p_loss": float((x < 0).mean()),
    }
