"""Model score drift detection — KS test and PSI."""

from __future__ import annotations

import numpy as np
from scipy import stats


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """PSI between reference and current score distributions."""
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    psi = 0.0
    for i in range(n_bins):
        exp_pct = np.mean(
            (expected >= breakpoints[i]) & (expected < breakpoints[i + 1])
        )
        act_pct = np.mean(
            (actual >= breakpoints[i]) & (actual < breakpoints[i + 1])
        )
        exp_pct = max(exp_pct, 1e-6)
        act_pct = max(act_pct, 1e-6)
        psi += (act_pct - exp_pct) * np.log(act_pct / exp_pct)
    return float(psi)


def ks_drift_test(
    reference: np.ndarray,
    current: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Two-sample Kolmogorov-Smirnov test for distribution shift."""
    statistic, p_value = stats.ks_2samp(reference, current)
    return {
        "ks_statistic": float(statistic),
        "p_value": float(p_value),
        "drift_detected": p_value < alpha,
        "alpha": alpha,
    }


def drift_report(
    reference: np.ndarray,
    current: np.ndarray,
    psi_threshold: float = 0.2,
) -> dict:
    """Combined drift report: KS + PSI."""
    ks = ks_drift_test(reference, current)
    psi = population_stability_index(reference, current)
    return {
        **ks,
        "psi": psi,
        "psi_alert": psi > psi_threshold,
        "reference_mean": float(np.mean(reference)),
        "current_mean": float(np.mean(current)),
    }
