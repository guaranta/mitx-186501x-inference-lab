"""HW2-inspired: Delta method confidence interval for g(X) = X^2."""

import numpy as np
from scipy import stats


def delta_method_ci(
    samples: np.ndarray,
    g_func,
    g_prime,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """
    CI for g(theta) using delta method.
    Returns (point_estimate, lower, upper).
    """
    n = len(samples)
    theta_hat = np.mean(samples)
    g_hat = g_func(theta_hat)
    var_g = (g_prime(theta_hat) ** 2) * np.var(samples, ddof=1) / n
    se = np.sqrt(var_g)
    z = stats.norm.ppf(1 - alpha / 2)
    return g_hat, g_hat - z * se, g_hat + z * se


if __name__ == "__main__":
    np.random.seed(0)
    gamma_samples = np.random.gamma(shape=3.0, scale=2.0, size=500)
    point, lo, hi = delta_method_ci(
        gamma_samples,
        g_func=lambda x: x**2,
        g_prime=lambda x: 2 * x,
    )
    print(f"g(X)=X² estimate: {point:.4f}")
    print(f"95% CI (delta method): [{lo:.4f}, {hi:.4f}]")
