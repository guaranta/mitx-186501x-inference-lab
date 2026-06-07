"""HW3-inspired: Bernoulli hypothesis test and two-sample comparison."""

import numpy as np
from scipy import stats


def bernoulli_test(successes: int, trials: int, p0: float = 0.5) -> dict:
    """One-sample binomial test H0: p = p0."""
    result = stats.binomtest(successes, trials, p=p0, alternative="two-sided")
    return {
        "successes": successes,
        "trials": trials,
        "p_hat": successes / trials,
        "p_value": result.pvalue,
        "reject_h0": result.pvalue < 0.05,
    }


def two_sample_ttest(group_a: np.ndarray, group_b: np.ndarray) -> dict:
    """Welch's t-test for two independent groups."""
    stat, pval = stats.ttest_ind(group_a, group_b, equal_var=False)
    return {
        "t_statistic": float(stat),
        "p_value": float(pval),
        "mean_a": float(np.mean(group_a)),
        "mean_b": float(np.mean(group_b)),
        "reject_h0": pval < 0.05,
    }


if __name__ == "__main__":
    np.random.seed(1)
    print("Bernoulli test:", bernoulli_test(62, 100, p0=0.5))
    a = np.random.normal(75, 10, 50)
    b = np.random.normal(80, 10, 50)
    print("Two-sample test:", two_sample_ttest(a, b))
