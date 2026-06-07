"""Demo: drift detection on synthetic model scores."""

import numpy as np

from drift_detection import drift_report

if __name__ == "__main__":
    np.random.seed(42)
    reference = np.random.normal(0.7, 0.1, 1000)
    current = np.random.normal(0.65, 0.12, 1000)  # simulated drift

    report = drift_report(reference, current)
    print("=== Model Drift Report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")
