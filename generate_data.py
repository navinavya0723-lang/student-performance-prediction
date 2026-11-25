"""Generate a reproducible demo dataset for the Student Performance Prediction project."""
from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_STATE = 42
OUTPUT = Path(__file__).parent / "data" / "student_performance.csv"

def main():
    rng = np.random.default_rng(RANDOM_STATE)
    n = 1200

    study_hours = rng.uniform(0.5, 8.0, n)
    attendance = rng.uniform(45, 100, n)
    previous_marks = rng.uniform(25, 95, n)
    assignment_completion = rng.uniform(20, 100, n)
    sleep_hours = rng.uniform(4.0, 9.0, n)

    # Synthetic teaching/demo relationship with random variation.
    score = (
        4.0 * study_hours
        + 0.20 * attendance
        + 0.30 * previous_marks
        + 0.12 * assignment_completion
        + 1.2 * sleep_hours
        - 12
        + rng.normal(0, 6, n)
    )
    score = np.clip(score, 0, 100)

    df = pd.DataFrame({
        "study_hours": study_hours.round(1),
        "attendance": attendance.round(1),
        "previous_marks": previous_marks.round(1),
        "assignment_completion": assignment_completion.round(1),
        "sleep_hours": sleep_hours.round(1),
        "final_score": score.round(1),
    })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Created {len(df)} demo records at: {OUTPUT}")

if __name__ == "__main__":
    main()
