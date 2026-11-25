"""Train and evaluate the student score prediction model."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "student_performance.csv"
MODEL_PATH = ROOT / "student_performance_model.joblib"

FEATURES = [
    "study_hours",
    "attendance",
    "previous_marks",
    "assignment_completion",
    "sleep_hours",
]
TARGET = "final_score"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Dataset not found. Run: python generate_data.py")

    data = pd.read_csv(DATA_PATH)
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    bundle = {"model": model, "features": FEATURES}
    joblib.dump(bundle, MODEL_PATH)

    print("Model training complete.")
    print(f"Mean Absolute Error: {mae:.2f} marks")
    print(f"R² score: {r2:.3f}")
    print(f"Saved model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
