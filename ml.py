import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


# -----------------------------
# CONFIG: paths and column names
# -----------------------------
DATA_PATH = os.path.join("data", "processed", "merged_daily_data.csv")
DATE_COL = "date"
TARGET_COL = "congestion_index"
FIG_DIR = "figures"


def ensure_dirs():
    # Create output directory if it does not exist
    os.makedirs(FIG_DIR, exist_ok=True)


def load_data():
    # Load merged dataset and check required columns
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"[ERROR] {DATA_PATH} not found.\n"
            "Make sure you have data/processed/merged_daily_data.csv"
        )

    df = pd.read_csv(DATA_PATH)

    # Parse and sort by date if available
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df = df.sort_values(DATE_COL).reset_index(drop=True)

    # Ensure target column exists
    if TARGET_COL not in df.columns:
        raise ValueError(f"[ERROR] Target '{TARGET_COL}' not found in columns: {list(df.columns)}")

    return df


def add_time_features(df):
    # Add basic calendar-based features
    if DATE_COL not in df.columns:
        return df

    df = df.copy()
    df["month"] = df[DATE_COL].dt.month
    df["dayofweek"] = df[DATE_COL].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df


def pick_features(df):
    # Select available numeric features from predefined candidates
    candidate_features = [
        "temperature",
        "precipitation",
        "wind_speed",
        "pm25",
        "pm10",
        "aqi",
        "is_public_holiday",
        "is_school_holiday",
        "month",
        "dayofweek",
        "is_weekend",
    ]

    return [c for c in candidate_features if c in df.columns]


def time_split(df, test_ratio=0.2):
    # Split dataset chronologically into train and test sets
    n = len(df)
    cut = int(n * (1 - test_ratio))
    train = df.iloc[:cut].copy()
    test = df.iloc[cut:].copy()
    return train, test


def evaluate_model(name, model, X_test, y_test, y_pred):
    # Compute and print RMSE and R² metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n==== {name} ====")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:  {r2:.3f}")

    # Print coefficients for linear regression
    if isinstance(model, LinearRegression):
        print("Intercept:", float(model.intercept_))
        for col, coef in zip(X_test.columns, model.coef_):
            print(f"{col}: {coef:.3f}")

    return rmse, r2


def main():
    # Prepare directories and load data
    ensure_dirs()
    df = load_data()
    df = add_time_features(df)

    # Remove rows with missing target
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    # Select usable features
    features = pick_features(df)
    if len(features) < 2:
        raise ValueError(
            f"[ERROR] Not enough usable feature columns found.\n"
            f"Available columns: {list(df.columns)}\n"
            f"Candidates used: {features}"
        )

    # Keep only relevant columns
    used_cols = [DATE_COL] if DATE_COL in df.columns else []
    used_cols += features + [TARGET_COL]
    df = df[used_cols].copy()

    # Ensure features are numeric
    for c in features:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=features + [TARGET_COL]).reset_index(drop=True)

    # Time-based train/test split
    train_df, test_df = time_split(df, test_ratio=0.2)

    X_train = train_df[features]
    y_train = train_df[TARGET_COL]
    X_test = test_df[features]
    y_test = test_df[TARGET_COL]

    # Train and evaluate linear regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lin = lin_reg.predict(X_test)
    evaluate_model("Linear Regression", lin_reg, X_test, y_test, y_pred_lin)

    # Train and evaluate polynomial regression (degree 2)
    poly_model = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("lr", LinearRegression())
    ])
    poly_model.fit(X_train, y_train)
    y_pred_poly = poly_model.predict(X_test)

    rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))
    r2_poly = r2_score(y_test, y_pred_poly)

    print("\n==== Polynomial Regression (degree 2) ====")
    print(f"RMSE: {rmse_poly:.3f}")
    print(f"R²:  {r2_poly:.3f}")

    # Plot predicted vs actual values
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred_lin, label="Linear", alpha=0.7)
    plt.scatter(y_test, y_pred_poly, label="Polynomial (deg 2)", alpha=0.7)

    mn = min(y_test.min(), y_pred_lin.min(), y_pred_poly.min())
    mx = max(y_test.max(), y_pred_lin.max(), y_pred_poly.max())
    plt.plot([mn, mx], [mn, mx], "r--", lw=2)

    plt.xlabel("Actual congestion_index")
    plt.ylabel("Predicted congestion_index")
    plt.title("Model Predictions vs. Actual (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "ml_pred_vs_actual.png"))
    plt.close()

    print(f"\n[INFO] Features used: {features}")
    print("[INFO] Saved figure: figures/ml_pred_vs_actual.png")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()