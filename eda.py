import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


DATA_DIR = "data/raw"
OUTPUT_DIR = "data/processed"
FIG_DIR = "figures"


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


def load_and_merge():
    """Load all CSV files and merge them on the 'date' column."""
    air_path = os.path.join(DATA_DIR, "air.csv")
    holidays_path = os.path.join(DATA_DIR, "holidays.csv")
    traffic_path = os.path.join(DATA_DIR, "traffic.csv")
    weather_path = os.path.join(DATA_DIR, "weather.csv")

    air = pd.read_csv(air_path, parse_dates=["date"])
    holidays = pd.read_csv(holidays_path, parse_dates=["date"])
    traffic = pd.read_csv(traffic_path, parse_dates=["date"])
    weather = pd.read_csv(weather_path, parse_dates=["date"])

    # Merge step by step on 'date'
    df = traffic.merge(weather, on="date", how="inner")
    df = df.merge(air, on="date", how="inner")
    df = df.merge(holidays, on="date", how="left")

    # Sort by date for nicer plots
    df = df.sort_values("date").reset_index(drop=True)

    # Save merged data for later stages (ML part etc.)
    merged_path = os.path.join(OUTPUT_DIR, "merged_daily_data.csv")
    df.to_csv(merged_path, index=False)
    print(f"[INFO] Merged data saved to: {merged_path}")
    print(f"[INFO] Final shape: {df.shape}")

    return df


def basic_eda(df: pd.DataFrame):
    print("\n===== BASIC EDA =====")
    print("[INFO] Columns:", df.columns.tolist())
    print("\n[INFO] Head:")
    print(df.head())
    print("\n[INFO] Info:")
    print(df.info())
    print("\n[INFO] Missing values per column:")
    print(df.isna().sum())
    print("\n[INFO] Descriptive statistics:")
    print(df.describe())


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek  # 0 = Monday, 6 = Sunday
    df["is_weekend"] = df["dayofweek"] >= 5
    return df


def plot_time_series(df: pd.DataFrame):
    """Traffic & air quality & temperature over time."""
    plt.figure(figsize=(12, 5))
    plt.plot(df["date"], df["congestion_index"])
    plt.title("Daily Traffic Congestion Index Over Time")
    plt.xlabel("Date")
    plt.ylabel("Congestion Index")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "time_traffic_congestion.png"))
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(df["date"], df["pm25"], label="PM2.5")
    plt.plot(df["date"], df["pm10"], label="PM10")
    plt.legend()
    plt.title("Daily Air Pollution Levels (PM2.5 & PM10)")
    plt.xlabel("Date")
    plt.ylabel("µg/m³")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "time_air_pollution.png"))
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(df["date"], df["temperature"])
    plt.title("Daily Temperature Over Time")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "time_temperature.png"))
    plt.close()

    print("[INFO] Time series plots saved.")


def plot_distributions(df: pd.DataFrame):
    numeric_cols = ["congestion_index", "pm25", "pm10",
                    "temperature", "precipitation", "wind_speed"]

    for col in numeric_cols:
        if col not in df.columns:
            continue
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"dist_{col}.png"))
        plt.close()
            # Special case: precipitation -> log-log plot
    if "precipitation" in df.columns:
        plt.figure(figsize=(6, 4))
        plt.xscale("log")
        plt.yscale("log")
        sns.histplot(df["precipitation"].replace(0, np.nan).dropna(), bins=50)
        plt.title("Distribution of precipitation (log-log)")
        plt.xlabel("precipitation (log scale)")
        plt.ylabel("count (log scale)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "dist_precipitation_loglog.png"))
        plt.close()

    print("[INFO] Distribution plots saved.")


def plot_correlations(df: pd.DataFrame):
    numeric_cols = ["congestion_index", "pm25", "pm10",
                    "temperature", "precipitation", "wind_speed"]
    existing = [c for c in numeric_cols if c in df.columns]
    corr = df[existing].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "correlation_heatmap.png"))
    plt.close()

    print("\n===== CORRELATION MATRIX =====")
    print(corr)
    print("[INFO] Correlation heatmap saved.")


def hypothesis_tests(df: pd.DataFrame):
    print("\n===== HYPOTHESIS TESTS =====")

    # 1) Rainy vs. non-rainy days: congestion_index
    if "precipitation" in df.columns:
        df["is_rainy"] = df["precipitation"] > 0
        rainy = df.loc[df["is_rainy"], "congestion_index"].dropna()
        dry = df.loc[~df["is_rainy"], "congestion_index"].dropna()

        if len(rainy) > 5 and len(dry) > 5:
            t_stat, p_val = stats.ttest_ind(rainy, dry, equal_var=False)
            print("\n[TEST 1] Mean traffic congestion on rainy vs. dry days")
            print(f" - Rainy days count: {len(rainy)}, mean: {rainy.mean():.2f}")
            print(f" - Dry days count: {len(dry)}, mean: {dry.mean():.2f}")
            print(f" - t-statistic: {t_stat:.3f}, p-value: {p_val:.4f}")
            if p_val < 0.05:
                print(" -> Result: Significant difference in congestion "
                      "between rainy and dry days (p < 0.05).")
            else:
                print(" -> Result: No statistically significant difference (p ≥ 0.05).")
        else:
            print("[TEST 1] Not enough data for rainy vs. dry comparison.")

    # 2) Public holiday vs. normal day: congestion_index
    if "is_public_holiday" in df.columns:
        df["is_public_holiday"] = df["is_public_holiday"].fillna(0)
        df["is_public_holiday"] = df["is_public_holiday"].astype(int) == 1

        holiday = df.loc[df["is_public_holiday"], "congestion_index"].dropna()
        non_holiday = df.loc[~df["is_public_holiday"], "congestion_index"].dropna()

        if len(holiday) > 5 and len(non_holiday) > 5:
            t_stat, p_val = stats.ttest_ind(holiday, non_holiday, equal_var=False)
            print("\n[TEST 2] Mean traffic congestion on public holidays vs. normal days")
            print(f" - Holiday days count: {len(holiday)}, mean: {holiday.mean():.2f}")
            print(f" - Non-holiday days count: {len(non_holiday)}, mean: {non_holiday.mean():.2f}")
            print(f" - t-statistic: {t_stat:.3f}, p-value: {p_val:.4f}")
            if p_val < 0.05:
                print(" -> Result: Significant difference in congestion "
                      "between holidays and normal days (p < 0.05).")
            else:
                print(" -> Result: No statistically significant difference (p ≥ 0.05).")
        else:
            print("[TEST 2] Not enough data for holiday vs. normal day comparison.")

    # 3) Correlation: congestion vs. PM2.5
    if "pm25" in df.columns:
        x = df["pm25"].astype(float)
        y = df["congestion_index"].astype(float)
        mask = x.notna() & y.notna()
        if mask.sum() > 10:
            r, p_val = stats.pearsonr(x[mask], y[mask])
            print("\n[TEST 3] Correlation between PM2.5 and congestion_index (Pearson)")
            print(f" - r: {r:.3f}, p-value: {p_val:.4f}")
            if p_val < 0.05:
                print(" -> Result: Significant linear relationship (p < 0.05).")
            else:
                print(" -> Result: No statistically significant linear relationship (p ≥ 0.05).")
        else:
            print("[TEST 3] Not enough data for correlation test PM2.5 vs congestion.")


def main():
    ensure_dirs()
    df = load_and_merge()
    df = add_time_features(df)
    basic_eda(df)
    plot_time_series(df)
    plot_distributions(df)
    plot_correlations(df)
    hypothesis_tests(df)
    print("\n[INFO] EDA and hypothesis tests completed successfully.")



main()