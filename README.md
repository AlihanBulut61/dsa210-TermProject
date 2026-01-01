# Istanbul Traffic Congestion, Weather, Holidays and Air Quality

## Overview of the Project  
This project investigates the relationship between **Istanbul’s daily traffic congestion** and several external factors such as **weather conditions, public & school holidays, and air quality levels**.  
The objective is to analyze how these variables influence traffic intensity and to develop statistical and predictive models to better understand and forecast congestion patterns.

---

## Motivation  
Istanbul is one of the most congested cities in the world, and its traffic varies drastically with season, weekday, and even weather changes.  
Understanding how **rain, temperature, wind**, and **holidays** impact congestion can help improve urban mobility planning and reduce travel delays.  


---

## Objectives  
- Analyze how different weather patterns influence traffic congestion levels in Istanbul.  
- Examine how public and school holidays affect average congestion.  
- Study whether air quality indicators (PM2.5, PM10, AQI) correlate with traffic intensity.  
- Build regression and time-series models to predict daily congestion levels.  

---

## Information Collection  
The project will integrate **multiple open datasets** related to traffic, weather, holidays, and air quality.  
Data will be collected programmatically (via APIs) or from official CSV sources, cleaned, merged, and analyzed.  

| Variable | Description | Source |
|-----------|--------------|---------|
| `date` | Observation date | Common key for merging |
| `congestion_index` | Daily average congestion (%) | İstanbul Metropolitan Municipality (İBB) Open Data Portal |
| `temperature` | Mean daily temperature (°C) | Turkish State Meteorological Service (MGM) |
| `precipitation` | Daily total rainfall (mm) | MGM or open-weather API |
| `wind_speed` | Daily mean wind speed (m/s) | MGM |
| `is_public_holiday` | Indicator of public holiday | Turkey official holiday calendar (Gov API / CSV) |
| `is_school_holiday` | Indicator of school closure periods | Ministry of Education academic calendar |
| `aqi`, `pm25`, `pm10` | Air quality metrics | İBB or WAQI (aqicn.org) API |

---

## Data Sources  
- **Traffic Data:** [İBB Open Data Portal](https://data.ibb.gov.tr) — “Traffic Congestion Index” dataset.  
- **Weather Data:** [Meteoroloji Genel Müdürlüğü](https://www.mgm.gov.tr/) or Open-Meteo API.  
- **Holiday Data:** [Gov.tr official holiday API](https://www.resmigazete.gov.tr/) and [Nager.Date API](https://date.nager.at/).  
- **Air Quality Data (optional):** [WAQI](https://aqicn.org/api/) or İBB environmental monitoring feeds.  

All sources will be properly cited in the final repo and report. Any scraping will observe robots.txt / Terms.

## 4) Data Collection Plan
1. **Traffic**
   - If an official İBB endpoint is available, request JSON/CSV and store to `data/raw/traffic/*.csv`.  
   - If a community dataset is used (e.g., Kaggle Istanbul Traffic Index), download the CSV and document its provenance and schema.
2. **Weather**
   - Use MGM’s public data pages or a documented API endpoint to collect **daily** precipitation (mm), max/min temperature (°C), and wind speed. Save to `data/raw/weather/*.csv`.
3. **Holidays & School Calendar**
   - Import a Turkey public‑holiday calendar (CSV/ICS/API) and convert to a daily boolean indicator. For school holidays, ingest the academic calendar to mark term/break days. Save to `data/raw/calendar/*.csv`.
4. **Air Quality (optional)**
   - Query WAQI/İBB environmental monitoring for daily AQI/PM2.5/PM10; aggregate to daily averages. Save to `data/raw/air/*.csv`.
5. **Versioning**
   - Keep all raw files immutable under `data/raw/`. Create cleaned, merged, and feature‑engineered tables under `data/processed/` with scripts in `src/`.
6. **Reproducibility**
   - Provide `src/collect_*.py` scripts with CLI args and a `make all` or `python -m src.pipeline` entry point. Pin dependencies in `requirements.txt`.

## 5) Planned Methods (EDA & Inference)
- **EDA:** seasonal/hourly/daily profiles; heatmaps by weekday×hour; distributions; missingness checks.
- **Hypothesis tests:** two‑sample tests (rain vs no‑rain), correlation analysis, and linear models with holiday/weather dummies.
- **Time‑series controls:** multiple regression on daily congestion with fixed effects for weekday/month, weather, holiday, and AQI controls. Robust SEs.
- **Visualization:** clear, labeled charts; confidence intervals; reproducible notebooks.
- **Ethics:** only aggregated, non‑personal, open data; no PII.

## 6) Deliverables & Milestones
- **By 31 Oct 2025 (Proposal):** This README in GitHub, with an initial `requirements.txt` and skeleton folders.
- **By 28 Nov 2025:** Raw data collected; EDA + initial hypothesis tests committed.
- **By 02 Jan 2026:** Baseline ML model(s) for prediction.
- **By 09 Jan 2026, 23:59:** Final report (notebook or article), polished visualizations, and documented code.
## Phase 3 — Machine Learning Modeling & Results

### Types of Analysis Performed

**Feature Engineering & Preprocessing:**  
The merged daily dataset was prepared for modeling using a time-aware setup (date sorting) and a set of numeric predictors derived from weather / air quality / holiday indicators and basic calendar features.

**Modeling (Supervised Regression):**  
Multiple regression models were trained to predict **`congestion_index`** and compared using cross-validation:
- Linear Regression
- Ridge
- Lasso
- Gradient Boosting
- Random Forest

**Model Evaluation:**  
Performance was evaluated using **MAE**, **RMSE**, and **R²** on both cross-validation and a held-out test period. Residual patterns were also checked via saved diagnostic plots.

> **Note:** The strongest baseline in this phase is a tree-based ensemble (Random Forest), suggesting non-linear effects and interactions in the predictors.

---

### Analysis Summary & Results

#### Cross-Validation Model Comparison (CV)

Results from `outputs/cv_model_comparison.csv`:

| Model | CV RMSE | CV MAE | CV R² | Interpretation |
|------|--------:|-------:|------:|---|
| **Random Forest** | **4.50** | **3.49** | **0.224** | Best overall (lowest errors, positive R²) |
| Gradient Boosting | 5.03 | 3.90 | -0.006 | Near-zero explanatory power |
| Ridge | 6.40 | 5.14 | -0.730 | Weak fit |
| Lasso | 6.56 | 5.27 | -0.878 | Weak fit |
| Linear Regression | Extremely large | Extremely large | Very negative | Unstable / poor fit |

**Comment:**  
Cross-validation indicates **Random Forest** generalizes better than linear and regularized linear approaches for this dataset.

---

#### Best Model Test Performance

Best model from `outputs/best_model_metrics.csv`:

- **Best model:** RandomForest  
- **Test MAE:** **2.850**  
- **Test RMSE:** **3.496**  
- **Test R²:** **0.443**

This means the final model explains approximately **44%** of the variance in daily congestion on the test set, with an average absolute error of about **2.85** congestion-index points.

---

#### Prediction Behavior on the Test Period (from `outputs/test_predictions.csv`)

A quick diagnostic summary of prediction errors on the provided test set:
- **Mean error (bias):** **+2.095** → model tends to **overpredict** congestion slightly overall  
- **Median absolute error:** **2.572**  
- **% predictions within ±3 points:** **57.5%**  
- **% predictions within ±5 points:** **80.8%**  
- **Largest underprediction:** **-3.27** (2023-11-0303)  
- **Largest overprediction:** **+7.59** (2023-12-27)

**Interpretation:**  
The model tracks typical congestion levels reasonably well, but it can miss some low-congestion days (e.g., sudden drops), which increases error peaks.

---

### Saved Outputs

- `models/best_model.joblib` — serialized best model  
- `outputs/cv_model_comparison.csv` — cross-validation comparison table  
- `outputs/best_model_metrics.csv` — final test metrics  
- `outputs/test_predictions.csv` — day-level test predictions  
- `figures/ml_pred_vs_actual.png` — predicted vs actual plot  
- `figures/ml_residuals.png` — residual diagnostics

---

### Conclusion

Across the evaluated models, **Random Forest** achieved the best predictive performance with **Test R² = 0.443** and **RMSE = 3.496**, outperforming linear and regularized regression baselines. Overall, the results suggest that daily congestion patterns are partially predictable from the available external factors, while some sudden regime changes (especially low-congestion days) remain harder to capture within this feature set.

## 7) Risks & Mitigations
- **Data availability gaps:** If an official traffic history is limited, fall back to reputable mirrors (e.g., Kaggle/TomTom indices) with clear caveats.
- **API rate limits / access:** Cache raw pulls to disk; document schemas; keep a small sample CSV in the repo to allow quick tests.
- **Schema changes:** Use validation checks in the ETL to fail fast when columns drift.

## 8) Repository Structure (initial)
```
.
├── README.md
├── requirements.txt
├── eda.py                       # Exploratory Data Analysis & hypothesis tests
├── ml.py                        # Machine Learning models and evaluation
├── output.txt                   # Console outputs from EDA & ML runs
├── data/
│   ├── raw/                     # Raw datasets
│   │   ├── air.csv              # Air quality data (AQI, PM2.5, PM10)
│   │   ├── holidays.csv         # Public & school holiday indicators
│   │   ├── traffic.csv          # Daily traffic congestion index
│   │   └── weather.csv          # Daily weather data
│   └── processed/
│       └── merged_daily_data.csv   # Cleaned and merged dataset used for EDA & ML
├── figures/
│   ├── correlation_heatmap.png
│   ├── dist_congestion_index.png
│   ├── dist_pm10.png
│   ├── dist_pm25.png
│   ├── dist_precipitation.png
│   ├── dist_precipitation_loglog.png
│   ├── dist_temperature.png
│   ├── dist_wind_speed.png
│   ├── time_air_pollution.png
│   ├── time_temperature.png
│   ├── time_traffic_congestion.png
│   ├── ml_pred_vs_actual.png    # ML predictions vs actual values
│   └── ml_residuals.png         # Residual analysis plot
├── models/
│   └── best_model.joblib        # Trained best-performing ML model
└── outputs/
    ├── best_model_metrics.csv   # Test-set performance of the best model
    ├── cv_model_comparison.csv  # Cross-validation comparison of models
    └── test_predictions.csv     # Actual vs predicted congestion values
```
## 9) Reproducibility
- Python ≥ 3.10; `pip install -r requirements.txt`
- A `.env` (not committed) may store API keys if needed (e.g., WAQI token).

---

## Limitations & Future Work  
- **Data Gaps:** Weather or AQI data might have missing days.  
- **Granularity:** Daily averages may hide short-term traffic variations.  
- **Future Work:** Include hourly traffic feeds, integrate mobility data (Google Maps, TomTom), and test nonlinear models (e.g., Random Forests, LSTM).
