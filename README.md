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

## 7) Risks & Mitigations
- **Data availability gaps:** If an official traffic history is limited, fall back to reputable mirrors (e.g., Kaggle/TomTom indices) with clear caveats.
- **API rate limits / access:** Cache raw pulls to disk; document schemas; keep a small sample CSV in the repo to allow quick tests.
- **Schema changes:** Use validation checks in the ETL to fail fast when columns drift.

## 8) Repository Structure (initial)
```
.
├── README.md
├── requirements.txt
├── src/
│   ├── collect_traffic.py           # placeholder
│   ├── collect_weather.py           # placeholder
│   ├── collect_holidays.py          # placeholder
│   ├── collect_air_quality.py       # optional
│   ├── merge_build_features.py      # placeholder
│   └── utils.py                     # placeholder
├── notebooks/
│   └── 00_eda.ipynb                 # to be added
└── data/
    ├── raw/                         # do not commit large files
    └── processed/
```
## 9) Reproducibility
- Python ≥ 3.10; `pip install -r requirements.txt`
- A `.env` (not committed) may store API keys if needed (e.g., WAQI token).

---

## Limitations & Future Work  
- **Data Gaps:** Weather or AQI data might have missing days.  
- **Granularity:** Daily averages may hide short-term traffic variations.  
- **Future Work:** Include hourly traffic feeds, integrate mobility data (Google Maps, TomTom), and test nonlinear models (e.g., Random Forests, LSTM).
