# Forecast Driver Pipeline

This repository provides a Colab-friendly pipeline to transform a wide-format Excel file into rolling driver analytics and a Streamlit dashboard for visualization.

## Overview
- **Input**: Wide Excel file with monthly columns for each brand (`<BRAND>_POWER`, `<BRAND>_<DRIVER>`).
- **Outputs**:
  - `rolling_contrib_long.csv`: Per-date, per-driver contribution values from rolling Lasso models.
  - `rolling_score_long.csv`: Driver importance scores combining Granger causality, impulse response, Lasso coefficients, and random-forest importance.
- **Dashboard**: Streamlit app (`app.py`) to explore driver importance (Index) and monthly contributions.

## Colab Pipeline
The included `app.py` is structured as Colab cells. Execute cells sequentially.

1. **Install & Mount Drive**
   - Installs required packages.
   - Mounts Google Drive and defines paths:
     - `EXCEL_PATH`: Source Excel (first sheet used).
     - `OUT_CONTRIB_CSV` / `OUT_SCORE_CSV`: Output CSVs.
     - `APP_PATH`: Streamlit app destination in Drive.

2. **Build Rolling Outputs**
   - Parses dates from `month` or `Date` columns.
   - Detects brands via `<BRAND>_POWER` and driver columns (`TOM`, `AFFINITY`, `DYNAMISM`, `NEEDS`, `UNIQUE`, `P3M`, `P4W`, `P7D`).
   - For each brand, uses a 12-month rolling window (minimum 14 rows):
     - **LassoCV** on standardized data to estimate driver contributions.
     - **RandomForestRegressor** feature importance.
     - **VAR** per driver to compute Granger causality, impulse-response significance/direction, and best lags.
     - Computes normalized score components (G/I/M/S) with weights `wG=0.20`, `wI=0.30`, `wM=0.25`, `wS=0.25`, plus Index (mean=100).
     - Records Power, MoM, and YoY deltas for context.
   - Saves `rolling_contrib_long.csv` and `rolling_score_long.csv` to Drive.

3. **Generate Streamlit App**
   - Writes `app.py` to `APP_PATH` using the CSVs above.
   - Dashboard features:
     - Adjustable weights (for display) of G/I/M/S components.
     - KPI cards for Power, MoM, YoY, and variable counts.
     - Stacked bar breakdown of Index components and detail table.
     - Waterfall chart of current-month contributions.
     - Trend line (Power) with stacked contributions over time.
     - Bubble chart of Index vs IRF_Total.
     - Raw data preview for the selected date.

4. **Run Dashboard in Colab**
   - Installs `localtunnel`, runs Streamlit on port 8501, and exposes via `lt --port 8501` for external access.

## Usage Notes
- Input data must include a `month` (`YYYY-MM`) or `Date` column and brand-specific POWER/driver columns.
- Rows with zero POWER are dropped by default before modeling.
- Adjust rolling window, Granger lag, IRF steps, or weight parameters inside `rolling_compute_for_brand` as needed.
- The dashboard requires both output CSVs and assumes overlapping brands/dates between them.

## Files
- `app.py` – Colab-ready pipeline and dashboard generator.
- `LICENSE` – Apache 2.0.
