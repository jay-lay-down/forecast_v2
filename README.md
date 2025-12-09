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
     - **LassoCV** on standardized data to estimate driver contributions (records per-window R²/MAE).
     - **RandomForestRegressor** feature importance (records per-window R²/MAE).
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

## 한국어 요약
- **무엇을 하나요?** 한 시트짜리 Excel에서 브랜드별 파워 지표(`_POWER`)와 여러 드라이버(`_TOM`, `_AFFINITY` 등)를 읽어, 12개월 롤링 윈도우마다 Lasso·랜덤포레스트·VAR(Granger, IRF) 분석을 수행합니다. 분석 결과는 두 개의 장형 CSV(`rolling_contrib_long.csv`, `rolling_score_long.csv`)로 저장되고, Streamlit 대시보드가 이를 시각화합니다.
- **왜 장형(long)으로 바꾸나요?** 월·변수 단위로 정규화된 Score와 월별 기여도(Contribution)를 갖추면, 대시보드에서 브랜드/시점별 필터링과 차트(스택 바, 워터폴, 버블)를 일관되게 그릴 수 있기 때문입니다.
- **Score 구조**: 네 가지 신호를 0~1로 정규화한 뒤 가중합합니다.
  - `G_s`: Granger 유의성(유/무) 
  - `I_s`: IRF에서 유의한 스텝의 수(지속성·방향성 참고)
  - `M_s`: 표준화된 Lasso 계수의 절댓값
  - `S_s`: 랜덤포레스트 중요도
  - 가중치 기본값은 `wG=0.20`, `wI=0.30`, `wM=0.25`, `wS=0.25`이며, 대시보드에서 시각화용으로 조정 가능합니다. 평균을 100으로 맞춘 `Index`도 함께 저장됩니다.
- **기여도(Contribution)**: 윈도우 내 표준화 후 산출된 Lasso 계수 × 마지막 시점 입력값으로 월별 상승·하락 요인을 계산합니다.
- **대시보드 흐름**: 브랜드/시점을 고르고, 좌측에서 Score 분해(가중치 슬라이더·스택 바·세부 테이블)를, 우측에서 당월 워터폴을, 추가로 트렌드+스택 기여도와 Index vs IRF 버블 맵을 제공합니다.

## 향후 개선 아이디어
- **교차검증 전략 튜닝**: 데이터가 짧을 때 `LassoCV`의 폴드 수와 알파 그리드를 자동 축소하거나, ElasticNet을 옵션으로 제공해 과적합·희소성의 균형을 잡을 수 있습니다.
- **정규화 일관성**: Score와 Index 계산 시 브랜드·시점별 스케일 차이를 줄이기 위해 전 기간/전 브랜드 기준 정규화 옵션을 추가하면 비교 가능성이 높아집니다.
- **변수 선택 자동화**: 드라이버 컬럼 탐지 시 정규식/메타데이터 맵핑을 지원하거나, 결측률·상관 기반 사전 필터링을 넣어 안정성을 높일 수 있습니다.
- **시뮬레이션 모드**: 대시보드에서 가중치와 드라이버 값을 조정해 가상의 Power 변동을 시뮬레이션하는 인터랙티브 도구를 추가하면 의사결정에 도움이 됩니다.
- **모델 성능 로그**: 각 윈도우별 R², MAE 등의 간단한 성능 지표를 CSV에 함께 저장해 모델 신뢰도를 모니터링하도록 확장할 수 있습니다.
