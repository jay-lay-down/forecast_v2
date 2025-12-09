# =========================
# [CELL 1] Install + Drive + Paths
# =========================
!pip -q install pandas numpy openpyxl scikit-learn statsmodels plotly streamlit

from google.colab import drive
drive.mount("/content/drive")

EXCEL_PATH      = "/content/drive/MyDrive/fake/fake.xlsx"
OUT_CONTRIB_CSV = "/content/drive/MyDrive/fake/rolling_contrib_long.csv"
OUT_SCORE_CSV   = "/content/drive/MyDrive/fake/rolling_score_long.csv"
APP_PATH        = "/content/drive/MyDrive/fake/app.py"

print("EXCEL_PATH     :", EXCEL_PATH)
print("OUT_CONTRIB_CSV:", OUT_CONTRIB_CSV)
print("OUT_SCORE_CSV  :", OUT_SCORE_CSV)
print("APP_PATH       :", APP_PATH)

# =========================
# [CELL 2] Build rolling_contrib_long + rolling_score_long
# =========================
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

from statsmodels.tsa.api import VAR

# -----------------
# Utils
# -----------------
def parse_month(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "month" in df.columns:
        df["Date"] = pd.to_datetime(df["month"].astype(str) + "-01", errors="coerce")
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        raise ValueError("Need a 'month' (YYYY-MM) column or a 'Date' column.")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df

def load_first_sheet_excel(path: str) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    sheet = xls.sheet_names[0]
    return pd.read_excel(path, sheet_name=sheet)

def extract_brands_from_wide(df: pd.DataFrame):
    brands = set()
    for c in df.columns:
        if "_" in c and c not in ("month", "Date"):
            brands.add(c.split("_", 1)[0])
    return sorted(brands)

def safe_minmax(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    mn, mx = np.nanmin(x.values), np.nanmax(x.values)
    if not np.isfinite(mn) or not np.isfinite(mx) or abs(mx - mn) < 1e-12:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mn) / (mx - mn)

def mom_yoy(power_series: pd.Series, idx: int) -> tuple[float, float]:
    """Return MoM, YoY from a series aligned to rolling window end index."""
    mom = np.nan
    yoy = np.nan
    if idx - 1 >= 0:
        mom = float(power_series.iloc[idx] - power_series.iloc[idx - 1])
    if idx - 12 >= 0:
        yoy = float(power_series.iloc[idx] - power_series.iloc[idx - 12])
    return mom, yoy

# -----------------
# Core: per brand rolling
# -----------------
CANDIDATE = ["TOM", "AFFINITY", "DYNAMISM", "NEEDS", "UNIQUE", "P3M", "P4W", "P7D"]

def rolling_compute_for_brand(
    wide_df: pd.DataFrame,
    brand: str,
    window: int = 12,
    maxlag_granger: int = 6,
    irf_steps: int = 12,
    drop_zero_power: bool = True,
    min_train_rows: int = 14,
    weights: dict = None
):
    """
    Returns:
      contrib_long_df: Date, Brand, Power, Variable, Contribution, Model
      score_long_df:   Date, Brand, Variable, components + normalized + Score + Index + metadata
    """
    if weights is None:
        # 너가 나중에 대시보드에서 바꿀 수 있게 app에서도 입력받게 함
        weights = {"wG": 0.20, "wI": 0.30, "wM": 0.25, "wS": 0.25}

    df = parse_month(wide_df)

    power_col = f"{brand}_POWER"
    if power_col not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    drivers = [m for m in CANDIDATE if f"{brand}_{m}" in df.columns]
    if not drivers:
        return pd.DataFrame(), pd.DataFrame()

    use_cols = ["Date", power_col] + [f"{brand}_{m}" for m in drivers]
    d = df[use_cols].copy()

    # optional: unreleased(0) 구간 제거
    if drop_zero_power:
        d = d[d[power_col].astype(float) > 0].reset_index(drop=True)

    if len(d) < max(window, min_train_rows):
        return pd.DataFrame(), pd.DataFrame()

    dates = d["Date"].tolist()
    y_all = d[power_col].astype(float).values
    X_all = np.column_stack([d[f"{brand}_{m}"].astype(float).values for m in drivers])

    contrib_rows = []
    score_rows = []

    # for MoM/YoY reference (aligned to d)
    y_series_all = pd.Series(y_all)

    for end in range(window - 1, len(dates)):
        start = end - window + 1
        date_t = pd.to_datetime(dates[end])
        power_t = float(y_all[end])
        mom, yoy = mom_yoy(y_series_all, end)

        X_win = X_all[start:end+1]
        y_win = y_all[start:end+1]

        # ---------- 1) LassoCV (coef abs / contribution) ----------
        scaler = StandardScaler()
        Xs_win = scaler.fit_transform(X_win)

        cv_folds = min(5, len(y_win))
        lasso = LassoCV(
            alphas=np.logspace(-3, 0, 50),
            cv=cv_folds,
            random_state=42,
            max_iter=20000
        )
        try:
            lasso.fit(Xs_win, y_win)
            coefs_std = lasso.coef_.astype(float)  # in standardized space
            y_pred_lasso = lasso.predict(Xs_win)
        except Exception:
            coefs_std = np.zeros(len(drivers), dtype=float)
            y_pred_lasso = np.full_like(y_win, np.nan, dtype=float)

        # contribution: coef_std * x_t_std
        x_t_std = Xs_win[-1]
        contrib = coefs_std * x_t_std

        # coef magnitude for score component (M)
        M_raw = np.abs(coefs_std)
        try:
            r2_lasso = float(r2_score(y_win, y_pred_lasso)) if np.all(np.isfinite(y_pred_lasso)) else np.nan
            mae_lasso = float(mean_absolute_error(y_win, y_pred_lasso)) if np.all(np.isfinite(y_pred_lasso)) else np.nan
        except Exception:
            r2_lasso = np.nan
            mae_lasso = np.nan

        # ---------- 2) RandomForest importance (S) ----------
        try:
            rf = RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                min_samples_leaf=2
            )
            rf.fit(X_win, y_win)
            S_raw = rf.feature_importances_.astype(float)
            y_pred_rf = rf.predict(X_win)
        except Exception:
            S_raw = np.zeros(len(drivers), dtype=float)
            y_pred_rf = np.full_like(y_win, np.nan, dtype=float)

        try:
            r2_rf = float(r2_score(y_win, y_pred_rf)) if np.all(np.isfinite(y_pred_rf)) else np.nan
            mae_rf = float(mean_absolute_error(y_win, y_pred_rf)) if np.all(np.isfinite(y_pred_rf)) else np.nan
        except Exception:
            r2_rf = np.nan
            mae_rf = np.nan

        # ---------- 3) VAR 기반 Best Lag + Granger + IRF (G, I, Direction) ----------
        # per driver, fit VAR([power, driver]) on window
        G_flag = np.zeros(len(drivers), dtype=float)
        best_lag = np.full(len(drivers), np.nan)
        p_granger = np.full(len(drivers), np.nan)
        direction = np.full(len(drivers), np.nan)
        IRF_total = np.zeros(len(drivers), dtype=float)  # I_raw (0..irf_steps)
        IRF_duration = np.zeros(len(drivers), dtype=float)  # same as total for now

        for j, m in enumerate(drivers):
            x = X_win[:, j].astype(float)
            pair = pd.DataFrame({"POWER": y_win.astype(float), "X": x})
            # VAR can fail if singular or not enough data
            try:
                model = VAR(pair)
                # choose lag by AIC (<= maxlag_granger)
                res = model.fit(maxlags=maxlag_granger, ic="aic")
                lag = int(res.k_ar) if res.k_ar is not None else 1
                lag = max(1, min(lag, maxlag_granger))
                best_lag[j] = lag

                # Granger causality: does X cause POWER?
                # statsmodels: test_causality(caused, causing)
                test = res.test_causality(caused="POWER", causing=["X"], kind="f")
                pv = float(test.pvalue) if np.isfinite(test.pvalue) else np.nan
                p_granger[j] = pv
                G_flag[j] = 1.0 if (np.isfinite(pv) and pv < 0.05) else 0.0

                # Directionality: sign of impulse response (sum of IRF mean response)
                # IRF: response of POWER to shock in X
                try:
                    irf = res.irf(irf_steps)
                    # IRF array: (steps+1, neqs, neqs) with order same as columns
                    # columns order: ["POWER","X"]
                    resp = irf.irfs[:, 0, 1]  # POWER response to X shock
                    # err bands (MC): may fail; if fails, fallback to simple duration proxy by abs(resp)
                    try:
                        lower, upper = irf.errband_mc(repl=200, orth=False, signif=0.05)
                        # lower/upper shape: (steps+1, neqs, neqs)
                        lo = lower[:, 0, 1]
                        up = upper[:, 0, 1]
                        sig = ((lo > 0) | (up < 0)).astype(int)  # 0 not in CI
                        IRF_total[j] = float(sig.sum())
                        IRF_duration[j] = float(sig.sum())
                    except Exception:
                        # fallback: count steps where |resp| above tiny threshold
                        sig = (np.abs(resp) > 1e-6).astype(int)
                        IRF_total[j] = float(sig.sum())
                        IRF_duration[j] = float(sig.sum())

                    direction[j] = float(np.sign(np.nansum(resp))) if np.isfinite(np.nansum(resp)) else 0.0
                except Exception:
                    direction[j] = 0.0
                    IRF_total[j] = 0.0
                    IRF_duration[j] = 0.0

            except Exception:
                # leave defaults
                continue

        # ---------- Build contrib_long ----------
        for j, m in enumerate(drivers):
            contrib_rows.append({
                "Date": date_t,
                "Brand": brand,
                "Power": power_t,
                "Variable": m,
                "Contribution": float(contrib[j]),
                "Model": "LassoCV_std_coef_x_std_x"
            })

        # ---------- Normalize per-date across variables (for score) ----------
        # G already 0/1
        tmp = pd.DataFrame({
            "Variable": drivers,
            "G_raw": G_flag,
            "I_raw": IRF_total,  # IRF total significance count
            "M_raw": M_raw,      # |lasso coef| in standardized space
            "S_raw": S_raw,      # RF importance
            "BestLag": best_lag,
            "GrangerP": p_granger,
            "Direction": direction,
            "IRF_Total": IRF_total,
            "IRF_Duration": IRF_duration,
        })

        tmp["I_s"] = safe_minmax(tmp["I_raw"])
        tmp["M_s"] = safe_minmax(tmp["M_raw"])
        tmp["S_s"] = safe_minmax(tmp["S_raw"])
        tmp["G_s"] = tmp["G_raw"].astype(float)  # already 0/1

        wG, wI, wM, wS = weights["wG"], weights["wI"], weights["wM"], weights["wS"]
        tmp["Score_G"] = wG * tmp["G_s"]
        tmp["Score_I"] = wI * tmp["I_s"]
        tmp["Score_M"] = wM * tmp["M_s"]
        tmp["Score_S"] = wS * tmp["S_s"]
        tmp["Score"] = tmp["Score_G"] + tmp["Score_I"] + tmp["Score_M"] + tmp["Score_S"]

        avg_score = float(tmp["Score"].mean()) if np.isfinite(tmp["Score"].mean()) and tmp["Score"].mean() > 1e-12 else 1.0
        tmp["Index"] = 100.0 * tmp["Score"] / avg_score

        # add meta columns
        tmp["Date"] = date_t
        tmp["Brand"] = brand
        tmp["Power"] = power_t
        tmp["Power_MoM"] = mom
        tmp["Power_YoY"] = yoy
        tmp["R2_Lasso"] = r2_lasso
        tmp["MAE_Lasso"] = mae_lasso
        tmp["R2_RF"] = r2_rf
        tmp["MAE_RF"] = mae_rf

        score_rows.append(tmp)

    contrib_long_df = pd.DataFrame(contrib_rows)
    score_long_df = pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()
    return contrib_long_df, score_long_df

# -----------------
# Run pipeline (all brands)
# -----------------
wide_raw = load_first_sheet_excel(EXCEL_PATH)
wide = parse_month(wide_raw)
brands = extract_brands_from_wide(wide)

print("Detected brands:", len(brands))
print(brands[:20])

all_contrib = []
all_score = []

for b in brands:
    cdf, sdf = rolling_compute_for_brand(
        wide, b,
        window=12,
        maxlag_granger=6,
        irf_steps=12,
        drop_zero_power=True,
        min_train_rows=14,
        weights={"wG":0.20, "wI":0.30, "wM":0.25, "wS":0.25}
    )
    if not cdf.empty:
        all_contrib.append(cdf)
    if not sdf.empty:
        all_score.append(sdf)

contrib_df = pd.concat(all_contrib, ignore_index=True) if all_contrib else pd.DataFrame()
score_df   = pd.concat(all_score, ignore_index=True) if all_score else pd.DataFrame()

print("contrib_df shape:", contrib_df.shape)
print("score_df shape  :", score_df.shape)

if contrib_df.empty or score_df.empty:
    raise RuntimeError("No outputs created. Check POWER columns/data format or window size.")

contrib_df.to_csv(OUT_CONTRIB_CSV, index=False, encoding="utf-8-sig")
score_df.to_csv(OUT_SCORE_CSV, index=False, encoding="utf-8-sig")
print("✅ Saved:", OUT_CONTRIB_CSV)
print("✅ Saved:", OUT_SCORE_CSV)

# =========================
# [CELL 3] Write app.py (Streamlit)
# =========================
app_code = r'''
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Forecasting Driver Dashboard", layout="wide")
st.title("📊 Driver Importance (Index) + Contribution (Decomposition)")
st.caption("왼쪽: 종합 중요도(Index)와 분해(Score_G/I/M/S) / 오른쪽: 월별 기여도(Contribution)")

DEFAULT_CONTRIB = "/content/drive/MyDrive/fake/rolling_contrib_long.csv"
DEFAULT_SCORE   = "/content/drive/MyDrive/fake/rolling_score_long.csv"

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date"]).sort_values("Date")

st.sidebar.header("⚙️ Data")
contrib_path = st.sidebar.text_input("rolling_contrib_long.csv", DEFAULT_CONTRIB)
score_path   = st.sidebar.text_input("rolling_score_long.csv", DEFAULT_SCORE)

contrib = load_csv(contrib_path)
score   = load_csv(score_path)

# Brand filter
brands = sorted(set(contrib["Brand"].dropna().unique()).intersection(set(score["Brand"].dropna().unique())))
if not brands:
    st.error("No common brands found between contrib and score files.")
    st.stop()

selected_brand = st.sidebar.selectbox("브랜드", brands, index=0)
contrib_b = contrib[contrib["Brand"] == selected_brand].copy()
score_b   = score[score["Brand"] == selected_brand].copy()

# Date selector (use intersection)
dates = sorted(set(contrib_b["Date"].unique()).intersection(set(score_b["Date"].unique())))
if not dates:
    st.error("No common dates for selected brand.")
    st.stop()

selected_date = st.sidebar.select_slider(
    "분석 시점 (Rolling Window End)",
    options=dates,
    format_func=lambda x: pd.to_datetime(x).strftime("%Y-%m")
)

# weights (dashboard only; pipeline weights already baked in, but you can re-weight visually)
st.sidebar.subheader("🎛️ Score Weights (display)")
wG = st.sidebar.slider("wG (Granger)", 0.0, 1.0, 0.20, 0.01)
wI = st.sidebar.slider("wI (IRF)",     0.0, 1.0, 0.30, 0.01)
wM = st.sidebar.slider("wM (Lasso)",   0.0, 1.0, 0.25, 0.01)
wS = st.sidebar.slider("wS (RF)",      0.0, 1.0, 0.25, 0.01)
w_sum = wG+wI+wM+wS
if w_sum <= 1e-12:
    st.sidebar.warning("Weights sum is 0. Using defaults for display.")
    wG,wI,wM,wS = 0.20,0.30,0.25,0.25
    w_sum = 1.0

# current slices
cur_contrib = contrib_b[contrib_b["Date"] == selected_date].copy()
cur_score   = score_b[score_b["Date"] == selected_date].copy()

# ----- KPI -----
k1,k2,k3,k4 = st.columns(4)
power = float(cur_score["Power"].mean()) if "Power" in cur_score.columns and not cur_score.empty else np.nan
mom   = float(cur_score["Power_MoM"].mean()) if "Power_MoM" in cur_score.columns and not cur_score.empty else np.nan
yoy   = float(cur_score["Power_YoY"].mean()) if "Power_YoY" in cur_score.columns and not cur_score.empty else np.nan
r2_display = float(cur_score["R2_Lasso"].mean()) if "R2_Lasso" in cur_score.columns and not cur_score.empty else np.nan
r2_rf_display = float(cur_score["R2_RF"].mean()) if "R2_RF" in cur_score.columns and not cur_score.empty else np.nan

with k1:
    st.metric("Power", f"{power:.2f}" if np.isfinite(power) else "-", f"{mom:+.2f}" if np.isfinite(mom) else None)
with k2:
    st.metric("YoY", f"{yoy:+.2f}" if np.isfinite(yoy) else "-")
with k3:
    st.metric("Vars (score)", f"{len(cur_score)}")
with k4:
    st.metric("Vars (contrib)", f"{len(cur_contrib)}")

def _fmt_metric(x: float) -> str:
    return f"{x:.3f}" if np.isfinite(x) else "-"

st.caption(
    "학습 창 내 모델 성능 지표(평균) — "
    f"Lasso R²: {_fmt_metric(r2_display)}, RF R²: {_fmt_metric(r2_rf_display)}"
)

st.divider()

# ----- Recompute display score using adjustable weights (from normalized columns) -----
# Use *_s columns if present
need_cols = ["G_s","I_s","M_s","S_s"]
missing = [c for c in need_cols if c not in cur_score.columns]
if missing:
    st.warning(f"Score normalization columns missing: {missing}. Showing stored Score/Index only.")
    cur_score["Score_disp"] = cur_score.get("Score", np.nan)
    cur_score["Index_disp"] = cur_score.get("Index", np.nan)
    cur_score["Score_G_disp"] = cur_score.get("Score_G", np.nan)
    cur_score["Score_I_disp"] = cur_score.get("Score_I", np.nan)
    cur_score["Score_M_disp"] = cur_score.get("Score_M", np.nan)
    cur_score["Score_S_disp"] = cur_score.get("Score_S", np.nan)
else:
    cur_score["Score_G_disp"] = (wG/w_sum) * cur_score["G_s"]
    cur_score["Score_I_disp"] = (wI/w_sum) * cur_score["I_s"]
    cur_score["Score_M_disp"] = (wM/w_sum) * cur_score["M_s"]
    cur_score["Score_S_disp"] = (wS/w_sum) * cur_score["S_s"]
    cur_score["Score_disp"]   = cur_score["Score_G_disp"] + cur_score["Score_I_disp"] + cur_score["Score_M_disp"] + cur_score["Score_S_disp"]
    avg = float(cur_score["Score_disp"].mean()) if cur_score["Score_disp"].mean() > 1e-12 else 1.0
    cur_score["Index_disp"]   = 100.0 * cur_score["Score_disp"] / avg

# ----- Layout: left(score) / right(contribution) -----
left, right = st.columns([1.05, 0.95])

with left:
    st.subheader("A) 종합 중요도(Index) + 분해(Score Breakdown)")

    topN = st.slider("Top N variables", 5, 30, 12, 1)
    show = cur_score.sort_values("Index_disp", ascending=False).head(topN).copy()

    # stacked bar breakdown
    fig_break = go.Figure()
    fig_break.add_trace(go.Bar(x=show["Variable"], y=show["Score_G_disp"], name="Granger (G)"))
    fig_break.add_trace(go.Bar(x=show["Variable"], y=show["Score_I_disp"], name="IRF (I)"))
    fig_break.add_trace(go.Bar(x=show["Variable"], y=show["Score_M_disp"], name="Lasso (M)"))
    fig_break.add_trace(go.Bar(x=show["Variable"], y=show["Score_S_disp"], name="RF (S)"))
    fig_break.update_layout(
        barmode="stack", height=420,
        title="Score Breakdown (stacked) — higher = more important",
        xaxis_title="Variable", yaxis_title="Score (weighted)"
    )
    st.plotly_chart(fig_break, use_container_width=True)

    # table (with meta)
    cols = ["Variable","Index_disp","Score_disp","GrangerP","BestLag","Direction","IRF_Total","M_raw","S_raw","R2_Lasso","R2_RF","MAE_Lasso","MAE_RF"]
    existing = [c for c in cols if c in show.columns]
    show_tbl = show[existing].copy()
    if "Index_disp" in show_tbl.columns:
        show_tbl["Index_disp"] = show_tbl["Index_disp"].round(2)
    if "Score_disp" in show_tbl.columns:
        show_tbl["Score_disp"] = show_tbl["Score_disp"].round(4)
    if "GrangerP" in show_tbl.columns:
        show_tbl["GrangerP"] = pd.to_numeric(show_tbl["GrangerP"], errors="coerce").round(4)

    st.dataframe(show_tbl, use_container_width=True)

with right:
    st.subheader("B) 월별 기여도(Contribution) — 이번 달에 올린/내린 요인")

    if cur_contrib.empty:
        st.info("No contribution rows for selected date.")
    else:
        wf = cur_contrib.sort_values("Contribution", ascending=False).copy()
        fig_wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative"] * len(wf),
            x=wf["Variable"],
            y=wf["Contribution"],
            text=[f"{x:+.3f}" for x in wf["Contribution"]],
            textposition="outside",
        ))
        fig_wf.update_layout(height=420, title="Waterfall — Contribution by Variable")
        st.plotly_chart(fig_wf, use_container_width=True)

    st.caption("Tip: Index는 '중요도(장기 관리)' / Contribution은 '해당 월 변화 요인(단기 등락)'이라 둘이 동시에 봐야 함.")

st.divider()

# ----- Trend + Contribution (stacked) -----
st.subheader("C) Trend + Decomposition (Contribution over time)")
trend = score_b.groupby("Date")["Power"].mean().reset_index() if "Power" in score_b.columns else None

fig_trend = go.Figure()
if trend is not None and not trend.empty:
    fig_trend.add_trace(go.Scatter(
        x=trend["Date"], y=trend["Power"],
        mode="lines+markers", name="Power"
    ))

# stacked contributions (only for selected brand)
vars_all = sorted(contrib_b["Variable"].dropna().unique().tolist())
# optional: limit too many vars
max_vars_plot = st.slider("Max vars in stacked contribution plot", 5, 30, 12, 1)
# pick top vars by latest Index to keep readable
latest = score_b[score_b["Date"] == max(dates)].copy()
if "Index" in latest.columns:
    top_vars = latest.sort_values("Index", ascending=False)["Variable"].head(max_vars_plot).tolist()
else:
    top_vars = vars_all[:max_vars_plot]

for v in top_vars:
    vd = contrib_b[contrib_b["Variable"] == v]
    fig_trend.add_trace(go.Bar(x=vd["Date"], y=vd["Contribution"], name=str(v), opacity=0.65))

fig_trend.update_layout(barmode="relative", height=520, title="Power (line) + Contribution (stacked bars)")
st.plotly_chart(fig_trend, use_container_width=True)

# ----- Bubble: Importance vs Duration -----
st.subheader("D) 전략 맵 — 중요도 vs 지속성 (Index vs IRF)")
if "IRF_Total" in cur_score.columns:
    bubble = cur_score.copy()
    bubble["BubbleSize"] = np.clip(bubble.get("Index_disp", bubble.get("Index", 0)), 0, None)
    fig_b = px.scatter(
        bubble,
        x="IRF_Total",
        y="Index_disp" if "Index_disp" in bubble.columns else "Index",
        size="BubbleSize",
        color="Variable",
        hover_name="Variable",
        size_max=55,
        title="Index vs IRF_Total (bubble size = Index)"
    )
    fig_b.update_layout(height=520)
    st.plotly_chart(fig_b, use_container_width=True)
else:
    st.info("IRF_Total column not found in score file.")

with st.expander("🔎 Raw preview (current date)"):
    st.write("Score rows:")
    st.dataframe(cur_score.reset_index(drop=True), use_container_width=True)
    st.write("Contribution rows:")
    st.dataframe(cur_contrib.reset_index(drop=True), use_container_width=True)
'''
with open(APP_PATH, "w", encoding="utf-8") as f:
    f.write(app_code)

print("✅ Wrote Streamlit app:", APP_PATH)

# =========================
# [CELL 4] Run Streamlit in Colab
# =========================
!npm -q install -g localtunnel

# 1) run streamlit (leave this running)
!streamlit run /content/drive/MyDrive/fake/app.py --server.port 8501 --server.address 0.0.0.0

# =========================
# [CELL 5] Expose via localtunnel
# =========================
!lt --port 8501
