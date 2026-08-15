import os
import re
import ast
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.spatial.distance import cdist

# =========================================================================
# 1. 環境與全域設定
# =========================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
TSV_PATH = os.path.join(SCRIPT_DIR, "humob2026-dataset.tsv")
BY_CLASS_DIR = os.path.join(SCRIPT_DIR, "humob2026", "data", "output", "module05", "classification", "by_class")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "humob_pipeline_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_INFO_MAP = {
    1: "Class 01: Persistent Zero",
    2: "Class 02: Persistent Decrease",
    3: "Class 03: Emergent / Temporary Activity",
    4: "Class 04: Partial Recovery",
    5: "Class 05: Fully Recovered",
    6: "Class 06: Stable Inflow",
    7: "Class 07: Temporary Increase",
    8: "Class 08: Partial Dissipation",
    9: "Class 09: Persistent Increase"
}

# 🔴 2~4 月缺測與斷線區間設定 (2024-02-01 ~ 2024-04-30)
GAP_START = pd.to_datetime("2024-02-01")
GAP_END = pd.to_datetime("2024-04-30")
PRED_START = pd.to_datetime("2024-01-01")
PRED_END = pd.to_datetime("2024-10-31")

print("=" * 80)
print("🚀 HuMob 完整預測評估流程：強健統計 ➔ 空間最近鄰矩陣 ➔ 動態均值遷移 ➔ NRMSE ➔ 9大類別繪圖")
print(f"📁 缺測斷線區間: {GAP_START.strftime('%Y-%m-%d')} ~ {GAP_END.strftime('%Y-%m-%d')} (2~4月)")
print(f"📁 輸出目錄: {OUTPUT_DIR}")
print("=" * 80)

# =========================================================================
# 2. 精確解析 9 大類別網格標籤與 TSV 全時空資料集
# =========================================================================
print("\n[1/5] 解析 9 大類別標籤與 TSV 全時空人流資料...")

def get_class_id_from_filename(fname: str) -> int:
    fname = fname.lower()
    if "zero" in fname:
        return 1
    if "decrease" in fname:
        return 2
    if "emergent" in fname or "temporary_activity" in fname:
        return 3
    if "partial_recovery" in fname or "partial_rec" in fname:
        return 4
    if "recovered" in fname:
        return 5
    if "stable" in fname:
        return 6
    if "temporary_increase" in fname or "temp_inc" in fname:
        return 7
    if "partial_dissipation" in fname or "dissip" in fname:
        return 8
    if "persistent_increase" in fname or "increase" in fname:
        return 9
    return None

grid_class_lookup = {}

if os.path.exists(BY_CLASS_DIR):
    csv_files = glob.glob(os.path.join(BY_CLASS_DIR, "*.csv"))
    for fpath in csv_files:
        fname = os.path.basename(fpath)
        c_id = get_class_id_from_filename(fname)

        if c_id is not None:
            try:
                df_cls = pd.read_csv(fpath)
                col = [c for c in df_cls.columns if any(k in str(c).lower() for k in ["grid", "orig", "id"])][0]
                grids_in_file = df_cls[col].dropna().astype(str).unique()
                for g in grids_in_file:
                    grid_class_lookup[g] = c_id
                print(f"  📂 載入 {fname} ➔ {CLASS_INFO_MAP[c_id]} (共 {len(grids_in_file)} 個網格)")
            except Exception as e:
                print(f"  ⚠️ 讀取失敗 {fname}: {e}")

raw_df = pd.read_csv(TSV_PATH, sep="\t", names=["date", "od_matrix_raw"])
raw_df['date_dt'] = pd.to_datetime(raw_df['date'].astype(str), format='%Y%m%d')
raw_df = raw_df.sort_values('date_dt').reset_index(drop=True)

daily_od_records = {}   # 保存 OD 結構用於 NRMSE 評估
daily_grid_flows = {}   # 保存各網格總人流

for dt, val in zip(raw_df['date_dt'], raw_df['od_matrix_raw']):
    daily_od_records[dt] = {}
    daily_grid_flows[dt] = {}
    if pd.isna(val) or val == "NA":
        continue
    try:
        od_dict = ast.literal_eval(val) if isinstance(val, str) else val
        for orig, dests in od_dict.items():
            if orig == "-1_-1":
                continue
            y_idx, x_idx = map(int, orig.split('_'))
            if 30 <= x_idx <= 70 and 35 <= y_idx <= 70:
                daily_od_records[dt][orig] = dests
                daily_grid_flows[dt][orig] = sum(dests.values())
    except Exception:
        pass

flow_df = pd.DataFrame.from_dict(daily_grid_flows, orient='index').fillna(0.0)

# 篩選在分類清單中或震前具備有效訊號的網格
valid_grids = [g for g in flow_df.columns if g in grid_class_lookup]
if not valid_grids:
    pre_mask_temp = flow_df.index < PRED_START
    valid_grids = flow_df.columns[flow_df[pre_mask_temp].mean() >= 0.001].tolist()

flow_df = flow_df[valid_grids]
pre_mask = flow_df.index < PRED_START

print(f"\n✓ 成功配對有效網格總數: {len(valid_grids)} | 觀測天數: {len(flow_df)} 天")
for c_id in range(1, 10):
    c_count = sum(1 for g in valid_grids if grid_class_lookup.get(g) == c_id)
    print(f"   - {CLASS_INFO_MAP[c_id]}: {c_count} 個網格")

# =========================================================================
# 3. 震前強健統計分解、月內週次週期特徵與空間最近鄰矩陣
# =========================================================================
print("\n[2/5] 執行震前 IQR 離群值修剪、(週次, 星期) 週期提取與空間最近鄰矩陣構建...")

# --- A. 震前時間特徵工程 ---
pre_df = flow_df[pre_mask].copy()
pre_df['dow'] = pre_df.index.dayofweek
pre_df['week_of_month'] = (pre_df.index.day - 1) // 7 + 1  # 該月第幾週 (1~5)

# --- B. 震前強健統計分析：IQR 離群值修剪 + 中位數基準 ---
def compute_robust_baseline(df_feat: pd.DataFrame, grids: list) -> pd.DataFrame:
    clean_records = []
    for (wom, dow), group in df_feat.groupby(['week_of_month', 'dow']):
        grp_grids = group[grids].copy()
        
        # 1. IQR 離群值檢測與截斷 (Clipping)
        q25 = grp_grids.quantile(0.25)
        q75 = grp_grids.quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        clipped_grp = grp_grids.clip(lower=lower_bound, upper=upper_bound, axis=1)
        
        # 2. 計算代表值 (中位數 Median 抵禦極端震動或活動峰值)
        robust_center = clipped_grp.median(axis=0)
        robust_center['week_of_month'] = wom
        robust_center['dow'] = dow
        clean_records.append(robust_center)
        
    pattern_df = pd.DataFrame(clean_records).set_index(['week_of_month', 'dow'])
    return pattern_df

robust_cycle_patterns = compute_robust_baseline(pre_df, valid_grids)
M_pre_robust = pre_df[valid_grids].median().replace(0, 1.0)
dow_medians_pre = pre_df.groupby('dow')[valid_grids].median()

# --- C. 空間座標解析與空間最近鄰矩陣 (Spatial KNN Matrix) ---
def build_spatial_knn_matrix(grids: list, k_neighbors: int = 4) -> pd.DataFrame:
    coords = np.array([[int(c) for c in g.split('_')] for g in grids])
    dist_matrix = cdist(coords, coords, metric='euclidean')
    
    adj_weights = np.zeros_like(dist_matrix)
    for i in range(len(grids)):
        nearest_idx = np.argsort(dist_matrix[i])[:k_neighbors + 1]
        dists = dist_matrix[i, nearest_idx]
        weights = 1.0 / (dists + 1e-5)
        weights /= weights.sum()
        adj_weights[i, nearest_idx] = weights
        
    return pd.DataFrame(adj_weights, index=grids, columns=grids)

spatial_knn_weights = build_spatial_knn_matrix(valid_grids, k_neighbors=4)

# --- D. 震前歷史 OD 轉移機率矩陣 ---
hist_od_trans_prob = {g: {} for g in valid_grids}
for dt in flow_df[pre_mask].index:
    day_od = daily_od_records.get(dt, {})
    for orig in valid_grids:
        if orig in day_od:
            for dest, cnt in day_od[orig].items():
                hist_od_trans_prob[orig][dest] = hist_od_trans_prob[orig].get(dest, 0.0) + cnt

smoothed_od_prob = {}
for orig in valid_grids:
    total_c = sum(hist_od_trans_prob[orig].values())
    if total_c > 0:
        smoothed_od_prob[orig] = {d: c / total_c for d, c in hist_od_trans_prob[orig].items()}
    else:
        smoothed_od_prob[orig] = {orig: 1.0}

# --- E. 關鍵階段錨點提取 (以強健週期因子校正) ---
def get_cycle_factor(dt, grid_list):
    wom = (dt.day - 1) // 7 + 1
    dow = dt.dayofweek
    wom_clamped = min(wom, 4) if (wom, dow) not in robust_cycle_patterns.index else wom
    
    if (wom_clamped, dow) in robust_cycle_patterns.index:
        pattern = robust_cycle_patterns.loc[(wom_clamped, dow)][grid_list]
    else:
        pattern = dow_medians_pre.loc[dow]
        
    factor = pattern / M_pre_robust[grid_list]
    return factor.replace(0, 1.0).fillna(1.0)

# 1. 1 月底衝擊穩態
jan_sub = flow_df.loc["2024-01-25":"2024-01-31"]
jan_factors = pd.DataFrame([get_cycle_factor(dt, valid_grids) for dt in jan_sub.index], index=jan_sub.index)
l_jan_end = (jan_sub / jan_factors).median().fillna(M_pre_robust)

# 2. 5 月初復原錨點 (5/01 ~ 5/07)
may_sub = flow_df.loc["2024-05-01":"2024-05-07"] if "2024-05-01" in flow_df.index else flow_df.loc["2024-04-01":"2024-04-07"]
may_factors = pd.DataFrame([get_cycle_factor(dt, valid_grids) for dt in may_sub.index], index=may_sub.index)
l_resume_start = (may_sub / may_factors).median().fillna(l_jan_end)
l_resume_start = l_resume_start.where(l_resume_start > 0, l_jan_end)

# 3. 5~10 月長期位準
post_sub = flow_df.loc["2024-05-01":"2024-10-31"] if "2024-05-01" in flow_df.index else flow_df.loc["2024-04-01":"2024-10-31"]
post_factors = pd.DataFrame([get_cycle_factor(dt, valid_grids) for dt in post_sub.index], index=post_sub.index)
l_long_term = (post_sub / post_factors).median().fillna(l_resume_start)
l_long_term = np.maximum(l_long_term, l_resume_start * 0.85)

# --- F. 邊界對接殘差 ---
dt_jan31 = pd.to_datetime("2024-01-31")
dt_resume = pd.to_datetime("2024-05-01") if pd.to_datetime("2024-05-01") in flow_df.index else pd.to_datetime("2024-04-01")

eps_start = flow_df.loc[dt_jan31] - (l_jan_end * get_cycle_factor(dt_jan31, valid_grids))
eps_end = flow_df.loc[dt_resume] - (l_resume_start * get_cycle_factor(dt_resume, valid_grids)) if dt_resume in flow_df.index else eps_start * 0.0

# --- G. 執行全期 (2024-01-01 ~ 2024-10-31) 預測生成 ---
full_pred_dates = pd.date_range(PRED_START, PRED_END, freq="D")
gap_span = (GAP_END - GAP_START).days + 1

pred_grid_records = {}
pred_od_records = {}

jan_init_raw = flow_df.loc["2024-01-01":"2024-01-05"].median()
jan_init_mu = jan_init_raw.fillna(l_jan_end)

for dt in full_pred_dates:
    r_t = get_cycle_factor(dt, valid_grids)
    
    if dt < GAP_START:
        tau = (dt - PRED_START).days / 30.0
        mu_t = jan_init_mu + (tau ** 1.5) * (l_jan_end - jan_init_mu)
        bridge = 0.0
    elif dt <= GAP_END:
        tau = ((dt - GAP_START).days + 1) / gap_span
        s_curve = 3.0 * (tau ** 2) - 2.0 * (tau ** 3)
        mu_t = l_jan_end + s_curve * (l_resume_start - l_jan_end)
        bridge = eps_start * ((1.0 - tau) ** 2) + eps_end * (tau ** 2)
    else:
        tau_post = min(1.0, (dt - (GAP_END + pd.Timedelta(days=1))).days / 60.0)
        mu_t = l_resume_start + tau_post * (l_long_term - l_resume_start)
        bridge = 0.0

    raw_pred = (mu_t * r_t + bridge).clip(lower=0.0)
    
    # 空間最近鄰矩陣融合平滑 (90% 主訊號 + 10% 空間鄰居輔助)
    smooth_pred = 0.90 * raw_pred + 0.10 * spatial_knn_weights.dot(raw_pred)
    pred_grid_records[dt] = smooth_pred

    # 還原 OD 矩陣
    day_od_pred = {}
    for orig in valid_grids:
        orig_vol = smooth_pred[orig]
        if orig_vol > 0 and orig in smoothed_od_prob:
            day_od_pred[orig] = {d: prob * orig_vol for d, prob in smoothed_od_prob[orig].items()}
        else:
            day_od_pred[orig] = {orig: orig_vol}
    pred_od_records[dt] = day_od_pred

pred_df = pd.DataFrame.from_dict(pred_grid_records, orient='index')
pred_csv_path = os.path.join(OUTPUT_DIR, "full_predictions_jan_to_oct.csv")
pred_df.to_csv(pred_csv_path, encoding="utf-8-sig")
print(f"✓ 全期預測結果已成功匯出至: {pred_csv_path}")

# =========================================================================
# 4. HuMob 官方 Combined NRMSE 指標評估 (排除 NaN 穩健計算)
# =========================================================================
print("\n[3/5] 執行 HuMob 競賽指標評估 (Combined NRMSE)...")

def evaluate_humob_competition_score(pred_ods, actual_ods, eval_dates, grids):
    daily_metrics = []
    
    for dt in eval_dates:
        if dt not in actual_ods or not actual_ods[dt]:
            continue
        
        act_od = actual_ods[dt]
        prd_od = pred_ods.get(dt, {})
        
        diag_diffs, offdiag_diffs = [], []
        
        for orig in grids:
            act_dests = act_od.get(orig, {})
            prd_dests = prd_od.get(orig, {})
            
            # Step 1: Diagonal RMSE (orig == dest)
            act_diag = float(act_dests.get(orig, 0.0))
            prd_diag = float(prd_dests.get(orig, 0.0))
            diag_diffs.append((prd_diag - act_diag) ** 2)
            
            # Step 2: Off-Diagonal RMSE (orig != dest)
            all_dests = set(act_dests.keys()).union(set(prd_dests.keys()))
            for dest in all_dests:
                if dest == orig or dest == "-1_-1":
                    continue
                act_off = float(act_dests.get(dest, 0.0))
                prd_off = float(prd_dests.get(dest, 0.0))
                offdiag_diffs.append((prd_off - act_off) ** 2)
        
        rmse_diag = np.sqrt(np.mean(diag_diffs)) if len(diag_diffs) > 0 else 0.0
        rmse_offdiag = np.sqrt(np.mean(offdiag_diffs)) if len(offdiag_diffs) > 0 else 0.0
        
        if not np.isnan(rmse_diag) and not np.isnan(rmse_offdiag):
            daily_metrics.append({
                "date": dt,
                "RMSE_diag": rmse_diag,
                "RMSE_offdiag": rmse_offdiag
            })
    
    df_eval = pd.DataFrame(daily_metrics)
    
    # Step 3: D 天平均與歸一化
    mean_rmse_diag = float(df_eval["RMSE_diag"].mean()) if not df_eval.empty else 0.0
    mean_rmse_offdiag = float(df_eval["RMSE_offdiag"].mean()) if not df_eval.empty else 0.0
    nrmse_diag = mean_rmse_diag / 207.6
    nrmse_offdiag = mean_rmse_offdiag / 19.7
    
    # Step 4: Combined NRMSE
    combined_nrmse = 0.5 * nrmse_diag + 0.5 * nrmse_offdiag
    
    return mean_rmse_diag, mean_rmse_offdiag, nrmse_diag, nrmse_offdiag, combined_nrmse, df_eval

# 評估區間：2024 年已知資料段 (排除 2~4 月缺測)
eval_dates = [dt for dt in flow_df.index if dt >= PRED_START and not (GAP_START <= dt <= GAP_END)]
m_diag, m_off, n_diag, n_off, final_score, df_eval = evaluate_humob_competition_score(
    pred_od_records, daily_od_records, eval_dates, valid_grids
)

print("-" * 75)
print(f"📊 評估天數 (D):             {len(df_eval)} 天 (已排除 2~4 月缺測)")
print(f"🔹 Step 1 | Mean RMSE (Diag):        {m_diag:8.4f}")
print(f"🔹 Step 2 | Mean RMSE (Off-Diag):    {m_off:8.4f}")
print(f"🔸 Step 3 | NRMSE (Diag)   [/207.6]: {n_diag:8.4f}")
print(f"🔸 Step 3 | NRMSE (Off-Diag) [/19.7]: {n_off:8.4f}")
print("-" * 75)
print(f"🏆 Step 4 | Combined NRMSE:          {final_score:8.4f}")
print("-" * 75)

# =========================================================================
# 5. 繪製 9 大類別圖譜 (紅線斷開 2~4 月，綠線從 1 月完整延展)
# =========================================================================
print("\n[4/5] 繪製 9 大類別獨立圖檔與 3x3 整合全景圖...")

plt.style.use('dark_background')
COLOR_ACTUAL = '#f43f5e'     # 玫紅色實線 (真實數據 Actual，2~4月斷線)
COLOR_PRED = '#10b981'       # 翠綠色實線 (預測數據 Pred，從 1/1 開始)
COLOR_BASE = '#64748b'       # 灰藍色虛線 (震前 Baseline)
COLOR_GAP = '#f59e0b'        # 琥珀金背景 (2~4月缺測區間標示)

# 🔴 紅色線核心設定：將 2024-02-01 ~ 2024-04-30 設為 NaN 自動斷線
actual_plot_df = flow_df.copy()
gap_mask = (actual_plot_df.index >= GAP_START) & (actual_plot_df.index <= GAP_END)
actual_plot_df.loc[gap_mask] = np.nan

# 🟢 綠色線核心設定：從 2024-01-01 開始
pred_plot_df = pred_df[pred_df.index >= PRED_START].copy()

baseline_df = pd.DataFrame(
    dow_medians_pre.loc[flow_df.index.dayofweek, valid_grids].values,
    index=flow_df.index,
    columns=valid_grids
)

class_series = {}
for c_id in range(1, 10):
    c_grids = [g for g in valid_grids if grid_class_lookup.get(g) == c_id]
    if not c_grids:
        continue
    class_series[c_id] = {
        "name": CLASS_INFO_MAP[c_id],
        "dates": flow_df.index,
        "actual": actual_plot_df[c_grids].mean(axis=1),
        "baseline": baseline_df[c_grids].mean(axis=1),
        "pred_dates": pred_plot_df.index,
        "pred": pred_plot_df[c_grids].mean(axis=1),
        "count": len(c_grids)
    }

# A. 匯出 9 張獨立圖檔
for c_id, data in class_series.items():
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=300)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    ax.axvspan(GAP_START, GAP_END, color=COLOR_GAP, alpha=0.15, label='Feb-Apr Gap (No Actual Data)')
    ax.plot(data["dates"], data["baseline"], color=COLOR_BASE, linestyle=':', linewidth=1.2, label='Pre-EQ Baseline')
    ax.plot(data["dates"], data["actual"], color=COLOR_ACTUAL, linewidth=1.5, label='Actual Flow (No Feb-Apr)')
    ax.plot(data["pred_dates"], data["pred"], color=COLOR_PRED, linewidth=2.0, label='Predicted Flow (Jan 1 - Oct 31)')

    ax.set_title(f"{data['name']} (Grids: {data['count']})", fontsize=13, fontweight='bold', color='#ffffff', pad=12)
    ax.set_xlabel('Date', fontsize=10, color='#94a3b8')
    ax.set_ylabel('Average Flow Volume', fontsize=10, color='#94a3b8')
    ax.grid(True, color='#334155', linestyle=':', alpha=0.6)
    ax.tick_params(colors='#94a3b8', labelsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.legend(loc='upper right', frameon=True, facecolor='#0f172a', edgecolor='#475569', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"class_{c_id:02d}_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

# B. 匯出 3x3 整合全景大圖
fig_grid, axes = plt.subplots(3, 3, figsize=(19, 11.5), dpi=300)
fig_grid.patch.set_facecolor('#0f172a')
fig_grid.suptitle(f'9-Class Mobility Dynamics Overview | Combined NRMSE: {final_score:.4f}\n(Red: Actual without Feb-Apr | Green: Model Prediction from Jan 1)', 
                  fontsize=15, fontweight='bold', color='#ffffff', y=0.98)

for c_id in range(1, 10):
    row, col = (c_id - 1) // 3, (c_id - 1) % 3
    ax = axes[row, col]
    ax.set_facecolor('#1e293b')

    if c_id not in class_series:
        ax.set_title(CLASS_INFO_MAP[c_id], fontsize=10, fontweight='bold', color='#64748b')
        continue

    data = class_series[c_id]
    ax.axvspan(GAP_START, GAP_END, color=COLOR_GAP, alpha=0.15, label='Feb-Apr Gap' if c_id == 1 else "")
    ax.plot(data["dates"], data["baseline"], color=COLOR_BASE, linestyle=':', linewidth=1.0, label='Pre-EQ Baseline' if c_id == 1 else "")
    ax.plot(data["dates"], data["actual"], color=COLOR_ACTUAL, linewidth=1.2, label='Actual Flow (No Feb-Apr)' if c_id == 1 else "")
    ax.plot(data["pred_dates"], data["pred"], color=COLOR_PRED, linewidth=1.6, label='Prediction (Jan-Oct)' if c_id == 1 else "")

    ax.set_title(f"{data['name']} (N={data['count']})", fontsize=11, fontweight='bold', color='#f8fafc', pad=8)
    ax.grid(True, color='#334155', linestyle=':', alpha=0.5)
    ax.tick_params(colors='#94a3b8', labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

fig_grid.legend(loc='lower center', bbox_to_anchor=(0.5, 0.015), ncol=4, 
                fontsize=11, frameon=True, facecolor='#1e293b', edgecolor='#475569')
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

overview_img_path = os.path.join(OUTPUT_DIR, "all_9classes_comparison_overview.png")
plt.savefig(overview_img_path, dpi=300, bbox_inches='tight')
plt.close(fig_grid)

print("\n" + "=" * 80)
print("🎉 全流程執行完畢！產出成果清單：")
print(f"👉 預測數據 CSV: {pred_csv_path}")
print(f"👉 3x3 整合全景圖: {overview_img_path}")
print(f"👉 9 張獨立圖檔: {OUTPUT_DIR}/class_01_comparison.png ~ class_09_comparison.png")
print("=" * 80)
