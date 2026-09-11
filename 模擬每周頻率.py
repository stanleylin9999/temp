import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =========================================================================
# 1. 環境配置與路徑定位 (確保輸出直接存於腳本同目錄)
# =========================================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定位當前腳本所在目錄
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()

PRED_START = pd.to_datetime("2024-01-01")
GAP_START = pd.to_datetime("2024-02-01")
GAP_END = pd.to_datetime("2024-03-31")
PRED_END = pd.to_datetime("2024-10-31")
GAP_LEN = (GAP_END - GAP_START).days + 1  # 2024年為閏年，共 60 天

# =========================================================================
# 2. 合成數據產生器 (若無外部資料集時自動建立高擬真資料供全流程運算)
# =========================================================================
def build_benchmark_dataset():
    """建立包含災前、地震衝擊、60天真實空窗期與復原期的人流序列"""
    dates = pd.date_range("2023-11-01", PRED_END, freq="D")
    grids = [f"grid_{i:02d}" for i in range(9)]
    
    # 7 天標準相對週波 (以週一為基準 0: Mon=0.0, Tue=+0.3, Wed=+0.4, Thu=+0.2, Fri=+1.0(波峰), Sat=-0.6, Sun=-1.0(波谷))
    true_weekly_wave = np.array([0.0, 0.35, 0.45, 0.25, 1.00, -0.65, -1.00])
    
    df_dict = {}
    for g_idx, g in enumerate(grids):
        base_capacity = 40.0 + g_idx * 8.0
        daily_records = []
        for d in dates:
            dow = d.dayofweek
            if d < PRED_START:
                macro = base_capacity
                amp = 8.5
            elif d < GAP_START:
                days_post = (d - PRED_START).days
                macro = base_capacity * 0.30 + days_post * 0.18
                amp = 2.0 + (days_post / 31.0) * 1.0
            elif d <= GAP_END:
                tau = (d - GAP_START).days / float(GAP_LEN)
                macro = base_capacity * 0.45 + tau * (base_capacity * 0.30)
                amp = 3.0 + tau * 4.0
            else:
                tau_rec = min(1.0, (d - GAP_END).days / 100.0)
                macro = base_capacity * 0.75 + tau_rec * (base_capacity * 0.20)
                amp = 7.0 + tau_rec * 1.5
                
            val = macro + amp * true_weekly_wave[dow] + np.random.normal(0, 0.45)
            daily_records.append(max(0.0, val))
        df_dict[g] = daily_records
        
    return pd.DataFrame(df_dict, index=dates), grids

print("[1/6] 讀取並初始化人流時間序列...")
raw_flow_df, valid_grids = build_benchmark_dataset()

# =========================================================================
# 3. 星期一錨點動力學與 7 天週波展開引擎
# =========================================================================
class MondayAnchoredWeeklyEngine:
    """
    星期一錨點動力學引擎：
    1. 解耦出以星期一為基準 (s[0] = 0) 的 7 天無量綱特徵波形
    2. 識別每一週波的理論波峰 (Peak) 與波谷 (Trough)
    3. 利用 S 曲線 Hermite 樣條插值空窗期各週一的起始強度與當週振幅 A_w
    4. 依據星期一變化率 Delta M_w 展開一週 7 天連續基線
    """
    def __init__(self, flow_df: pd.DataFrame, valid_grids: list):
        self.flow_df = flow_df
        self.valid_grids = valid_grids
        self.canonical_profiles = {}  # {grid: (7,)}
        self.extrema_flags = {}        # {grid: (7,)}
        self.mon_records = {}          # {grid: pd.DataFrame}
        self._extract_weekly_geometry()

    def _extract_weekly_geometry(self):
        obs_df = self.flow_df.copy()
        # 排除空窗期干擾以獲取穩健的週幾何先驗
        obs_df = obs_df.loc[~((obs_df.index >= GAP_START) & (obs_df.index <= GAP_END))]
        obs_df['dow'] = obs_df.index.dayofweek

        for g in self.valid_grids:
            dow_medians = obs_df.groupby('dow')[g].median().values
            mon_val = dow_medians[0]
            
            # 以星期一為零點基準
            relative_pattern = dow_medians - mon_val
            max_swing = np.max(np.abs(relative_pattern)) + 1e-6
            norm_profile = relative_pattern / max_swing
            self.canonical_profiles[g] = norm_profile

            # 標記波峰 (1.0) 與波谷 (-1.0)
            flags = np.zeros(7, dtype=np.float32)
            for d in range(7):
                p_v = norm_profile[(d - 1) % 7]
                c_v = norm_profile[d]
                n_v = norm_profile[(d + 1) % 7]
                if c_v > p_v and c_v >= n_v:
                    flags[d] = 1.0   # 波峰
                elif c_v < p_v and c_v <= n_v:
                    flags[d] = -1.0  # 波谷
            self.extrema_flags[g] = flags

    def rollout_7day_backbone(self, date_range: pd.DatetimeIndex) -> (pd.DataFrame, pd.DataFrame):
        pred_df = pd.DataFrame(index=date_range, columns=self.valid_grids, dtype=np.float32)
        mondays = date_range[date_range.dayofweek == 0]
        meta_records = []

        for g in self.valid_grids:
            s_d = self.canonical_profiles[g]
            mon_obs = self.flow_df.loc[self.flow_df.index.dayofweek == 0, g]
            
            # 空窗期邊界值
            mon_jan = mon_obs.loc["2024-01-15":"2024-01-31"]
            mon_apr = mon_obs.loc["2024-04-01":"2024-04-20"]
            
            M_jan_end = mon_jan.iloc[-1] if len(mon_jan) > 0 else 12.0
            M_apr_start = mon_apr.iloc[0] if len(mon_apr) > 0 else M_jan_end * 1.6
            
            A_jan = max(1.0, float(self.flow_df.loc["2024-01-15":"2024-01-31", g].std() * 1.8))
            A_apr = max(1.0, float(self.flow_df.loc["2024-04-01":"2024-04-20", g].std() * 1.8))
            
            gap_mondays = [m for m in mondays if GAP_START <= m <= GAP_END]
            n_gap = len(gap_mondays)
            
            mon_dict, amp_dict = {}, {}
            for idx, m in enumerate(gap_mondays):
                tau = (idx + 1) / (n_gap + 1)
                s_curve = 3.0 * (tau ** 2) - 2.0 * (tau ** 3)
                mon_dict[m] = M_jan_end + s_curve * (M_apr_start - M_jan_end)
                amp_dict[m] = A_jan + s_curve * (A_apr - A_jan)
                
            for m in mondays:
                if m not in mon_dict:
                    mon_dict[m] = self.flow_df.loc[m, g] if m in self.flow_df.index else M_jan_end
                    w_flow = self.flow_df.loc[m : m + pd.Timedelta(days=6), g]
                    amp_dict[m] = max(1.0, float(w_flow.max() - w_flow.min()) * 0.5) if len(w_flow) > 0 else A_jan
                
                meta_records.append({
                    "grid": g, "monday_date": m.strftime("%Y-%m-%d"),
                    "monday_level": round(float(mon_dict[m]), 3),
                    "weekly_amplitude": round(float(amp_dict[m]), 3)
                })

            # 週次前向滾動展開
            for i in range(len(mondays)):
                m_curr = mondays[i]
                m_next = mondays[i+1] if i + 1 < len(mondays) else m_curr + pd.Timedelta(days=7)
                
                M_w = mon_dict[m_curr]
                delta_M = mon_dict.get(m_next, M_w) - M_w
                A_w = amp_dict[m_curr]
                
                for offset in range(7):
                    t_day = m_curr + pd.Timedelta(days=offset)
                    if t_day in pred_df.index:
                        drift = (offset / 7.0) * delta_M
                        cycle = A_w * s_d[offset]
                        pred_df.loc[t_day, g] = max(0.0, M_w + drift + cycle)

        return pred_df, pd.DataFrame(meta_records)

print("[2/6] 執行星期一趨勢解耦與 7 天週波展開...")
engine = MondayAnchoredWeeklyEngine(raw_flow_df, valid_grids)
all_dates = pd.date_range("2023-11-01", PRED_END, freq="D")
macro_baseline_df, monday_meta_df = engine.rollout_7day_backbone(all_dates)

# =========================================================================
# 4. 最優傳輸流匹配 (OT-FM) 星期一殘差微調網絡
# =========================================================================
class WeeklyChunkOTDataset(Dataset):
    def __init__(self, gt_df, base_df, valid_grids, samples=2000):
        self.samples = []
        res_df = gt_df - base_df
        valid_mondays = [
            d for d in gt_df.index 
            if d.dayofweek == 0 and d + pd.Timedelta(days=6) <= gt_df.index.max()
            and not (GAP_START <= d <= GAP_END)
            and not (GAP_START <= d + pd.Timedelta(days=6) <= GAP_END)
        ]
        for _ in range(samples):
            m = random.choice(valid_mondays)
            g = random.choice(valid_grids)
            span = pd.date_range(m, periods=7, freq="D")
            res_seq = res_df.loc[span, g].values.astype(np.float32)
            base_seq = base_df.loc[span, g].values.astype(np.float32)
            mon_val = base_df.loc[m, g]
            self.samples.append((res_seq, base_seq, np.array([mon_val], dtype=np.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        res, base, mon = self.samples[idx]
        return (torch.from_numpy(res).unsqueeze(0), 
                torch.from_numpy(base).unsqueeze(0), 
                torch.from_numpy(mon))

class MondayGuidedFlowUNet(nn.Module):
    def __init__(self, hidden=48):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.mon_proj = nn.Linear(1, hidden)
        self.c_in = nn.Conv1d(2, hidden, kernel_size=3, padding=1)
        self.block1 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.block2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.gn = nn.GroupNorm(4, hidden)
        self.c_out = nn.Conv1d(hidden, 1, kernel_size=3, padding=1)

    def forward(self, x_t, t, base_seq, mon_val):
        cond = self.time_mlp(t) + self.mon_proj(mon_val)
        h = self.c_in(torch.cat([x_t, base_seq], dim=1)) + cond.unsqueeze(-1)
        res = h
        h = self.block2(F.silu(self.gn(self.block1(h)))) + res
        return self.c_out(h)

def train_otfm(gt_df, base_df, valid_grids, epochs=25):
    dataset = WeeklyChunkOTDataset(gt_df, base_df, valid_grids)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = MondayGuidedFlowUNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    
    model.train()
    for _ in range(epochs):
        for res_real, base_seq, mon_val in loader:
            res_real, base_seq, mon_val = res_real.to(DEVICE), base_seq.to(DEVICE), mon_val.to(DEVICE)
            B = res_real.size(0)
            
            t = torch.rand(B, 1, device=DEVICE)
            x_0 = torch.randn_like(res_real)
            x_t = (1.0 - t.unsqueeze(-1)) * x_0 + t.unsqueeze(-1) * res_real
            target_v = res_real - x_0
            
            pred_v = model(x_t, t, base_seq, mon_val)
            loss = F.mse_loss(pred_v, target_v)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model

print("[3/6] 訓練 OT-FM 神經網絡學習週微觀隨機擾動...")
ot_model = train_otfm(raw_flow_df, macro_baseline_df, valid_grids)

# =========================================================================
# 5. 空窗期 RK4 ODE 數值積分求解
# =========================================================================
@torch.no_grad()
def solve_rk4_rollout(model, base_df, valid_grids, steps=12):
    model.eval()
    dt = 1.0 / steps
    pred_df = base_df.copy()
    gap_mondays = pd.date_range(GAP_START - pd.Timedelta(days=6), GAP_END, freq="W-MON")
    
    for m in gap_mondays:
        w_span = pd.date_range(m, periods=7, freq="D")
        for g in valid_grids:
            base_np = base_df.loc[w_span, g].values.astype(np.float32)
            mon_np = np.array([base_df.loc[m, g]], dtype=np.float32)
            
            base_t = torch.from_numpy(base_np).unsqueeze(0).unsqueeze(0).repeat(8, 1, 1).to(DEVICE)
            mon_t = torch.from_numpy(mon_np).unsqueeze(0).repeat(8, 1).to(DEVICE)
            x = torch.randn(8, 1, 7, device=DEVICE)
            
            for s in range(steps):
                t_cur = s / steps
                t1 = torch.full((8, 1), t_cur, device=DEVICE)
                k1 = model(x, t1, base_t, mon_t)
                
                t2 = torch.full((8, 1), t_cur + 0.5 * dt, device=DEVICE)
                k2 = model(x + 0.5 * dt * k1, t2, base_t, mon_t)
                
                t3 = torch.full((8, 1), t_cur + 0.5 * dt, device=DEVICE)
                k3 = model(x + 0.5 * dt * k2, t3, base_t, mon_t)
                
                t4 = torch.full((8, 1), t_cur + dt, device=DEVICE)
                k4 = model(x + dt * k3, t4, base_t, mon_t)
                
                x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                
            residual = np.median(x.squeeze(1).cpu().numpy(), axis=0)
            for idx, d in enumerate(w_span):
                if GAP_START <= d <= GAP_END:
                    pred_df.loc[d, g] = max(0.0, base_np[idx] + residual[idx])
                    
    return pred_df

print("[4/6] 執行 60 天空窗期 RK4 積分求解...")
final_prediction_df = solve_rk4_rollout(ot_model, macro_baseline_df, valid_grids)

# =========================================================================
# 6. 計算指標與匯出數據檔案 (直接存於腳本同目錄下)
# =========================================================================
print("[5/6] 評估模型精確度並儲存數據檔案至本目錄...")
eval_dates = [d for d in raw_flow_df.index if d >= PRED_START and not (GAP_START <= d <= GAP_END)]
metrics = []
for g in valid_grids:
    gt_arr = raw_flow_df.loc[eval_dates, g].values
    pr_arr = final_prediction_df.loc[eval_dates, g].values
    base_arr = macro_baseline_df.loc[eval_dates, g].values
    
    rmse_model = np.sqrt(np.mean((gt_arr - pr_arr) ** 2))
    rmse_base = np.sqrt(np.mean((gt_arr - base_arr) ** 2))
    mae = np.mean(np.abs(gt_arr - pr_arr))
    nrmse = rmse_model / (np.mean(gt_arr) + 1e-6)
    
    metrics.append({
        "Grid_ID": g,
        "RMSE_Model": round(rmse_model, 3),
        "RMSE_Baseline": round(rmse_base, 3),
        "MAE": round(mae, 3),
        "NRMSE": round(nrmse, 4)
    })

metrics_df = pd.DataFrame(metrics)

# 檔案路徑定義 (直接存於 SCRIPT_DIR)
pred_csv_path = os.path.join(SCRIPT_DIR, "monday_anchored_7day_pred.csv")
meta_csv_path = os.path.join(SCRIPT_DIR, "monday_trend_and_amplitude.csv")
metrics_csv_path = os.path.join(SCRIPT_DIR, "evaluation_metrics.csv")

final_prediction_df.to_csv(pred_csv_path, encoding="utf-8-sig")
monday_meta_df.to_csv(meta_csv_path, index=False, encoding="utf-8-sig")
metrics_df.to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")

print(f"✓ 已儲存預測時間序列: {pred_csv_path}")
print(f"✓ 已儲存週一趨勢與振幅數據: {meta_csv_path}")
print(f"✓ 已儲存評估指標表: {metrics_csv_path}")

# =========================================================================
# 7. 繪製並匯出專業對比圖表 (直接存於腳本同目錄下)
# =========================================================================
print("[6/6] 渲染視覺化圖表...")

# 圖表 1: 366 天全域 3x3 網格波形圖
plt.style.use('dark_background')
fig1, axes1 = plt.subplots(3, 3, figsize=(20, 11), dpi=220)
fig1.patch.set_facecolor('#070c18')
fig1.suptitle("HuMob 2026: Monday-Anchored 7-Day Rollout with OT-FM (RK4)\nGround Truth vs Monday Baseline vs OT-FM Prediction", 
              fontsize=14, fontweight='bold', color='#f8fafc', y=0.98)

for idx, g in enumerate(valid_grids):
    r, c = idx // 3, idx % 3
    ax = axes1[r, c]
    ax.set_facecolor('#0d1527')
    
    gt_series = raw_flow_df[g].copy()
    gt_series.loc[(gt_series.index >= GAP_START) & (gt_series.index <= GAP_END)] = np.nan
    base_series = macro_baseline_df[g]
    pred_series = final_prediction_df[g]
    
    ax.axvspan(GAP_START, GAP_END, color='#45271d', alpha=0.5, label='60-Day Blind Zone' if idx == 0 else "")
    ax.plot(base_series.index, base_series, color='#94a3b8', linestyle='--', linewidth=1.1, label='Monday-Anchored Baseline' if idx == 0 else "")
    ax.plot(pred_series.index, pred_series, color='#2dd4bf', linewidth=1.3, label='OT-FM + RK4 Prediction' if idx == 0 else "")
    ax.plot(gt_series.index, gt_series, color='#f43f5e', linewidth=1.1, label='Ground Truth' if idx == 0 else "")
    
    m_info = metrics_df[metrics_df["Grid_ID"] == g].iloc[0]
    ax.set_title(f"Node: {g} | NRMSE: {m_info['NRMSE']:.4f} | RMSE: {m_info['RMSE_Model']:.2f}", 
                 fontsize=9.5, fontweight='bold', color='#cbd5e1')
    ax.grid(True, color='#1e293b', linestyle=':', alpha=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.tick_params(colors='#64748b', labelsize=7.5)

fig1.legend(loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=4, fontsize=9.5, facecolor='#0a1020', edgecolor='#1e293b')
plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.95])
plot1_path = os.path.join(SCRIPT_DIR, "humob_monday_rollout_waveform.png")
fig1.savefig(plot1_path, dpi=220, bbox_inches='tight')
plt.close(fig1)

# 圖表 2: 60 天空窗期 (2~3月) 局部放大特寫 (展示精確波峰波谷與星期一錨點)
fig2, ax2 = plt.subplots(figsize=(15, 6), dpi=220)
fig2.patch.set_facecolor('#070c18')
ax2.set_facecolor('#0d1527')

target_grid = valid_grids[0]
gap_plot_range = pd.date_range(GAP_START - pd.Timedelta(days=3), GAP_END + pd.Timedelta(days=3), freq="D")
gap_mondays = gap_plot_range[gap_plot_range.dayofweek == 0]

p_series = final_prediction_df.loc[gap_plot_range, target_grid]
b_series = macro_baseline_df.loc[gap_plot_range, target_grid]

ax2.axvspan(GAP_START, GAP_END, color='#45271d', alpha=0.35, label='60-Day Missing Gap')
ax2.plot(b_series.index, b_series, color='#94a3b8', linestyle=':', linewidth=1.4, label='Monday Drift Carrier')
ax2.plot(p_series.index, p_series, color='#2dd4bf', linewidth=2.0, label='Predicted 7-Day Waveform')

# 標記星期一錨點 (垂直線與點)
for m in gap_mondays:
    val = p_series.loc[m]
    ax2.axvline(m, color='#38bdf8', linestyle='--', alpha=0.6, linewidth=1.0)
    ax2.scatter(m, val, color='#38bdf8', s=45, zorder=5)
    ax2.text(m, val + 1.2, 'Mon', color='#38bdf8', fontsize=8, ha='center', fontweight='bold')

# 標記週五波峰與週日波谷
for d in gap_plot_range:
    if GAP_START <= d <= GAP_END:
        if d.dayofweek == 4: # 週五波峰
            ax2.scatter(d, p_series.loc[d], color='#fbbf24', s=35, zorder=4)
        elif d.dayofweek == 6: # 週日波谷
            ax2.scatter(d, p_series.loc[d], color='#f87171', s=35, zorder=4)

ax2.set_title(f"Detailed 60-Day Gap Rollout: Monday Anchors, Friday Peaks (Gold), Sunday Troughs (Red) [{target_grid}]", 
              fontsize=12, fontweight='bold', color='#f8fafc', pad=12)
ax2.set_ylabel("Persons / Day", fontsize=10, color='#94a3b8')
ax2.grid(True, color='#1e293b', linestyle=':', alpha=0.6)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d (%a)'))
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax2.tick_params(colors='#94a3b8', labelsize=8)
ax2.legend(loc='upper left', facecolor='#0a1020', edgecolor='#1e293b')

plt.tight_layout()
plot2_path = os.path.join(SCRIPT_DIR, "humob_gap_weekly_peaks_troughs.png")
fig2.savefig(plot2_path, dpi=220, bbox_inches='tight')
plt.close(fig2)

print(f"✓ 已產出全域對比波形圖: {plot1_path}")
print(f"✓ 已產出空窗期峰谷特寫圖: {plot2_path}")

print("\n" + "=" * 70)
print(f" 🏆 星期一錨點動力學 (Monday-Anchored 7-Day Engine) 運算完畢！")
print("=" * 70)
print(metrics_df.to_string(index=False))
print("=" * 70 + "\n")
