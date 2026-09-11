import os
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =========================================================================
# 1. 全域參數與環境配置
# =========================================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAP_START = pd.to_datetime("2024-02-01")
GAP_END = pd.to_datetime("2024-03-31")
PRED_START = pd.to_datetime("2024-01-01")
PRED_END = pd.to_datetime("2024-10-31")
GAP_LEN = (GAP_END - GAP_START).days + 1  # 60 天

OUTPUT_DIR = "humob_monday_rollout_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================================
# 2. 模擬/載入資料生成器 (確保獨立可執行)
# =========================================================================
def generate_synthetic_mobility_stream():
    """生成具備真實災後行為特徵與星期一週期調變的人流序列"""
    dates = pd.date_range("2023-11-01", "2024-10-31", freq="D")
    grids = [f"grid_{i:02d}" for i in range(12)]
    
    # 7 天基準週期 (以週一為 0: Mon=0, Tue=0.2, Wed=0.3, Thu=0.1, Fri=0.8(峰), Sat=-0.5, Sun=-0.9(谷))
    true_weekly_pattern = np.array([0.0, 0.25, 0.35, 0.20, 0.85, -0.60, -0.90])
    
    data = {}
    for g_idx, g in enumerate(grids):
        base_level = 35.0 + g_idx * 5.0
        series = []
        for d in dates:
            dow = d.dayofweek
            # 模擬 2024-01-01 地震衝擊
            if d < pd.to_datetime("2024-01-01"):
                macro = base_level
                amp = 8.0
            elif d < GAP_START:
                # 震後 1 月驟降與微弱起伏
                days_post = (d - pd.to_datetime("2024-01-01")).days
                macro = base_level * 0.35 + days_post * 0.15
                amp = 2.5
            elif d <= GAP_END:
                # 2~3 月空窗期 (真實場景缺失)
                tau = (d - GAP_START).days / 60.0
                macro = base_level * 0.45 + tau * (base_level * 0.35)
                amp = 2.5 + tau * 3.5
            else:
                # 4 月後緩慢復原
                days_rec = min(90.0, (d - GAP_END).days)
                macro = base_level * 0.80 + (days_rec / 90.0) * (base_level * 0.15)
                amp = 6.0 + (days_rec / 90.0) * 1.5
                
            val = macro + amp * true_weekly_pattern[dow] + np.random.normal(0, 0.35)
            series.append(max(0.0, val))
        data[g] = series
        
    df = pd.DataFrame(data, index=dates)
    return df, grids

print("[1/5] 初始化人流時間序列資料...")
raw_flow_df, valid_grids = generate_synthetic_mobility_stream()

# =========================================================================
# 3. 星期一錨點動力學與 7 天週波展開引擎 (自主研發核心)
# =========================================================================
class MondayAnchoredWeeklyEngine:
    """
    星期一錨點趨勢預測與 7 天序列展開器：
    1. 提取所有週一節點流量 M_w
    2. 計算以週一為基準點 (s[0] = 0) 的無量綱正規化週波 profile
    3. 利用 Hermite 樣條插值預測空窗期內每週一的起始點與當週振幅
    4. 依據星期一的變化率 Delta M_w 展開一週 7 天的連續動態軌跡
    """
    def __init__(self, flow_df: pd.DataFrame, valid_grids: list):
        self.flow_df = flow_df
        self.valid_grids = valid_grids
        self.canonical_profiles = {}  # {grid: np.array of shape (7,)}
        self.extrema_masks = {}       # {grid: np.array of shape (7,)} (1=峰, -1=谷, 0=平)
        self._extract_canonical_patterns()

    def _extract_canonical_patterns(self):
        """從已觀測數據解耦出以星期一為零點基準的 7 天特徵波"""
        obs_df = self.flow_df.copy()
        # 排除空窗期
        obs_df = obs_df.loc[~((obs_df.index >= GAP_START) & (obs_df.index <= GAP_END))]
        obs_df['dow'] = obs_df.index.dayofweek

        for g in self.valid_grids:
            dow_medians = obs_df.groupby('dow')[g].median().values  # 長度 7
            monday_val = dow_medians[0]
            
            # 以週一為錨點扣減，強制週一相對值為 0.0
            relative_pattern = dow_medians - monday_val
            
            # 正規化至最大振幅為 1.0 (保持正負相對極值)
            max_swing = np.max(np.abs(relative_pattern)) + 1e-6
            normalized_s = relative_pattern / max_swing
            self.canonical_profiles[g] = normalized_s

            # 檢測 7 天波形的極值點
            extrema = np.zeros(7, dtype=np.float32)
            for d in range(7):
                prev_val = normalized_s[(d - 1) % 7]
                curr_val = normalized_s[d]
                next_val = normalized_s[(d + 1) % 7]
                if curr_val > prev_val and curr_val >= next_val:
                    extrema[d] = 1.0   # 波峰 (通常為週五)
                elif curr_val < prev_val and curr_val <= next_val:
                    extrema[d] = -1.0  # 波谷 (通常為週日)
            self.extrema_masks[g] = extrema

    def reconstruct_full_backbone(self, date_range: pd.DatetimeIndex) -> pd.DataFrame:
        """
        全域 7 天展開演算法：
        對時間序列按週切片，逐週由星期一錨點 M_w 與動態振幅 A_w 展開 7 天預測
        """
        pred_df = pd.DataFrame(index=date_range, columns=self.valid_grids, dtype=np.float32)
        
        # 尋找所有星期一索引
        mondays = date_range[date_range.dayofweek == 0]
        
        for g in self.valid_grids:
            s_d = self.canonical_profiles[g]
            
            # 提取已知歷史中每週一的值與每週的振幅
            mon_series = self.flow_df.loc[self.flow_df.index.dayofweek == 0, g]
            
            # 計算 1 月底與 4 月初的星期一過渡邊界
            mon_jan = mon_series.loc["2024-01-15":"2024-01-31"]
            mon_apr = mon_series.loc["2024-04-01":"2024-04-20"]
            
            M_jan_end = mon_jan.iloc[-1] if len(mon_jan) > 0 else 10.0
            M_apr_start = mon_apr.iloc[0] if len(mon_apr) > 0 else M_jan_end * 1.5
            
            # 振幅估計
            A_jan = max(1.0, float(self.flow_df.loc["2024-01-15":"2024-01-31", g].std() * 1.8))
            A_apr = max(1.0, float(self.flow_df.loc["2024-04-01":"2024-04-20", g].std() * 1.8))
            
            # 建立每週一的連續序列插值器
            monday_dict = {}
            amp_dict = {}
            
            gap_mondays = [m for m in mondays if GAP_START <= m <= GAP_END]
            n_gap_m = len(gap_mondays)
            
            for idx, m in enumerate(gap_mondays):
                tau = (idx + 1) / (n_gap_m + 1)
                # 三次 S 曲線推進星期一的趨勢轉變
                s_curve = 3.0 * (tau ** 2) - 2.0 * (tau ** 3)
                monday_dict[m] = M_jan_end + s_curve * (M_apr_start - M_jan_end)
                amp_dict[m] = A_jan + s_curve * (A_apr - A_jan)
                
            for m in mondays:
                if m not in monday_dict:
                    monday_dict[m] = self.flow_df.loc[m, g] if m in self.flow_df.index else M_jan_end
                    w_sub = self.flow_df.loc[m : m + pd.Timedelta(days=6), g]
                    amp_dict[m] = max(1.0, float(w_sub.max() - w_sub.min()) * 0.5) if len(w_sub) > 0 else A_jan

            # 依據星期一的節點向前展開 7 天
            for i in range(len(mondays)):
                m_curr = mondays[i]
                m_next = mondays[i+1] if i + 1 < len(mondays) else m_curr + pd.Timedelta(days=7)
                
                M_w = monday_dict[m_curr]
                M_w_next = monday_dict.get(m_next, M_w)
                delta_M = M_w_next - M_w  # 星期一至星期一的斜率差分
                A_w = amp_dict[m_curr]
                
                # 滾動預測一週 7 天
                for day_offset in range(7):
                    target_dt = m_curr + pd.Timedelta(days=day_offset)
                    if target_dt in pred_df.index:
                        # 核心解析公式：週一基底 + 週內漂移趨勢 + 振幅調變峰谷
                        drift_t = (day_offset / 7.0) * delta_M
                        cycle_t = A_w * s_d[day_offset]
                        pred_df.loc[target_dt, g] = max(0.0, M_w + drift_t + cycle_t)
                        
        return pred_df

print("[2/5] 執行星期一動力學解耦與 7 天展開管線...")
monday_engine = MondayAnchoredWeeklyEngine(raw_flow_df, valid_grids)
all_dates = pd.date_range("2023-11-01", PRED_END, freq="D")
macro_rollout_df = monday_engine.reconstruct_full_backbone(all_dates)

# =========================================================================
# 4. 最優傳輸流匹配 (OT-FM) 星期一殘差微調網絡
# =========================================================================
class WeeklyResidualDataset(Dataset):
    """將時間序列切片為以星期一為起始的 7 天窗口進行殘差訓練"""
    def __init__(self, gt_df, base_df, valid_grids, samples_num=1800):
        self.samples = []
        res_df = gt_df - base_df
        # 取得非空窗期之週一
        valid_mondays = [
            d for d in gt_df.index 
            if d.dayofweek == 0 and d + pd.Timedelta(days=6) <= gt_df.index.max()
            and not (GAP_START <= d <= GAP_END)
            and not (GAP_START <= d + pd.Timedelta(days=6) <= GAP_END)
        ]
        
        for _ in range(samples_num):
            m = random.choice(valid_mondays)
            g = random.choice(valid_grids)
            span = pd.date_range(m, periods=7, freq="D")
            
            res_seq = res_df.loc[span, g].values.astype(np.float32)
            base_seq = base_df.loc[span, g].values.astype(np.float32)
            mon_val = base_df.loc[m, g]
            
            self.samples.append((
                res_seq, 
                base_seq, 
                np.array([mon_val], dtype=np.float32)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        res, base, mon = self.samples[idx]
        return (
            torch.from_numpy(res).unsqueeze(0),    # [1, 7]
            torch.from_numpy(base).unsqueeze(0),   # [1, 7]
            torch.from_numpy(mon)                  # [1]
        )

class MondayConditionedUNet(nn.Module):
    """輕量化 1D-ResUNet：注入星期一強度 (M_w) 與微觀物理時間條件"""
    def __init__(self, hidden_dim=48):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.mon_proj = nn.Linear(1, hidden_dim)
        
        self.conv_in = nn.Conv1d(2, hidden_dim, kernel_size=3, padding=1)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.gn = nn.GroupNorm(4, hidden_dim)
        self.conv_out = nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)

    def forward(self, x_t, t, base_seq, mon_val):
        cond = self.time_mlp(t) + self.mon_proj(mon_val)
        h = self.conv_in(torch.cat([x_t, base_seq], dim=1))
        h = h + cond.unsqueeze(-1)
        res = h
        h = F.silu(self.gn(self.conv1(h)))
        h = self.conv2(h) + res
        return self.conv_out(h)

def train_otfm_model(gt_df, base_df, valid_grids, epochs=20):
    dataset = WeeklyResidualDataset(gt_df, base_df, valid_grids)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = MondayConditionedUNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    
    model.train()
    for ep in range(epochs):
        for res_real, base_seq, mon_val in loader:
            res_real = res_real.to(DEVICE)
            base_seq = base_seq.to(DEVICE)
            mon_val = mon_val.to(DEVICE)
            B = res_real.shape[0]
            
            # 最優傳輸線性插值路徑
            t = torch.rand(B, 1, device=DEVICE)
            x_0 = torch.randn_like(res_real)
            x_t = (1.0 - t.unsqueeze(-1)) * x_0 + t.unsqueeze(-1) * res_real
            target_v = res_real - x_0
            
            pred_v = model(x_t, t, base_seq, mon_val)
            loss = F.mse_loss(pred_v, target_v)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

print("[3/5] 訓練星期一條件約束之流匹配 (OT-FM) 神經網絡...")
otfm_model = train_otfm_model(raw_flow_df, macro_rollout_df, valid_grids)

# =========================================================================
# 5. 空窗期 RK4 ODE 積分與全域 7 天流動重構
# =========================================================================
@torch.no_grad()
def solve_otfm_weekly_rollout(model, base_df, valid_grids, steps=10):
    """利用四階龍格-庫塔法 (RK4) 在以週為單位的尺度下積分生成殘差"""
    model.eval()
    dt = 1.0 / steps
    gap_dates = pd.date_range(GAP_START, GAP_END, freq="D")
    final_pred_df = base_df.copy()
    
    # 遍歷空窗期內的每一個星期一
    gap_mondays = pd.date_range(GAP_START - pd.Timedelta(days=6), GAP_END, freq="W-MON")
    
    for m in gap_mondays:
        w_span = pd.date_range(m, periods=7, freq="D")
        for g in valid_grids:
            base_seq_np = base_df.loc[w_span, g].values.astype(np.float32)
            mon_val_np = np.array([base_df.loc[m, g]], dtype=np.float32)
            
            base_t = torch.from_numpy(base_seq_np).unsqueeze(0).unsqueeze(0).to(DEVICE)
            mon_t = torch.from_numpy(mon_val_np).unsqueeze(0).to(DEVICE)
            
            # 8 次 Monte Carlo 取中位數消除波動雜訊
            x = torch.randn(8, 1, 7, device=DEVICE)
            for s in range(steps):
                t_cur = s / steps
                t1 = torch.full((8, 1), t_cur, device=DEVICE)
                k1 = model(x, t1, base_t.repeat(8, 1, 1), mon_t.repeat(8, 1))
                
                t2 = torch.full((8, 1), t_cur + 0.5 * dt, device=DEVICE)
                k2 = model(x + 0.5 * dt * k1, t2, base_t.repeat(8, 1, 1), mon_t.repeat(8, 1))
                
                t3 = torch.full((8, 1), t_cur + 0.5 * dt, device=DEVICE)
                k3 = model(x + 0.5 * dt * k2, t3, base_t.repeat(8, 1, 1), mon_t.repeat(8, 1))
                
                t4 = torch.full((8, 1), t_cur + dt, device=DEVICE)
                k4 = model(x + dt * k3, t4, base_t.repeat(8, 1, 1), mon_t.repeat(8, 1))
                
                x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                
            residual_opt = np.median(x.squeeze(1).cpu().numpy(), axis=0)
            
            # 覆蓋進入空窗期的實際日期
            for d_idx, d in enumerate(w_span):
                if GAP_START <= d <= GAP_END:
                    rec_val = base_seq_np[d_idx] + residual_opt[d_idx]
                    final_pred_df.loc[d, g] = max(0.0, rec_val)
                    
    return final_pred_df

print("[4/5] 執行 60 天空窗期 (2~3月) RK4 數值積分預測...")
final_solution_df = solve_otfm_weekly_rollout(otfm_model, macro_rollout_df, valid_grids)

# =========================================================================
# 6. 輸出精確度評估報告與儲存
# =========================================================================
print("[5/5] 計算週波預測指標與輸出結果...")
eval_dates = [d for d in raw_flow_df.index if d >= PRED_START and not (GAP_START <= d <= GAP_END)]
rmse_per_grid = []
for g in valid_grids:
    gt_v = raw_flow_df.loc[eval_dates, g].values
    pr_v = final_solution_df.loc[eval_dates, g].values
    rmse = np.sqrt(np.mean((gt_v - pr_v) ** 2))
    rmse_per_grid.append(rmse)

overall_rmse = float(np.mean(rmse_per_grid))
print("\n" + "=" * 65)
print(f"  週波星期一錨點動力學 (Monday Rollout + OT-FM) 成果")
print("=" * 65)
print(f"  觀測網格總數          : {len(valid_grids)} 個")
print(f"  全域均方根誤差 (RMSE)  : {overall_rmse:.4f}")
print("=" * 65)

# 匯出預測 CSV 檔案
final_solution_df.to_csv(os.path.join(OUTPUT_DIR, "monday_anchored_7day_pred.csv"), encoding="utf-8-sig")
print(f"✓ 預測結果已匯出至: {OUTPUT_DIR}/monday_anchored_7day_pred.csv")
