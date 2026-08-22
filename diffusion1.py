import os
import ast
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =========================================================================
# 1. 全域配置與官方正規化參數
# =========================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIFFUSION_STEPS = 100
MC_SAMPLES = 16           # 蒙地卡羅期望值採樣次數
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-3
COND_DROPOUT_PROB = 0.15  # 條件隨機丟棄
SPARSITY_THRESH = 0.015   # 跨區轉移矩陣微小機率截斷門檻

# 官方最新正規化分母
NORM_DIAG = 26.57
NORM_OFFDIAG = 0.0176

EQ_DATE = pd.to_datetime("2024-01-01")
EVAL_GAP_START = pd.to_datetime("2024-02-01")
EVAL_GAP_END = pd.to_datetime("2024-03-31")
PRED_START = pd.to_datetime("2024-01-01")
PRED_END = pd.to_datetime("2024-10-31")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
TSV_PATH = os.path.join(SCRIPT_DIR, "humob2026-dataset.tsv")
SPECIFIC_CLASS_DIR = r"C:\Users\User\Desktop\人口預測專案\人口預測專案3\humob2026\data\output\module05\classification\by_class"
FALLBACK_CLASS_DIR = os.path.join(SCRIPT_DIR, "humob2026", "data", "output", "module05", "classification", "by_class")
BY_CLASS_DIR = SPECIFIC_CLASS_DIR if os.path.exists(SPECIFIC_CLASS_DIR) else FALLBACK_CLASS_DIR

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "humob_preeq_diffusion_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_INFO_MAP = {
    1: "Class 01: Persistent Zero",
    2: "Class 02: Persistent Decrease",
    3: "Class 03: Emergent Activity",
    4: "Class 04: Partial Recovery",
    5: "Class 05: Fully Recovered",
    6: "Class 06: Stable Inflow",
    7: "Class 07: Temporary Increase",
    8: "Class 08: Partial Dissipation",
    9: "Class 09: Persistent Increase"
}

# =========================================================================
# 2. 資料解析與解耦
# =========================================================================
def get_class_id_from_filename(fname: str) -> int:
    fname = fname.lower()
    if "zero" in fname: return 1
    if "decrease" in fname: return 2
    if "emergent" in fname or "temporary_activity" in fname: return 3
    if "partial_recovery" in fname or "partial_rec" in fname: return 4
    if "recovered" in fname: return 5
    if "stable" in fname: return 6
    if "temporary_increase" in fname or "temp_inc" in fname: return 7
    if "partial_dissipation" in fname or "dissip" in fname: return 8
    if "persistent_increase" in fname or "increase" in fname: return 9
    return None

def load_and_split_flows():
    print("[1/5] 解析 9 大類別與 TSV (分離對角線與非對角線流量)...")
    grid_class_lookup = {}
    if os.path.exists(BY_CLASS_DIR):
        for fpath in glob.glob(os.path.join(BY_CLASS_DIR, "*.csv")):
            c_id = get_class_id_from_filename(os.path.basename(fpath))
            if c_id is not None:
                try:
                    df_cls = pd.read_csv(fpath)
                    col = [c for c in df_cls.columns if any(k in str(c).lower() for k in ["grid", "orig", "id", "mesh"])][0]
                    for g in df_cls[col].dropna().astype(str).unique():
                        grid_class_lookup[g] = c_id
                except Exception:
                    pass
        print(f"✓ 成功匹配 {len(grid_class_lookup)} 個網格的類別標籤")

    raw_df = pd.read_csv(TSV_PATH, sep="\t", names=["date", "od_matrix_raw"])
    raw_df['date_dt'] = pd.to_datetime(raw_df['date'].astype(str), format='%Y%m%d')
    raw_df = raw_df.sort_values('date_dt').reset_index(drop=True)

    daily_od_records, daily_diag_flows, daily_offdiag_flows = {}, {}, {}
    for dt, val in zip(raw_df['date_dt'], raw_df['od_matrix_raw']):
        daily_od_records[dt], daily_diag_flows[dt], daily_offdiag_flows[dt] = {}, {}, {}
        if pd.isna(val) or val == "NA": continue
        try:
            od_dict = ast.literal_eval(val) if isinstance(val, str) else val
            for orig, dests in od_dict.items():
                if orig == "-1_-1": continue
                y_idx, x_idx = map(int, orig.split('_'))
                if 30 <= x_idx <= 70 and 35 <= y_idx <= 70:
                    daily_od_records[dt][orig] = dests
                    diag_val = float(dests.get(orig, 0.0))
                    offdiag_val = sum(float(cnt) for dest, cnt in dests.items() if dest != orig and dest != "-1_-1")
                    daily_diag_flows[dt][orig] = diag_val
                    daily_offdiag_flows[dt][orig] = offdiag_val
        except Exception:
            pass

    diag_df = pd.DataFrame.from_dict(daily_diag_flows, orient='index').fillna(0.0)
    offdiag_df = pd.DataFrame.from_dict(daily_offdiag_flows, orient='index').fillna(0.0)
    
    pre_mask = diag_df.index < PRED_START
    valid_grids = diag_df.columns[diag_df.loc[pre_mask].mean() > 0.0].tolist() if pre_mask.sum() > 0 else diag_df.columns.tolist()

    for g in valid_grids:
        if g not in grid_class_lookup:
            grid_class_lookup[g] = 5

    return diag_df[valid_grids], offdiag_df[valid_grids], daily_od_records, valid_grids, grid_class_lookup

# =========================================================================
# 3. 模組一：震後長期回復函數 (決定 2~3 月波形平均位置)
# =========================================================================
class PostEarthquakeRecoveryEngine:
    def __init__(self, flow_df, grid_class_lookup, valid_grids):
        self.flow_df = flow_df
        self.grid_class_lookup = grid_class_lookup
        self.valid_grids = valid_grids
        
        pre_mask = flow_df.index < PRED_START
        self.M_pre = flow_df.loc[pre_mask, valid_grids].median().clip(lower=1.0)
        
        # 1 月底 (Left Anchor) 與 4 月初 (Right Anchor)
        jan_tail = flow_df.loc["2024-01-25":"2024-01-31", valid_grids]
        self.l_left = jan_tail.median().fillna(self.M_pre).clip(lower=0.0)
        
        right_sub = flow_df.loc["2024-04-01":"2024-04-07", valid_grids] if "2024-04-01" in flow_df.index else jan_tail
        self.l_right = right_sub.median().fillna(self.l_left).clip(lower=0.0)

    def compute_recovery_seed(self, dt):
        """計算在日期 dt 的宏觀平均回復人數 (種子位準)"""
        if dt < EVAL_GAP_START:
            if dt in self.flow_df.index:
                return self.flow_df.loc[dt, self.valid_grids].values
            return self.l_left.values
        elif dt <= EVAL_GAP_END:
            gap_days = (EVAL_GAP_END - EVAL_GAP_START).days + 1
            tau = min(1.0, max(0.0, ((dt - EVAL_GAP_START).days + 1) / gap_days))
            s_curve = 3.0 * (tau ** 2) - 2.0 * (tau ** 3)
            mu_t = self.l_left + s_curve * (self.l_right - self.l_left)
            
            # 各類別形態微調
            for g in self.valid_grids:
                c = self.grid_class_lookup.get(g, 0)
                if c == 3:   # 避難聚集
                    mu_t[g] += np.sin(np.pi * tau) * 0.30 * self.l_left[g]
                elif c == 7: # 激增消散
                    mu_t[g] = self.l_left[g] + (1.0 - np.exp(-3.0 * tau)) * (self.l_right[g] - self.l_left[g])
                elif c == 4: # 部分復原凸曲線
                    mu_t[g] = self.l_left[g] + (tau ** 2.0) * (self.l_right[g] - self.l_left[g])
                elif c == 1: # Persistent Zero
                    mu_t[g] = 0.0
            return mu_t.values
        else:
            if dt in self.flow_df.index:
                return self.flow_df.loc[dt, self.valid_grids].values
            return self.l_right.values

# =========================================================================
# 4. 模組二：震前常態波型資料集 (純震前資料訓練)
# =========================================================================
def extract_calendar_slice(dt):
    dow = dt.dayofweek
    return np.array([
        np.sin(2 * np.pi * dow / 7.0),
        np.cos(2 * np.pi * dow / 7.0),
        1.0 if dow < 5 else 0.0,            # 工作日
        1.0 if dow in [5, 6] else 0.0,       # 週末
        np.sin(2 * np.pi * dt.day / 31.0),
        np.cos(2 * np.pi * dt.day / 31.0)
    ], dtype=np.float32)

class PreEarthquakeWaveformDataset(Dataset):
    """只使用震前 (2024-01-01 以前) 的常態歷史資料建立波型先驗分佈"""
    def __init__(self, flow_df, valid_grids, max_ceiling):
        self.samples = []
        self.max_ceiling = max_ceiling
        
        # 嚴格過濾：僅使用震前資料
        pre_dates = [dt for dt in flow_df.index if dt < PRED_START]
        self.M_pre = flow_df.loc[pre_dates, valid_grids].median().values
        
        raw_res, raw_seeds, raw_times = [], [], []
        for dt in pre_dates:
            y_true = np.nan_to_num(flow_df.loc[dt, valid_grids].values, nan=0.0)
            seed = self.M_pre.copy() # 震前基準水平
            time_slice = extract_calendar_slice(dt)
            
            raw_res.append(y_true - seed)
            raw_seeds.append(seed)
            raw_times.append(time_slice)
            
        raw_res = np.array(raw_res)
        self.scale = np.std(raw_res, axis=0)
        self.scale = np.where(self.scale < 1.0, 1.0, self.scale)
        
        for res, seed, t_slice in zip(raw_res, raw_seeds, raw_times):
            norm_res = np.nan_to_num(res / self.scale, nan=0.0)
            norm_seed = np.nan_to_num(seed / (self.max_ceiling + 1e-4), nan=0.0)
            self.samples.append((norm_res, norm_seed, t_slice))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        res, n_seed, t_slice = self.samples[idx]
        return (torch.tensor(res, dtype=torch.float32), 
                torch.tensor(n_seed, dtype=torch.float32), 
                torch.tensor(t_slice, dtype=torch.float32))

# =========================================================================
# 5. 模組三：去噪擴散網路架構
# =========================================================================
class PureWaveformDenoiser(nn.Module):
    def __init__(self, num_nodes, time_dim=6, hidden_dim=128):
        super().__init__()
        self.num_nodes = num_nodes
        
        self.step_mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        self.cond_norm = nn.LayerNorm(num_nodes + time_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(num_nodes + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 64)
        )
        
        self.in_proj = nn.Linear(num_nodes, hidden_dim)
        self.res1 = nn.Sequential(
            nn.Linear(hidden_dim + 128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.res2 = nn.Sequential(
            nn.Linear(hidden_dim + 128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.out_proj = nn.Linear(hidden_dim, num_nodes)

    def _get_timestep_embedding(self, timesteps, dim=64):
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half)
        args = timesteps[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, x_noisy, t_step, norm_seed, time_slice, drop_mask=None):
        t_emb = self.step_mlp(self._get_timestep_embedding(t_step, 64))
        if drop_mask is not None:
            norm_seed = norm_seed * drop_mask
            time_slice = time_slice * drop_mask
            
        c_in = self.cond_norm(torch.cat([norm_seed, time_slice], dim=-1))
        c_emb = self.cond_mlp(c_in)
        ctx = torch.cat([t_emb, c_emb], dim=-1)
        
        h = self.in_proj(x_noisy)
        h = self.res1(torch.cat([h, ctx], dim=-1)) + h
        h = self.res2(torch.cat([h, ctx], dim=-1)) + h
        return self.out_proj(h)

class DualDiffusionEngine:
    def __init__(self, timesteps=DIFFUSION_STEPS):
        self.timesteps = timesteps
        self.betas = torch.linspace(1e-4, 0.02, timesteps).to(DEVICE)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=DEVICE), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_var = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        if noise is None: noise = torch.randn_like(x_0)
        return self.sqrt_alphas_cumprod[t].unsqueeze(-1) * x_0 + self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1) * noise, noise

    @torch.no_grad()
    def p_sample(self, model, x_t, t, norm_seed, time_slice):
        betas_t = self.betas[t].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        sqrt_recip = torch.sqrt(1.0 / self.alphas[t]).unsqueeze(-1)
        
        pred_eps = model(x_t, t, norm_seed, time_slice)
        mean = sqrt_recip * (x_t - (betas_t / sqrt_one_minus) * pred_eps)
        if (t == 0).all(): return mean
        return mean + torch.sqrt(self.posterior_var[t].unsqueeze(-1)) * torch.randn_like(x_t)

    @torch.no_grad()
    def sample_monte_carlo(self, model, norm_seed, time_slice, k_samples=MC_SAMPLES):
        b, dim = norm_seed.shape[0], model.num_nodes
        accum = torch.zeros(b, dim, device=DEVICE)
        for _ in range(k_samples):
            x_t = torch.randn(b, dim, device=DEVICE)
            for step in reversed(range(self.timesteps)):
                t_tensor = torch.full((b,), step, device=DEVICE, dtype=torch.long)
                x_t = self.p_sample(model, x_t, t_tensor, norm_seed, time_slice)
            accum += x_t
        return torch.clamp(accum / k_samples, -2.5, 2.5)

# =========================================================================
# 6. 主管線流程
# =========================================================================
def main():
    print("=" * 80)
    print("🚀 HuMob 2026：純震前常態波型訓練 ＋ 長期回復種子 (Seed) 遷移生成")
    print(f"🎯 評測正規化標準: Diag / {NORM_DIAG} | Off-Diag / {NORM_OFFDIAG}")
    print("=" * 80)

    # 1. 載入資料
    diag_df, offdiag_df, daily_od_records, valid_grids, grid_class_lookup = load_and_split_flows()
    num_nodes = len(valid_grids)
    print(f"✓ 有效網格數: {num_nodes}")

    pre_mask = diag_df.index < PRED_START
    max_ceiling_diag = diag_df.loc[pre_mask, valid_grids].quantile(0.99).fillna(50.0).values * 1.5 + 5.0
    max_ceiling_offdiag = offdiag_df.loc[pre_mask, valid_grids].quantile(0.99).fillna(20.0).values * 1.5 + 5.0

    # 2. 建立純震前資料載入器
    diag_ds = PreEarthquakeWaveformDataset(diag_df, valid_grids, max_ceiling_diag)
    offdiag_ds = PreEarthquakeWaveformDataset(offdiag_df, valid_grids, max_ceiling_offdiag)
    
    diag_loader = DataLoader(diag_ds, batch_size=BATCH_SIZE, shuffle=True)
    offdiag_loader = DataLoader(offdiag_ds, batch_size=BATCH_SIZE, shuffle=True)
    print(f"✓ 震前常態訓練樣本數: {len(diag_ds)} 天")

    # 3. 建立災後長期回復函數引擎
    rec_engine_diag = PostEarthquakeRecoveryEngine(diag_df, grid_class_lookup, valid_grids)
    rec_engine_offdiag = PostEarthquakeRecoveryEngine(offdiag_df, grid_class_lookup, valid_grids)

    # 4. 訓練 Model 1 (對角線常態波型) 與 Model 2 (非對角線常態波型)
    diff_engine = DualDiffusionEngine(DIFFUSION_STEPS)
    model_diag = PureWaveformDenoiser(num_nodes=num_nodes).to(DEVICE)
    model_offdiag = PureWaveformDenoiser(num_nodes=num_nodes).to(DEVICE)

    opt_diag = optim.AdamW(model_diag.parameters(), lr=LR, weight_decay=1e-4)
    opt_offdiag = optim.AdamW(model_offdiag.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.MSELoss()

    print("\n[2/5] 在震前常態資料上訓練對角線留存波型模型 (Diag Pre-EQ)...")
    model_diag.train()
    for ep in range(1, EPOCHS + 1):
        total_loss = 0.0
        for res, n_seed, t_slice in diag_loader:
            res, n_seed, t_slice = res.to(DEVICE), n_seed.to(DEVICE), t_slice.to(DEVICE)
            t = torch.randint(0, diff_engine.timesteps, (res.shape[0],), device=DEVICE).long()
            x_noisy, noise = diff_engine.q_sample(res, t)
            
            drop_mask = (torch.rand(res.shape[0], 1, device=DEVICE) > COND_DROPOUT_PROB).float()
            pred_noise = model_diag(x_noisy, t, n_seed, t_slice, drop_mask)
            
            loss = criterion(pred_noise, noise)
            opt_diag.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_diag.parameters(), max_norm=1.0)
            opt_diag.step()
            total_loss += loss.item()
        if ep % 30 == 0 or ep == EPOCHS:
            print(f"  Epoch [{ep:03d}/{EPOCHS}] - Diag Pre-EQ Loss: {total_loss / len(diag_loader):.6f}")

    print("\n[3/5] 在震前常態資料上訓練非對角線跨區波型模型 (Off-Diag Pre-EQ)...")
    model_offdiag.train()
    for ep in range(1, EPOCHS + 1):
        total_loss = 0.0
        for res, n_seed, t_slice in offdiag_loader:
            res, n_seed, t_slice = res.to(DEVICE), n_seed.to(DEVICE), t_slice.to(DEVICE)
            t = torch.randint(0, diff_engine.timesteps, (res.shape[0],), device=DEVICE).long()
            x_noisy, noise = diff_engine.q_sample(res, t)
            
            drop_mask = (torch.rand(res.shape[0], 1, device=DEVICE) > COND_DROPOUT_PROB).float()
            pred_noise = model_offdiag(x_noisy, t, n_seed, t_slice, drop_mask)
            
            loss = criterion(pred_noise, noise)
            opt_offdiag.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_offdiag.parameters(), max_norm=1.0)
            opt_offdiag.step()
            total_loss += loss.item()
        if ep % 30 == 0 or ep == EPOCHS:
            print(f"  Epoch [{ep:03d}/{EPOCHS}] - Off-Diag Pre-EQ Loss: {total_loss / len(offdiag_loader):.6f}")

    # 5. 結合回復函數 (Seed) 與擴散模型生成 2~3 月補全
    print("\n[4/5] 結合回復位準 (Seed) 與擴散模型生成 2~3 月補全波型...")
    model_diag.eval()
    model_offdiag.eval()
    
    all_dates = pd.date_range(PRED_START, PRED_END, freq="D")
    pred_diag_flows, pred_offdiag_flows = {}, {}

    with torch.no_grad():
        for dt in all_dates:
            if dt in diag_df.index and not (EVAL_GAP_START <= dt <= EVAL_GAP_END):
                pred_diag_flows[dt] = diag_df.loc[dt, valid_grids].copy()
                pred_offdiag_flows[dt] = offdiag_df.loc[dt, valid_grids].copy()
            else:
                # 1. 回復函數提供宏觀水位 Seed
                seed_d = rec_engine_diag.compute_recovery_seed(dt)
                seed_o = rec_engine_offdiag.compute_recovery_seed(dt)
                
                n_seed_d = torch.tensor(seed_d / (max_ceiling_diag + 1e-4), dtype=torch.float32).unsqueeze(0).to(DEVICE)
                n_seed_o = torch.tensor(seed_o / (max_ceiling_offdiag + 1e-4), dtype=torch.float32).unsqueeze(0).to(DEVICE)
                t_slice = torch.tensor(extract_calendar_slice(dt), dtype=torch.float32).unsqueeze(0).to(DEVICE)
                
                # 2. 擴散模型生成常態週期通勤波動
                res_d = diff_engine.sample_monte_carlo(model_diag, n_seed_d, t_slice).squeeze(0).cpu().numpy() * diag_ds.scale
                # 依水位按比例縮放波動幅度 (避免低水位時波動過大)
                level_ratio_d = np.clip(seed_d / (diag_ds.M_pre + 1e-4), 0.1, 1.2)
                res_d = res_d * level_ratio_d

                res_o = diff_engine.sample_monte_carlo(model_offdiag, n_seed_o, t_slice).squeeze(0).cpu().numpy() * offdiag_ds.scale
                level_ratio_o = np.clip(seed_o / (offdiag_ds.M_pre + 1e-4), 0.1, 1.2)
                res_o = res_o * level_ratio_o
                
                final_d = np.clip(seed_d + res_d, 0.0, max_ceiling_diag)
                final_o = np.clip(seed_o + res_o, 0.0, max_ceiling_offdiag)
                
                for i, g in enumerate(valid_grids):
                    if grid_class_lookup.get(g, 0) == 1:
                        final_d[i] = 0.0
                        final_o[i] = 0.0

                pred_diag_flows[dt] = pd.Series(final_d, index=valid_grids)
                pred_offdiag_flows[dt] = pd.Series(final_o, index=valid_grids)

    # 6. 稀疏化評估與 NRMSE
    print("\n[5/5] 計算最新官方 Combined NRMSE...")
    hist_off_prob = {g: {} for g in valid_grids}
    for dt in diag_df.loc[pre_mask].index:
        day_od = daily_od_records.get(dt, {})
        for orig in valid_grids:
            if orig in day_od:
                for dest, cnt in day_od[orig].items():
                    if dest != orig and dest != "-1_-1":
                        hist_off_prob[orig][dest] = hist_off_prob[orig].get(dest, 0.0) + cnt

    for orig in valid_grids:
        tot = sum(hist_off_prob[orig].values())
        if tot > 0:
            raw_probs = {d: c / tot for d, c in hist_off_prob[orig].items()}
            filtered = {d: p for d, p in raw_probs.items() if p >= SPARSITY_THRESH}
            f_tot = sum(filtered.values())
            hist_off_prob[orig] = {d: p / f_tot for d, p in filtered.items()} if f_tot > 0 else raw_probs
        else:
            hist_off_prob[orig] = {}

    eval_dates = [dt for dt in diag_df.index if dt >= PRED_START and not (EVAL_GAP_START <= dt <= EVAL_GAP_END)]
    daily_metrics = []

    for dt in eval_dates:
        act_od = daily_od_records.get(dt, {})
        p_diag, p_off = pred_diag_flows[dt], pred_offdiag_flows[dt]
        diag_diffs, offdiag_diffs = [], []
        
        for orig in valid_grids:
            act_dests = act_od.get(orig, {})
            diag_diffs.append((p_diag[orig] - float(act_dests.get(orig, 0.0))) ** 2)
            
            probs = hist_off_prob.get(orig, {})
            all_off = set([d for d in act_dests.keys() if d != orig and d != "-1_-1"]).union(probs.keys())
            for dest in all_off:
                pred_cnt = p_off[orig] * probs.get(dest, 0.0)
                act_cnt = float(act_dests.get(dest, 0.0))
                offdiag_diffs.append((pred_cnt - act_cnt) ** 2)

        daily_metrics.append({
            "RMSE_diag": np.sqrt(np.mean(diag_diffs)) if diag_diffs else 0.0,
            "RMSE_offdiag": np.sqrt(np.mean(offdiag_diffs)) if offdiag_diffs else 0.0
        })

    eval_res = pd.DataFrame(daily_metrics)
    m_diag, m_off = eval_res["RMSE_diag"].mean(), eval_res["RMSE_offdiag"].mean()
    n_diag, n_off = m_diag / NORM_DIAG, m_off / NORM_OFFDIAG
    combined_nrmse = 0.5 * n_diag + 0.5 * n_off

    print("-" * 75)
    print(f"🔹 Mean RMSE (Diag):               {m_diag:8.4f}")
    print(f"🔹 Mean RMSE (Off-Diag):           {m_off:8.4f}")
    print(f"🔸 NRMSE (Diag)     [/{NORM_DIAG}]:    {n_diag:8.4f}")
    print(f"🔸 NRMSE (Off-Diag) [/{NORM_OFFDIAG}]:  {n_off:8.4f}")
    print(f"🏆 Pre-EQ Diffusion Combined NRMSE: {combined_nrmse:8.4f}")
    print("-" * 75)

    # 7. 匯出 CSV 與繪製 9 大類別圖譜
    pred_diag_df = pd.DataFrame.from_dict(pred_diag_flows, orient='index')
    pred_offdiag_df = pd.DataFrame.from_dict(pred_offdiag_flows, orient='index')
    pred_total_df = pred_diag_df + pred_offdiag_df
    pred_total_df.to_csv(os.path.join(OUTPUT_DIR, "pred_total_flows.csv"), encoding="utf-8-sig")

    total_truth_df = diag_df + offdiag_df
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 3, figsize=(19, 11.5), dpi=250)
    fig.patch.set_facecolor('#0f172a')
    fig.suptitle(f'HuMob 2026: Pre-EQ Diffusion + Recovery Seed across 9 Classes | NRMSE: {combined_nrmse:.4f}', 
                 fontsize=14, fontweight='bold', color='#ffffff', y=0.98)

    pre_gap_mask = total_truth_df.index < EVAL_GAP_START
    post_gap_mask = total_truth_df.index > EVAL_GAP_END

    for c_id in range(1, 10):
        row, col = (c_id - 1) // 3, (c_id - 1) % 3
        ax = axes[row, col]
        ax.set_facecolor('#1e293b')
        c_grids = [g for g in valid_grids if grid_class_lookup.get(g) == c_id]
        
        if not c_grids:
            ax.text(0.5, 0.5, f'No Grids in Class {c_id}', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, color='#64748b', fontsize=10)
            ax.set_title(CLASS_INFO_MAP[c_id], fontsize=10, color='#94a3b8')
            continue

        actual_pre = total_truth_df.loc[pre_gap_mask, c_grids].mean(axis=1)
        actual_post = total_truth_df.loc[post_gap_mask, c_grids].mean(axis=1)
        pred_line = pred_total_df[c_grids].mean(axis=1)

        ax.axvspan(EVAL_GAP_START, EVAL_GAP_END, color='#f59e0b', alpha=0.15, label='Feb-Mar Gap' if c_id == 1 else "")
        ax.plot(actual_pre.index, actual_pre, color='#f43f5e', linewidth=1.2, label='Ground Truth' if c_id == 1 else "")
        ax.plot(actual_post.index, actual_post, color='#f43f5e', linewidth=1.2)
        ax.plot(pred_line.index, pred_line, color='#10b981', linewidth=1.4, linestyle='--', label='Pre-EQ Diffusion Pred' if c_id == 1 else "")

        ax.set_title(f"{CLASS_INFO_MAP[c_id]} (N={len(c_grids)})", fontsize=10, fontweight='bold', color='#f8fafc', pad=6)
        ax.grid(True, color='#334155', linestyle=':', alpha=0.5)
        ax.tick_params(colors='#94a3b8', labelsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=3, 
               fontsize=10, frameon=True, facecolor='#1e293b', edgecolor='#475569')
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    
    chart_path = os.path.join(OUTPUT_DIR, "all_9classes_preeq_diffusion_overview.png")
    plt.savefig(chart_path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ 圖表已存檔至: {chart_path}")

if __name__ == "__main__":
    main()
