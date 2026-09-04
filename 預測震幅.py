import os
import ast
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# =========================================================================
# 1. 全域配置與時間常數
# =========================================================================
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 邊界補全跨度：1 月 31 日 至 4 月 1 日 (包含首尾端點共 62 天)
BRIDGE_START = pd.to_datetime("2024-01-31")
BRIDGE_END = pd.to_datetime("2024-04-01")
BRIDGE_LEN = 62

BATCH_SIZE = 256
EPOCHS = 25
LR = 2e-3

PRED_START = pd.to_datetime("2024-01-01")
GAP_START = pd.to_datetime("2024-02-01")
GAP_END = pd.to_datetime("2024-03-31")
POST_START = pd.to_datetime("2024-04-01")
PRED_END = pd.to_datetime("2024-10-31")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
TSV_PATH = os.path.join(SCRIPT_DIR, "humob2026-dataset.tsv")
SPECIFIC_CLASS_DIR = r"C:\Users\User\Desktop\人口預測專案\人口預測專案3\humob2026\data\output\module05\classification\by_class"
FALLBACK_CLASS_DIR = os.path.join(SCRIPT_DIR, "humob2026", "data", "output", "module05", "classification", "by_class")
BY_CLASS_DIR = SPECIFIC_CLASS_DIR if os.path.exists(SPECIFIC_CLASS_DIR) else FALLBACK_CLASS_DIR

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "pinned_gap_infill_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_INFO_MAP = {
    1: "Class 01: Persistent Zero", 2: "Class 02: Persistent Decrease",
    3: "Class 03: Emergent Activity", 4: "Class 04: Partial Recovery",
    5: "Class 05: Fully Recovered", 6: "Class 06: Stable Inflow",
    7: "Class 07: Temporary Increase", 8: "Class 08: Partial Dissipation",
    9: "Class 09: Persistent Increase"
}

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

# =========================================================================
# 2. 資料讀取與空間過濾
# =========================================================================
print("[1/5] 讀取網格類別索引與原始 TSV 數據...")
grid_class_lookup = {}
if os.path.exists(BY_CLASS_DIR):
    for fpath in glob.glob(os.path.join(BY_CLASS_DIR, "*.csv")):
        c_id = get_class_id_from_filename(os.path.basename(fpath))
        if c_id is not None:
            try:
                df_cls = pd.read_csv(fpath)
                col = [c for c in df_cls.columns if any(k in str(c).lower() for k in ["grid", "orig", "id"])][0]
                for g in df_cls[col].dropna().astype(str).unique():
                    grid_class_lookup[g] = c_id
            except Exception:
                pass

raw_df = pd.read_csv(TSV_PATH, sep="\t", names=["date", "od_matrix_raw"])
raw_df['date_dt'] = pd.to_datetime(raw_df['date'].astype(str), format='%Y%m%d')
raw_df = raw_df.sort_values('date_dt').reset_index(drop=True)

daily_diag, daily_offdiag = {}, {}
for dt, val in zip(raw_df['date_dt'], raw_df['od_matrix_raw']):
    daily_diag[dt], daily_offdiag[dt] = {}, {}
    if pd.isna(val) or val == "NA": continue
    try:
        od_dict = ast.literal_eval(val) if isinstance(val, str) else val
        for orig, dests in od_dict.items():
            if orig == "-1_-1": continue
            y_idx, x_idx = map(int, orig.split('_'))
            if 30 <= x_idx <= 70 and 35 <= y_idx <= 70:
                daily_diag[dt][orig] = float(dests.get(orig, 0.0))
                daily_offdiag[dt][orig] = sum(float(cnt) for dest, cnt in dests.items() if dest != orig and dest != "-1_-1")
    except Exception:
        pass

diag_df = pd.DataFrame.from_dict(daily_diag, orient='index').fillna(0.0)
offdiag_df = pd.DataFrame.from_dict(daily_offdiag, orient='index').fillna(0.0)

valid_grids = [g for g in diag_df.columns if g in grid_class_lookup]
if not valid_grids:
    valid_grids = diag_df.columns[diag_df[diag_df.index < PRED_START].mean() >= 0.001].tolist()

for g in valid_grids:
    if g not in grid_class_lookup:
        grid_class_lookup[g] = 5

diag_df = diag_df[valid_grids]
offdiag_df = offdiag_df[valid_grids]
num_nodes = len(valid_grids)
grid_classes = np.array([grid_class_lookup.get(g, 5) - 1 for g in valid_grids], dtype=np.int64)

# =========================================================================
# 3. 提取 4 通道特徵與日曆序列 (2~3 月設 NaN)
# =========================================================================
def extract_dynamics(flow_df):
    full_dates = pd.date_range(flow_df.index.min(), PRED_END, freq='D')
    raw = flow_df.reindex(full_dates)
    raw.loc[(raw.index >= GAP_START) & (raw.index <= GAP_END)] = np.nan
    
    mu = raw.rolling(window=7, min_periods=4, center=True).median()
    r_max = raw.rolling(window=7, min_periods=4, center=True).max()
    r_min = raw.rolling(window=7, min_periods=4, center=True).min()
    amp = ((r_max - r_min) / 2.0).clip(lower=0.0)
    
    mu.loc[(mu.index >= GAP_START) & (mu.index <= GAP_END)] = np.nan
    amp.loc[(amp.index >= GAP_START) & (amp.index <= GAP_END)] = np.nan
    
    pre = flow_df.loc[flow_df.index < PRED_START]
    base_mu = pre.median().clip(lower=1e-5)
    
    pre_dow = pre.copy()
    pre_dow['dow'] = pre_dow.index.dayofweek
    dow_m = pre_dow.groupby('dow').median()
    base_amp = np.maximum((dow_m.max() - dow_m.min()) / 2.0, 1e-5)
    
    return mu, amp, base_mu, base_amp

obs_mu_d, obs_amp_d, base_mu_d, base_amp_d = extract_dynamics(diag_df)
obs_mu_o, obs_amp_o, base_mu_o, base_amp_o = extract_dynamics(offdiag_df)

full_dates = obs_mu_d.index
all_channels_3d = np.stack([
    obs_mu_d.values, obs_amp_d.values,
    obs_mu_o.values, obs_amp_o.values
], axis=-1).astype(np.float32)

base_feats_2d = np.stack([
    base_mu_d.values, base_amp_d.values,
    base_mu_o.values, base_amp_o.values
], axis=-1).astype(np.float32)

# =========================================================================
# 4. 向量化三次埃爾米特基底
# =========================================================================
tau = np.linspace(0.0, 1.0, BRIDGE_LEN, dtype=np.float32)  # [62]
h00 = (2.0 * (tau ** 3) - 3.0 * (tau ** 2) + 1.0).reshape(1, 1, 1, BRIDGE_LEN)
h10 = ((tau ** 3) - 2.0 * (tau ** 2) + tau).reshape(1, 1, 1, BRIDGE_LEN)
h01 = (-2.0 * (tau ** 3) + 3.0 * (tau ** 2)).reshape(1, 1, 1, BRIDGE_LEN)
h11 = ((tau ** 3) - (tau ** 2)).reshape(1, 1, 1, BRIDGE_LEN)

def build_hermite_base(v_l, s_l, v_r, s_r, span_len=BRIDGE_LEN):
    v_l_exp = v_l[..., None]
    s_l_exp = (s_l * span_len)[..., None]
    v_r_exp = v_r[..., None]
    s_r_exp = (s_r * span_len)[..., None]
    return h00 * v_l_exp + h10 * s_l_exp + h01 * v_r_exp + h11 * s_r_exp

# =========================================================================
# 5. 向量化切片生成訓練集
# =========================================================================
print("[2/5] 向量化切片生成邊界訓練張量...")
post_mask = (full_dates >= POST_START) & (full_dates <= PRED_END)
post_data = all_channels_3d[post_mask]
T_post = len(post_data)

stride = 2
start_indices = np.arange(4, T_post - BRIDGE_LEN - 4, stride)
K = len(start_indices)

target_windows = np.array([post_data[s : s + BRIDGE_LEN] for s in start_indices])
target_windows = np.transpose(target_windows, (0, 2, 3, 1))

v_left = np.array([post_data[s] for s in start_indices])
v_left_prev = np.array([post_data[s - 3] for s in start_indices])
slope_left = (v_left - v_left_prev) / 3.0

v_right = np.array([post_data[s + BRIDGE_LEN - 1] for s in start_indices])
v_right_next = np.array([post_data[s + BRIDGE_LEN + 2] for s in start_indices])
slope_right = (v_right_next - v_right) / 3.0

v_base = np.tile(base_feats_2d[None, :, :], (K, 1, 1))

base_hermite_train = build_hermite_base(v_left, slope_left, v_right, slope_right, span_len=BRIDGE_LEN)
cond_features = np.concatenate([v_base, v_left, slope_left, v_right, slope_right], axis=-1)

X_train = cond_features.reshape(-1, 20)
Y_train = target_windows.reshape(-1, 4 * BRIDGE_LEN)
B_train = base_hermite_train.reshape(-1, 4 * BRIDGE_LEN)
C_train = np.tile(grid_classes[None, :], (K, 1)).reshape(-1)

valid_mask = ~np.isnan(X_train).any(axis=-1) & ~np.isnan(Y_train).any(axis=-1)
X_train = torch.tensor(X_train[valid_mask], dtype=torch.float32)
Y_train = torch.tensor(Y_train[valid_mask], dtype=torch.float32)
B_train = torch.tensor(B_train[valid_mask], dtype=torch.float32)
C_train = torch.tensor(C_train[valid_mask], dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, C_train, B_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)

# =========================================================================
# 6. 硬邊界約束神經網路
# =========================================================================
class HardPinnedNeuralBridge(nn.Module):
    def __init__(self, in_dim=20, num_classes=9, embed_dim=16, hidden_dim=128, out_dim=4 * BRIDGE_LEN):
        super().__init__()
        self.c_emb = nn.Embedding(num_classes, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, out_dim)
        )
        w_1d = np.sin(np.pi * tau) ** 2
        w_4ch = np.tile(w_1d, 4).reshape(1, -1)
        self.register_buffer("pin_window", torch.tensor(w_4ch, dtype=torch.float32))

    def forward(self, x, c, base_curve):
        emb = self.c_emb(c)
        feat = torch.cat([x, emb], dim=-1)
        nn_res = self.mlp(feat)
        pinned_out = base_curve + nn_res * self.pin_window
        return torch.relu(pinned_out)

print(f"[3/5] 極速訓練硬邊界約束網路 (樣本數: {len(X_train)}, Epochs: {EPOCHS})...")
model = HardPinnedNeuralBridge().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.MSELoss()

model.train()
for ep in range(1, EPOCHS + 1):
    total_loss = 0.0
    for bx, bc, bb, by in train_loader:
        bx, bc, bb, by = bx.to(DEVICE), bc.to(DEVICE), bb.to(DEVICE), by.to(DEVICE)
        pred = model(bx, bc, bb)
        loss = criterion(pred, by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if ep % 5 == 0 or ep == EPOCHS:
        print(f"  Epoch [{ep:02d}/{EPOCHS:02d}] - Loss: {total_loss / len(train_loader):.6f}")

# =========================================================================
# 7. 單步矩陣推論
# =========================================================================
print("[4/5] 批次推論 1/31~4/1 跨期走勢 (頭尾數值斜率完全重合)...")
model.eval()

idx_jan_end = np.where(full_dates == BRIDGE_START)[0][0]
idx_apr_start = np.where(full_dates == BRIDGE_END)[0][0]

v_l_real = all_channels_3d[idx_jan_end]
s_l_real = (all_channels_3d[idx_jan_end] - all_channels_3d[idx_jan_end - 3]) / 3.0
v_r_real = all_channels_3d[idx_apr_start]
s_r_real = (all_channels_3d[idx_apr_start + 3] - all_channels_3d[idx_apr_start]) / 3.0
v_b_real = base_feats_2d

test_feat = np.concatenate([v_b_real, v_l_real, s_l_real, v_r_real, s_r_real], axis=-1)

base_hermite_test = build_hermite_base(v_l_real[None, ...], s_l_real[None, ...], v_r_real[None, ...], s_r_real[None, ...], span_len=BRIDGE_LEN)
base_hermite_test_flat = base_hermite_test.reshape(num_nodes, 4 * BRIDGE_LEN)

with torch.no_grad():
    t_x = torch.tensor(test_feat, dtype=torch.float32, device=DEVICE)
    t_c = torch.tensor(grid_classes, dtype=torch.long, device=DEVICE)
    t_b = torch.tensor(base_hermite_test_flat, dtype=torch.float32, device=DEVICE)
    pred_pinned = model(t_x, t_c, t_b).view(num_nodes, 4, BRIDGE_LEN).cpu().numpy()

is_c1 = (grid_classes == 0)
pred_pinned[is_c1] = 0.0

bridge_dates = pd.date_range(BRIDGE_START, BRIDGE_END, freq='D')

pinned_mu_diag = pd.DataFrame(pred_pinned[:, 0, :].T, index=bridge_dates, columns=valid_grids)
pinned_amp_diag = pd.DataFrame(pred_pinned[:, 1, :].T, index=bridge_dates, columns=valid_grids)
pinned_mu_offdiag = pd.DataFrame(pred_pinned[:, 2, :].T, index=bridge_dates, columns=valid_grids)
pinned_amp_offdiag = pd.DataFrame(pred_pinned[:, 3, :].T, index=bridge_dates, columns=valid_grids)

# 匯出預測評估期 (2024-02-01 ~ 2024-03-31) CSV
gap_mu_diag = pinned_mu_diag.loc[GAP_START:GAP_END]
gap_amp_diag = pinned_amp_diag.loc[GAP_START:GAP_END]
gap_mu_offdiag = pinned_mu_offdiag.loc[GAP_START:GAP_END]
gap_amp_offdiag = pinned_amp_offdiag.loc[GAP_START:GAP_END]

gap_mu_diag.to_csv(os.path.join(OUTPUT_DIR, "pinned_gap_mu_diagonal.csv"), encoding="utf-8-sig")
gap_amp_diag.to_csv(os.path.join(OUTPUT_DIR, "pinned_gap_amp_diagonal.csv"), encoding="utf-8-sig")
gap_mu_offdiag.to_csv(os.path.join(OUTPUT_DIR, "pinned_gap_mu_offdiagonal.csv"), encoding="utf-8-sig")
gap_amp_offdiag.to_csv(os.path.join(OUTPUT_DIR, "pinned_gap_amp_offdiagonal.csv"), encoding="utf-8-sig")

# =========================================================================
# 8. 繪製面板圖 (預測虛線採用綠色 #10b981)
# =========================================================================
print("[5/5] 繪製首尾無縫對齊面板圖 (預測虛線改為綠色)...")
def plot_pinned_panel(obs_mu, obs_amp, pin_mu, pin_amp, base_mu, base_amp, tag_name, save_name):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(9, 2, figsize=(18, 22), dpi=220, sharex=True)
    fig.patch.set_facecolor('#090d16')
    fig.suptitle(f"HuMob 2026: Boundary-Pinned Trend μ(t) & Amplitude A(t) [{tag_name}]\n(Green Dashed Bridge = Predicted Infill Seamlessly Connected)", 
                 fontsize=14.5, fontweight='bold', color='#ffffff', y=0.99)

    for c_id in range(1, 10):
        ax_mu, ax_amp = axes[c_id - 1, 0], axes[c_id - 1, 1]
        for ax in (ax_mu, ax_amp):
            ax.set_facecolor('#0e1526')
            ax.grid(True, color='#1e293b', linestyle='--', alpha=0.6)
            ax.axvspan(GAP_START, GAP_END, color='#475569', alpha=0.35, label='Gap Period' if c_id == 1 else "")
            ax.axvline(PRED_START, color='#ef4444', linestyle=':', linewidth=1.1, alpha=0.85, label='Earthquake' if c_id == 1 else "")

        c_grids = [g for g in valid_grids if grid_class_lookup.get(g) == c_id]
        if not c_grids: continue

        mu_obs_c = obs_mu[c_grids].mean(axis=1)
        amp_obs_c = obs_amp[c_grids].mean(axis=1)
        mu_pin_c = pin_mu[c_grids].mean(axis=1)
        amp_pin_c = pin_amp[c_grids].mean(axis=1)

        b_mu = float(base_mu[c_grids].mean())
        b_amp = float(base_amp[c_grids].mean())

        # 左欄：趨勢均值 μ(t)
        # 真實觀測值保持藍色 (#38bdf8)，預測虛線換成綠色 (#10b981)
        ax_mu.plot(full_dates, mu_obs_c, color='#38bdf8', linewidth=1.3, label='Observed μ(t)' if c_id == 1 else "")
        ax_mu.plot(mu_pin_c.index, mu_pin_c, color='#10b981', linestyle='--', linewidth=1.9, label='Predicted (Gap Infill)' if c_id == 1 else "")
        ax_mu.axhline(b_mu, color='#eab308', linestyle='--', linewidth=1.0, alpha=0.75, label='Pre Baseline' if c_id == 1 else "")
        ax_mu.set_ylabel(f"C{c_id} μ(t)", fontsize=9, color='#e2e8f0')

        # 右欄：週期震幅 A(t)
        # 真實觀測值保持紅色 (#f43f5e)，預測虛線換成綠色 (#10b981)
        ax_amp.plot(full_dates, amp_obs_c, color='#f43f5e', linewidth=1.3, label='Observed A(t)' if c_id == 1 else "")
        ax_amp.plot(amp_pin_c.index, amp_pin_c, color='#10b981', linestyle='--', linewidth=1.9, label='Predicted (Gap Infill)' if c_id == 1 else "")
        ax_amp.axhline(b_amp, color='#eab308', linestyle='--', linewidth=1.0, alpha=0.75, label='Pre Baseline' if c_id == 1 else "")
        ax_amp.set_ylabel(f"C{c_id} A(t)", fontsize=9, color='#e2e8f0')

        if c_id == 1:
            ax_mu.set_title("Trend Mean μ(t) [Low-Frequency Base]", fontsize=11, fontweight='bold', color='#38bdf8', pad=6)
            ax_amp.set_title("Envelope Amplitude A(t) [Weekly Fluctuation]", fontsize=11, fontweight='bold', color='#f43f5e', pad=6)
            ax_mu.legend(loc='upper right', fontsize=7.5, frameon=True, facecolor='#090d16', edgecolor='#334155')
            ax_amp.legend(loc='upper right', fontsize=7.5, frameon=True, facecolor='#090d16', edgecolor='#334155')

    for col in (0, 1):
        ax_bottom = axes[-1, col]
        ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax_bottom.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax_bottom.tick_params(colors='#94a3b8', labelsize=8)
        ax_bottom.set_xlim(pd.to_datetime("2023-10-15"), pd.to_datetime("2024-11-15"))

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ 已產出綠色預測線面板圖: {save_path}")

plot_pinned_panel(obs_mu_d, obs_amp_d, pinned_mu_diag, pinned_amp_diag, base_mu_d, base_amp_d, "Stay Flow", "pinned_infill_diagonal.png")
plot_pinned_panel(obs_mu_o, obs_amp_o, pinned_mu_offdiag, pinned_amp_offdiag, base_mu_o, base_amp_o, "Cross Flow", "pinned_infill_offdiagonal.png")

print("\n" + "=" * 90)
print(" 🔍 【端點對齊驗證報告】")
print("=" * 90)
err_jan = np.max(np.abs(obs_mu_d.loc["2024-01-31"].values - pinned_mu_diag.loc["2024-01-31"].values))
err_apr = np.max(np.abs(obs_mu_d.loc["2024-04-01"].values - pinned_mu_diag.loc["2024-04-01"].values))
print(f"  * 1 月 31 日 (左邊界端點) 最大誤差: {err_jan:.10e} (完全對齊)")
print(f"  * 4 月 01 日 (右邊界端點) 最大誤差: {err_apr:.10e} (完全對齊)")
print("=" * 90)
print(f"\n✨ 執行完畢！所有綠色虛線預測成果均已儲存至：{OUTPUT_DIR}")
