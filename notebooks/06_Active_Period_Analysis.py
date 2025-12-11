"""
06: 活跃期准实验分析
基于 #新冠后遗症# 话题的两个活跃期进行理论验证
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf

# 路径设置
ROOT = Path("..").resolve() if Path("..").resolve().name == "emotion_dynamics" else Path(".").resolve()
sys.path.insert(0, str(ROOT))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 输出路径
FIG_DIR = ROOT / "outputs/figs/empirical"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ============ 1. 数据加载与关联 ============
from src.empirical.user_mapper import UserTypeMapper
from src.empirical.time_series import (
    TimeSeriesConfig, aggregate_time_series,
    calculate_rolling_stats, calculate_rolling_ac1,
    calculate_r_proxy, get_topic_summary
)

# 1.1 加载原始数据
DATA_PATH = ROOT / "dataset/Topic_data/#新冠后遗症#_filtered.csv"
df_raw = pd.read_csv(DATA_PATH)
df_raw['publish_time'] = pd.to_datetime(df_raw['publish_time'])
print(f"原始数据: {len(df_raw)} 条")

# 1.2 映射用户类型
mapper = UserTypeMapper()
df_raw['user_type'] = [
    mapper.map_verify_type(row['verify_typ'], row['user_name']).user_type
    for _, row in df_raw.iterrows()
]
print("用户类型分布:")
print(df_raw['user_type'].value_counts())

# 1.3 加载 LLM 标注结果
ANNOT_PATH = ROOT / "outputs/annotations/v3/annotated_intent_rule_v3.jsonl"
df_ann = pd.read_json(ANNOT_PATH, lines=True)
print(f"标注数据: {len(df_ann)} 条")

# 1.4 合并标注（优先用 mid，对齐失败时再用截断文本一对一匹配）
if 'mid' in df_ann.columns and 'mid' in df_raw.columns:
    # 去重后再合并，避免一对多
    df_ann_mid = df_ann.drop_duplicates(subset=['mid'])
    df_raw_mid = df_raw.drop_duplicates(subset=['mid'])
    df = df_raw_mid.merge(
        df_ann_mid[['mid', 'emotion_class', 'risk_class']],
        on='mid',
        how='inner',
        validate='one_to_one'
    )
else:
    df_raw['match_key'] = df_raw['content'].str[:120].str.strip()
    df_ann['match_key'] = df_ann.get('original_text', df_ann.get('text', '')).str[:120].str.strip()
    df_ann = df_ann.drop_duplicates(subset=['match_key'])
    df_raw = df_raw.drop_duplicates(subset=['match_key'])
    df = df_raw.merge(
        df_ann[['match_key', 'emotion_class', 'risk_class']],
        on='match_key',
        how='inner',
        validate='one_to_one'
    )
print(f"成功关联: {len(df)} 条 ({len(df)/len(df_raw)*100:.1f}%)")

# ============ 2. 活跃期筛选 ============
PERIOD_1 = ('2022-03-01', '2022-04-01')  # 上海疫情
PERIOD_2 = ('2022-12-01', '2023-03-01')  # 放开后

df_p1 = df[(df['publish_time'] >= PERIOD_1[0]) & (df['publish_time'] < PERIOD_1[1])].copy()
df_p2 = df[(df['publish_time'] >= PERIOD_2[0]) & (df['publish_time'] < PERIOD_2[1])].copy()

print(f"活跃期1: {len(df_p1)} 条")
print(f"活跃期2: {len(df_p2)} 条")

# ============ 3. 时间序列聚合（降低阈值） ============
config_1h = TimeSeriesConfig(
    time_col='publish_time',
    emotion_col='emotion_class',
    risk_col='risk_class',
    user_type_col='user_type',
    freq='1h',
    min_posts=3,  # 降低阈值
)

config_4h = TimeSeriesConfig(
    time_col='publish_time',
    emotion_col='emotion_class',
    risk_col='risk_class',
    user_type_col='user_type',
    freq='4h',
    min_posts=5,
)

ts_p1 = aggregate_time_series(df_p1, config_1h)
ts_p2 = aggregate_time_series(df_p2, config_4h)

ts_p1['r_proxy'] = calculate_r_proxy(ts_p1)
ts_p2['r_proxy'] = calculate_r_proxy(ts_p2)

print(f"活跃期1 (1h): {len(ts_p1)} 窗口, a非空 {ts_p1['a'].notna().sum()}")
print(f"活跃期2 (4h): {len(ts_p2)} 窗口, a非空 {ts_p2['a'].notna().sum()}")

# ============ 4. 计算突变指标 ============
def calculate_jump_metrics(ts, Q_col='Q'):
    """计算突变指标"""
    ts = ts.copy()
    ts['dQ_dt'] = ts[Q_col].diff()
    ts['abs_dQ_dt'] = ts['dQ_dt'].abs()
    ts['Q_volatility'] = ts[Q_col].rolling(6, min_periods=3).std()
    return ts

ts_p1 = calculate_jump_metrics(ts_p1)
ts_p2 = calculate_jump_metrics(ts_p2)

# ============ 5. 假设检验 ============

# H1: a 与突变指标的相关性
def test_h1(ts, title):
    valid = ts[['a', 'abs_dQ_dt', 'Q_volatility']].dropna()
    if len(valid) < 5:
        print(f"{title}: 有效数据不足 ({len(valid)})")
        return
    r1, p1 = stats.pearsonr(valid['a'], valid['abs_dQ_dt'])
    r2, p2 = stats.pearsonr(valid['a'], valid['Q_volatility'])
    print(f"=== {title} (n={len(valid)}) ===")
    print(f"  a vs |dQ/dt|: r={r1:.3f}, p={p1:.4f}")
    print(f"  a vs volatility: r={r2:.3f}, p={p2:.4f}")

# H2: r_proxy 与波动性
def test_h2(ts, title):
    valid = ts[['r_proxy', 'Q_volatility']].dropna()
    if len(valid) < 5:
        print(f"{title}: 有效数据不足")
        return
    r, p = stats.pearsonr(valid['r_proxy'], valid['Q_volatility'])
    print(f"=== {title} (n={len(valid)}) ===")
    print(f"  r_proxy vs volatility: r={r:.3f}, p={p:.4f}")

# H3: 交互效应
def test_h3(ts, title):
    valid = ts[['a', 'r_proxy', 'Q_volatility']].dropna()
    if len(valid) < 10:
        print(f"{title}: 有效数据不足")
        return
    a_med, r_med = valid['a'].median(), valid['r_proxy'].median()
    high = valid[(valid['r_proxy'] > r_med) & (valid['a'] > a_med)]['Q_volatility']
    low = valid[(valid['r_proxy'] <= r_med) & (valid['a'] <= a_med)]['Q_volatility']
    print(f"=== {title} ===")
    print(f"  高r高a: {len(high)}, 均值={high.mean():.4f}")
    print(f"  低r低a: {len(low)}, 均值={low.mean():.4f}")
    if len(high) >= 3 and len(low) >= 3:
        t, p = stats.ttest_ind(high, low)
        print(f"  t={t:.3f}, p={p:.4f}")

print("\n--- H1 检验 ---")
test_h1(ts_p1, "活跃期1")
test_h1(ts_p2, "活跃期2")

print("\n--- H2 检验 ---")
test_h2(ts_p1, "活跃期1")
test_h2(ts_p2, "活跃期2")

print("\n--- H3 检验 ---")
test_h3(ts_p1, "活跃期1")
test_h3(ts_p2, "活跃期2")

# ============ 6. 临界慢化信号 ============ 
ts_p1_csd = calculate_rolling_ac1(ts_p1.copy(), column='Q', window_size=6)
ts_p1_csd = calculate_rolling_stats(ts_p1_csd, window_size=6, columns=['Q'])

ts_p2_csd = calculate_rolling_ac1(ts_p2.copy(), column='Q', window_size=12)
ts_p2_csd = calculate_rolling_stats(ts_p2_csd, window_size=12, columns=['Q'])

# ============ 7. 变量分布与回归（交互效应） ============
print("\n--- r_proxy 与 a 分布（活跃期2） ---")
print(ts_p2[['a', 'r_proxy']].describe())
print("r_proxy 分位数:", ts_p2['r_proxy'].quantile([0.25, 0.5, 0.75, 0.9]).to_dict())

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ts_p2['a'].dropna().hist(ax=axes[0], bins=20)
axes[0].set_title('a 分布')
ts_p2['r_proxy'].dropna().hist(ax=axes[1], bins=20)
axes[1].set_title('r_proxy 分布')
plt.tight_layout()
plt.savefig(FIG_DIR / "active_period_p2_dist.png", dpi=200)

# 回归检验交互效应（活跃期2）
valid_reg = ts_p2.dropna(subset=['a', 'r_proxy', 'Q_volatility', 'abs_dQ_dt'])
if len(valid_reg) >= 20:
    model_vol = smf.ols('Q_volatility ~ a * r_proxy', data=valid_reg).fit()
    model_dq = smf.ols('abs_dQ_dt ~ a * r_proxy', data=valid_reg).fit()
    print("\n--- 回归：Q_volatility ~ a * r_proxy (活跃期2) ---")
    print(model_vol.summary())
    print("\n--- 回归：|dQ/dt| ~ a * r_proxy (活跃期2) ---")
    print(model_dq.summary())
else:
    print("回归样本不足，跳过交互回归")

# ============ 8. 保存结果 ============
OUTPUT_DIR = ROOT / "outputs/annotations/v3"
ts_p1_csd.to_csv(OUTPUT_DIR / "time_series_p1_1h.csv", index=False)
ts_p2_csd.to_csv(OUTPUT_DIR / "time_series_p2_4h.csv", index=False)
print(f"\nSaved to {OUTPUT_DIR}")
