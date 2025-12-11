"""分析活跃期数据"""
import pandas as pd

df = pd.read_csv('outputs/annotations/v3/time_series_1h.csv')
df['time'] = pd.to_datetime(df['time_window'])

# 活跃期1: 2022年3月 (上海疫情)
period1 = df[(df['time'] >= '2022-03-01') & (df['time'] < '2022-04-01')]
print('=== 活跃期1: 2022-03 上海疫情 ===')
print(f'窗口数: {len(period1)}')
print(f'帖子数: {period1["n_posts"].sum()}')
print(f'公众帖: {period1["n_public"].sum()}')
print(f'自媒体: {period1["n_wemedia"].sum()}')
print(f'主流媒体: {period1["n_mainstream"].sum()}')
print(f'a 非空窗口: {period1["a"].notna().sum()}')
print()

# 活跃期2: 2022-12 ~ 2023-02 (放开后)
period2 = df[(df['time'] >= '2022-12-01') & (df['time'] < '2023-03-01')]
print('=== 活跃期2: 2022-12~2023-02 放开后 ===')
print(f'窗口数: {len(period2)}')
print(f'帖子数: {period2["n_posts"].sum()}')
print(f'公众帖: {period2["n_public"].sum()}')
print(f'自媒体: {period2["n_wemedia"].sum()}')
print(f'主流媒体: {period2["n_mainstream"].sum()}')
print(f'a 非空窗口: {period2["a"].notna().sum()}')
print()

# 两个时期合计
combined = pd.concat([period1, period2])
print('=== 两个活跃期合计 ===')
print(f'窗口数: {len(combined)}')
print(f'帖子数: {combined["n_posts"].sum()}')
print(f'a 非空窗口: {combined["a"].notna().sum()}')
print()

# 检查活跃期1的时间连续性
print('=== 活跃期1 详细分布 ===')
period1_daily = period1.groupby(period1['time'].dt.date).agg({
    'n_posts': 'sum',
    'n_public': 'sum',
    'n_wemedia': 'sum',
    'n_mainstream': 'sum',
    'a': lambda x: x.notna().sum()
}).rename(columns={'a': 'a_valid_windows'})
print(period1_daily[period1_daily['n_posts'] > 10])
print()

# 检查活跃期2的月分布
print('=== 活跃期2 月度分布 ===')
period2['month'] = period2['time'].dt.to_period('M')
period2_monthly = period2.groupby('month').agg({
    'n_posts': 'sum',
    'n_public': 'sum',
    'n_wemedia': 'sum',
    'n_mainstream': 'sum',
    'a': lambda x: x.notna().sum()
}).rename(columns={'a': 'a_valid_windows'})
print(period2_monthly)


