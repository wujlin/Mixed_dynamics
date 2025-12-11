"""分析 time_series_1h.csv 的数据稀疏性"""
import pandas as pd
from datetime import datetime

df = pd.read_csv('outputs/annotations/v3/time_series_1h.csv')

print('=== 基本统计 ===')
print(f'时间跨度: {df["time_window"].min()} ~ {df["time_window"].max()}')
print(f'总窗口数: {len(df)}')
print(f'总帖子数: {df["n_posts"].sum()}')
print()

print('=== 数据稀疏性分析 ===')
start = datetime.strptime(df['time_window'].min(), '%Y-%m-%d %H:%M:%S')
end = datetime.strptime(df['time_window'].max(), '%Y-%m-%d %H:%M:%S')
total_hours = (end - start).total_seconds() / 3600
print(f'理论窗口数(连续1h): {int(total_hours)} 小时')
print(f'实际有数据窗口数: {len(df)}')
print(f'数据覆盖率: {len(df)/total_hours*100:.2f}%')
print()

print('=== n_posts 分布 ===')
print(df['n_posts'].describe())
print()

print('=== 按帖子数分布 ===')
bins = [0, 1, 2, 3, 5, 10, 20, 50, 100, 1000]
df['posts_bin'] = pd.cut(df['n_posts'], bins=bins, right=False)
print(df['posts_bin'].value_counts().sort_index())
print()

print('=== a 非空窗口分析 ===')
valid_a = df[df['a'].notna()]
print(f'a 非空窗口: {len(valid_a)}')
if len(valid_a) > 0:
    print(f'a 非空窗口平均 n_public: {valid_a["n_public"].mean():.1f}')
print()

print('=== 帖子活跃时段分析 ===')
# 按年月统计帖子分布
df['time'] = pd.to_datetime(df['time_window'])
df['year_month'] = df['time'].dt.to_period('M')
monthly = df.groupby('year_month')['n_posts'].sum()
print('每月帖子数:')
print(monthly[monthly > 50])  # 只显示>50帖的月份
print()

print('=== 活跃期分析 ===')
# 找出帖子最集中的时段
df_sorted = df.sort_values('n_posts', ascending=False)
print('帖子最多的10个窗口:')
print(df_sorted[['time_window', 'n_posts', 'n_public', 'n_wemedia', 'n_mainstream']].head(10))


