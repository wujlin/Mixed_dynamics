# Batch4（上海 2022）UID 用户元信息：最终分析摘要（给 PI）

更新时间：2025-12-21

> 背景：Batch4（上海疫情期间 2022-01~05）在 Note07 经验验证中存在 `r_proxy` 近乎饱和、H2/H3 难以检验的问题。我们完成了 batch4 用户 UID 的补抓，并用离线脚本统一认证口径与 `user_type` 映射，给出最终可复现的统计结论。

---

## 1. 数据源与口径

**数据源**

- UID 抓取输出（原始）：`data/derived/user_meta_batch4.csv`
- UID 口径修正（最终用于分析）：`data/derived/user_meta_batch4_fixed.csv`
  - 由脚本离线生成：`python3 scripts/fix_user_meta_csv.py --input data/derived/user_meta_batch4.csv --output data/derived/user_meta_batch4_fixed.csv`
- Batch4 posts 元数据（用于覆盖率/生态构成）：`outputs/annotations/intermediate/to_annotate_batch4_shanghai_2022_loose.csv`
  - 行数：32083（与此前经验分析一致）
  - 该文件含 `publish_time/user_link/weibo_link`，可解析 UID 并聚合到 4H 时间窗。

**认证与用户类型映射口径（离线修正脚本）**

- `verified_type` → `verify_typ`：
  - `0` → 黄V认证
  - `1..7` → 蓝V认证
  - 其它/缺失 → 无认证
- `verify_typ + user_name + official_list(uid)` → `user_type`：
  - 黄V认证 → `wemedia`（自媒体/大V）
  - 蓝V认证 → 依据关键词/白名单进一步划分：`mainstream / government / other`
  - 无认证 → `public`
  - 官媒/政府白名单（uid）：`data/config/official_media_list.txt`（uid 精确匹配，优先生效）

---

## 2. UID 抓取结果（覆盖率与失败原因）

来自 `scripts/fix_user_meta_csv.py` 的统计输出（对最新 `user_meta_batch4.csv` 修正后）：

- UID 总行数：13017
- 抓取成功（error 为空）：12597（96.77%）
- 抓取失败：420（3.23%）
  - 失败类型：`api_not_ok: ok=0 msg=这里还没有内容`
  - 含义：接口返回“无内容/不可访问”，在本项目中统一视为元信息缺失。

---

## 3. UID 级别用户类型分布（仅成功行，n=12597）

| user_type | 计数 | 占比 |
|---|---:|---:|
| public | 11540 | 91.61% |
| wemedia | 892 | 7.08% |
| mainstream | 32 | 0.25% |
| government | 9 | 0.07% |
| other（蓝V非媒体/政府） | 124 | 0.98% |

对应的认证类型分布（成功行）：

- 黄V认证：892（全部映射为 `wemedia`）
- 蓝V认证：164（映射为 `mainstream=32 / government=9 / other=123`，另有 1 个 uid 白名单覆盖导致 `mainstream` 但 verify_typ 非蓝V）
- 无认证：11541（几乎全部映射为 `public`）

**解释（对研究主线的意义）**

- Batch4 的 UID 生态“结构性”上以 `public` 为主，且官方叙事账号（mainstream+government）占比极低（<0.4%）。
- 因此 `r_proxy` 在时间窗上很容易出现“官方分母很小 → r_proxy 接近 1”的饱和结构，这不是窗口伪影，而更像数据生态本身导致的可辨识性不足。

---

## 4. 帖子级别覆盖率与生态构成（n=32083）

对 `to_annotate_batch4_shanghai_2022_loose.csv` 的每条微博，通过 `user_link` 或 `weibo_link` 解析 UID（两者互补，最终缺失 UID=0），再与 `user_meta_batch4_fixed.csv` 连接得到：

### 4.1 帖子按 user_type 分布

| user_type | 帖子数 | 占比 |
|---|---:|---:|
| public | 17751 | 55.33% |
| wemedia | 2337 | 7.28% |
| mainstream | 50 | 0.16% |
| government | 28 | 0.09% |
| other | 409 | 1.27% |
| unknown（抓取失败 uid，对应 error 行） | 712 | 2.22% |
| missing_meta（UID 未抓到元信息） | 10796 | 33.65% |

### 4.2 UID 覆盖率

- Batch4 posts 唯一 UID：19668
- 已覆盖到 `user_meta` 的 UID：12437（63.23%）
- 仍缺失元信息 UID：7231（36.77%）

**对缺失 UID 的“官方账号风险”快速诊断**

- 在 7231 个缺失 UID 中，仅发现 2 个用户名命中主流媒体关键词（如“新闻”），未发现明显政府关键词高频聚集。
- 这意味着：即便补齐剩余 UID，官方账号占比也不太可能大幅上升，`r_proxy` 的高饱和趋势大概率不变。

---

## 5. r_proxy 的最终可辨识性（按 4H 时间窗）

按 4H 时间窗聚合统计（仅基于媒体类帖子：`wemedia/mainstream/government`），计算：

$$
r_{proxy} = \\frac{n_{wemedia}}{n_{wemedia}+n_{mainstream}+n_{government}}
$$

结果（denom>0 的时间窗）：

- 可计算时间窗数：469
- mean = 0.9681
- median = 1.0000
- std = 0.1157
- p10 = 0.9203
- p90 = 1.0000
- pct(r_proxy==1) = 86.78%

**结论**

- Batch4 的 `r_proxy` 在时间上高度饱和（大量时间窗=1），自变量方差很小。
- 这会直接削弱 H2/H3 的统计可辨识性（即使模型机制真实存在，经验侧也很难检出）。
- 该结论与之前在 Note07 中观察到的“batch4 r_proxy 接近 1、H2/H3 难以评估”一致，说明问题更偏**数据生态结构**而非实现错误。

---

## 6. 官方/自媒体账号在 batch4 的贡献（按帖子数 Top）

### 6.1 mainstream（Top10）

- 1845864154,看看新闻Knews,8
- 1314608344,新闻晨报,4
- 2539961154,上海发布,2
- 1649173367,每日经济新闻,2
- 1737737970,新民晚报新民网,2
- 1852893050,中国教育电视台,2
- 2606218210,中国妇女报,2
- 1681029540,新民周刊,2
- 1918021250,东方网,2
- 2541763545,上海静安,2

### 6.2 government（Top9，全部）

- 2620648747,上海松江发布,9
- 2175830437,上海徐汇发布,9
- 2557375422,浦东发布,3
- 2643346122,上海奉贤发布,2
- 2652595170,上海司法行政发布,1
- 2568435544,宁波发布,1
- 2177412401,北海发布,1
- 2403912521,天衢公安,1
- 1770545650,柳州公安,1

### 6.3 wemedia（Top10）

- 1156927893,难般聊聊,172
- 2346780085,懂上海,91
- 1685820442,期货刘洋,75
- 1770074717,饭饭,74
- 2385754614,我是律师的朋友,55
- 1777937442,clcc豪车老虎,54
- 3221972074,JXSoung,41
- 2715653023,章克俭律师,38
- 1772440677,陶侃疫苗,37
- 1014792675,枪迷平原君,30

**解释**

- 官方账号在 batch4 的出现频率极低（mainstream+government 合计仅 78 条帖子），而 `wemedia` 在媒体池中占绝对多数，从而把 `r_proxy` 推向 1 的饱和值。

---

## 7. 复现命令

1) 生成修正后的 user_meta：

```bash
python3 scripts/fix_user_meta_csv.py \
  --input data/derived/user_meta_batch4.csv \
  --output data/derived/user_meta_batch4_fixed.csv
```

2) 输出一份可直接贴给 PI 的控制台摘要（可选写入 md）：

```bash
python3 scripts/analyze_batch4_user_meta.py \
  --user-meta data/derived/user_meta_batch4_fixed.csv \
  --posts-csv outputs/annotations/intermediate/to_annotate_batch4_shanghai_2022_loose.csv \
  --freq 4H \
  --output docs/batch4_user_meta_analysis.md
```

