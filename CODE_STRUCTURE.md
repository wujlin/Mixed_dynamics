# Code Structure & Architecture

**项目**: Mixed Feedback Dynamics in Collective Emotion  
**更新时间**: 2026-01-16

---

## 📐 整体架构

```
emotion_dynamics/
├── src/                    # 核心库（可复用模块）
├── scripts/                # 数据处理脚本（独立运行）
├── notebooks/              # 分析笔记本 + 图表生成
├── tests/                  # 单元测试
├── data/                   # 处理后的数据
├── outputs/                # 模拟结果 + 缓存
└── docs/                   # 文档
```

**设计原则**:
1. **模块化**: `src/` 中的核心功能独立、可复用
2. **可测试**: 所有核心模块有对应的单元测试
3. **可复现**: 图表生成脚本可独立运行
4. **版本控制**: 敏感数据和论文草稿不入库（见 `.gitignore`）

---

## 🔬 核心库 (`src/`)

### 1. `theory.py` - 理论框架

**功能**: Mean-field 分析、临界点计算、有效势

**核心类**:

#### `MixedFeedbackModel`
```python
class MixedFeedbackModel:
    """
    混合反馈模型的主类
    
    Parameters
    ----------
    phi : float
        高唤醒阈值（感知风险 >= phi → High 状态）
    theta : float
        低唤醒阈值（感知风险 <= theta → Low 状态）
    k : int
        信息密度（每次更新采样的信号数）
    n_m : float
        主流媒体通道强度
    n_w : float
        用户生成通道强度
    """
    
    def compute_critical_point(self, regime='symmetric') -> float:
        """计算临界点 r_c"""
        
    def compute_chi(self) -> float:
        """计算心理敏感度 χ"""
        
    def steady_state_q(self, r, regime='symmetric') -> tuple:
        """计算稳态极化方向 q*"""
        
    def effective_potential(self, q, r) -> float:
        """计算有效势 V_eff(q)"""
        
    def feedback_gradient(self, r, regime='symmetric') -> float:
        """计算反馈梯度 Γ = ∂p_env/∂q|_{q=0}"""
```

**关键函数**:
- `p_main(q)`: 主流媒体通道信号
- `p_we_sym(q)`: 用户生成通道信号（对称版本）
- `p_we_asym(q, a)`: 用户生成通道信号（活动耦合版本）
- `p_env(q, a, r)`: 混合环境信号

**用途**: 
- 论文 Fig 1a (bifurcation diagram)
- 论文 Fig 1b (effective potential)
- 论文 Fig 4 (parameter landscape)

---

### 2. `sde_solver.py` - 随机微分方程求解器

**功能**: 时间演化、弛豫时间估计、临界慢化

**核心函数**:

#### `solve_sde()`
```python
def solve_sde(
    model: MixedFeedbackModel,
    r: float,
    T: float = 1000.0,
    dt: float = 0.01,
    q0: float = 0.01,
    a0: float = 0.1,
    noise_strength: float = 0.1,
    regime: str = 'symmetric'
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    求解 SDE: dq = f(q,a) dt + σ dW
    
    Returns
    -------
    t : np.ndarray
        时间序列
    q : np.ndarray
        极化方向轨迹 q(t)
    a : np.ndarray
        活动轨迹 a(t)
    """
```

#### `estimate_relaxation_time()`
```python
def estimate_relaxation_time(
    model: MixedFeedbackModel,
    r: float,
    method: str = 'autocorrelation'
) -> float:
    """
    估计弛豫时间 τ
    
    Parameters
    ----------
    method : str
        'autocorrelation' - 从自相关函数 e^{-t/τ} 拟合
        'eigenvalue' - 从线性化系统的特征值
    """
```

**用途**:
- 论文 Fig 1c (symmetric vs asymmetric dynamics)
- 论文 Fig 3a (critical slowing down scaling)
- 论文 Fig 3c (time series near criticality)

---

### 3. `network_sim.py` - Agent-based 网络模拟

**功能**: 网络上的微观动力学、有限尺寸效应

**核心类**:

#### `NetworkSimulation`
```python
class NetworkSimulation:
    """
    基于网络的 Agent-based 模拟
    
    Parameters
    ----------
    graph : nx.Graph
        网络拓扑（ER, BA, 等）
    model : MixedFeedbackModel
        理论模型参数
    r : float
        通道平衡参数
    beta : float
        局部耦合强度（邻居影响）
    update_fraction : float
        异步更新比例（每步更新的节点比例）
    """
    
    def initialize_states(self, H_frac=0.1, L_frac=0.1):
        """初始化节点状态（H, M, L）"""
        
    def step(self):
        """执行一步异步更新"""
        
    def run(self, n_sweeps=100):
        """运行 n_sweeps 轮（1 sweep = N 次更新）"""
        
    def get_polarization(self) -> float:
        """计算宏观极化 Q = (N_H - N_L) / N"""
        
    def get_activity(self) -> float:
        """计算宏观活动 a = (N_H + N_L) / N"""
```

**关键方法**:
- `_compute_local_p(node)`: 计算节点的感知风险概率（全局+局部）
- `_update_node(node)`: 根据阈值规则更新单个节点
- `compute_binder_cumulant()`: 计算 Binder cumulant $U_4$ (有限尺寸标度诊断)

**用途**:
- 论文 Fig 2a (network validation)
- 论文 Fig 2b (Binder cumulant)
- 论文 Fig 2c (activity dynamics)
- 论文 Fig 4d (beta effect)

---

### 4. `plot_style.py` - 统一绘图风格

**功能**: 论文图表的统一样式、字体、颜色

**核心对象**:

#### `PaperStyle` (dataclass)
```python
@dataclass
class PaperStyle:
    """论文图表样式配置"""
    font_size: float = 11.0
    axes_labelsize: float = 12.0
    tick_labelsize: float = 10.0
    axes_titlesize: float = 11.0
    linewidth: float = 1.5
    markersize: float = 6.0
    color_theory: str = '#1f77b4'      # 理论曲线
    color_abm: str = '#ff7f0e'         # ABM 模拟
    color_empirical: str = '#2ca02c'   # 经验数据
```

#### `paper_rcparams()`
```python
def paper_rcparams(style: PaperStyle = None) -> dict:
    """生成 matplotlib rcParams 字典"""
```

#### `add_panel_label()`
```python
def add_panel_label(ax, label, x=-0.12, y=1.05):
    """在子图外侧添加面板标签 (a), (b), (c), ..."""
```

**用途**: 所有 `make_fig*.py` 脚本

---

### 5. `utils.py` - 工具函数

**功能**: 数学运算、数据处理、文件 I/O

**核心函数**:
- `binomial_cdf(n, k, p)`: 二项分布累积分布函数（用于阈值计算）
- `moving_average(x, window)`: 移动平均（平滑时间序列）
- `bootstrap_ci(data, statistic, n_boot=1000, alpha=0.05)`: Bootstrap 置信区间
- `save_npz(filepath, **arrays)`: 保存模拟结果
- `load_npz(filepath)`: 加载模拟结果

---

### 6. `empirical/` - 经验分析模块

#### `data_loader.py`
```python
def load_weibo_corpus(batch='batch3') -> pd.DataFrame:
    """
    加载 Weibo 数据集
    
    Returns
    -------
    df : pd.DataFrame
        Columns: ['timestamp', 'arousal', 'risk', 'media_type', ...]
        （不包含用户 ID 或原始文本）
    """

def load_timeseries(aggregation='4H') -> pd.DataFrame:
    """
    加载聚合时间序列
    
    Returns
    -------
    df : pd.DataFrame
        Columns: ['window_start', 'X_H', 'X_M', 'X_L', 
                  'n_mainstream', 'n_wemedia', 'n_gov']
    """
```

#### `proxies.py`
```python
def compute_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算操作代理变量
    
    Adds columns:
    - Q: 极化 (X_H - X_L)
    - a: 活动 (X_H + X_L)
    - r_proxy: 媒体组成
    """

def compute_segment_statistics(df: pd.DataFrame, freq='7D') -> pd.DataFrame:
    """
    计算时段级统计量
    
    Adds columns:
    - volatility: std(Q)
    - jump_q95: 95th percentile of |dQ/dt|
    - mean_a: 平均活动
    - sampling_density: 有效窗口比例
    """
```

#### `statistics.py`
```python
def test_activity_jump_correlation(segments: pd.DataFrame) -> dict:
    """
    测试 H1: 活动-跳跃耦合
    
    Returns
    -------
    result : dict
        {'pearson_r': ..., 'p_value': ..., 'n': ...}
    """

def test_media_volatility_correlation(segments: pd.DataFrame) -> dict:
    """
    测试 H2: 媒体组成-波动耦合
    
    Returns
    -------
    result : dict
        {'pearson_r': ..., 'p_value': ..., 'partial_r': ..., 
         'partial_p': ..., 'n': ...}
    """
```

#### `validation.py`
```python
def validate_llm_annotation(
    manual_ratings: pd.DataFrame,
    llm_annotations: pd.DataFrame
) -> dict:
    """
    验证 LLM 标注质量
    
    Returns
    -------
    metrics : dict
        {'accuracy': ..., 'cohen_kappa': ..., 
         'confusion_matrix': ...}
    """
```

---

## 📊 数据处理流程 (`scripts/`)

### 阶段 1: 数据收集
- `01_scrape_weibo.py`: 使用 Weibo API 收集帖子
  - 输入: `data/config/keywords.json` (关键词)
  - 输出: `dataset/raw/weibo_raw_{date}.json`
  - **不公开**（包含用户 ID）

### 阶段 2: 数据清洗
- `02_clean_data.py`: 去重、过滤、标准化
  - 输入: `dataset/raw/*.json`
  - 输出: `dataset/processed/weibo_clean.csv`
  - **不公开**（仍包含用户信息）

### 阶段 3: LLM 标注
- `03_llm_annotation.py`: 使用 Qwen3-8B 标注唤醒和风险
  - 输入: `dataset/processed/weibo_clean.csv`
  - 输出: `dataset/processed/weibo_annotated.csv`
  - **不公开**（包含原始文本）

### 阶段 4: 构建时间序列
- `04_build_timeseries.py`: 聚合为窗口级代理
  - 输入: `dataset/processed/weibo_annotated.csv`
  - 输出: `data/derived/timeseries_4h.csv`
  - **公开**（仅聚合统计量，无用户信息）

### 阶段 5: 时段统计
- `05_segment_statistics.py`: 计算周级统计量
  - 输入: `data/derived/timeseries_4h.csv`
  - 输出: `data/derived/segments_pooled.csv`
  - **公开**

### 阶段 6: 假设检验
- `06_hypothesis_testing.py`: 运行 H1, H2 测试
  - 输入: `data/derived/segments_*.csv`
  - 输出: `outputs/hypothesis_results.json`
  - **公开**

---

## 📓 分析笔记本 (`notebooks/`)

### 理论与模拟
1. **`01_Theory_and_Potential.ipynb`**
   - Mean-field 分叉分析
   - 有效势计算
   - 生成 Fig 1a, 1b

2. **`02_Network_Topology.ipynb`**
   - ER / BA 网络模拟
   - 有限尺寸标度（Binder cumulant）
   - 生成 Fig 2a, 2b, 2c

3. **`03_Critical_Slowing_Down.ipynb`**
   - 弛豫时间估计
   - 自相关函数分析
   - 生成 Fig 3a, 3b, 3c

4. **`04_Sensitivity_Chi_Landscape.ipynb`**
   - 参数扫描（φ, θ, k, n_w/n_m, β）
   - 敏感度分析
   - 生成 Fig 4a, 4b, 4c, 4d

### 经验验证
5. **`05_Annotation_Pipeline.ipynb`**
   - LLM 标注流程
   - 人工验证（n=5000）
   - 生成 Supplementary Fig S1

6. **`07_Empirical_Validation.ipynb`**
   - H1 / H2 假设检验
   - 散点图、时间序列示例
   - 生成 Fig 5a, 5b, 5c, 5d

### 图表生成
7. **`99_Paper_Figures.ipynb`**
   - 汇总所有图表生成代码
   - 检查一致性

### 独立脚本（可批量运行）
- `make_fig*.py`: 每个主图对应一个脚本
- `make_supp_*.py`: 补充材料图表
- `regenerate_all_figures.sh`: 批量重新生成

**命名规范**:
- `make_fig1a_*.py` → `outputs/figs/fig1a_*.pdf`
- `make_supp_s2_*.py` → `outputs/figs/fig_supp_*.pdf`

---

## 🧪 单元测试 (`tests/`)

### 测试覆盖

| 模块 | 测试文件 | 覆盖的功能 |
|---|---|---|
| `src/theory.py` | `test_theory.py` | 临界点计算、有效势、稳态求解 |
| `src/sde_solver.py` | `test_sde_solver.py` | SDE 数值积分、弛豫时间估计 |
| `src/network_sim.py` | `test_network_sim.py` | ABM 动力学、Binder cumulant |
| `src/utils.py` | `test_utils.py` | 数学函数、Bootstrap |

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_theory.py::test_critical_point -v

# 检查覆盖率
pytest tests/ --cov=src --cov-report=html
```

### 测试原则
1. **确定性**: 固定随机种子 (`np.random.seed(42)`)
2. **快速**: 单个测试 < 1 秒
3. **独立**: 测试间无依赖
4. **覆盖**: 关键路径 + 边界情况

---

## 📁 数据文件结构

### `data/derived/` (公开)
```
derived/
├── timeseries_4h.csv         # 4 小时聚合时间序列
├── timeseries_12h.csv        # 12 小时聚合（鲁棒性检查）
├── segments_pooled.csv       # 混合数据集时段统计
└── segments_high_density.csv # 高密度子集时段统计
```

**列定义** (`timeseries_4h.csv`):
| 列名 | 类型 | 说明 |
|---|---|---|
| `window_start` | datetime | 窗口起始时间 |
| `X_H` | float | High 唤醒比例 |
| `X_M` | float | Medium 唤醒比例 |
| `X_L` | float | Low 唤醒比例 |
| `n_mainstream` | int | 主流媒体帖子数 |
| `n_wemedia` | int | 自媒体帖子数 |
| `n_gov` | int | 政府帖子数 |
| `n_public` | int | 公众用户帖子数（用于计算 X_H, X_M, X_L） |
| `Q` | float | 极化代理 (X_H - X_L) |
| `a` | float | 活动代理 (X_H + X_L) |
| `r_proxy` | float | 媒体组成代理 |

**列定义** (`segments_pooled.csv`):
| 列名 | 类型 | 说明 |
|---|---|---|
| `segment_id` | str | 时段标识符 (e.g., "2020-03-01_2020-03-07") |
| `mean_a` | float | 平均活动 |
| `volatility` | float | 极化波动 std(Q) |
| `jump_q95` | float | 跳跃强度（95 分位数） |
| `mean_r_proxy` | float | 平均媒体组成 |
| `sampling_density` | float | 有效窗口比例 |
| `n_windows` | int | 窗口总数 |

### `outputs/data/` (模拟结果缓存)
```
data/
├── bifurcation_symmetric.npz    # 对称分叉图
├── bifurcation_asymmetric.npz   # 非对称分叉图
├── network_er_N1000.npz         # ER 网络结果
├── network_ba_N1000.npz         # BA 网络结果
├── param_scan_chi.npz           # χ 扫描
├── param_scan_media_ratio.npz   # n_w/n_m 扫描
└── ...
```

**`.npz` 文件结构**（示例）:
```python
data = np.load('bifurcation_symmetric.npz')
# Keys: ['r_values', 'q_steady', 'a_steady', 'params']
```

---

## 🔄 工作流程

### 典型开发流程

```mermaid
graph LR
    A[修改核心模块] --> B[运行单元测试]
    B --> C{测试通过?}
    C -->|否| A
    C -->|是| D[更新笔记本]
    D --> E[重新生成图表]
    E --> F[检查视觉一致性]
    F --> G[提交代码]
```

### 图表更新流程

```bash
# 修改 src/plot_style.py 的全局样式
# → 重新生成所有图表
cd notebooks
bash regenerate_all_figures.sh

# 检查输出
ls -lh outputs/figs/fig*.pdf
```

### 添加新实验

1. **在 `src/` 中实现核心逻辑**
   ```python
   # src/theory.py
   def new_feature(...):
       """New theoretical calculation"""
       ...
   ```

2. **添加单元测试**
   ```python
   # tests/test_theory.py
   def test_new_feature():
       result = new_feature(...)
       assert result == expected_value
   ```

3. **在笔记本中分析**
   ```python
   # notebooks/08_New_Analysis.ipynb
   from src.theory import new_feature
   # ... 分析代码
   ```

4. **生成图表**
   ```python
   # notebooks/make_fig_new.py
   # ... 图表生成代码
   ```

---

## 🔗 模块依赖关系

```
           theory.py
               ↓
          sde_solver.py
               ↓
         network_sim.py
               ↓
          plot_style.py
               ↓
         empirical/ (独立)
               ↓
            notebooks/
```

**依赖说明**:
- `theory.py`: 无内部依赖（纯数学）
- `sde_solver.py`: 依赖 `theory.py`（调用模型参数）
- `network_sim.py`: 依赖 `theory.py`（共享阈值逻辑）
- `plot_style.py`: 无内部依赖（纯样式）
- `empirical/`: 独立模块（不依赖理论模块）

**外部依赖**（见 `requirements.txt`）:
- 核心: `numpy`, `scipy`, `networkx`
- 可视化: `matplotlib`, `seaborn`
- 数据: `pandas`, `statsmodels`
- LLM: `transformers`, `torch`（可选，仅标注流程）

---

## 📝 代码规范

### Docstring 格式（NumPy 风格）

```python
def function_name(param1, param2):
    """
    简短描述（一行）
    
    详细描述（多行，可选）
    
    Parameters
    ----------
    param1 : type
        参数说明
    param2 : type, optional
        可选参数说明（default: value）
    
    Returns
    -------
    result : type
        返回值说明
    
    Examples
    --------
    >>> function_name(1, 2)
    3
    
    Notes
    -----
    额外说明（数学公式、引用等）
    """
    ...
```

### 命名规范

| 类型 | 规范 | 示例 |
|---|---|---|
| 变量/函数 | snake_case | `critical_point`, `compute_chi()` |
| 类 | PascalCase | `MixedFeedbackModel` |
| 常量 | UPPER_CASE | `DEFAULT_PHI` |
| 私有 | 前缀 `_` | `_internal_function()` |

### 类型提示（Python 3.9+）

```python
def compute_polarization(
    x_h: float,
    x_l: float
) -> float:
    """计算极化 Q = x_h - x_l"""
    return x_h - x_l
```

---

## 🛠️ 常见任务

### 任务 1: 修改模型参数默认值

**位置**: `src/theory.py` → `MixedFeedbackModel.__init__()`

```python
# 修改默认阈值
phi_default = 0.7  # 原值
phi_default = 0.75 # 新值
```

**影响**: 所有使用默认参数的模拟

**需要重新运行**: 
- `make_fig1a_bifurcation.py`
- `make_fig4a_rc_landscape.py`

### 任务 2: 添加新的绘图颜色

**位置**: `src/plot_style.py` → `PaperStyle`

```python
@dataclass
class PaperStyle:
    ...
    color_new: str = '#d62728'  # 新颜色
```

**使用**:
```python
from src.plot_style import paper_rcparams, PaperStyle
style = PaperStyle()
plt.plot(x, y, color=style.color_new)
```

### 任务 3: 修改图表尺寸

**位置**: 各个 `make_fig*.py` 脚本

```python
# 修改前
fig, ax = plt.subplots(figsize=(6, 4))

# 修改后（更大）
fig, ax = plt.subplots(figsize=(8, 5))
```

**注意**: 修改后需要重新检查标签位置和字体大小

### 任务 4: 添加新的经验分析

**步骤**:
1. 在 `src/empirical/` 添加新函数
2. 在 `notebooks/07_Empirical_Validation.ipynb` 中调用
3. 创建新的 `make_fig*.py` 脚本
4. 更新 `regenerate_all_figures.sh`

---

## 🐛 调试技巧

### 1. 模拟不收敛

**检查**:
- `r` 是否在合理范围 [0, 1]
- 初始条件 `q0`, `a0` 是否过大
- 时间步长 `dt` 是否过大（建议 < 0.01）

**诊断**:
```python
# 绘制轨迹检查
t, q, a = solve_sde(model, r=0.8, T=1000)
plt.plot(t, q, label='q(t)')
plt.plot(t, a, label='a(t)')
plt.legend()
plt.show()
```

### 2. 网络模拟结果不稳定

**检查**:
- 网络尺寸是否足够大（建议 N ≥ 1000）
- 是否运行了足够长时间（建议 ≥ 100 sweeps）
- 多次运行取平均（建议 ≥ 50 seeds）

**诊断**:
```python
# 检查多次运行的方差
results = [run_simulation(seed=i) for i in range(50)]
Q_values = [r['Q'] for r in results]
print(f"Mean: {np.mean(Q_values):.3f}, Std: {np.std(Q_values):.3f}")
```

### 3. 图表标签重叠

**检查**:
- `add_panel_label()` 的 `x`, `y` 坐标
- 子图的 `left`, `right`, `bottom`, `top` margin

**修复**:
```python
# 调整面板标签位置
add_panel_label(ax, 'a', x=-0.15, y=1.08)  # 更靠左上

# 调整子图边距
fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.92)
```

---

## 📚 进一步阅读

- **理论背景**: `docs/theory_validation_report_note01-04.md`
- **经验验证**: `docs/note07_empirical_validation_report.md`
- **数据集描述**: `docs/dataset_description.md`
- **可视化指南**: `docs/visual_style_guide.md`
- **LLM 标注**: `Essay/supplementary_materials_s1_llm_annotation.md`

---

## 🔄 更新日志

| 版本 | 日期 | 变更 |
|---|---|---|
| 1.0 | 2026-01-16 | 初始版本 |

---

**维护者**: Jinlin Wu, Zhihang Liu  
**最后更新**: 2026-01-16

