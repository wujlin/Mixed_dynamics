# Overleaf 编译与设置（主文 + Supplementary + Slides）

目标：主文（`essay.tex`）与补充材料（`supplementary.tex`）**分开提交/分开编译**，但主文里仍能正确显示 `Supplementary Fig.~S#` 的编号（不再出现 `??`）。

## 推荐的 Overleaf 项目结构（最省事）

在 Overleaf 新建一个项目后，把本仓库 `Essay/` 目录下的以下内容**直接上传到项目根目录**（不要额外套一层 `Essay/` 文件夹）：

- `essay.tex`
- `supplementary.tex`
- `supplementary.aux`（关键：用于主文 cross-ref）
- `references.bib`
- `figures/`（来自本仓库 `Essay/figures/`）
- `figures_supp/`（来自本仓库 `Essay/figures_supp/`）

然后：
- Overleaf → **Menu → Main document** 选择 `essay.tex`（编译主文）
- 需要编译补充材料时，把 Main document 改为 `supplementary.tex`

## 为什么主文会出现 “Supplementary Fig. ??”

Overleaf **不会把** `supplementary.tex` 编译生成的构建产物（例如 `.aux`）**共享给** `essay.tex` 的编译过程。

主文要引用补充材料的 `\label{fig:supp:...}`，必须让 `essay.tex` 在编译时“读到”一个可用的 `supplementary.aux`（作为源文件上传进项目）。

`essay.tex` 已通过 `xr-hyper` 做了跨文档引用：
- `Essay/essay.tex` 会读取 `supplementary.aux`（或 `Essay/supplementary.aux`，取决于你在 Overleaf 的目录结构）。

## 更新 Supplementary 后，如何同步 `supplementary.aux`

你只要新增/删除了补充材料里的 figure（影响 `S#` 编号），就必须更新 `supplementary.aux`，否则主文引用会错/会变 `??`。

### 方案 A（本地更新，推荐）

在仓库根目录运行：

```bash
python scripts/generate_supplementary_aux.py
```

然后把生成的 `Essay/supplementary.aux` 上传/同步到 Overleaf（如果你把文件放到 Overleaf 根目录，则上传后的文件名仍应为 `supplementary.aux`）。

### 方案 B（Overleaf 内更新，无需本地 Python）

1. 先把 Overleaf 的 Main document 设为 `supplementary.tex` 并编译一次。
2. 打开 **Logs and output files**，下载生成的 `supplementary.aux`。
3. 把下载的 `supplementary.aux` 作为“源文件”重新上传到项目里（覆盖旧的）。
4. 再把 Main document 切回 `essay.tex` 编译主文。

## 编译排错清单（主文仍出现 ?? 时）

- 确认项目里存在 `supplementary.aux`（源文件，不是输出文件）。
- 确认 `supplementary.tex` 里的 `\label{fig:supp:...}` 与主文 `\ref{fig:supp:...}` 完全一致（大小写/冒号都要一致）。
- Overleaf 点一次 **Recompile from scratch**，并至少编译两次（交叉引用需要 2-pass）。

## Slides（Beamer）

幻灯片在 `Essay/slides/`，Overleaf 使用说明见：
- `Essay/slides/README.md`

