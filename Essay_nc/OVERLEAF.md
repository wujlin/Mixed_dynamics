# Overleaf 编译与设置（Nature Communications：主文 + Supplementary）

目标：主文（`main_manuscript.tex`）与补充材料（`supplementary.tex`）**分开提交/分开编译**，但主文里仍能正确显示 `Supplementary Fig.~S#`（不出现 `??`）。

> 建议：在 Overleaf 用同一个项目，切换 **Main document** 分别编译主文与 SI；最终下载两份 PDF 作为投稿材料。

---

## 1) 推荐的 Overleaf 项目结构（最省事）

在 Overleaf 新建项目后，把本仓库 `Essay_nc/` 目录下的内容**直接上传到项目根目录**（不要再套一层 `Essay_nc/` 文件夹），包括：

- `main_manuscript.tex`
- `supplementary.tex`
- `supplementary.aux`（关键：用于主文 cross-ref）
- `references.bib`
- `figures/`（主文图）
- `figures_supp/`（补充图）

然后：
- Overleaf → **Menu → Main document** 选择 `main_manuscript.tex`（编译主文）
- 需要编译补充材料时，把 Main document 改为 `supplementary.tex`

---

## 2) 为什么主文会出现 “Supplementary Fig. ??”

Overleaf **不会把** `supplementary.tex` 编译生成的构建产物（例如 `.aux`）**共享给** `main_manuscript.tex` 的编译过程。

主文要引用补充材料的 `\label{fig:supp:...}`，必须让主文编译时“读到”一个可用的 `supplementary.aux`（作为源文件上传进项目）。

主文已通过 `xr-hyper` 做跨文档引用：只要项目里有 `supplementary.aux`，`Supplementary Fig.~\ref{fig:supp:...}` 就能解析出 `S#`。

---

## 3) 更新 Supplementary 后如何同步 `supplementary.aux`

只要补充材料里新增/删除 figure（会影响 `S#` 编号），就必须更新 `supplementary.aux`，否则主文引用会错或变 `??`。

### 方案 A（本地更新，推荐）

在仓库根目录运行：

```bash
python scripts/generate_supplementary_aux.py \
  --supp-tex Essay_nc/supplementary.tex \
  --out-aux Essay_nc/supplementary.aux
```

然后把生成的 `Essay_nc/supplementary.aux` 上传/同步到 Overleaf 项目根目录（文件名保持为 `supplementary.aux`）。

### 方案 B（Overleaf 内更新，无需本地 Python）

1. 把 Overleaf 的 Main document 设为 `supplementary.tex` 并编译一次。
2. 打开 **Logs and output files**，下载生成的 `supplementary.aux`。
3. 把下载的 `supplementary.aux` 作为“源文件”重新上传到项目里（覆盖旧的）。
4. 再把 Main document 切回 `main_manuscript.tex` 编译主文。

---

## 4) 编译排错清单（主文仍出现 ?? 时）

- 项目里必须存在 `supplementary.aux`（源文件，不是输出文件）。
- `supplementary.tex` 的 `\label{fig:supp:...}` 与主文 `\ref{...}` 必须完全一致（大小写/冒号一致）。
- Overleaf 点一次 **Recompile from scratch**，并至少编译两次（交叉引用需要 2-pass）。
