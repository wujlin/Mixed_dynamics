# GitHub 仓库整理 - 快速参考卡

## 🚀 快速开始

### 1. 验证敏感文件保护

```bash
# Linux/Mac
bash scripts/repo_hygiene/verify_gitignore.sh

# Windows
powershell -ExecutionPolicy Bypass -File .\scripts\repo_hygiene\verify_gitignore.ps1
```

### 2. 查看将要提交的文件

```bash
# 查看暂存区文件（将要提交的）
git status

# 查看被忽略的文件（不会提交的）
git status --ignored
```

### 3. 提交并推送

```bash
# 添加修改
git add .gitignore README.md docs/architecture/CODE_STRUCTURE.md

# 提交
git commit -m "docs: add comprehensive README and protect sensitive data"

# 推送
git push origin main
```

---

## ⚠️ 必须忽略的文件

| 类别 | 文件/文件夹 | 原因 |
|---|---|---|
| **论文** | `Essay/`, `Essay_nc/` | 版权保护 |
| **数据** | `dataset/`, `data/raw/` | 包含用户信息 |
| **秘密** | `secrets/`, `cookies*.json` | 敏感凭证 |
| **开发** | `AGENTS.md`, `docs/internal/DEVELOPMENT.md` | 内部文档 |
| **输出** | `outputs/figs/`, `outputs/annotations/` | 可重新生成 |

---

## ✅ 必须公开的文件

| 类别 | 文件/文件夹 | 说明 |
|---|---|---|
| **代码** | `src/`, `scripts/`, `notebooks/` | 核心逻辑 |
| **测试** | `tests/` | 单元测试 |
| **文档** | `README.md`, `docs/architecture/CODE_STRUCTURE.md` | 项目说明 |
| **数据** | `data/derived/` | 去标识化数据 |
| **配置** | `requirements.txt`, `.gitignore` | 环境配置 |

---

## 🔍 常用检查命令

### 检查特定文件是否被忽略

```bash
# 应该被忽略（返回文件路径表示被忽略）
git check-ignore -v Essay/
git check-ignore -v dataset/
git check-ignore -v AGENTS.md

# 不应该被忽略（无输出表示未被忽略）
git check-ignore -v README.md
git check-ignore -v src/theory.py
```

### 检查数据文件是否包含用户信息

```bash
# 检查列名
head -n 1 data/derived/timeseries_4h.csv

# 搜索用户相关列
head -n 1 data/derived/timeseries_4h.csv | grep -E "user_id|uid|username|nickname"

# 无输出 = 安全 ✅
# 有输出 = 包含用户信息 ❌
```

---

## 🛠️ 清理已追踪的敏感文件

如果敏感文件已经被 git 追踪（之前提交过）：

```bash
# 从 git 移除（保留本地文件）
git rm --cached -r Essay/
git rm --cached -r Essay_nc/
git rm --cached -r dataset/
git rm --cached -r secrets/
git rm --cached AGENTS.md
git rm --cached docs/internal/DEVELOPMENT.md

# 提交清理
git commit -m "chore: remove sensitive files from tracking"

# 推送
git push origin main
```

**注意**：这只能移除未来的追踪，历史记录中仍然存在。如果需要完全清理历史，需要使用 `git filter-branch` 或 `BFG Repo-Cleaner`（高级操作，请先备份）。

---

## 📊 验证清单

在推送前确认：

- [ ] 运行 `scripts/repo_hygiene/verify_gitignore.sh` 或 `scripts/repo_hygiene/verify_gitignore.ps1`
- [ ] 所有检查项通过（0 个失败）
- [ ] `git status --ignored` 显示敏感文件被忽略
- [ ] `git status` 只显示应该提交的文件
- [ ] `data/derived/` 中的数据文件无用户信息
- [ ] `README.md` 和 `docs/architecture/CODE_STRUCTURE.md` 已更新

---

## 🆘 常见问题

### Q1: 验证脚本显示"未被忽略 - 危险"

**原因**：`.gitignore` 规则未生效或文件路径不匹配

**解决**：
1. 检查 `.gitignore` 文件是否包含该规则
2. 检查路径是否正确（注意 `/` 和 `*` 的使用）
3. 如果文件已被追踪，使用 `git rm --cached <file>` 移除

### Q2: 数据文件包含用户信息

**原因**：数据未完全去标识化

**解决**：
1. 重新生成数据文件，确保只包含聚合统计量
2. 移除 `user_id`, `uid`, `username`, `nickname` 等列
3. 验证：`head -n 1 <file> | grep -E "user_id|uid"`

### Q3: 推送后发现敏感文件被上传

**紧急处理**：
1. 立即删除 GitHub 上的仓库（如果是私有仓库）
2. 本地清理历史：使用 `git filter-branch` 或 `BFG Repo-Cleaner`
3. 重新创建仓库并推送

---

## 📁 目录结构参考

### 推送后 GitHub 上应该看到的结构

```
emotion_dynamics/
├── .gitignore          ✅
├── README.md           ✅
├── docs/architecture/CODE_STRUCTURE.md   ✅
├── requirements.txt    ✅
├── src/                ✅
├── scripts/            ✅
├── notebooks/          ✅
├── tests/              ✅
├── data/
│   └── derived/       ✅ (无用户信息)
└── docs/              ⚠️ (部分文件)
```

### 不应该出现的文件/文件夹

```
❌ Essay/
❌ Essay_nc/
❌ dataset/
❌ secrets/
❌ AGENTS.md
❌ docs/internal/DEVELOPMENT.md
❌ Manuscript*.pdf
```

---

## 🔗 相关文档

- **详细报告**: `GITHUB_CLEANUP_REPORT.md`
- **README**: `README.md`
- **代码架构**: `docs/architecture/CODE_STRUCTURE.md`
- **验证脚本**: `scripts/repo_hygiene/verify_gitignore.sh` / `scripts/repo_hygiene/verify_gitignore.ps1`

---

**最后更新**: 2026-01-16
