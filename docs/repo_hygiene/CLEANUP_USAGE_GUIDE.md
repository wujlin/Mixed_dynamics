# Git 敏感文件清理指南

目标是把敏感文件从 Git 追踪中移除，同时保留本地副本。现在统一采用直接命令，不再维护额外的清理脚本。

## 原则

- 使用 `git rm --cached`，只移除追踪，不删除本地文件
- 提交前先看 `git status`
- 推送后再去 GitHub 核对结果

## 快速开始

### Windows PowerShell

```powershell
cd E:\newdesktop\emotion_dynamics

git rm --cached -r Essay/ Essay_nc/ dataset/ secrets/ outputs/figs/ outputs/annotations/ outputs/llm_validation/
git rm --cached AGENTS.md docs/internal/DEVELOPMENT.md docs/progress.md docs/review.md docs/email_pi_empirical_update.md docs/writing_drills.md Manuscript.pdf legacy/source_materials/Manuscript.docx legacy/source_materials/Google Gemini.pdf

git status
git commit -m "chore: remove sensitive files from git tracking"
git push origin main
```

### Linux / macOS

```bash
cd /path/to/emotion_dynamics

git rm --cached -r Essay/ Essay_nc/ dataset/ secrets/ outputs/figs/ outputs/annotations/ outputs/llm_validation/
git rm --cached AGENTS.md docs/internal/DEVELOPMENT.md docs/progress.md docs/review.md docs/email_pi_empirical_update.md docs/writing_drills.md Manuscript.pdf legacy/source_materials/Manuscript.docx legacy/source_materials/Google Gemini.pdf

git status
git commit -m "chore: remove sensitive files from git tracking"
git push origin main
```

## 命令逻辑

### Step 1: 移除目录追踪

```bash
git rm --cached -r Essay/ Essay_nc/ dataset/ secrets/ outputs/figs/ outputs/annotations/ outputs/llm_validation/
```

这一步用于处理整类敏感目录或可再生成目录。

### Step 2: 移除单文件追踪

```bash
git rm --cached AGENTS.md docs/internal/DEVELOPMENT.md docs/progress.md docs/review.md docs/email_pi_empirical_update.md docs/writing_drills.md Manuscript.pdf legacy/source_materials/Manuscript.docx legacy/source_materials/Google Gemini.pdf
```

这一步用于处理内部文档、手稿副本和其他不应公开跟踪的单文件。

### Step 3: 核对暂存区

```bash
git status
```

你应该看到的是一组 `deleted` 或 `removed from index` 的变更；本地文件本身仍然存在。

### Step 4: 提交并推送

```bash
git commit -m "chore: remove sensitive files from git tracking"
git push origin main
```

---

## 验证清理结果

### 1. 检查本地文件

```bash
ls Essay/
ls dataset/
ls AGENTS.md
```

预期结果是所有文件仍然存在。

### 2. 检查 Git 状态

```bash
git status
```

推送完成后，如果没有其他改动，`git status` 应该恢复为干净状态。

### 3. 检查被忽略的文件

```bash
git status --ignored
```

你应该能看到这些敏感路径出现在 ignored files 中。

### 4. 检查 GitHub 页面

1. 打开仓库主页。
2. 刷新页面。
3. 确认敏感路径不再显示。
4. 确认 `src/`、`notebooks/`、`tests/`、`README.md` 等公开内容仍然可见。

---

## 常见问题

### Q1: 执行命令后显示 "pathspec did not match any files"

原因是这些路径可能从未被 Git 追踪，或者已经被移除出索引。

解决方法：

```bash
cat .gitignore | grep Essay
```

如果规则正确，说明这些路径已经被忽略，无需重复清理。

### Q2: 推送失败 "Permission denied"

原因是没有 GitHub 推送权限。

解决方法：

```bash
git config user.name
git config user.email
git push origin main
```

### Q3: 本地文件被误删了

原因通常是误用了 `git rm` 而不是 `git rm --cached`。

解决方法：

```bash
git reset HEAD~1
git checkout HEAD Essay/ dataset/ AGENTS.md
```

### Q4: 文件仍然出现在 GitHub 历史记录中

`git rm --cached` 只移除未来的追踪，不清除历史。

如果需要完全清除历史，请参考 `GITHUB_CLEANUP_REPORT.md` 中的相关说明。

---

## 安全检查

### 检查 dataset/ 是否包含用户数据

```bash
ls dataset/
git log --all --full-history -- dataset/ | head -20
```

### 检查 secrets/ 是否包含密钥

```bash
ls secrets/
git log --all --full-history -- secrets/ | head -20
```

如果这些路径曾经被提交，就需要考虑清理历史并轮换密钥。

---

## 需要帮助

1. 查看完整报告：`GITHUB_CLEANUP_REPORT.md`
2. 查看快速参考：`QUICK_REFERENCE.md`
3. 运行验证脚本：`scripts/repo_hygiene/verify_gitignore.sh` 或 `scripts/repo_hygiene/verify_gitignore.ps1`

---

**最后更新**: 2026-03-11
