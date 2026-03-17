# Git 清理快速指南

目标是把敏感文件从 Git 追踪中移除，但保留本地文件。

## Windows PowerShell

```powershell
cd E:\newdesktop\emotion_dynamics

git rm --cached -r Essay/ Essay_nc/ dataset/ secrets/ outputs/figs/ outputs/annotations/ outputs/llm_validation/
git rm --cached AGENTS.md docs/internal/DEVELOPMENT.md docs/progress.md docs/review.md docs/email_pi_empirical_update.md docs/writing_drills.md Manuscript.pdf legacy/source_materials/Manuscript.docx legacy/source_materials/Google Gemini.pdf

git status
git commit -m "chore: remove sensitive files from git tracking"
git push origin main
```

## Linux / macOS

```bash
cd /path/to/emotion_dynamics

git rm --cached -r Essay/ Essay_nc/ dataset/ secrets/ outputs/figs/ outputs/annotations/ outputs/llm_validation/
git rm --cached AGENTS.md docs/internal/DEVELOPMENT.md docs/progress.md docs/review.md docs/email_pi_empirical_update.md docs/writing_drills.md Manuscript.pdf legacy/source_materials/Manuscript.docx legacy/source_materials/Google Gemini.pdf

git status
git commit -m "chore: remove sensitive files from git tracking"
git push origin main
```

## 结果判断

- 本地文件仍然存在
- `git status` 只显示从追踪中移除的变更
- 推送后 GitHub 页面不再显示这些敏感路径

## 补充说明

- `git rm --cached` 不会删除本地文件
- 如果某些路径未被 Git 追踪，Git 会提示跳过
- 详细说明见 `CLEANUP_USAGE_GUIDE.md`
