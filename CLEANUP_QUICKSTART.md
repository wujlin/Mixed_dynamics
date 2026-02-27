# 🚀 Git 清理 - 一分钟快速指南

## Windows 用户（复制粘贴即可）

```powershell
# 打开 PowerShell，粘贴这 3 行：
cd E:\newdesktop\emotion_dynamics
powershell -ExecutionPolicy Bypass -File .\cleanup_git_tracking.ps1
# 然后按照提示输入 y 确认每一步
```

## Linux/Mac 用户（复制粘贴即可）

```bash
# 打开终端，粘贴这 3 行：
cd /path/to/emotion_dynamics  # 替换为你的路径
chmod +x cleanup_git_tracking.sh && ./cleanup_git_tracking.sh
# 然后按照提示输入 y 确认每一步
```

---

## ⚡ 脚本做什么？

1. ✅ 从 GitHub 移除敏感文件（Essay/, dataset/, secrets/, AGENTS.md 等）
2. ✅ **本地文件不会被删除**
3. ✅ 自动提交并推送到 GitHub
4. ✅ 每一步都会询问确认

---

## 📊 预期结果

**推送后（等待 1-2 分钟）**:
- ❌ GitHub 上不再显示 Essay/, Essay_nc/, dataset/, AGENTS.md
- ✓ 本地文件仍然存在
- ✓ src/, notebooks/, README.md 等公开文件正常显示

---

## ⚠️ 如果遇到问题

查看详细指南：`CLEANUP_USAGE_GUIDE.md`

或手动执行（Windows PowerShell）：
```powershell
git rm --cached -r Essay/ Essay_nc/ dataset/ secrets/ AGENTS.md DEVELOPMENT.md
git commit -m "chore: remove sensitive files"
git push origin main
```

---

**完整文档**: `CLEANUP_USAGE_GUIDE.md`

