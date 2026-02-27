# Git 敏感文件清理 - 使用指南

**目的**: 从 GitHub 上移除敏感文件（Essay/, dataset/, secrets/ 等）

**重要**: 本地文件**不会被删除**，只是从 Git 追踪中移除

---

## 🚀 快速开始

### Windows 用户（PowerShell）

```powershell
# 1. 打开 PowerShell
# 2. 进入项目目录
cd E:\newdesktop\emotion_dynamics

# 3. 运行清理脚本
powershell -ExecutionPolicy Bypass -File .\cleanup_git_tracking.ps1
```

### Linux/Mac 用户（Bash）

```bash
# 1. 打开终端
# 2. 进入项目目录
cd /path/to/emotion_dynamics

# 3. 给脚本添加执行权限
chmod +x cleanup_git_tracking.sh

# 4. 运行清理脚本
./cleanup_git_tracking.sh
```

---

## 📋 脚本执行流程

脚本会按照以下步骤执行（**每一步都会询问确认**）：

### Step 1: 列出将要移除的文件
脚本会显示所有将要从 Git 追踪中移除的文件和文件夹：
```
- Essay/
- Essay_nc/
- dataset/
- secrets/
- AGENTS.md
- DEVELOPMENT.md
- ...
```

**询问**: "继续执行? (y/N)"
- 输入 `y` 继续
- 输入 `N` 或直接回车取消

### Step 2: 执行移除操作
使用 `git rm --cached` 命令移除文件追踪（本地文件保留）

**结果显示**:
- ✓ 成功 - 文件已从追踪中移除
- ⚠ 跳过 - 文件未被 Git 追踪

### Step 3: 检查暂存区
显示已暂存的更改（将要提交的文件列表）

### Step 4: 提交更改
**询问**: "是否提交这些更改? (y/N)"
- 输入 `y` 继续提交
- 输入 `N` 取消（可以稍后手动提交）

提交信息会自动生成：
```
chore: remove sensitive files from git tracking

- Remove Essay/ and Essay_nc/ (manuscript drafts)
- Remove dataset/ (contains user data)
- ...
```

### Step 5: 推送到 GitHub
**询问**: "是否推送到 GitHub? (y/N)"
- 输入 `y` 立即推送
- 输入 `N` 稍后手动推送（`git push origin main`）

---

## 🖥️ 详细执行步骤（Windows）

### 方法 1: 使用脚本（推荐）

1. **打开 PowerShell**
   - 按 `Win + X`，选择 "Windows PowerShell" 或 "终端"
   - 或在开始菜单搜索 "PowerShell"

2. **进入项目目录**
   ```powershell
   cd E:\newdesktop\emotion_dynamics
   ```

3. **运行脚本**
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\cleanup_git_tracking.ps1
   ```

4. **按照提示操作**
   - 阅读将要移除的文件列表
   - 输入 `y` 确认每个步骤
   - 等待推送完成

5. **验证结果**
   - 等待 1-2 分钟
   - 刷新 GitHub 页面
   - 确认 Essay/, Essay_nc/, dataset/ 等文件夹已消失

### 方法 2: 手动执行命令

如果脚本无法运行，可以手动执行以下命令：

```powershell
# 进入项目目录
cd E:\newdesktop\emotion_dynamics

# 移除文件夹
git rm --cached -r Essay/
git rm --cached -r Essay_nc/
git rm --cached -r dataset/
git rm --cached -r secrets/
git rm --cached -r outputs/figs/
git rm --cached -r outputs/annotations/
git rm --cached -r outputs/llm_validation/

# 移除单个文件
git rm --cached AGENTS.md
git rm --cached DEVELOPMENT.md
git rm --cached docs/progress.md
git rm --cached docs/review.md
git rm --cached Manuscript.pdf
git rm --cached Manuscript.docx

# 检查暂存区
git status

# 提交
git commit -m "chore: remove sensitive files from git tracking"

# 推送
git push origin main
```

---

## 🐧 详细执行步骤（Linux/Mac）

### 方法 1: 使用脚本（推荐）

1. **打开终端**
   - Mac: `Cmd + Space`，输入 "Terminal"
   - Linux: `Ctrl + Alt + T`

2. **进入项目目录**
   ```bash
   cd /path/to/emotion_dynamics
   ```

3. **添加执行权限**
   ```bash
   chmod +x cleanup_git_tracking.sh
   ```

4. **运行脚本**
   ```bash
   ./cleanup_git_tracking.sh
   ```

5. **按照提示操作**
   - 阅读将要移除的文件列表
   - 输入 `y` 确认每个步骤
   - 等待推送完成

6. **验证结果**
   - 等待 1-2 分钟
   - 刷新 GitHub 页面
   - 确认敏感文件夹已消失

### 方法 2: 手动执行命令

```bash
# 进入项目目录
cd /path/to/emotion_dynamics

# 移除文件夹
git rm --cached -r Essay/
git rm --cached -r Essay_nc/
git rm --cached -r dataset/
git rm --cached -r secrets/
git rm --cached -r outputs/figs/

# 移除单个文件
git rm --cached AGENTS.md
git rm --cached DEVELOPMENT.md

# 提交
git commit -m "chore: remove sensitive files from git tracking"

# 推送
git push origin main
```

---

## ✅ 验证清理结果

### 1. 检查本地文件（应该仍然存在）

**Windows**:
```powershell
ls Essay/
ls dataset/
ls AGENTS.md
```

**Linux/Mac**:
```bash
ls Essay/
ls dataset/
ls AGENTS.md
```

**预期结果**: 所有文件仍然存在 ✓

### 2. 检查 Git 状态

```bash
git status
```

**预期结果**:
```
On branch main
nothing to commit, working tree clean
```

### 3. 检查被忽略的文件

```bash
git status --ignored
```

**预期结果**:
```
Ignored files:
  Essay/
  Essay_nc/
  dataset/
  secrets/
  AGENTS.md
  ...
```

### 4. 检查 GitHub 页面

1. 打开 https://github.com/wujlin/Mixed_dynamics
2. 刷新页面（可能需要等待 1-2 分钟）
3. 确认以下文件夹**不再显示**:
   - ❌ Essay/
   - ❌ Essay_nc/
   - ❌ dataset/
   - ❌ secrets/
   - ❌ AGENTS.md

4. 确认以下文件夹**仍然显示**:
   - ✓ src/
   - ✓ notebooks/
   - ✓ tests/
   - ✓ README.md
   - ✓ CODE_STRUCTURE.md

---

## ❗ 常见问题

### Q1: 脚本运行后显示 "没有文件被移除"

**原因**: 这些文件可能从未被 Git 追踪

**解决**:
1. 检查 `.gitignore` 是否正确：
   ```bash
   cat .gitignore | grep Essay
   ```
2. 如果规则正确，说明文件已经被成功忽略，无需清理

### Q2: 推送失败 "Permission denied"

**原因**: 没有 GitHub 推送权限

**解决**:
1. 检查 Git 凭证：
   ```bash
   git config user.name
   git config user.email
   ```
2. 重新配置 GitHub 认证：
   ```bash
   git config --global credential.helper store
   git push origin main
   ```
3. 输入 GitHub 用户名和 Personal Access Token

### Q3: 脚本提示 "ExecutionPolicy" 错误（Windows）

**原因**: PowerShell 执行策略限制

**解决**:
```powershell
# 方法 1: 临时绕过
powershell -ExecutionPolicy Bypass -File .\cleanup_git_tracking.ps1

# 方法 2: 永久修改（需要管理员权限）
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Q4: 本地文件被误删了

**原因**: 可能使用了 `git rm` 而不是 `git rm --cached`

**解决**:
```bash
# 撤销 commit（保留更改）
git reset HEAD~1

# 从 Git 恢复文件
git checkout HEAD Essay/ dataset/ AGENTS.md
```

### Q5: 文件仍然出现在 GitHub 历史记录中

**原因**: `git rm --cached` 只移除未来的追踪，不清除历史

**解决**: 如果需要完全清除历史（例如泄露了密钥），请参考 `GITHUB_CLEANUP_REPORT.md` 中的 "完全清除历史" 部分

---

## 🔒 安全检查

在推送前，请确认以下内容：

### 检查 dataset/ 是否包含用户数据

```bash
# 检查文件列表
ls dataset/

# 检查是否曾经被提交
git log --all --full-history -- dataset/ | head -20
```

**如果显示有提交记录**，说明用户数据曾经被上传，需要完全清除历史！

### 检查 secrets/ 是否包含密钥

```bash
# 检查文件列表
ls secrets/

# 检查是否曾经被提交
git log --all --full-history -- secrets/ | head -20
```

**如果显示有提交记录**，说明密钥曾经被上传，需要立即清除历史并更换密钥！

---

## 🔥 紧急情况：需要完全清除历史

如果 `dataset/` 或 `secrets/` 曾经被提交，需要完全清除历史：

### 使用 BFG Repo-Cleaner（推荐）

```bash
# 1. 备份仓库
cd ..
cp -r emotion_dynamics emotion_dynamics_backup

# 2. 下载 BFG
# https://rtyley.github.io/bfg-repo-cleaner/

# 3. 清理敏感文件夹
cd emotion_dynamics
java -jar bfg.jar --delete-folders dataset
java -jar bfg.jar --delete-folders secrets

# 4. 清理和压缩
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 5. 强制推送（危险！）
git push origin main --force
```

**⚠️ 警告**: `--force` 推送会重写 GitHub 历史，如果有其他协作者需要提前通知！

---

## 📞 需要帮助？

如果遇到问题：

1. 查看完整报告：`GITHUB_CLEANUP_REPORT.md`
2. 查看快速参考：`QUICK_REFERENCE.md`
3. 运行验证脚本：`verify_gitignore.sh` 或 `verify_gitignore.ps1`

---

**最后更新**: 2026-01-16

