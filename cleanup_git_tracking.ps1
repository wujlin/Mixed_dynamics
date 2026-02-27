# Git 敏感文件清理脚本 (PowerShell 版本)
# 用途: 从 Git 追踪中移除敏感文件（但保留本地文件）

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  Git 敏感文件清理脚本" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# 检查是否在 git 仓库中
try {
    git rev-parse --is-inside-work-tree 2>$null | Out-Null
} catch {
    Write-Host "错误: 当前目录不是 Git 仓库！" -ForegroundColor Red
    exit 1
}

Write-Host "⚠️  警告: 此脚本将从 Git 追踪中移除以下文件/文件夹:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  - Essay/"
Write-Host "  - Essay_nc/"
Write-Host "  - dataset/"
Write-Host "  - secrets/"
Write-Host "  - outputs/figs/"
Write-Host "  - outputs/annotations/"
Write-Host "  - outputs/llm_validation/"
Write-Host "  - AGENTS.md"
Write-Host "  - DEVELOPMENT.md"
Write-Host "  - docs/progress.md"
Write-Host "  - docs/review.md"
Write-Host "  - Manuscript*.pdf"
Write-Host "  - Manuscript*.docx"
Write-Host ""
Write-Host "✓ 本地文件不会被删除，仅从 Git 追踪中移除" -ForegroundColor Green
Write-Host ""

# 询问用户确认
$confirm = Read-Host "继续执行? (y/N)"
if ($confirm -ne 'y' -and $confirm -ne 'Y') {
    Write-Host "操作已取消"
    exit 0
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  Step 1: 从 Git 追踪中移除文件" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# 移除文件夹
$foldersToRemove = @(
    "Essay/",
    "Essay_nc/",
    "dataset/",
    "secrets/",
    "outputs/figs/",
    "outputs/annotations/",
    "outputs/llm_validation/"
)

foreach ($folder in $foldersToRemove) {
    $files = git ls-files $folder 2>$null
    if ($files) {
        Write-Host "  移除文件夹: $folder" -ForegroundColor Yellow
        git rm --cached -r $folder 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "    ✓ 成功" -ForegroundColor Green
        } else {
            Write-Host "    ⚠  跳过（未追踪或已移除）" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  跳过文件夹: $folder (未被 Git 追踪)" -ForegroundColor Gray
    }
}

# 移除单个文件
$filesToRemove = @(
    "AGENTS.md",
    "DEVELOPMENT.md",
    "docs/progress.md",
    "docs/review.md",
    "docs/email_pi_empirical_update.md",
    "docs/writing_drills.md",
    "Manuscript.pdf",
    "Manuscript.docx",
    "Google Gemini.pdf"
)

foreach ($file in $filesToRemove) {
    $exists = git ls-files $file 2>$null
    if ($exists) {
        Write-Host "  移除文件: $file" -ForegroundColor Yellow
        git rm --cached $file 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "    ✓ 成功" -ForegroundColor Green
        } else {
            Write-Host "    ⚠  跳过（未追踪或已移除）" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  跳过文件: $file (未被 Git 追踪)" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  Step 2: 检查暂存区状态" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

$cached = git diff --cached --name-only
if (-not $cached) {
    Write-Host "⚠️  没有文件被移除（可能这些文件未被追踪）" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "检查 .gitignore 是否正确配置："
    Write-Host "  Get-Content .gitignore | Select-String Essay"
    exit 0
} else {
    Write-Host "✓ 已暂存的更改:" -ForegroundColor Green
    $cached | Select-Object -First 20 | ForEach-Object { Write-Host "    $_" }
    Write-Host ""
}

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  Step 3: 提交更改" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

$commitConfirm = Read-Host "是否提交这些更改? (y/N)"
if ($commitConfirm -ne 'y' -and $commitConfirm -ne 'Y') {
    Write-Host "操作已取消，使用 'git reset' 可以撤销暂存的更改"
    exit 0
}

$commitMessage = @"
chore: remove sensitive files from git tracking

- Remove Essay/ and Essay_nc/ (manuscript drafts)
- Remove dataset/ (contains user data)
- Remove secrets/ (credentials)
- Remove AGENTS.md, DEVELOPMENT.md (internal docs)
- Remove outputs/figs/, outputs/annotations/ (generated files)
- These files are now properly ignored by .gitignore
- Local files are preserved
"@

git commit -m $commitMessage

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ 提交成功" -ForegroundColor Green
} else {
    Write-Host "✗ 提交失败" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  Step 4: 推送到 GitHub" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

$pushConfirm = Read-Host "是否推送到 GitHub? (y/N)"
if ($pushConfirm -ne 'y' -and $pushConfirm -ne 'Y') {
    Write-Host "跳过推送。稍后可以手动执行: git push origin main"
    exit 0
}

Write-Host "正在推送..." -ForegroundColor Yellow
git push origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ 推送成功！" -ForegroundColor Green
    Write-Host ""
    Write-Host "请等待 1-2 分钟后刷新 GitHub 页面，敏感文件应该已经消失。"
} else {
    Write-Host "✗ 推送失败" -ForegroundColor Red
    Write-Host "请检查网络连接和权限，然后手动执行: git push origin main"
    exit 1
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  ✅ 清理完成！" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "验证步骤:"
Write-Host "  1. 刷新 GitHub 页面，确认敏感文件夹已消失"
Write-Host "  2. 检查本地文件是否仍然存在: ls Essay/, ls dataset/"
Write-Host "  3. 检查 git 状态: git status --ignored"
Write-Host ""
Write-Host "注意: 这些文件仍存在于 Git 历史记录中。" -ForegroundColor Yellow
Write-Host "如果需要完全清除历史（例如泄露了密钥），请查看 GITHUB_CLEANUP_REPORT.md"
Write-Host ""

