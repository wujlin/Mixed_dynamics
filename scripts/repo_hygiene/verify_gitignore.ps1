# GitHub 仓库安全验证脚本 (PowerShell 版本)
# 使用方法: .\scripts\repo_hygiene\verify_gitignore.ps1

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  GitHub 仓库安全验证脚本" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# 计数器
$SUCCESS = 0
$FAILED = 0

# 应该被忽略的文件/文件夹
$SHOULD_IGNORE = @(
    "Essay/",
    "Essay_nc/",
    "dataset/",
    "secrets/",
    "AGENTS.md",
    "docs/internal/DEVELOPMENT.md",
    "docs/progress.md",
    "docs/review.md",
    "docs/email_pi_empirical_update.md",
    "docs/writing_drills.md",
    "Manuscript.pdf",
    "Manuscript.docx",
    "outputs/annotations/",
    "outputs/llm_validation/",
    "outputs/figs/"
)

# 不应该被忽略的文件
$SHOULD_NOT_IGNORE = @(
    "src/theory.py",
    "src/network_sim.py",
    "src/plot_style.py",
    "README.md",
    "docs/architecture/CODE_STRUCTURE.md",
    "requirements.txt",
    ".gitignore"
)

Write-Host "1️⃣  检查敏感文件是否被正确忽略" -ForegroundColor Yellow
Write-Host "----------------------------------------"
foreach ($file in $SHOULD_IGNORE) {
    $result = git check-ignore $file 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ $file (已忽略)" -ForegroundColor Green
        $SUCCESS++
    } else {
        Write-Host "  ❌ $file (未忽略 - 危险！)" -ForegroundColor Red
        $FAILED++
    }
}

Write-Host ""
Write-Host "2️⃣  检查公开文件未被错误忽略" -ForegroundColor Yellow
Write-Host "----------------------------------------"
foreach ($file in $SHOULD_NOT_IGNORE) {
    if (Test-Path $file) {
        $result = git check-ignore $file 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ❌ $file (被忽略 - 不应该！)" -ForegroundColor Red
            $FAILED++
        } else {
            Write-Host "  ✅ $file (未忽略)" -ForegroundColor Green
            $SUCCESS++
        }
    } else {
        Write-Host "  ⚠️  $file (文件不存在)" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "3️⃣  检查公开数据文件" -ForegroundColor Yellow
Write-Host "----------------------------------------"

if (Test-Path "data/derived") {
    Write-Host "  ✅ data/derived/ 目录存在" -ForegroundColor Green
    
    $DATA_FILES = @(
        "data/derived/timeseries_4h.csv",
        "data/derived/segments_pooled.csv"
    )
    
    foreach ($file in $DATA_FILES) {
        if (Test-Path $file) {
            Write-Host "  ✅ $file 存在" -ForegroundColor Green
            
            # 检查是否包含用户信息列
            $firstLine = Get-Content $file -First 1
            if ($firstLine -match "user_id|uid|username|nickname") {
                Write-Host "    ❌ 包含用户信息列（user_id/uid/username）！" -ForegroundColor Red
                $FAILED++
            } else {
                Write-Host "    ✅ 无用户信息列" -ForegroundColor Green
                $SUCCESS++
            }
        } else {
            Write-Host "  ⚠️  $file 不存在（可能尚未生成）" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "  ⚠️  data/derived/ 目录不存在" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "4️⃣  检查 secrets/ 和 cookies" -ForegroundColor Yellow
Write-Host "----------------------------------------"
$result = git check-ignore "secrets/" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✅ secrets/ 被忽略" -ForegroundColor Green
    $SUCCESS++
} else {
    Write-Host "  ❌ secrets/ 未被忽略" -ForegroundColor Red
    $FAILED++
}

$result = git check-ignore "cookies.json" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✅ cookies*.json 被忽略" -ForegroundColor Green
    $SUCCESS++
} else {
    Write-Host "  ⚠️  cookies*.json 匹配规则（确保所有 cookie 文件被忽略）" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "5️⃣  检查开发文档" -ForegroundColor Yellow
Write-Host "----------------------------------------"
$DEV_DOCS = @(
    "AGENTS.md",
    "docs/internal/DEVELOPMENT.md",
    "docs/progress.md"
)

foreach ($file in $DEV_DOCS) {
    $result = git check-ignore $file 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ $file 被忽略" -ForegroundColor Green
        $SUCCESS++
    } else {
        if (Test-Path $file) {
            Write-Host "  ❌ $file 未被忽略（存在且未忽略）" -ForegroundColor Red
            $FAILED++
        } else {
            Write-Host "  ⚠️  $file 未被忽略（但文件不存在）" -ForegroundColor Yellow
        }
    }
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  验证结果" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "通过: $SUCCESS" -ForegroundColor Green
Write-Host "失败: $FAILED" -ForegroundColor Red
Write-Host ""

if ($FAILED -eq 0) {
    Write-Host "✅ 所有检查通过！可以安全推送到 GitHub。" -ForegroundColor Green
    exit 0
} else {
    Write-Host "❌ 发现 $FAILED 个问题，请修复后再推送！" -ForegroundColor Red
    Write-Host ""
    Write-Host "修复建议："
    Write-Host "1. 检查 .gitignore 文件是否包含所有敏感文件规则"
    Write-Host "2. 如果敏感文件已被追踪，使用 'git rm --cached <file>' 移除"
    Write-Host "3. 检查数据文件是否包含用户信息"
    exit 1
}
