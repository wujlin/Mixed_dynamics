# 批量重新生成所有论文图表 (Windows PowerShell 版本)
# 用法：在项目根目录运行 .\notebooks\regenerate_all_figures.ps1
# 注意：需要激活 conda 环境或确保 python 在 PATH 中

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Blue
Write-Host "  批量重新生成论文图表" -ForegroundColor Blue
Write-Host "==========================================" -ForegroundColor Blue
Write-Host ""

$TOTAL = 0
$SUCCESS = 0
$FAILED = 0

function Run-Script {
    param (
        [string]$Script,
        [string]$Description
    )
    
    $script:TOTAL++
    
    Write-Host "[$script:TOTAL] 正在生成: $Description" -ForegroundColor Yellow
    Write-Host "    脚本: $Script"
    
    try {
        python $Script
        $script:SUCCESS++
        Write-Host "    ✓ 成功" -ForegroundColor Green
    } catch {
        $script:FAILED++
        Write-Host "    ✗ 失败: $_" -ForegroundColor Red
    }
    Write-Host ""
}

# P0: 必须重新生成
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Blue
Write-Host "  P0: 必须重新生成（修改了代码）" -ForegroundColor Blue
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Blue
Write-Host ""

Run-Script "notebooks\make_fig2c_activity.py" "Fig 2c - Activity dynamics"
Run-Script "notebooks\make_fig3c_csd_timeseries.py" "Fig 3c - CSD time series"
Run-Script "notebooks\make_fig4a_rc_landscape.py" "Fig 4a - Parameter landscape"
Run-Script "notebooks\make_fig4b_k_effect.py" "Fig 4b - Information density effect"
Run-Script "notebooks\make_fig4c_media_ratio.py" "Fig 4c - Media ecology effect"
Run-Script "notebooks\make_fig4d_beta_effect.py" "Fig 4d - Local coupling effect"

# P1: 强烈建议重新生成
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Blue
Write-Host "  P1: 强烈建议重新生成（序号位置优化）" -ForegroundColor Blue
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Blue
Write-Host ""

Run-Script "notebooks\make_fig1a_bifurcation.py" "Fig 1a - Bifurcation diagram"
Run-Script "notebooks\make_fig1b_potential.py" "Fig 1b - Effective potential"
Run-Script "notebooks\make_fig1c_sym_asym_qa.py" "Fig 1c - Symmetric vs Asymmetric"
Run-Script "notebooks\make_fig2a_network_validation.py" "Fig 2a - Network validation"
Run-Script "notebooks\make_fig2b_binder_u4.py" "Fig 2b - Binder cumulant"
Run-Script "notebooks\make_fig3a_csd_scaling.py" "Fig 3a - CSD scaling"
Run-Script "notebooks\make_fig3b_csd_sde_vs_abm.py" "Fig 3b - CSD SDE vs ABM"
Run-Script "notebooks\make_fig5a_h1_all.py" "Fig 5a - Activity-Jump association"
Run-Script "notebooks\make_fig5b_h2_batch3_density.py" "Fig 5b - Media-Volatility association"

# 汇总
Write-Host "==========================================" -ForegroundColor Blue
Write-Host "  汇总报告" -ForegroundColor Blue
Write-Host "==========================================" -ForegroundColor Blue
Write-Host "总计:   $TOTAL 张图"
Write-Host "成功:   $SUCCESS 张" -ForegroundColor Green
if ($FAILED -gt 0) {
    Write-Host "失败:   $FAILED 张" -ForegroundColor Red
    Write-Host ""
    Write-Host "请检查失败的脚本输出日志" -ForegroundColor Red
    exit 1
} else {
    Write-Host "失败:   0 张" -ForegroundColor Green
    Write-Host ""
    Write-Host "✓ 所有图表已成功重新生成！" -ForegroundColor Green
    Write-Host ""
    Write-Host "生成的图表位置：" -ForegroundColor Yellow
    Write-Host "  - PDF (用于 LaTeX):  Essay\figures\"
    Write-Host "  - PNG (预览):        outputs\figs\"
}

