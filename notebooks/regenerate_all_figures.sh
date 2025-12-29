#!/bin/bash
# 批量重新生成所有论文图表
# 用法：在 WSL 中运行 bash notebooks/regenerate_all_figures.sh

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Python 解释器（根据你的环境调整）
PYTHON="/home/wujlin/miniconda3/envs/emotion/bin/python"

# 项目根目录
cd "$(dirname "$0")/.."
ROOT=$(pwd)

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  批量重新生成论文图表${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 统计
TOTAL=0
SUCCESS=0
FAILED=0

run_script() {
    local script=$1
    local desc=$2
    TOTAL=$((TOTAL + 1))
    
    echo -e "${YELLOW}[$TOTAL] 正在生成: $desc${NC}"
    echo -e "    脚本: $script"
    
    if PYTHONDONTWRITEBYTECODE=1 $PYTHON "$script"; then
        SUCCESS=$((SUCCESS + 1))
        echo -e "${GREEN}    ✓ 成功${NC}"
    else
        FAILED=$((FAILED + 1))
        echo -e "${RED}    ✗ 失败${NC}"
    fi
    echo ""
}

# ============================================
# 优先级 P0：必须重新生成（直接受修改影响）
# ============================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  P0: 必须重新生成（修改了代码）${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Figure 2c (修改了 title fontsize)
run_script "notebooks/make_fig2c_activity.py" "Fig 2c - Activity dynamics"

# Figure 3c (修改了 title fontsize + y轴范围)
run_script "notebooks/make_fig3c_csd_timeseries.py" "Fig 3c - CSD time series"

# Figure 4 所有 panel (截图显示序号被遮挡)
run_script "notebooks/make_fig4a_rc_landscape.py" "Fig 4a - Parameter landscape"
run_script "notebooks/make_fig4b_k_effect.py" "Fig 4b - Information density effect"
run_script "notebooks/make_fig4c_media_ratio.py" "Fig 4c - Media ecology effect"
run_script "notebooks/make_fig4d_beta_effect.py" "Fig 4d - Local coupling effect"

# ============================================
# 优先级 P1：强烈建议重新生成（受 add_panel_label 修改影响）
# ============================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  P1: 强烈建议重新生成（序号位置优化）${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Figure 1
run_script "notebooks/make_fig1a_bifurcation.py" "Fig 1a - Bifurcation diagram"
run_script "notebooks/make_fig1b_potential.py" "Fig 1b - Effective potential"
run_script "notebooks/make_fig1c_sym_asym_qa.py" "Fig 1c - Symmetric vs Asymmetric"

# Figure 2a, 2b
run_script "notebooks/make_fig2a_network_validation.py" "Fig 2a - Network validation"
run_script "notebooks/make_fig2b_binder_u4.py" "Fig 2b - Binder cumulant"

# Figure 3a, 3b
run_script "notebooks/make_fig3a_csd_scaling.py" "Fig 3a - CSD scaling"
run_script "notebooks/make_fig3b_csd_sde_vs_abm.py" "Fig 3b - CSD SDE vs ABM"

# Figure 5
run_script "notebooks/make_fig5a_h1_all.py" "Fig 5a - Activity-Jump association"
run_script "notebooks/make_fig5b_h2_batch3_density.py" "Fig 5b - Media-Volatility association"

# ============================================
# 汇总报告
# ============================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  汇总报告${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "总计:   $TOTAL 张图"
echo -e "${GREEN}成功:   $SUCCESS 张${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}失败:   $FAILED 张${NC}"
    echo ""
    echo -e "${RED}请检查失败的脚本输出日志${NC}"
    exit 1
else
    echo -e "${GREEN}失败:   0 张${NC}"
    echo ""
    echo -e "${GREEN}✓ 所有图表已成功重新生成！${NC}"
    echo ""
    echo -e "${YELLOW}生成的图表位置：${NC}"
    echo -e "  - PDF (用于 LaTeX):  Essay/figures/"
    echo -e "  - PNG (预览):        outputs/figs/"
fi

