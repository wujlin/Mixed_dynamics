#!/bin/bash
# 快速批量重新生成优先级 P0 图表（必须重新生成）
# 用法：在 WSL 中运行 bash notebooks/regenerate_priority_figures.sh

set -e  # 遇到错误立即退出

# Python 解释器（根据你的环境调整）
PYTHON="/home/wujlin/miniconda3/envs/emotion/bin/python"

# 项目根目录
cd "$(dirname "$0")/.."

echo "=========================================="
echo "  重新生成优先级 P0 图表"
echo "=========================================="
echo ""

# Figure 2c (修改了 title fontsize)
echo "[1/6] Fig 2c - Activity dynamics"
PYTHONDONTWRITEBYTECODE=1 $PYTHON notebooks/make_fig2c_activity.py
echo ""

# Figure 3c (修改了 title fontsize + y轴范围)
echo "[2/6] Fig 3c - CSD time series"
PYTHONDONTWRITEBYTECODE=1 $PYTHON notebooks/make_fig3c_csd_timeseries.py
echo ""

# Figure 4 所有 panel (截图显示序号被遮挡)
echo "[3/6] Fig 4a - Parameter landscape"
PYTHONDONTWRITEBYTECODE=1 $PYTHON notebooks/make_fig4a_rc_landscape.py
echo ""

echo "[4/6] Fig 4b - Information density effect"
PYTHONDONTWRITEBYTECODE=1 $PYTHON notebooks/make_fig4b_k_effect.py
echo ""

echo "[5/6] Fig 4c - Media ecology effect"
PYTHONDONTWRITEBYTECODE=1 $PYTHON notebooks/make_fig4c_media_ratio.py
echo ""

echo "[6/6] Fig 4d - Local coupling effect"
PYTHONDONTWRITEBYTECODE=1 $PYTHON notebooks/make_fig4d_beta_effect.py
echo ""

echo "=========================================="
echo "✓ 优先级 P0 图表重新生成完成！"
echo "=========================================="






