#!/bin/bash
# 批量修复图表边距，防止右侧 tick 被截断
# 将 right=0.98 改为 right=0.96

set -e

echo "修复图表边距设置..."
echo ""

# 需要修改的文件列表
files=(
    "notebooks/make_fig1b_potential.py"
    "notebooks/make_fig2a_network_validation.py"
    "notebooks/make_fig2b_binder_u4.py"
    "notebooks/make_fig2c_activity.py"
    "notebooks/make_fig3a_csd_scaling.py"
    "notebooks/make_fig3b_csd_sde_vs_abm.py"
    "notebooks/make_fig3c_csd_timeseries.py"
    "notebooks/make_fig4c_media_ratio.py"
    "notebooks/make_fig4d_beta_effect.py"
    "notebooks/make_fig5a_h1_all.py"
    "notebooks/make_fig5b_h2_batch3_density.py"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "处理: $file"
        sed -i 's/right=0\.98/right=0.96/g' "$file"
        echo "  ✓ 完成"
    else
        echo "  ✗ 文件不存在: $file"
    fi
done

echo ""
echo "=========================================="
echo "✓ 边距修复完成！"
echo "=========================================="
echo ""
echo "修改内容：right=0.98 → right=0.96"
echo "影响文件：${#files[@]} 个"


