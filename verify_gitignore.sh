#!/bin/bash
# 验证 .gitignore 规则是否正确保护敏感文件

echo "========================================="
echo "  GitHub 仓库安全验证脚本"
echo "========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 计数器
SUCCESS=0
FAILED=0

# 应该被忽略的文件/文件夹
SHOULD_IGNORE=(
    "Essay/"
    "Essay_nc/"
    "dataset/"
    "secrets/"
    "AGENTS.md"
    "DEVELOPMENT.md"
    "docs/progress.md"
    "docs/review.md"
    "docs/email_pi_empirical_update.md"
    "docs/writing_drills.md"
    "Manuscript.pdf"
    "Manuscript.docx"
    "outputs/annotations/"
    "outputs/llm_validation/"
    "outputs/figs/"
)

# 不应该被忽略的文件
SHOULD_NOT_IGNORE=(
    "src/theory.py"
    "src/network_sim.py"
    "src/plot_style.py"
    "README.md"
    "CODE_STRUCTURE.md"
    "requirements.txt"
    ".gitignore"
)

echo "1️⃣  检查敏感文件是否被正确忽略"
echo "----------------------------------------"
for file in "${SHOULD_IGNORE[@]}"; do
    if git check-ignore -q "$file" 2>/dev/null; then
        echo -e "  ${GREEN}✅${NC} $file (已忽略)"
        ((SUCCESS++))
    else
        echo -e "  ${RED}❌${NC} $file (未忽略 - 危险！)"
        ((FAILED++))
    fi
done

echo ""
echo "2️⃣  检查公开文件未被错误忽略"
echo "----------------------------------------"
for file in "${SHOULD_NOT_IGNORE[@]}"; do
    if [ -f "$file" ]; then
        if git check-ignore -q "$file" 2>/dev/null; then
            echo -e "  ${RED}❌${NC} $file (被忽略 - 不应该！)"
            ((FAILED++))
        else
            echo -e "  ${GREEN}✅${NC} $file (未忽略)"
            ((SUCCESS++))
        fi
    else
        echo -e "  ${YELLOW}⚠️${NC}  $file (文件不存在)"
    fi
done

echo ""
echo "3️⃣  检查公开数据文件"
echo "----------------------------------------"

# 检查 data/derived/ 目录
if [ -d "data/derived" ]; then
    echo -e "  ${GREEN}✅${NC} data/derived/ 目录存在"
    
    # 检查关键数据文件
    DATA_FILES=(
        "data/derived/timeseries_4h.csv"
        "data/derived/segments_pooled.csv"
    )
    
    for file in "${DATA_FILES[@]}"; do
        if [ -f "$file" ]; then
            echo -e "  ${GREEN}✅${NC} $file 存在"
            
            # 检查是否包含用户信息列
            if head -n 1 "$file" 2>/dev/null | grep -qE "user_id|uid|username|nickname"; then
                echo -e "    ${RED}❌${NC} 包含用户信息列（user_id/uid/username）！"
                ((FAILED++))
            else
                echo -e "    ${GREEN}✅${NC} 无用户信息列"
                ((SUCCESS++))
            fi
        else
            echo -e "  ${YELLOW}⚠️${NC}  $file 不存在（可能尚未生成）"
        fi
    done
else
    echo -e "  ${YELLOW}⚠️${NC}  data/derived/ 目录不存在"
fi

echo ""
echo "4️⃣  检查 secrets/ 和 cookies"
echo "----------------------------------------"
if git check-ignore -q "secrets/" 2>/dev/null; then
    echo -e "  ${GREEN}✅${NC} secrets/ 被忽略"
    ((SUCCESS++))
else
    echo -e "  ${RED}❌${NC} secrets/ 未被忽略"
    ((FAILED++))
fi

if git check-ignore -q "cookies.json" 2>/dev/null; then
    echo -e "  ${GREEN}✅${NC} cookies*.json 被忽略"
    ((SUCCESS++))
else
    echo -e "  ${YELLOW}⚠️${NC}  cookies*.json 匹配规则（确保所有 cookie 文件被忽略）"
fi

echo ""
echo "5️⃣  检查开发文档"
echo "----------------------------------------"
DEV_DOCS=(
    "AGENTS.md"
    "DEVELOPMENT.md"
    "docs/progress.md"
)

for file in "${DEV_DOCS[@]}"; do
    if git check-ignore -q "$file" 2>/dev/null; then
        echo -e "  ${GREEN}✅${NC} $file 被忽略"
        ((SUCCESS++))
    else
        if [ -f "$file" ]; then
            echo -e "  ${RED}❌${NC} $file 未被忽略（存在且未忽略）"
            ((FAILED++))
        else
            echo -e "  ${YELLOW}⚠️${NC}  $file 未被忽略（但文件不存在）"
        fi
    fi
done

echo ""
echo "========================================="
echo "  验证结果"
echo "========================================="
echo -e "通过: ${GREEN}$SUCCESS${NC}"
echo -e "失败: ${RED}$FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ 所有检查通过！可以安全推送到 GitHub。${NC}"
    exit 0
else
    echo -e "${RED}❌ 发现 $FAILED 个问题，请修复后再推送！${NC}"
    echo ""
    echo "修复建议："
    echo "1. 检查 .gitignore 文件是否包含所有敏感文件规则"
    echo "2. 如果敏感文件已被追踪，使用 'git rm --cached <file>' 移除"
    echo "3. 检查数据文件是否包含用户信息"
    exit 1
fi

