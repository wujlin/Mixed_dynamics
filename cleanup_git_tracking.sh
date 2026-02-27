#!/bin/bash
# Git 敏感文件清理脚本 (Bash 版本)
# 用途: 从 Git 追踪中移除敏感文件（但保留本地文件）

echo "========================================="
echo "  Git 敏感文件清理脚本"
echo "========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查是否在 git 仓库中
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo -e "${RED}错误: 当前目录不是 Git 仓库！${NC}"
    exit 1
fi

echo -e "${YELLOW}⚠️  警告: 此脚本将从 Git 追踪中移除以下文件/文件夹:${NC}"
echo ""
echo "  - Essay/"
echo "  - Essay_nc/"
echo "  - dataset/"
echo "  - secrets/"
echo "  - outputs/figs/"
echo "  - outputs/annotations/"
echo "  - outputs/llm_validation/"
echo "  - AGENTS.md"
echo "  - DEVELOPMENT.md"
echo "  - docs/progress.md"
echo "  - docs/review.md"
echo "  - Manuscript*.pdf"
echo "  - Manuscript*.docx"
echo ""
echo -e "${GREEN}✓ 本地文件不会被删除，仅从 Git 追踪中移除${NC}"
echo ""

# 询问用户确认
read -p "继续执行? (y/N): " confirm
if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "操作已取消"
    exit 0
fi

echo ""
echo "========================================="
echo "  Step 1: 从 Git 追踪中移除文件"
echo "========================================="

# 移除文件夹
folders_to_remove=(
    "Essay/"
    "Essay_nc/"
    "dataset/"
    "secrets/"
    "outputs/figs/"
    "outputs/annotations/"
    "outputs/llm_validation/"
)

for folder in "${folders_to_remove[@]}"; do
    if git ls-files "$folder" > /dev/null 2>&1; then
        echo -e "  移除文件夹: ${YELLOW}$folder${NC}"
        git rm --cached -r "$folder" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo -e "    ${GREEN}✓${NC} 成功"
        else
            echo -e "    ${YELLOW}⚠${NC}  跳过（未追踪或已移除）"
        fi
    else
        echo -e "  跳过文件夹: ${YELLOW}$folder${NC} (未被 Git 追踪)"
    fi
done

# 移除单个文件
files_to_remove=(
    "AGENTS.md"
    "DEVELOPMENT.md"
    "docs/progress.md"
    "docs/review.md"
    "docs/email_pi_empirical_update.md"
    "docs/writing_drills.md"
    "Manuscript.pdf"
    "Manuscript.docx"
    "Google Gemini.pdf"
)

for file in "${files_to_remove[@]}"; do
    if git ls-files "$file" > /dev/null 2>&1; then
        echo -e "  移除文件: ${YELLOW}$file${NC}"
        git rm --cached "$file" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo -e "    ${GREEN}✓${NC} 成功"
        else
            echo -e "    ${YELLOW}⚠${NC}  跳过（未追踪或已移除）"
        fi
    else
        echo -e "  跳过文件: ${YELLOW}$file${NC} (未被 Git 追踪)"
    fi
done

echo ""
echo "========================================="
echo "  Step 2: 检查暂存区状态"
echo "========================================="

if git diff --cached --quiet; then
    echo -e "${YELLOW}⚠️  没有文件被移除（可能这些文件未被追踪）${NC}"
    echo ""
    echo "检查 .gitignore 是否正确配置："
    echo "  cat .gitignore | grep Essay"
    exit 0
else
    echo -e "${GREEN}✓ 已暂存的更改:${NC}"
    git diff --cached --name-only | head -20
    echo ""
fi

echo "========================================="
echo "  Step 3: 提交更改"
echo "========================================="

read -p "是否提交这些更改? (y/N): " commit_confirm
if [[ ! $commit_confirm =~ ^[Yy]$ ]]; then
    echo "操作已取消，使用 'git reset' 可以撤销暂存的更改"
    exit 0
fi

git commit -m "chore: remove sensitive files from git tracking

- Remove Essay/ and Essay_nc/ (manuscript drafts)
- Remove dataset/ (contains user data)
- Remove secrets/ (credentials)
- Remove AGENTS.md, DEVELOPMENT.md (internal docs)
- Remove outputs/figs/, outputs/annotations/ (generated files)
- These files are now properly ignored by .gitignore
- Local files are preserved"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 提交成功${NC}"
else
    echo -e "${RED}✗ 提交失败${NC}"
    exit 1
fi

echo ""
echo "========================================="
echo "  Step 4: 推送到 GitHub"
echo "========================================="

read -p "是否推送到 GitHub? (y/N): " push_confirm
if [[ ! $push_confirm =~ ^[Yy]$ ]]; then
    echo "跳过推送。稍后可以手动执行: git push origin main"
    exit 0
fi

echo "正在推送..."
git push origin main

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 推送成功！${NC}"
    echo ""
    echo "请等待 1-2 分钟后刷新 GitHub 页面，敏感文件应该已经消失。"
else
    echo -e "${RED}✗ 推送失败${NC}"
    echo "请检查网络连接和权限，然后手动执行: git push origin main"
    exit 1
fi

echo ""
echo "========================================="
echo "  ✅ 清理完成！"
echo "========================================="
echo ""
echo "验证步骤:"
echo "  1. 刷新 GitHub 页面，确认敏感文件夹已消失"
echo "  2. 检查本地文件是否仍然存在: ls Essay/ dataset/"
echo "  3. 检查 git 状态: git status --ignored"
echo ""
echo -e "${YELLOW}注意: 这些文件仍存在于 Git 历史记录中。${NC}"
echo "如果需要完全清除历史（例如泄露了密钥），请查看 GITHUB_CLEANUP_REPORT.md"
echo ""

