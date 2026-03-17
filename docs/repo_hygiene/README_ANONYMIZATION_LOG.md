# README 匿名化修改记录

**修改时间**: 2026-01-16  
**目的**: 移除作者姓名、投稿信息和资助信息，使 README 更通用

---

## ✅ 已完成的修改

### 1. Paper 部分 → About 部分

**修改前**:
```markdown
## 📄 Paper

**Title**: Competing feedback channels drive phase transitions in collective emotion

**Authors**: Juan Li¹, Jinlin Wu²*, Zhihang Liu³*

**Affiliations**:
- ¹ School of Journalism and Communication, Lanzhou University
- ² Urban Governance and Design Thrust, HKUST (Guangzhou)
- ³ Institute of Space and Earth Information Science, CUHK

**Status**: Submitted to *Nature Communications* / *npj Complexity*

**Preprint**: [Coming soon]
```

**修改后**:
```markdown
## 📄 About

This repository contains the code and data for a research project on phase transitions in collective emotion on social media platforms.

**Preprint**: Coming soon
```

**移除内容**:
- ❌ 作者姓名
- ❌ 作者单位
- ❌ 投稿状态（Nature Communications / npj Complexity）

---

### 2. Citation 部分

**修改前**:
```bibtex
@article{li2026mixed,
  title={Competing feedback channels drive phase transitions in collective emotion},
  author={Li, Juan and Wu, Jinlin and Liu, Zhihang},
  journal={[Journal name]},
  year={2026},
  note={Under review}
}
```

**修改后**:
```bibtex
@software{mixed_dynamics,
  title={Mixed Feedback Dynamics in Collective Emotion},
  author={{Mixed Dynamics Team}},
  year={2026},
  url={https://github.com/wujlin/Mixed_dynamics}
}
```

**添加说明**:
```markdown
A paper describing this work is in preparation.
```

**移除内容**:
- ❌ 作者姓名
- ❌ "Under review" 状态

**替换内容**:
- ✓ 使用 `@software` 类型（更适合代码仓库）
- ✓ 作者改为 "Mixed Dynamics Team"
- ✓ 添加仓库 URL

---

### 3. Contact 部分

**修改前**:
```markdown
## 📧 Contact

**Corresponding Authors**:
- **Jinlin Wu**: jwu923@connect.hkust-gz.edu.cn
- **Zhihang Liu**: zhihangliu@cuhk.edu.hk

For questions about the code or data, please open an issue on GitHub or contact the corresponding authors.
```

**修改后**:
```markdown
## 📧 Contact

For questions about the code or data, please open an issue on GitHub.
```

**移除内容**:
- ❌ 通讯作者姓名
- ❌ 邮箱地址

---

### 4. Acknowledgments 部分

**修改前**:
```markdown
## 🙏 Acknowledgments

This study was funded by the National Social Science Foundation of China (Grant No. 24AXW005). The funder played no role in study design, data collection, analysis, or manuscript writing.

We thank the developers of:
- [NumPy](https://numpy.org/), ...
```

**修改后**:
```markdown
## 🙏 Acknowledgments

We thank the developers of:
- [NumPy](https://numpy.org/), ...
```

**移除内容**:
- ❌ 资助信息（National Social Science Foundation of China, Grant No. 24AXW005）

---

### 5. Roadmap 部分

**修改前**:
```markdown
- [ ] Release preprint (arXiv)
- [ ] Publish de-identified empirical data
- ...
```

**修改后**:
```markdown
- [ ] Publish de-identified empirical data
- ...
```

**移除内容**:
- ❌ "Release preprint (arXiv)" 条目

---

## 📊 修改总结

| 部分 | 修改类型 | 移除的内容 |
|---|---|---|
| **Paper → About** | 大幅简化 | 作者、单位、投稿状态 |
| **Citation** | 改为软件引用 | 作者姓名、期刊、审稿状态 |
| **Contact** | 移除联系方式 | 通讯作者姓名和邮箱 |
| **Acknowledgments** | 移除资助信息 | NSSFC 资助号 |
| **Roadmap** | 移除预印本计划 | arXiv 发布计划 |

---

## ✅ 验证

运行以下命令确认所有敏感信息已移除：

```bash
# 检查是否还有作者姓名
grep -E "Juan Li|Jinlin Wu|Zhihang Liu" README.md

# 检查是否还有投稿信息
grep -E "Nature Communications|npj Complexity|Under review" README.md

# 检查是否还有资助信息
grep -E "National Social Science Foundation|24AXW005" README.md

# 所有命令应该无输出 (No matches found)
```

---

## 📝 当前 README 状态

### 保留的内容（公开信息）
- ✓ 项目概述
- ✓ 仓库结构
- ✓ 快速开始（代码示例）
- ✓ 数据说明（隐私政策）
- ✓ 方法概述
- ✓ 关键结果
- ✓ 开发指南
- ✓ 通用引用格式
- ✓ MIT 许可证

### 移除的内容（敏感信息）
- ❌ 作者姓名和单位
- ❌ 投稿期刊和状态
- ❌ 通讯作者联系方式
- ❌ 资助信息
- ❌ 预印本发布计划

---

## 🔄 后续操作

### 论文接收后（可选）

如果论文被接收并发表，可以更新 README：

1. **恢复 Paper 部分**（可选）:
   ```markdown
   ## 📄 Paper
   
   This work has been published:
   
   **Citation**: [完整引用]
   
   **DOI**: [DOI链接]
   ```

2. **更新 Citation**:
   ```bibtex
   @article{...,
     title={...},
     author={...},
     journal={...},
     year={...},
     doi={...}
   }
   ```

3. **添加 Contact**（可选）:
   ```markdown
   For academic inquiries, please contact: [邮箱]
   ```

---

## 📋 提交建议

推荐的 commit message：

```bash
git add README.md
git commit -m "docs: anonymize README for pre-publication phase

- Remove author names and affiliations
- Remove submission status and journal names
- Remove funding information
- Simplify citation to software reference
- Remove author contact details

This ensures the repository is suitable for public sharing before paper acceptance."
```

---

**状态**: ✅ README 已完全匿名化，可以安全地公开分享

**最后更新**: 2026-01-16

