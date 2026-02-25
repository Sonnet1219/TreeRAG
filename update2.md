# TreeRAG — Robust Tree Builder 完整重构方案

## 问题概述

当前 Tree Builder 完全依赖正则匹配 `#` 数量和编号模式推断层级，无法应对真实世界中各种非标准的 Markdown 文档。本方案设计一个三层递进的层级推断系统：规则引擎（高置信度快速处理）→ 启发式推断（处理模糊场景）→ LLM 兜底（处理极端情况）。

---

## Part 1: 全部边界场景枚举

### Case 1: 标准 Markdown（基线，已支持）
```markdown
# 1 Introduction
## 1.1 Background
### 1.1.1 History
```
`#` 数量和编号深度一致，无歧义。

### Case 2: 扁平标题 + 数字编号（已部分支持）
```markdown
# 1 Introduction
# 1.1 Background
# 1.1.1 History
```
所有标题都用 `#`，但编号暗示层级。

### Case 3: 扁平标题 + 无编号
```markdown
# Introduction
# Background
# Motivation
# Methods
# Data Collection
# Model Architecture
# Experiments
# Conclusion
```
没有任何编号，所有标题都是 `#`，无法区分哪些是父、哪些是子。这是最难的情况。

### Case 4: 编号跳跃 / 不连续
```markdown
# 1 Introduction
# 3 Methods          ← 跳过了 2
# 3.1 Overview
# 3.3 Training       ← 跳过了 3.2
```
编号不连续但层级关系仍可推断。

### Case 5: `#` 层级跳跃
```markdown
# Introduction
### Detail A          ← 直接从 # 跳到 ###，缺少 ##
### Detail B
## Methods
```
中间层级缺失。

### Case 6: 混合编号格式
```markdown
# 1. Introduction
# 1.1 Background
# II. Related Work          ← 罗马数字
# A. Appendix               ← 字母编号
# A.1 Dataset Details
```
同一文档内多种编号风格混用。

### Case 7: 特殊固定章节（学术论文常见）
```markdown
# Abstract
# 1 Introduction
# 2 Methods
...
# 5 Conclusion
# Acknowledgments
# References
# Appendix A: Supplementary Results
# Appendix B: Proofs
```
Abstract、References、Acknowledgments、Appendix 等无编号但属于一级章节。

### Case 8: 代码块内的伪标题
````markdown
# Real Heading

```python
# This is a comment, not a heading
## Another comment
```

## Another Real Heading
````
代码块 fence 内的 `#` 不应被识别为标题。

### Case 9: Markdown 格式噪音
```markdown
#Introduction          ← # 后无空格（非标准但常见）
##  Background         ← 多余空格
# **1.1 Motivation**   ← 标题内含 bold 标记
# [2 Methods](#methods) ← 标题内含链接
# 3. Methods.          ← 末尾多余句号
```

### Case 10: 非标准编号模式
```markdown
# Section One
# Section Two
# 第一章 绪论           ← 中文编号
# 第二章 方法
# Chapter 3: Results
# Part IV: Discussion   ← 罗马数字 + Part
```

### Case 11: 层级反转 / 不一致
```markdown
## Overview             ← 文档以 ## 开头
## Background
### Details
## Methods
# Conclusion            ← 突然出现 #，比前面的 ## 更高
```
`#` 层级使用不一致。

### Case 12: 重复标题文本
```markdown
# Overview
## Methods
### Overview            ← 和顶层标题同名
## Results
### Summary
# Summary               ← 和子标题同名
```

### Case 13: 超深层级
```markdown
# 1 Introduction
## 1.1 Background
### 1.1.1 History
#### 1.1.1.1 Early Work       ← 超过 3 层
##### 1.1.1.1.1 Foundations   ← 第 5 层
```
超过 max_depth 限制的深层嵌套。

### Case 14: 纯 PDF 转换产物（无标题标记）
```markdown
Introduction

This paper presents...

1.1 Background

The field of...
```
没有 `#` 标记，标题靠独立短行 + 编号推断。

---

## Part 2: 三层递进架构

```
输入: Markdown 原文
         │
         ▼
   ┌─────────────────┐
   │  Layer 1: 预处理  │  清洗噪音、识别代码块、标准化格式
   │  (确定性规则)     │  
   └────────┬────────┘
            │ 干净的 heading 列表
            ▼
   ┌─────────────────┐
   │  Layer 2: 规则   │  编号解析、# 数量、特殊章节识别
   │  + 启发式推断    │  每个 heading 产出 (inferred_level, confidence)
   └────────┬────────┘
            │ 带置信度的层级列表
            ▼
   ┌─────────────────┐
   │  Layer 3: LLM   │  仅对 confidence < 阈值的 heading 调用 LLM
   │  辅助修正        │  修正层级、补全结构
   └────────┬────────┘
            │ 最终确定的层级列表
            ▼
   ┌─────────────────┐
   │  树构建 + 验证    │  栈算法构建树 + 结构完整性校验
   └─────────────────┘
```

---

## Part 3: Layer 1 — 预处理

### 3.1 代码块过滤

识别 fenced code blocks（``` 或 ~~~），标记其行范围，后续解析跳过这些行。

```python
def mark_code_blocks(lines: List[str]) -> Set[int]:
    """返回属于代码块内部的行号集合"""
    code_lines = set()
    in_code = False
    fence_pattern = re.compile(r'^(`{3,}|~{3,})')

    for i, line in enumerate(lines):
        if fence_pattern.match(line.strip()):
            if in_code:
                code_lines.add(i)  # 闭合行也标记
                in_code = False
            else:
                code_lines.add(i)
                in_code = True
        elif in_code:
            code_lines.add(i)

    return code_lines
```

### 3.2 标题行标准化

清洗各种格式噪音，提取干净的标题信息：

```python
def normalize_heading(raw_line: str) -> Optional[dict]:
    """
    输入: 原始行文本
    输出: {
        "hash_count": int,           # 原始 # 数量
        "raw_text": str,             # 清洗后的标题文本
        "has_hash_marker": bool,     # 是否有 # 标记
    }
    """
    # 匹配标准 ATX 标题: # 后必须有空格（或 # 后无空格但紧跟字母——宽松模式）
    # 标准: r'^(#{1,6})\s+(.+)'
    # 宽松: r'^(#{1,6})(.+)'  当标准匹配失败时尝试
    
    # 清洗操作:
    # 1. 去除 bold/italic 标记: **text** → text, *text* → text
    # 2. 去除链接: [text](url) → text
    # 3. 去除末尾多余标点: "3. Methods." → "3. Methods"
    # 4. 去除首尾空白
    # 5. 去除末尾的 # (ATX closing): "## Title ##" → "Title"
```

### 3.3 无 `#` 标记的标题检测（Case 14）

对于没有 `#` 的行，用启发式判断是否为标题：

```python
def detect_unmarked_heading(line: str, prev_line: str, next_line: str) -> bool:
    """
    无 # 标记的潜在标题检测
    特征:
    - 独立短行（< 80 字符）
    - 前后有空行
    - 以数字编号开头（如 "1.1 Background"）
    - 或匹配已知章节名（如 "Introduction", "Methods"）
    - 不含句号结尾（标题通常不以句号结尾）
    - 不含逗号（标题通常不含逗号）
    """
```

---

## Part 4: Layer 2 — 规则 + 启发式推断

### 4.1 信号提取

对每个识别到的 heading，提取多种层级信号：

```python
@dataclass
class HeadingSignals:
    """从单个 heading 中提取的所有层级信号"""

    # --- 信号 1: # 数量 ---
    hash_count: int                    # 原始 # 数量 (0 if no # marker)
    has_hash_marker: bool              # 是否有 # 标记

    # --- 信号 2: 编号模式 ---
    numbering: Optional[str]           # 原始编号, e.g., "1.2.3", "A.1", "IV"
    numbering_type: Optional[str]      # "arabic" | "roman" | "letter" | "chinese" | None
    numbering_depth: int               # 编号层级深度 (0 if no numbering)
    # "1" → 1, "1.2" → 2, "1.2.3" → 3, "A" → 1, "A.1" → 2

    # --- 信号 3: 特殊章节 ---
    is_special_section: bool           # Abstract, References, Acknowledgments, Appendix 等
    special_section_level: int         # 特殊章节的默认层级 (通常为 1)

    # --- 信号 4: 文本特征 ---
    text_length: int                   # 标题文本长度
    heading_text: str                  # 清洗后的纯标题文本（去除编号）
```

### 4.2 编号解析器（扩展版）

覆盖各种编号格式：

```python
NUMBERING_PATTERNS = [
    # 阿拉伯数字: "1", "1.2", "1.2.3"
    (re.compile(r'^(\d+(?:\.\d+)*)[\.\s\)\:\-]?\s*(.+)'), "arabic"),

    # 字母编号: "A", "A.1", "A.1.2"
    (re.compile(r'^([A-Z](?:\.\d+)*)[\.\s\)\:\-]?\s*(.+)'), "letter"),

    # 罗马数字: "I", "II", "IV", "XI"
    (re.compile(r'^((?:X{0,3})(?:IX|IV|V?I{0,3}))[\.\s\)\:\-]\s*(.+)', re.IGNORECASE), "roman"),

    # 中文编号: "第一章", "第二节"
    (re.compile(r'^第([一二三四五六七八九十百]+)[章节部分篇]\s*(.*)'), "chinese"),

    # Chapter/Part/Section 前缀: "Chapter 3", "Part II", "Section 4.1"
    (re.compile(r'^(?:Chapter|Part|Section)\s+(.+?)[\.\:\s]\s*(.+)', re.IGNORECASE), "prefix"),

    # Appendix: "Appendix A", "Appendix B.1"
    (re.compile(r'^Appendix\s+([A-Z](?:\.\d+)*)[\.\:\s]?\s*(.*)', re.IGNORECASE), "appendix"),
]
```

### 4.3 特殊章节识别

```python
SPECIAL_SECTIONS = {
    # name pattern → default level
    "abstract": 1,
    "摘要": 1,
    "introduction": 1,
    "引言": 1,
    "绪论": 1,
    "related work": 1,
    "background": 1,       # 注意: 可能是 1 级也可能是 2 级，需要上下文判断
    "methodology": 1,
    "methods": 1,
    "method": 1,
    "approach": 1,
    "experiments": 1,
    "evaluation": 1,
    "results": 1,
    "discussion": 1,
    "conclusion": 1,
    "conclusions": 1,
    "summary": 1,
    "acknowledgments": 1,
    "acknowledgements": 1,
    "references": 1,
    "bibliography": 1,
    "appendix": 1,
    "supplementary": 1,
    "future work": 1,      # 可能 1 级也可能 2 级
}

def match_special_section(heading_text: str) -> Optional[int]:
    """模糊匹配特殊章节名，返回默认层级或 None"""
    normalized = heading_text.lower().strip()
    for pattern, level in SPECIAL_SECTIONS.items():
        if normalized == pattern or normalized.startswith(pattern):
            return level
    return None
```

### 4.4 层级推断规则引擎

综合所有信号，按优先级推断层级并给出置信度：

```python
def infer_level(signals: HeadingSignals, context: 'DocumentContext') -> Tuple[int, float]:
    """
    返回 (inferred_level, confidence)
    confidence: 0.0 ~ 1.0，低于阈值(如 0.6)时触发 LLM 修正
    """

    # =============================================
    # Rule 1: 编号深度（最高优先级，最可靠）
    # =============================================
    if signals.numbering_depth > 0:
        level = min(signals.numbering_depth, MAX_DEPTH)

        if signals.has_hash_marker and signals.hash_count == level:
            # 编号和 # 数量一致 → 最高置信
            return level, 1.0
        elif signals.has_hash_marker and signals.hash_count != level:
            # 编号和 # 数量不一致 → 信任编号（常见于扁平 markdown）
            return level, 0.9
        else:
            # 有编号无 # → 信任编号
            return level, 0.85

    # =============================================
    # Rule 2: 特殊章节名（无编号时的强信号）
    # =============================================
    if signals.is_special_section:
        level = signals.special_section_level
        # 如果 # 数量与特殊章节默认层级一致，置信度更高
        if signals.has_hash_marker and signals.hash_count == level:
            return level, 0.9
        elif signals.has_hash_marker:
            return level, 0.7  # # 数量和预期不一致，降低置信
        else:
            return level, 0.75

    # =============================================
    # Rule 3: 纯 # 数量（无编号、非特殊章节）
    # =============================================
    if signals.has_hash_marker:
        level = min(signals.hash_count, MAX_DEPTH)

        # 检查是否和上下文中的其他 heading 一致
        consistency = context.check_hash_consistency()
        if consistency == "consistent":
            # 文档中 # 数量使用一致 → 较高置信
            return level, 0.8
        elif consistency == "all_same":
            # 所有标题都用同一个 # 数量（如全是 #）→ 低置信，无法区分层级
            return level, 0.3  # 需要 LLM 介入
        else:
            return level, 0.5  # 部分不一致，中等置信

    # =============================================
    # Rule 4: 无 # 标记、无编号（最低置信）
    # =============================================
    return 1, 0.2  # 几乎一定需要 LLM 介入
```

### 4.5 上下文感知推断（DocumentContext）

某些层级判断需要全局上下文：

```python
class DocumentContext:
    """收集文档级别的全局信号，辅助单个 heading 的层级推断"""

    def __init__(self, all_headings: List[HeadingSignals]):
        self.all_headings = all_headings

        # 统计 # 使用模式
        self.hash_distribution = Counter(h.hash_count for h in all_headings if h.has_hash_marker)

        # 统计编号使用模式
        self.has_any_numbering = any(h.numbering_depth > 0 for h in all_headings)
        self.numbering_coverage = sum(1 for h in all_headings if h.numbering_depth > 0) / len(all_headings)

    def check_hash_consistency(self) -> str:
        """
        检查 # 数量的使用是否一致
        返回:
        - "consistent": 多种 # 层级且使用合理
        - "all_same": 所有标题用同一 # 数量（如全是 #）
        - "inconsistent": 使用混乱
        """
        if len(self.hash_distribution) == 1:
            return "all_same"
        elif len(self.hash_distribution) >= 2:
            return "consistent"
        else:
            return "inconsistent"

    def get_dominant_numbering_type(self) -> Optional[str]:
        """获取文档中最主要的编号类型"""
        types = [h.numbering_type for h in self.all_headings if h.numbering_type]
        if not types:
            return None
        return Counter(types).most_common(1)[0][0]
```

---

## Part 5: Layer 3 — LLM 辅助修正

### 5.1 何时触发 LLM

```python
LLM_CONFIDENCE_THRESHOLD = 0.6

def needs_llm_correction(headings_with_levels: List[Tuple[HeadingSignals, int, float]]) -> bool:
    """判断是否需要 LLM 介入"""
    low_confidence_count = sum(1 for _, _, conf in headings_with_levels if conf < LLM_CONFIDENCE_THRESHOLD)
    low_confidence_ratio = low_confidence_count / len(headings_with_levels)

    # 条件 1: 超过 30% 的 heading 低置信
    if low_confidence_ratio > 0.3:
        return True

    # 条件 2: 全部 heading 用同一 # 数量且无编号
    if all(h.hash_count == headings_with_levels[0][0].hash_count for h, _, _ in headings_with_levels):
        if not any(h.numbering_depth > 0 for h, _, _ in headings_with_levels):
            return True

    # 条件 3: 存在层级跳跃（如 1 → 3，跳过了 2）
    levels = [lv for _, lv, _ in headings_with_levels]
    for i in range(1, len(levels)):
        if levels[i] - levels[i-1] > 1:  # 向下跳跃超过 1 级
            return True

    return False
```

### 5.2 LLM 修正模式

有两种调用模式，根据情况选择：

#### 模式 A: 全量结构推断（低置信比例高时使用）

把所有 heading 一次性发给 LLM，让它推断完整的层级结构：

```
你是一个文档结构分析专家。以下是从一篇文档中提取的所有章节标题（按出现顺序）。
请为每个标题推断其在文档中的层级（1=一级标题, 2=二级标题, 3=三级标题）。

推断依据:
1. 标题的编号模式（如 "1.2.3" 暗示三级标题）
2. 标题的内容语义（如 "Introduction" 通常是一级标题）
3. 标题之间的逻辑关系（如 "Background" 通常从属于 "Introduction"）
4. 文档的整体结构模式

标题列表:
{headings_list}

对每个标题你还可以参考以下规则引擎的初步推断结果和置信度：
{rule_based_results}

输出严格 JSON:
[
  {{"index": 0, "heading": "...", "level": 1, "reasoning": "..."}},
  {{"index": 1, "heading": "...", "level": 2, "reasoning": "..."}},
  ...
]
```

#### 模式 B: 局部修正（少量低置信时使用）

只把低置信的 heading 及其上下文发给 LLM：

```
你是一个文档结构分析专家。以下文档结构中有几个标题的层级不确定（标记为 [?]）。
请根据上下文推断它们的正确层级。

文档结构（已确定的部分）:
[L1] 1 Introduction
[L2] 1.1 Background
[?]  Background Details        ← 需要推断
[L2] 1.2 Motivation
[L1] 2 Methods
[?]  Data Preprocessing        ← 需要推断
[L2] 2.1 Model Architecture

对标记为 [?] 的标题，输出其层级:
[
  {{"heading": "Background Details", "level": 3, "reasoning": "从属于 1.1 Background"}},
  {{"heading": "Data Preprocessing", "level": 2, "reasoning": "与 2.1 并列，同属 2 Methods"}}
]
```

#### 选择逻辑

```python
def select_llm_mode(headings_with_levels):
    low_conf_count = sum(1 for _, _, c in headings_with_levels if c < LLM_CONFIDENCE_THRESHOLD)
    total = len(headings_with_levels)

    if low_conf_count / total > 0.5:
        return "full"       # 超过一半不确定 → 全量推断
    else:
        return "partial"    # 少量不确定 → 局部修正
```

### 5.3 LLM 结果合并

```python
def merge_llm_corrections(
    rule_results: List[Tuple[HeadingSignals, int, float]],
    llm_results: List[dict]
) -> List[Tuple[HeadingSignals, int, float]]:
    """
    将 LLM 的推断结果合并回规则引擎的结果
    
    策略:
    - 对于 confidence >= threshold 的 heading: 保留规则引擎结果
    - 对于 confidence < threshold 的 heading: 采用 LLM 结果，置信度设为 0.85
    - 如果 LLM 结果和规则引擎冲突且规则引擎置信度较高（>= 0.8），保留规则引擎
    """
```

---

## Part 6: 树构建 + 后处理验证

### 6.1 栈算法构建树（与之前一致）

层级确定后，使用栈算法构建树。

### 6.2 结构验证 + 自动修复

```python
def validate_and_fix_tree(root: TreeNode) -> List[str]:
    """
    验证树结构的合理性，自动修复常见问题
    返回修复日志
    """
    fixes = []

    # Check 1: 层级跳跃修复
    # 如果某个节点的 level 比 parent.level 大 2 以上，插入虚拟中间节点
    # 例如: L1 → L3（缺少 L2），插入一个 "[Inferred Section]" L2 节点
    for node in traverse_all(root):
        if node.parent and node.level > node.parent.level + 1:
            gap = node.level - node.parent.level - 1
            fixes.append(f"Level gap detected: {node.heading} (L{node.level}) under {node.parent.heading} (L{node.parent.level})")
            # 不插入虚拟节点，而是将 node 的 level 下调
            node.level = node.parent.level + 1
            fixes.append(f"  → Adjusted to L{node.level}")

    # Check 2: 超深层级截断
    for node in traverse_all(root):
        if node.level > MAX_DEPTH:
            old_level = node.level
            node.level = MAX_DEPTH
            fixes.append(f"Depth overflow: {node.heading} L{old_level} → L{MAX_DEPTH}")

    # Check 3: 孤儿节点检测
    # 如果某个节点没有 parent（除 root 外），挂到 root 下
    for node in traverse_all(root):
        if node.parent is None and node != root:
            node.parent = root
            root.children.append(node)
            node.level = 1
            fixes.append(f"Orphan node adopted: {node.heading}")

    # Check 4: 空节点剪枝
    # 如果某个非叶子节点既没有 content 也没有 children，删除
    for node in list(traverse_all(root)):
        if not node.is_leaf and not node.children and not node.content.strip():
            if node.parent:
                node.parent.children.remove(node)
                fixes.append(f"Empty node pruned: {node.heading}")

    return fixes
```

---

## Part 7: 完整构建主流程

```python
def build_robust_tree(markdown_text: str, doc_id: str, llm_client=None) -> Tuple[TreeNode, dict]:
    """
    返回: (root, build_report)
    build_report 包含构建过程的详细日志
    """
    lines = markdown_text.split('\n')
    report = {"warnings": [], "fixes": [], "llm_used": False}

    # ===== Layer 1: 预处理 =====
    code_block_lines = mark_code_blocks(lines)
    raw_headings = []
    for i, line in enumerate(lines):
        if i in code_block_lines:
            continue
        heading = normalize_heading(line)
        if heading:
            raw_headings.append(heading)
        elif detect_unmarked_heading(line, ...):
            raw_headings.append({"hash_count": 0, "raw_text": line.strip(), "has_hash_marker": False})

    # ===== Layer 2: 信号提取 + 规则推断 =====
    signals_list = [extract_signals(h) for h in raw_headings]
    context = DocumentContext(signals_list)

    rule_results = []  # List of (signals, level, confidence)
    for signals in signals_list:
        level, conf = infer_level(signals, context)
        rule_results.append((signals, level, conf))

    # ===== Layer 3: LLM 修正（按需） =====
    if needs_llm_correction(rule_results) and llm_client is not None:
        report["llm_used"] = True
        mode = select_llm_mode(rule_results)

        if mode == "full":
            llm_levels = llm_infer_full_structure(raw_headings, rule_results, llm_client)
        else:
            llm_levels = llm_infer_partial(raw_headings, rule_results, llm_client)

        final_results = merge_llm_corrections(rule_results, llm_levels)
    else:
        final_results = rule_results

    # ===== 构建树 =====
    sections = build_sections(lines, raw_headings, final_results, code_block_lines)
    root = build_tree_from_sections(sections, doc_id)

    # ===== 后处理 =====
    report["fixes"] = validate_and_fix_tree(root)

    return root, report
```

---

## Part 8: 与现有 Pipeline 的集成

新的 `build_robust_tree` 替换原有的 `build_tree`，后续流程不变：

```python
def build_document(markdown_text: str, doc_id: str, llm_client) -> TreeNode:
    # Step 1: 构建树（升级版）
    root, report = build_robust_tree(markdown_text, doc_id, llm_client)
    print_build_report(report)  # 输出构建日志

    # Step 2: 生成 summary（含 Case 2 改进）
    generate_summaries(root, llm_client)

    # Step 3: 注入 preamble
    inject_preamble_leaves(root)

    # Step 4: preamble summary
    generate_preamble_summaries(root, llm_client)

    return root
```

---

## Part 9: 终端构建日志输出

```
============================================================
🔨 Building Tree: test_paper.md
============================================================

>>> Layer 1: Preprocessing
  Lines: 156, Code blocks: 2 (filtered 12 lines)
  Headings detected: 18 (16 with # marker, 2 unmarked)

>>> Layer 2: Rule-based Inference
  High confidence (>= 0.8): 12/18 headings
  Medium confidence (0.6-0.8): 3/18 headings
  Low confidence (< 0.6): 3/18 headings
  ⚠ Low confidence headings:
    "Background Details" → L1 (conf=0.3, reason: no numbering, all_same #)
    "Data Preprocessing" → L1 (conf=0.3, reason: no numbering, all_same #)
    "Supplementary"      → L1 (conf=0.5, reason: special section but ambiguous)

>>> Layer 3: LLM Correction (partial mode)
  Corrected 3 headings:
    "Background Details"  L1 → L3 (reasoning: sub-topic of 1.1 Background)
    "Data Preprocessing"  L1 → L2 (reasoning: parallel to 2.1 Model Architecture)
    "Supplementary"       L1 → L1 (reasoning: confirmed as top-level appendix)

>>> Post-processing
  Fixes applied: 0
  Warnings: 0

>>> Result
  Nodes: 18, Leaves: 12
  Max depth: 3
  LLM calls: 1
============================================================
```

---

## Part 10: 测试用例矩阵

创建以下测试文件，验证各种边界情况：

| 文件名 | 覆盖的 Case | 关键验证点 |
|--------|------------|-----------|
| `test_standard.md` | Case 1 | 基线，# 和编号一致 |
| `test_flat_numbered.md` | Case 2, 4 | 全 # + 编号，编号跳跃 |
| `test_flat_no_number.md` | Case 3 | 全 # + 无编号（需要 LLM） |
| `test_mixed_numbering.md` | Case 6, 7 | 罗马数字 + 字母 + 特殊章节 |
| `test_noisy.md` | Case 8, 9 | 代码块伪标题 + 格式噪音 |
| `test_level_jump.md` | Case 5, 11 | 层级跳跃 + 层级反转 |
| `test_deep.md` | Case 13 | 超过 3 层的深层嵌套 |
| `test_chinese.md` | Case 10 | 中文编号 + 中文章节名 |

每个测试文件 30-50 行即可，附带预期的树结构（node_count, leaf_count, 各节点 level）作为断言。

---

## 文件结构

```
tree_builder/
├── preprocessor.py          # Layer 1: 代码块过滤、标题标准化、无标记标题检测
├── signals.py               # HeadingSignals 数据结构 + 信号提取
├── numbering.py             # 编号解析器（全格式支持）
├── special_sections.py      # 特殊章节识别
├── rule_engine.py           # Layer 2: 规则推断引擎 + DocumentContext
├── llm_corrector.py         # Layer 3: LLM 修正（full / partial 两种模式）
├── tree.py                  # 栈算法建树 + 后处理验证 (已有，需修改)
├── preamble.py              # Preamble 注入 (新增)
├── summary.py               # Summary 生成（含 Case 2 改进）(已有，需修改)
├── builder.py               # 主入口: build_robust_tree + build_document
├── visualizer.py            # 终端打印 + 构建日志 (已有，需修改)
└── test_data/
    ├── test_standard.md
    ├── test_flat_numbered.md
    ├── test_flat_no_number.md
    ├── test_mixed_numbering.md
    ├── test_noisy.md
    ├── test_level_jump.md
    ├── test_deep.md
    └── test_chinese.md
```

---

## 技术依赖

- Python 3.10+ 标准库（re, dataclasses, json, collections）
- LLM 调用: `anthropic` 或 `openai` SDK（仅 Layer 3 需要，可选）
- 无其他外部依赖

## Mock 模式

当 `llm_client=None` 时，Layer 3 整体跳过，仅使用 Layer 1 + Layer 2 的结果。对于低置信的 heading，在 report 中标记 warning 但不修正。