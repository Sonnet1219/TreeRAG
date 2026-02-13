# Tree-KG Builder - Tree Structure Construction Spec

## 项目目标

构建一个 **Markdown Tree Builder**，将 Markdown 文档解析为层次化的树结构。利用 Markdown 标题（`#`, `##`, `###`）的天然层级关系，将文档组织为最多 3 层的树。每个节点包含标题、内容和 LLM 生成的摘要。此树结构将作为后续 Tree-KG RAG 系统的骨架。

**本阶段范围：仅构建 Tree 结构 + Summary 生成，不涉及 KG 构建。**

---

## 核心难点：标题层级推断

许多 Markdown 文档（尤其是从 PDF 转换的学术论文）的标题层级不规范。常见情况：

### Case 1: 标准 Markdown（`#` 数量正确反映层级）
```markdown
# 1 Introduction
## 1.1 Background
## 1.2 Motivation
### 1.2.1 Problem Statement
```

### Case 2: 扁平 Markdown（所有标题都用 `#`，但有编号）
```markdown
# 1 Introduction
# 1.1 Background
# 1.2 Motivation
# 1.2.1 Problem Statement
```

### Case 3: 无编号 Markdown
```markdown
# Introduction
## Background
## Motivation
```

**策略：编号优先，`#` 数量兜底。**
- 如果标题包含层次编号（如 `1.2.3`），用编号的深度（`.` 的数量 + 1）作为层级
- 如果没有编号，用 `#` 的数量作为层级
- 所有层级 cap 到 3（超过 3 的归为 3）

---

## 数据结构设计

### TreeNode

```python
@dataclass
class TreeNode:
    # 身份信息
    node_id: str              # 唯一标识, e.g., "doc1_1.2.3"
    heading: str              # 原始标题文本, e.g., "1.2.3 Adaptive Routing"
    level: int                # 推断出的真实层级 (0=root, 1, 2, 3)

    # 内容
    content: str              # 该节点下的原始文本（不含子节点文本）
    summary: str              # LLM 生成的摘要（或占位空字符串）

    # 树结构
    parent: Optional['TreeNode']
    children: List['TreeNode']

    # 检索辅助
    heading_path: str         # 完整路径, e.g., "Introduction > Background > Problem Statement"

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0
```

### DocumentTree

```python
@dataclass
class DocumentTree:
    doc_id: str
    root: TreeNode            # 虚拟根节点 (level=0)
    leaf_count: int
    node_count: int
```

---

## 算法设计

### Step 1: 标题解析 (HeadingParser)

```
输入: 一行 Markdown heading, e.g., "# 1.2.3 Adaptive Routing"
输出: (hash_count, numbering, clean_title, inferred_level)

流程:
1. 提取 `#` 数量 → hash_count
2. 对标题文本做正则匹配，尝试提取编号:
   - Pattern: r'^([\d]+(?:\.[\d]+)*)[\.\s\)\-]?\s*(.+)'  → 匹配 "1.2.3" 格式
   - Pattern: r'^([A-Z](?:\.[\d]+)*)[\.\s\)\-]?\s*(.+)'  → 匹配 "A.1.2" 格式
3. 如果匹配到编号:
   - numbering_depth = numbering.count('.') + 1
   - inferred_level = min(numbering_depth, 3)
4. 如果没有编号:
   - inferred_level = min(hash_count, 3)
```

### Step 2: Markdown 分段 (Section Parsing)

```
输入: 完整的 Markdown 文本
输出: List[Section]，每个 Section 包含 heading 信息和对应的 content

流程:
1. 按行遍历 Markdown
2. 遇到 heading 行（以 # 开头）→ 创建新 Section
3. 非 heading 行 → 追加到当前 Section 的 content
4. 每个 Section 记录: hash_count, heading_raw, numbering, inferred_level, content
```

### Step 3: 树构建 (Tree Building)

使用 **栈算法** 将扁平的 Section 列表构建为树：

```
输入: List[Section]
输出: TreeNode (root)

流程:
1. 创建虚拟根节点 root (level=0)
2. 初始化栈 stack = [root]
3. 遍历每个 Section:
   a. 创建对应的 TreeNode
   b. 从栈顶开始回退，直到 stack[-1].level < 当前节点的 level
   c. parent = stack[-1]
   d. 将当前节点加入 parent.children
   e. 设置 heading_path = parent.heading_path + " > " + heading
   f. 将当前节点压入栈
```

### Step 4: Summary 生成

**自底向上遍历**（后序遍历）生成摘要：

- **叶子节点**: 截取 content 前 200 字 → 送 LLM 总结为 1-2 句话
- **非叶子节点**: 汇总 children 的 summary → 送 LLM 总结为 1-2 句话

```
Prompt 模板（叶子节点）:
---
请用1-2句话总结以下章节的核心内容。
标题: {heading}
内容片段: {content[:200]}
---

Prompt 模板（非叶子节点）:
---
请用1-2句话总结以下章节的核心内容。
标题: {heading}
子章节摘要:
{children_summaries}
---
```

**注意**: Summary 生成需要 LLM API。Demo 阶段可以:
- 提供真实 LLM 调用的实现（支持 OpenAI / Anthropic API）
- 同时提供 mock 模式（直接截取前 100 字作为 summary）方便测试

---

## Demo 要求

### 输入
- 一个 Markdown 文件路径
- 模式选择: `mock`（不调用 LLM）或 `llm`（需要 API key）

### 输出
1. **终端可视化**: 以缩进的树形结构打印整棵树，每个节点显示:
   - heading
   - level
   - 是否叶子节点
   - summary（前 50 字）
   - content 字数

2. **JSON 导出**: 将整棵树序列化为 JSON 文件，结构如下:
```json
{
  "doc_id": "example",
  "node_count": 15,
  "leaf_count": 8,
  "tree": {
    "node_id": "root",
    "heading": "ROOT",
    "level": 0,
    "content": "",
    "summary": "",
    "heading_path": "",
    "is_leaf": false,
    "children": [
      {
        "node_id": "...",
        "heading": "1 Introduction",
        "level": 1,
        "heading_path": "1 Introduction",
        "is_leaf": false,
        "children": [...]
      }
    ]
  }
}
```

### 测试用例

请内置至少 2 个测试 Markdown 文件:

**Test 1: 标准层级（`#` 数量正确）**
```markdown
# Introduction
This is the introduction section with some content about the paper.

## Background
Background information about the research area.

## Motivation
Why this research is important.

### Problem Statement
The specific problem we address.

### Research Questions
The questions we aim to answer.

# Methods
Our methodology overview.

## Data Collection
How we collected data.

## Model Architecture
The model we designed.

### Encoder Design
Details about the encoder.

### Decoder Design
Details about the decoder.

# Experiments
Experimental setup and results.

# Conclusion
Summary and future work.
```

**Test 2: 扁平层级（所有用 `#`，靠编号区分）**
```markdown
# 1 Introduction
This is the introduction.

# 1.1 Background
Background details here.

# 1.2 Motivation
Why we do this research.

# 2 Methods
Methods overview.

# 2.1 Data Collection
How data was collected.

# 2.1.1 Dataset A
Details about dataset A.

# 2.1.2 Dataset B
Details about dataset B.

# 2.2 Model Architecture
The model architecture.

# 3 Experiments
Results and analysis.

# 4 Conclusion
Final remarks.
```

两个测试用例应该产生**结构相似的树**，验证层级推断的正确性。

---

## 技术栈

- Python 3.10+
- 无外部依赖（标准库即可完成解析和树构建）
- LLM 调用: 支持 `anthropic` 或 `openai` SDK（可选，mock 模式不需要）
- 输出: JSON + 终端打印

---

## 文件结构建议

```
tree_builder/
├── main.py                 # CLI 入口
├── parser.py               # HeadingParser + Markdown 分段
├── tree.py                 # TreeNode, DocumentTree, 树构建算法
├── summary.py              # Summary 生成（LLM / Mock）
├── visualizer.py           # 终端打印 + JSON 导出
├── test_data/
│   ├── test_standard.md    # 测试用例 1
│   └── test_flat.md        # 测试用例 2
└── README.md
```

---

## 预期输出示例

对于 Test 2 的扁平层级输入，终端打印应类似:

```
📄 Document Tree: test_flat (10 nodes, 6 leaves)
=====================================
📁 [L1] 1 Introduction (120 chars)
│   Summary: "本章介绍研究背景..."
│   ├── 🍃 [L2] 1.1 Background (85 chars) ← LEAF
│   │   Summary: "研究领域的背景信息..."
│   └── 🍃 [L2] 1.2 Motivation (90 chars) ← LEAF
│       Summary: "研究动机和重要性..."
📁 [L1] 2 Methods (60 chars)
│   Summary: "本章描述研究方法..."
│   ├── 📁 [L2] 2.1 Data Collection (50 chars)
│   │   Summary: "数据收集方法概述..."
│   │   ├── 🍃 [L3] 2.1.1 Dataset A (75 chars) ← LEAF
│   │   │   Summary: "数据集A的详细信息..."
│   │   └── 🍃 [L3] 2.1.2 Dataset B (80 chars) ← LEAF
│   │       Summary: "数据集B的详细信息..."
│   └── 🍃 [L2] 2.2 Model Architecture (95 chars) ← LEAF
│       Summary: "模型架构设计..."
🍃 [L1] 3 Experiments (110 chars) ← LEAF
│   Summary: "实验结果和分析..."
🍃 [L1] 4 Conclusion (70 chars) ← LEAF
    Summary: "总结和未来工作..."
```