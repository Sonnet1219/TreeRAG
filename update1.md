# TreeRAG — Preamble Node 注入 & Summary 算法升级

## 问题描述

当前 Tree Builder 存在一个内容覆盖盲区：非叶子节点的直属内容（即标题和子标题之间的文本）不会被任何叶子节点承载，导致 indexing 阶段完全丢失这部分内容。

```markdown
# 3 Methods                        ← 非叶子节点
这里有一段方法论概述内容...            ← ❌ 丢失！无叶子节点承载

## 3.1 Overview                     ← 非叶子节点
这里有一段 overview 的引言...         ← ❌ 丢失！

### 3.1.1 Detail A                  ← 叶子节点 ✅
细节A的内容...

### 3.1.2 Detail B                  ← 叶子节点 ✅
细节B的内容...
```

这种"父标题下有直属正文，然后才展开子标题"的模式在学术论文中非常普遍。

## 解决方案

### 核心思路

对每个"有直属 content 的非叶子节点"，创建一个**虚拟 Preamble 叶子节点**，作为该节点的第一个子节点，专门承载这段悬空内容。同时调整 summary 生成的执行顺序和算法，确保 parent 自身的 content 在生成 summary 时被充分利用。

### 处理后效果

```
# 3 Methods                         (非叶子)
├── 🍃 [3_preamble] "3 Methods (Preamble)"      ← 新增，承载"方法论概述"
├── ## 3.1 Overview                  (非叶子)
│   ├── 🍃 [3.1_preamble] "3.1 Overview (Preamble)" ← 新增，承载"overview引言"
│   ├── 🍃 ### 3.1.1 Detail A
│   └── 🍃 ### 3.1.2 Detail B
└── 🍃 ## 3.2 Architecture
```

所有内容都有叶子节点承载，后续 indexing（chunk + embedding + BM25）无需任何改动。

---

## 实现计划

### 执行顺序（关键！）

```
Step 1: Build Tree             ← 已有，不动
Step 2: Generate Summaries     ← 修改算法（此时 parent 还保留 content）
Step 3: Inject Preamble        ← 新增（创建虚拟叶子，转移 content）
Step 4: Preamble Summary       ← 新增（单独为 preamble 生成 summary）
```

**必须先生成 summary 再注入 preamble**，因为 parent 自身的 content 是生成高质量 summary 的最佳素材。如果先注入 preamble（清空 parent content），parent 的 summary 就只能靠子节点摘要聚合，质量会下降。

---

### Step 2: Summary 生成算法（修改）

自底向上（后序遍历）生成 summary，根据节点类型分三种情况处理：

```python
def generate_summaries(root: TreeNode, llm_client):
    """
    自底向上生成 summary
    此函数在 inject_preamble 之前调用，parent 节点仍保留直属 content
    """
    for node in post_order_traverse(root):
        if node.level == 0:  # skip virtual root
            continue

        if node.is_leaf:
            # Case 1: 叶子节点 → 用自身 content 生成 summary
            node.summary = llm_client.summarize(
                f"标题: {node.heading}\n"
                f"内容: {node.content[:200]}"
            )

        elif node.content.strip():
            # Case 2: 非叶子 + 有直属 content
            # → 同时利用自身 content（作为本章概述）和 children summaries
            # → 自身 content 是更直接的 summary 素材，优先级更高
            children_summaries = '\n'.join(
                f"- {c.heading}: {c.summary}" for c in node.children
            )
            node.summary = llm_client.summarize(
                f"标题: {node.heading}\n"
                f"本章节概述: {node.content[:200]}\n"
                f"包含以下子章节:\n{children_summaries}"
            )

        else:
            # Case 3: 非叶子 + 无直属 content → 纯靠 children summaries 聚合
            children_summaries = '\n'.join(
                f"- {c.heading}: {c.summary}" for c in node.children
            )
            node.summary = llm_client.summarize(
                f"标题: {node.heading}\n"
                f"包含以下子章节:\n{children_summaries}"
            )
```

**Mock 模式**：与之前一致，叶子节点截取 content 前 100 字，非叶子节点拼接 children 的 summary 前 50 字。Case 2 的 mock 可以优先用自身 content 前 100 字。

---

### Step 3: Preamble 注入（新增）

在 summary 生成完成后执行。遍历所有非叶子节点，如果有直属 content，创建虚拟 preamble 叶子节点。

```python
def inject_preamble_leaves(root: TreeNode):
    """
    后序遍历：对每个有直属 content 的非叶子节点，
    创建虚拟 preamble 叶子节点承载其 content，
    插入为该节点的第一个子节点，然后清空该节点的 content。
    """
    for node in post_order_traverse(root):
        # 跳过叶子节点（无需处理）
        if node.is_leaf:
            continue
        # 跳过没有直属 content 的非叶子节点
        if not node.content.strip():
            continue

        # 创建 preamble 叶子节点
        preamble = TreeNode(
            node_id=f"{node.node_id}_preamble",
            heading=f"{node.heading} (Preamble)",
            level=node.level + 1,
            content=node.content,          # 转移 content
            summary="",                    # Step 4 单独生成
            parent=node,
            children=[],                   # 叶子节点，无子节点
            heading_path=f"{node.heading_path} > Preamble",
        )

        # 插入为第一个子节点
        node.children.insert(0, preamble)

        # 清空父节点的 content（已转移）
        node.content = ""
```

**关键细节**：
- `level = node.level + 1`：preamble 在层级上低于父节点一级
- `children.insert(0, preamble)`：插入为第一个子节点，保持"概述在前、细节在后"的语义顺序
- 清空 `node.content`：避免内容重复（summary 已经在 Step 2 中利用过了）

---

### Step 4: Preamble Summary 生成（新增）

单独为新创建的 preamble 节点生成 summary。

```python
def generate_preamble_summaries(root: TreeNode, llm_client):
    """
    遍历所有 preamble 节点，为其生成 summary
    在 inject_preamble_leaves 之后调用
    """
    for node in traverse_all(root):
        if not node.node_id.endswith("_preamble"):
            continue

        node.summary = llm_client.summarize(
            f"标题: {node.heading}\n"
            f"内容: {node.content[:200]}"
        )
```

**注意**：不需要重新生成父节点的 summary。父节点的 summary 在 Step 2 中已经基于原始 content 生成过了，质量是最优的。

---

## 完整构建主函数

```python
def build_document(markdown_text: str, doc_id: str, llm_client) -> TreeNode:
    """
    完整的文档构建流程，按顺序执行 4 步
    """
    # Step 1: 构建树结构（已有逻辑，不修改）
    root = build_tree(markdown_text, doc_id)

    # Step 2: 生成 summary（改进算法，利用 parent 自身 content）
    generate_summaries(root, llm_client)

    # Step 3: 注入 preamble 虚拟叶子节点（新增）
    inject_preamble_leaves(root)

    # Step 4: 为 preamble 节点生成 summary（新增）
    generate_preamble_summaries(root, llm_client)

    return root
```

---

## 测试验证

### 测试用例

使用以下 Markdown 验证 preamble 注入和 summary 生成的正确性：

```markdown
# Abstract
This paper proposes AdaRouter, a novel adaptive routing method for forex trading.

# 1 Introduction
The field of algorithmic trading has evolved rapidly over the past decade.
This section provides background and motivation for our research.

## 1.1 Background
Foreign exchange markets process over $6 trillion in daily volume.

## 1.2 Motivation
Static routing strategies fail to adapt to changing market conditions.

# 2 Methods
We propose a two-component architecture consisting of an encoder and a router.
The overall design philosophy emphasizes adaptability and real-time decision making.

## 2.1 Encoder Design
The encoder uses a Transformer architecture to process time-series data.

## 2.2 Router Design
The router employs a contextual bandit framework for dynamic routing decisions.

# 3 Experiments
We evaluate AdaRouter on multiple currency pairs spanning 2020-2023.
All experiments were conducted on NVIDIA A100 GPUs with identical hyperparameters.

## 3.1 Performance
AdaRouter achieves 73.2% accuracy on EUR/USD, outperforming the baseline.

## 3.2 Ablation Study
Removing the adaptive routing module results in a 8.1% accuracy drop.

# 4 Conclusion
We presented AdaRouter, demonstrating significant improvements over static methods.
```

### 预期输出（树结构）

```
📄 Document Tree
═══════════════════════════════════════════════
🍃 [L1] Abstract (叶子)
│
📁 [L1] 1 Introduction (非叶子)
│   Summary 来源: 自身 content + children summaries
│   ├── 🍃 1 Introduction (Preamble)    ← 新增 preamble
│   │     content = "The field of algorithmic trading..."
│   ├── 🍃 1.1 Background
│   └── 🍃 1.2 Motivation
│
📁 [L1] 2 Methods (非叶子)
│   Summary 来源: 自身 content + children summaries
│   ├── 🍃 2 Methods (Preamble)          ← 新增 preamble
│   │     content = "We propose a two-component architecture..."
│   ├── 🍃 2.1 Encoder Design
│   └── 🍃 2.2 Router Design
│
📁 [L1] 3 Experiments (非叶子)
│   Summary 来源: 自身 content + children summaries
│   ├── 🍃 3 Experiments (Preamble)      ← 新增 preamble
│   │     content = "We evaluate AdaRouter on multiple currency pairs..."
│   ├── 🍃 3.1 Performance
│   └── 🍃 3.2 Ablation Study
│
🍃 [L1] 4 Conclusion (叶子，无 preamble)
```

### 验证点

1. **内容完整性**：遍历所有叶子节点（含 preamble），拼接它们的 content，应该等于原始 Markdown 的全部正文（不含标题行本身）。如果不等，说明有内容丢失。

2. **Preamble 创建正确性**：
   - Abstract 和 Conclusion 是叶子节点 → 不创建 preamble ✅
   - 1 Introduction / 2 Methods / 3 Experiments 是有 content 的非叶子 → 创建 preamble ✅
   - 1.1 / 1.2 / 2.1 / 2.2 / 3.1 / 3.2 是叶子 → 不创建 preamble ✅

3. **Summary 质量**：
   - 非叶子节点的 summary 应该反映其自身 content 的语义（Case 2），而不仅仅是子节点的聚合
   - 例如 "2 Methods" 的 summary 应该包含"two-component architecture"这样来自自身 content 的关键信息

4. **node_count 和 leaf_count 变化**：
   - 注入前：11 nodes, 8 leaves
   - 注入后：14 nodes, 11 leaves（新增 3 个 preamble）

5. **后续 indexing 兼容性**：preamble 节点的 `is_leaf == True`，能被正常 chunk + embedding + BM25 索引，无需修改 indexing 代码。

