# Tree-RAG Pipeline — Phase 2: Retrieval System

## 项目背景

Phase 1 已完成 Tree Builder，能将 Markdown 文档解析为层次化树结构（TreeNode），每个节点包含 heading, content, summary, heading_path, is_leaf 等信息。

Phase 2 构建一个**固定三步 pipeline**（非 agentic），实现从 query 到 answer 的端到端检索问答。

---

## Pipeline 总览

```
Query
  │
  ▼
Step 1: Node Locating（PageIndex 风格）
  将树结构 (node_id + heading + summary) 序列化为文本
  塞进 LLM context，让 LLM 推理定位相关叶子节点
  输出: List[node_id] + 每个节点的 sub_query
  │
  ▼
Step 2: In-Node Hybrid Retrieval
  对每个命中的叶子节点:
    该节点的 content 已预先 chunk + embedding
    用 sub_query 做 Dense + BM25 混合检索
    Rerank 后返回 top_k chunks
  输出: List[RetrievedChunk]
  │
  ▼
Step 3: Synthesis
  将所有检索到的 chunks 拼接为 context
  连同原始 query 送给 LLM 生成最终回答
  输出: Answer (带来源标注)
```

**注意：这是一个固定流程的 pipeline，不是 agent。没有循环、没有 tool calling、没有动态决策。三步顺序执行。**

---

## 数据结构

### 复用 Phase 1

```python
# 来自 Phase 1，不需要修改
@dataclass
class TreeNode:
    node_id: str
    heading: str
    level: int
    content: str
    summary: str
    parent: Optional['TreeNode']
    children: List['TreeNode']
    heading_path: str
    
    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0
```

### Phase 2 新增

```python
@dataclass
class Chunk:
    chunk_id: str              # e.g., "node_0005_chunk_02"
    text: str                  # chunk 文本
    source_node_id: str        # 所属叶子节点 ID
    heading_path: str          # 所属节点的完整路径
    embedding: List[float]     # dense embedding 向量

@dataclass 
class RetrievedChunk:
    chunk: Chunk
    score: float               # rerank 后的最终得分
    retrieval_detail: dict     # {"dense_score": 0.82, "bm25_score": 12.3, "rerank_score": 0.91}
```

---

## Step 1: Node Locating 详细设计

### 树结构序列化

将整棵树序列化为紧凑的文本格式，塞进 LLM prompt：

```
文档结构:

[0001] Abstract (叶子节点)
  摘要: 本文提出AdaRouter，一种基于contextual bandit的自适应路由方法...

[0002] 1 Introduction
  摘要: 介绍研究背景和动机...

  [0003] 1.1 Background (叶子节点)
    摘要: 外汇市场算法交易的发展现状...

  [0004] 1.2 Motivation (叶子节点)
    摘要: 现有静态路由方法的局限性...

[0005] 2 Methods
  摘要: 本章描述方法论...

  [0006] 2.1 Encoder Design (叶子节点)
    摘要: 基于Transformer的时序编码器设计...

  [0007] 2.2 Router Design (叶子节点)
    摘要: Contextual bandit路由决策算法...

[0008] 3 Experiments
  摘要: 实验设置与结果分析...

  [0009] 3.1 Performance (叶子节点)
    摘要: 主实验结果，AdaRouter在EUR/USD上达到73.2%准确率...

  [0010] 3.2 Ablation Study (叶子节点)
    摘要: 各模块消融实验结果...

[0011] 4 Conclusion (叶子节点)
  摘要: 总结贡献和未来方向...
```

序列化规则：
- 缩进表示层级关系（2个空格 per level）
- 每个节点显示: `[node_id] heading (叶子节点标记)`
- 每个节点附带 1 句话 summary
- **不放 content 原文**，只放 summary，控制 token 量

### Node Locating Prompt

```
你是一个文档检索专家。给定用户问题和文档的树状结构，请找出最可能包含答案的叶子节点。

规则:
1. 只返回叶子节点（标记为"叶子节点"的节点）
2. 返回 1-5 个最相关的节点，不要过多
3. 为每个节点生成一个精化的 sub_query，描述你期望从该节点中找到什么信息

用户问题: {query}

文档结构:
{serialized_tree}

输出严格 JSON 格式:
{{
  "thinking": "简要分析问题需要哪些信息，以及这些信息可能在哪些章节",
  "results": [
    {{"node_id": "0007", "sub_query": "contextual bandit路由算法的具体设计"}},
    {{"node_id": "0009", "sub_query": "EUR/USD上的准确率数据和对比结果"}}
  ]
}}
```

### 实现

```python
def locate_nodes(query: str, tree: TreeNode, llm_client) -> List[dict]:
    """
    Step 1: 将树结构序列化后送 LLM 推理定位
    
    Returns: [{"node_id": "0007", "sub_query": "..."}, ...]
    """
    serialized = serialize_tree(tree)
    prompt = NODE_LOCATE_PROMPT.format(query=query, serialized_tree=serialized)
    response = llm_client.generate(prompt, response_format="json")
    result = json.loads(response)
    
    # 验证返回的 node_id 都是有效的叶子节点
    valid_leaf_ids = {n.node_id for n in get_all_leaves(tree)}
    results = [r for r in result["results"] if r["node_id"] in valid_leaf_ids]
    
    return results, result.get("thinking", "")
```

### Mock 模式

不调用 LLM，用关键词匹配：对 query 分词，匹配每个叶子节点的 heading + summary，返回得分最高的 3 个节点。sub_query 直接用原始 query。

---

## Step 2: In-Node Hybrid Retrieval 详细设计

### 2.1 Indexing 阶段（离线构建）

对每个叶子节点的 content 做 chunk + embedding + BM25 索引。

#### Chunking 策略

```python
def chunk_content(content: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """
    对叶子节点的 content 做固定窗口切分
    
    - chunk_size: 每个 chunk 的目标字符数（约 200 字符，中英文均适用）
    - overlap: 相邻 chunk 的重叠字符数
    - 先按段落 (\n\n) 切分，再对超长段落做滑窗切分
    - 过滤掉过短的 chunk（< 20 字符）
    """
    paragraphs = content.split('\n\n')
    chunks = []
    
    for para in paragraphs:
        para = para.strip()
        if len(para) < 20:
            continue
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            # 滑窗切分超长段落
            start = 0
            while start < len(para):
                end = start + chunk_size
                chunks.append(para[start:end])
                start += chunk_size - overlap
    
    return chunks
```

#### Dense Embedding

```python
# 对每个 chunk 生成 embedding 向量
for leaf_node in get_all_leaves(tree):
    chunks = chunk_content(leaf_node.content)
    for i, chunk_text in enumerate(chunks):
        chunk = Chunk(
            chunk_id=f"{leaf_node.node_id}_chunk_{i:02d}",
            text=chunk_text,
            source_node_id=leaf_node.node_id,
            heading_path=leaf_node.heading_path,
            embedding=embed(chunk_text)
        )
        leaf_node.chunks.append(chunk)
```

Embedding 选择：
- 真实模式: OpenAI `text-embedding-3-small` 或 Anthropic Voyage
- Mock/本地模式: `sentence-transformers` 的 `all-MiniLM-L6-v2`

#### BM25 索引

```python
# 对每个叶子节点单独构建 BM25 索引
# 使用 rank_bm25 库
from rank_bm25 import BM25Okapi
import jieba  # 如果是中文需要分词

for leaf_node in get_all_leaves(tree):
    tokenized_chunks = [tokenize(c.text) for c in leaf_node.chunks]
    leaf_node.bm25_index = BM25Okapi(tokenized_chunks)
```

分词策略：
- 英文：简单 split + lowercase + 去停用词
- 中文：jieba 分词
- 可以做一个简单的语言检测来自动选择

### 2.2 Retrieval 阶段（在线查询）

对 Step 1 返回的每个 (node_id, sub_query)，执行混合检索：

```python
def hybrid_retrieve(
    node: TreeNode, 
    query: str, 
    top_k: int = 5,
    dense_weight: float = 0.5,
    bm25_weight: float = 0.5
) -> List[RetrievedChunk]:
    """
    对单个叶子节点做 Dense + BM25 混合检索
    """
    chunks = node.chunks
    if not chunks:
        return []
    
    # 1. Dense retrieval
    query_emb = embed(query)
    dense_scores = [
        cosine_similarity(query_emb, chunk.embedding) 
        for chunk in chunks
    ]
    
    # 2. BM25 retrieval
    tokenized_query = tokenize(query)
    bm25_scores = node.bm25_index.get_scores(tokenized_query)
    
    # 3. Score normalization (min-max to [0, 1])
    dense_norm = min_max_normalize(dense_scores)
    bm25_norm = min_max_normalize(bm25_scores)
    
    # 4. Weighted fusion
    fused_scores = [
        dense_weight * d + bm25_weight * b 
        for d, b in zip(dense_norm, bm25_norm)
    ]
    
    # 5. 按融合分数排序，取 top_k
    ranked = sorted(
        zip(chunks, fused_scores, dense_scores, bm25_scores),
        key=lambda x: -x[1]
    )[:top_k]
    
    return [
        RetrievedChunk(
            chunk=chunk,
            score=fused,
            retrieval_detail={
                "dense_score": round(dense, 4),
                "bm25_score": round(bm25, 4),
                "fused_score": round(fused, 4)
            }
        )
        for chunk, fused, dense, bm25 in ranked
    ]
```

### 2.3 Rerank（可选但推荐）

对所有节点的混合检索结果汇总后，做一次 cross-encoder rerank：

```python
def rerank(query: str, retrieved_chunks: List[RetrievedChunk], top_k: int = 5) -> List[RetrievedChunk]:
    """
    用 cross-encoder 对所有检索结果做 rerank
    
    真实模式: 用 cross-encoder 模型 (如 BAAI/bge-reranker-v2-m3)
    Mock 模式: 直接返回按 fused_score 排序的结果（跳过 rerank）
    """
    # 汇总所有节点的检索结果
    all_chunks = retrieved_chunks
    
    # Cross-encoder scoring
    pairs = [(query, chunk.chunk.text) for chunk in all_chunks]
    rerank_scores = cross_encoder.predict(pairs)
    
    for chunk, score in zip(all_chunks, rerank_scores):
        chunk.retrieval_detail["rerank_score"] = round(float(score), 4)
        chunk.score = float(score)  # 最终排序用 rerank score
    
    return sorted(all_chunks, key=lambda x: -x.score)[:top_k]
```

Rerank 模型选择：
- 本地: `sentence-transformers` 的 CrossEncoder，模型用 `BAAI/bge-reranker-v2-m3` 或 `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Mock 模式: 跳过 rerank，直接用 fused_score 排序

---

## Step 3: Synthesis 详细设计

将检索结果拼接为 context，送 LLM 生成回答。

### Context 组装

```python
def build_context(retrieved_chunks: List[RetrievedChunk]) -> str:
    """将检索结果组装为 LLM 可读的 context"""
    context_parts = []
    for i, rc in enumerate(retrieved_chunks, 1):
        context_parts.append(
            f"[证据{i}] 来源: {rc.chunk.heading_path}\n{rc.chunk.text}"
        )
    return "\n\n".join(context_parts)
```

### Synthesis Prompt

```
基于以下从文档中检索到的证据，回答用户的问题。

规则:
1. 只基于提供的证据回答，不要编造信息
2. 回答时在关键信息后标注来源，格式为 [来源: 章节路径]
3. 如果证据不足以完整回答，明确说明
4. 回答要简洁准确

用户问题: {query}

检索到的证据:
{context}

请回答:
```

### 实现

```python
def synthesize(query: str, retrieved_chunks: List[RetrievedChunk], llm_client) -> str:
    context = build_context(retrieved_chunks)
    prompt = SYNTHESIS_PROMPT.format(query=query, context=context)
    answer = llm_client.generate(prompt)
    return answer
```

### Mock 模式

不调用 LLM，直接拼接检索结果作为回答：
```
根据检索结果:
[1] (来源: Methods > Router Design) "AdaRouter采用contextual bandit..."
[2] (来源: Experiments > Performance) "在EUR/USD上达到73.2%准确率..."
```

---

## Pipeline 主函数

```python
def run_pipeline(query: str, tree: TreeNode, index: Index, config: Config) -> dict:
    """
    三步 pipeline，顺序执行，无循环无分支
    """
    # Step 1: Node Locating
    located_nodes, thinking = locate_nodes(query, tree, config.llm_client)
    
    # Step 2: In-Node Hybrid Retrieval
    all_retrieved = []
    for node_info in located_nodes:
        node = index.get_node(node_info["node_id"])
        sub_query = node_info["sub_query"]
        chunks = hybrid_retrieve(node, sub_query, top_k=5)
        all_retrieved.extend(chunks)
    
    # Rerank（跨节点统一排序）
    reranked = rerank(query, all_retrieved, top_k=5)
    
    # Step 3: Synthesis
    answer = synthesize(query, reranked, config.llm_client)
    
    return {
        "query": query,
        "step1_thinking": thinking,
        "step1_nodes": located_nodes,
        "step2_retrieved": [
            {
                "text": rc.chunk.text,
                "heading_path": rc.chunk.heading_path,
                "scores": rc.retrieval_detail
            } for rc in reranked
        ],
        "answer": answer
    }
```

---

## CLI 接口

```bash
# 1. 构建索引（在 Phase 1 的 tree 基础上追加 chunk + embedding + BM25）
python main.py index --input test_data/test_paper.md --output index/ [--mock]

# 2. 查询
python main.py query --index index/ --query "AdaRouter的核心机制是什么？" [--mock]

# 3. 交互模式（循环输入 query）
python main.py interactive --index index/ [--mock]
```

---

## 终端输出格式

```
============================================================
Query: AdaRouter在EUR/USD上的表现为什么比baseline好？
============================================================

>>> Step 1: Node Locating
Thinking: 问题涉及AdaRouter的机制、EUR/USD实验结果和baseline对比...
Located 3 leaf nodes:
  [0007] Methods > Router Design
    sub_query: "contextual bandit路由算法的具体设计"
  [0009] Experiments > Performance
    sub_query: "EUR/USD上的准确率数据和baseline对比"
  [0010] Experiments > Ablation Study
    sub_query: "各模块消融实验的结果"

>>> Step 2: Hybrid Retrieval + Rerank
Retrieved 5 chunks (from 3 nodes):
  #1 [0009] Experiments > Performance
     "AdaRouter achieves 73.2% accuracy on EUR/USD..."
     dense=0.91  bm25=0.85  fused=0.88  rerank=0.94

  #2 [0009] Experiments > Performance
     "Compared to Static Router baseline (65.1%)..."
     dense=0.87  bm25=0.82  fused=0.85  rerank=0.91

  #3 [0007] Methods > Router Design
     "The router employs a contextual bandit framework..."
     dense=0.85  bm25=0.78  fused=0.82  rerank=0.88

  #4 [0010] Experiments > Ablation Study
     "Removing the adaptive routing module results in..."
     dense=0.80  bm25=0.75  fused=0.78  rerank=0.83

  #5 [0007] Methods > Router Design
     "Unlike static routing, AdaRouter dynamically..."
     dense=0.79  bm25=0.70  fused=0.75  rerank=0.80

>>> Step 3: Answer
AdaRouter在EUR/USD上表现优于baseline，主要原因如下：

AdaRouter采用contextual bandit框架进行动态路由决策 [来源: Methods > Router Design]，
能够根据实时市场状态调整策略。相比之下，Static Router baseline使用固定规则。

实验结果显示，AdaRouter在EUR/USD上达到73.2%准确率，而baseline仅为65.1%
[来源: Experiments > Performance]。

消融实验进一步验证了自适应路由模块的关键作用 [来源: Experiments > Ablation Study]。
============================================================
```

---

## 测试用 Markdown

创建一个模拟论文 `test_data/test_paper.md`，约 800-1000 字英文，章节结构如下。**每个叶子节点需要有 3-5 句实质内容**（包含具体实体名、数字、因果关系），这样才能有效测试检索。

```
# Abstract
# 1 Introduction
## 1.1 Background
## 1.2 Motivation
# 2 Related Work
# 3 Methods
## 3.1 Problem Formulation
## 3.2 Model Architecture
### 3.2.1 Encoder Design
### 3.2.2 Router Design
## 3.3 Training Procedure
# 4 Experiments
## 4.1 Dataset
## 4.2 Performance
## 4.3 Ablation Study
# 5 Conclusion
```

同时准备 3-5 个测试 query，覆盖不同难度：
- 简单: "AdaRouter使用什么算法？" → 单节点定位
- 中等: "AdaRouter在EUR/USD上的准确率是多少？" → 单节点 + 具体数据
- 复杂: "为什么AdaRouter比baseline表现好？" → 多节点 + 跨章节推理
- 细节: "训练用了多少个epoch？" → 精确匹配测试

---

## 技术依赖

```
# 核心
numpy

# BM25
rank-bm25

# Embedding（三选一）
sentence-transformers          # 本地模式（推荐 mock/测试用）
openai                         # OpenAI API
# anthropic                    # 如用 Voyage embedding

# Rerank（可选）
sentence-transformers          # CrossEncoder 模型

# LLM（Step 1 和 Step 3）
anthropic                      # Claude API
# 或 openai                   # OpenAI API

# 中文分词（如需处理中文）
jieba
```

---

## 文件结构

```
tree_rag/
├── main.py                     # CLI 入口 (index / query / interactive)
├── config.py                   # 配置: API keys, 模型选择, mock 开关
│
├── tree_builder/               # Phase 1（已有，不动）
│   ├── parser.py
│   ├── tree.py
│   ├── summary.py
│   └── visualizer.py
│
├── indexing/
│   ├── chunker.py              # content → chunks (段落优先 + 滑窗)
│   ├── embedder.py             # chunk embedding (API / local / mock)
│   ├── bm25_builder.py         # 每个叶子节点构建 BM25 索引
│   └── index_store.py          # 索引序列化/反序列化 (JSON + numpy)
│
├── retrieval/
│   ├── node_locator.py         # Step 1: 树序列化 + LLM 推理定位
│   ├── hybrid_retriever.py     # Step 2: Dense + BM25 + Rerank
│   └── synthesizer.py          # Step 3: Context 组装 + LLM 生成回答
│
├── pipeline.py                 # 三步 pipeline 主函数 (run_pipeline)
│
├── utils/
│   ├── llm_client.py           # LLM 调用封装 (Anthropic / OpenAI / Mock)
│   ├── similarity.py           # cosine_similarity, min_max_normalize
│   └── tokenizer.py            # 分词工具 (英文split / 中文jieba)
│
├── test_data/
│   └── test_paper.md           # 测试用模拟论文
│
└── README.md
```

---

## Mock 模式对照表

| 组件 | 真实模式 | Mock 模式 |
|------|----------|-----------|
| Node Locating (Step 1) | LLM 推理 | 关键词匹配 heading+summary，返回 top 3 |
| Embedding | API 或 sentence-transformers | sentence-transformers 本地模型 或 随机向量 |
| BM25 | rank_bm25 | rank_bm25（不需要 mock，本身就是本地的） |
| Rerank | CrossEncoder 模型 | 跳过，直接用 fused_score |
| Synthesis (Step 3) | LLM 生成 | 直接拼接检索结果 |

**建议开发顺序：先用 mock 模式跑通全流程 → 逐步替换为真实模型。**