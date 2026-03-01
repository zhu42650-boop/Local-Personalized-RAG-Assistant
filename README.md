# Research RAG Assistant (Local + Custom Embeddings)

面向科研场景的本地 RAG 助手，聚焦 **高质量检索**、**可控证据** 与 **长文本稳定回答**。  
目标：让科研资料可被可靠检索、可溯源引用、可在长文本条件下稳定回答。

---

## Why This Project
- 面向科研资料的 **结构化分块** 与 **混合检索**，提升召回质量
- 支持 **自研/微调 embedding**，用于科研语义对齐
- **上下文压缩** 控制长文提示词规模

---

## Key Improvements (vs. vanilla RAG demo)

### 1) Embedding 体系升级
- 改为 **Qwen3‑Embedding 版本，使用自制数据进行微调**
- 采用科研paper数据,llm 生成 query,得到4000条文本对
- 支持 LoRA 微调与离线部署

### 2) 分块按类别策略
- paper: 结构化分块（章节标题切分 + 递归分块）
- note: 轻量固定分块
- 其它类型可扩展语义分块策略

### 3) 混合检索 + 召回增强
实现 **BM25 + 向量混检**，并支持 **rerank 重排序**：
- BM25 捕获关键词精确匹配
- 向量检索补齐语义召回
- rerank 提升 top‑k 质量

### 4) 长文本处理
在检索后加入 **小模型摘要压缩**：
- 当上下文过长时自动摘要
- 保留关键信息，减少 prompt 长度

---

## Pipeline Overview
```
User Files -> Loaders -> Category Splitters -> Embeddings -> Chroma
                                       |                 |
                                       +-> BM25 Index ----+
                                                        |
Query -> Vector + BM25 -> Merge -> (Optional Rerank) -> Context Compression -> LLM
```

---

## Quick Start

```bash
python framework/main.py
```

### CLI Reindex
```bash
python framework/ingest/cli_ingest.py --reindex
```

---

## Project Layout
```
framework/
  config/           # 配置与环境检查
  ingest/           # 加载、分块、索引
  rag/              # 检索与回答
  ui/               # 桌面 UI
data/
  knowledge_base/   # 知识库（paper/note/...）
  vector_store/     # Chroma 向量库
  chunks.jsonl      # BM25 使用的 chunk 集合
scripts/
  collect_papers.py # 论文抓取
  chunk_papers.py   # PDF 分块
  gen_queries.py    # LLM 生成 query
  train_embedding.py# embedding 微调
test_retrieval.py   # 端到端检索测试
```

---

## Configuration (Core Fields)
```
embedding:
  model_name: "/path/to/qwen3-embed-ft"
  device: "cuda"

retriever:
  top_k_vector: 6
  top_k_bm25: 6
  top_k_final: 6

summary:
  enabled: true
  model: "gpt-4o-mini"
```

---

## Scripts Overview

| Script | Purpose |
| --- | --- |
| `scripts/collect_papers.py` | 从多源检索并下载论文 |
| `scripts/chunk_papers.py` | PDF 分块（可跳过坏文件） |
| `scripts/gen_queries.py` | 为 chunk 生成检索 query（训练数据） |
| `scripts/train_embedding.py` | 微调 embedding（支持 LoRA） |
| `test_retrieval.py` | 分块->向量->检索链路测试 |

---

## Notes
- 若使用 LoRA 训练，请先合并 adapter 再用于 embedding 服务
- 如果启用 rerank，请在 `rerank.model_name` 中配置模型路径

---

## License
MIT
