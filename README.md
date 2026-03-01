# Research RAG Assistant (Local + Custom Embeddings)

一个面向科研场景的本地 RAG 助手，围绕“高质量检索 + 可控证据 + 长文本压缩”做工程化优化。核心目标是：**让科研资料可被可靠检索、可溯源引用、可在长文本条件下稳定回答**。

---

## 项目目的
构建一个 **科研协作型 RAG 系统**，支持：
- 本地论文/笔记/资料的检索式对话
- 可追溯证据链（chunk 级来源）
- 适应科研长文档与结构化论文的召回

---

## 关键改进点（相较普通 RAG Demo）

### 1) Embedding 服务升级
- 不再使用通用 bge‑m3，改为 **Qwen3‑Embedding 微调版本**
- 采用对科研数据的“query‑chunk”对比学习
- 支持 LoRA 微调与离线部署

### 2) 分块按类别策略
- **paper**：结构化分块（按章节标题切分再递归分块）
- **note**：轻量固定分块
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

## 快速开始

```bash
python framework/main.py
```

### CLI 重建索引
```bash
python framework/ingest/cli_ingest.py --reindex
```

---

## 目录结构
```
framework/
  config/           # 配置与环境检查
  ingest/           # 加载、分块、索引
  rag/              # 检索与回答
  ui/               # 桌面 UI
_data/
  knowledge_base/   # 知识库（paper/note/...）
  vector_store/     # Chroma 向量库
  chunks.jsonl      # BM25 使用的 chunk 集合
scripts/
  collect_papers.py # 论文抓取
  chunk_papers.py   # PDF 分块
  gen_queries.py    # LLM 生成 query
  train_embedding.py# embedding 微调
```

---

## 配置示例（核心字段）
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

## 备注
- 若使用 LoRA 训练，请先合并 adapter 再用于 embedding 服务
- 如果启用 rerank，请在 `rerank.model_name` 中配置模型路径

---

## License
MIT
