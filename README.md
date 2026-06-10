# RAG 智能文档问答助手

上传 PDF/TXT/DOCX 文档，用自然语言提问，AI 自动从文档中找答案。支持扫描件 OCR 识别、流式打字机输出、多文档管理。

## 技术栈

| 组件 | 技术 |
|------|------|
| 大模型 | 通义千问 qwen-plus（DashScope） |
| Embedding | text-embedding-v3 |
| RAG 框架 | LangChain |
| 向量数据库 | ChromaDB（本地持久化） |
| Web 框架 | FastAPI + Uvicorn |
| 文档解析 | PyPDF / python-docx / pymupdf |
| OCR | EasyOCR（中文+英文） |
| 前端 | 原生 HTML/CSS/JS |
| 容器化 | Docker |

## 功能

- 📄 **多格式支持** — PDF、TXT、DOCX
- 🔍 **扫描件识别** — 自动检测扫描件 PDF，调用 OCR 提取文字
- 💬 **智能问答** — 基于 RAG（检索增强生成），答案附带引用来源
- ⌨️ **流式输出** — 打字机效果，逐字显示答案
- 🔄 **多轮对话** — 支持上下文追问
- 📚 **文档管理** — 查看已上传文档列表，支持单个删除和一键清空
- 💾 **持久化** — 向量数据库本地存储，重启不丢数据
- 🐳 **Docker 部署** — 一键构建运行

## 快速开始

### 1. 安装依赖

```bash
cd rag-doc-assistant
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，填入你的 API Key：

```bash
cp .env.example .env
```

编辑 `.env`：

```
DASHSCOPE_API_KEY=sk-你的密钥
```

> API Key 在 [阿里云 DashScope 控制台](https://dashscope.console.aliyun.com/) 申请。

### 3. 启动服务

```bash
uvicorn main:app --reload --port 8000
```

### 4. 访问

- Web 界面：http://localhost:8000
- API 文档：http://localhost:8000/docs

## Docker 部署

```bash
docker build -t rag-assistant .
docker run -p 8000:8000 -e DASHSCOPE_API_KEY=sk-你的密钥 rag-assistant
```

## 使用说明

1. **上传文档** — 点击上传区域或拖拽文件，支持 PDF/TXT/DOCX
2. **等待处理** — 文档会被自动切块、向量化。扫描件 PDF 会自动触发 OCR（首次需下载模型）
3. **开始提问** — 在输入框输入问题，按回车或点击发送
4. **查看来源** — 每个回答下方会显示引用的文档片段和页码
5. **管理文档** — 在"已加载文档"卡片中可删除单个文档或清空全部

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 界面 |
| `/upload` | POST | 上传文档 |
| `/ask` | POST | 提问（流式 SSE 返回） |
| `/documents` | GET | 获取已上传文档列表 |
| `/documents/{filename}` | DELETE | 删除指定文档 |
| `/documents` | DELETE | 清空所有文档 |
| `/clear` | POST | 清空对话历史 |
| `/docs` | GET | Swagger API 文档 |

### 流式问答协议

`POST /ask` 返回 `text/event-stream`，格式如下：

```
data: {"sources": ["第1页: ...", "第2页: ..."]}

data: {"token": "你"}
data: {"token": "好"}
data: {"token": "，"}
...

data: {"done": true}
```

## 项目结构

```
rag-doc-assistant/
├── main.py              # FastAPI 入口，接口定义
├── config.py            # 配置管理（从 .env 读取）
├── rag_engine.py        # RAG 核心引擎（文档加载、检索、问答、OCR）
├── llm_client.py        # 大模型 API 封装（独立工具）
├── logger.py            # 日志配置
├── requirements.txt     # Python 依赖
├── Dockerfile           # Docker 构建文件
├── .env.example         # 环境变量模板
├── .env                 # 环境变量（不提交 git）
├── .gitignore
├── .dockerignore
├── templates/
│   └── index.html       # Web 前端页面
├── uploads/             # 用户上传的文件
├── chroma_db/           # 向量数据库（自动生成）
└── tests/
    ├── __init__.py
    └── test_api.py      # 接口测试
```

## 工作原理

```
用户上传文档
    ↓
文档加载（PDF / TXT / DOCX）
    ↓
扫描件检测 → 是 → OCR 识别（EasyOCR）
              否 ↓
文本切块（每块 500 字符，重叠 50 字符）
    ↓
向量化（Embedding）→ 存入 ChromaDB
    ↓
用户提问
    ↓
从 ChromaDB 检索最相关的 3 个文档块
    ↓
文档块 + 问题 + 对话历史 → 发给大模型
    ↓
流式返回引用来源 + 逐字输出答案
```

## 配置说明

在 `.env` 中可调整以下参数（均有默认值）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DASHSCOPE_API_KEY` | — | DashScope API 密钥（必填） |
| `LLM_MODEL` | `qwen-plus` | 大模型名称 |
| `EMBEDDING_MODEL` | `text-embedding-v3` | Embedding 模型 |
| `CHUNK_SIZE` | `500` | 文本切块大小（字符数） |
| `CHUNK_OVERLAP` | `50` | 切块重叠字符数 |
| `RETRIEVER_K` | `3` | 检索返回的文档块数量 |
