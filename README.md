# RAG 智能文档问答助手

上传 PDF/TXT/DOCX 文档，用自然语言提问，AI 自动从文档中找答案。

## 技术栈

| 组件 | 技术 |
|------|------|
| 大模型 | 通义千问 qwen-plus |
| 框架 | LangChain |
| 向量数据库 | Chroma（本地持久化） |
| Web 框架 | FastAPI |
| 配置管理 | pydantic-settings |
| 容器化 | Docker |

## 功能

- 支持 PDF、TXT、DOCX 三种文档格式
- 文档自动切块、向量化、存储到向量数据库
- 基于 RAG（检索增强生成）的智能问答
- 多轮对话记忆，支持上下文追问
- 向量数据库持久化，重启不丢数据
- Web 界面，上传文档后直接提问
- Docker 一键部署

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，填入你的 API Key：

```bash
DASHSCOPE_API_KEY=sk-你的密钥
```

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

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 界面 |
| `/upload` | POST | 上传文档（PDF/TXT/DOCX） |
| `/ask?question=xxx` | GET | 提问 |
| `/clear` | POST | 清空对话历史 |
| `/docs` | GET | Swagger API 文档 |

## 项目结构

```
rag-doc-assistant/
├── main.py              # FastAPI 入口，接口定义
├── config.py            # 配置管理（从 .env 读取）
├── rag_engine.py        # RAG 核心引擎（文档加载、检索、问答）
├── llm_client.py        # 大模型 API 封装
├── logger.py            # 日志配置
├── requirements.txt     # Python 依赖
├── Dockerfile           # Docker 构建文件
├── .env                 # 环境变量（不提交 git）
├── .gitignore           # Git 忽略规则
├── .dockerignore        # Docker 忽略规则
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
文档加载（PDF/TXT/DOCX）
    ↓
文本切块（每块 500 字符，重叠 50 字符）
    ↓
向量化（Embedding）→ 存入 Chroma
    ↓
用户提问
    ↓
从 Chroma 检索最相关的 3 个文档块
    ↓
文档块 + 问题 + 对话历史 → 发给大模型
    ↓
大模型生成回答
```
