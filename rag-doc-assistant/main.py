from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from rag_engine import RAGEngine
from config import settings
import os
import threading

lock = threading.Lock()
engine = RAGEngine()
app = FastAPI(
    title="RAG 智能文档问答助手",
    description="上传 PDF/TXT/DOCX 文档，用自然语言提问，AI 自动从文档中找答案",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源（开发阶段用 *，上线要改成具体域名）
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有请求头
)


@app.get("/", summary="首页", description="访问 Web 界面")
async def home():
    with open("templates/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/upload", summary="上传文档", description="支持 PDF、TXT、DOCX 格式")
def upload(file: UploadFile = File(...)):
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    with lock:  # ← 加锁，同一时间只能有一个人上传
        chunks = engine.load_document(file_path)
        if chunks == -1:
            return {"error": "不支持的文件格式"}
    return {"status": "ok", "filename": file.filename, "chunks": chunks}


@app.get("/ask", summary="提问", description="根据已上传的文档回答问题")
def ask(question: str):
    result = engine.ask(question)
    return result


@app.post("/clear", summary="清空对话", description="清除对话历史，重新开始")
def clear():
    engine.clear_history()
    return {"status": "ok", "message": "对话历史已清空"}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "服务器内部错误，请稍后重试"}
    )
