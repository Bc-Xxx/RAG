from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
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


class AskRequest(BaseModel):
    question: str

@app.post("/ask", summary="提问", description="根据已上传的文档回答问题（流式返回）")
def ask(req: AskRequest):
    return StreamingResponse(
        engine.ask_stream(req.question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.post("/clear", summary="清空对话", description="清除对话历史，重新开始")
def clear():
    engine.clear_history()
    return {"status": "ok", "message": "对话历史已清空"}


@app.get("/documents", summary="文档列表", description="获取已上传的所有文档")
def list_documents():
    docs = []
    if os.path.exists(settings.UPLOAD_DIR):
        for filename in os.listdir(settings.UPLOAD_DIR):
            file_path = os.path.join(settings.UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                # 统计该文件在向量库中的 chunk 数
                chunk_count = 0
                if engine.vectorstore is not None:
                    try:
                        results = engine.vectorstore.get(
                            where={"source": file_path},
                            include=[]
                        )
                        chunk_count = len(results["ids"]) if results["ids"] else 0
                    except Exception:
                        chunk_count = -1  # 未知
                docs.append({
                    "filename": filename,
                    "size": size,
                    "chunks": chunk_count,
                    "modified": os.path.getmtime(file_path)
                })
    docs.sort(key=lambda d: d["modified"], reverse=True)
    return {"documents": docs}


@app.delete("/documents", summary="清空所有文档", description="删除所有文档和向量索引")
def delete_all_documents():
    with lock:
        engine.remove_all_documents()
        engine.clear_history()
    return {"status": "ok", "message": "所有文档已清空"}


@app.delete("/documents/{filename}", summary="删除文档", description="删除指定文档及其向量索引")
def delete_document(filename: str):
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "文档不存在"})
    with lock:
        result = engine.remove_document(filename)
    return {"status": "ok", "filename": filename, "removed_chunks": result["removed_chunks"]}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "服务器内部错误，请稍后重试"}
    )
