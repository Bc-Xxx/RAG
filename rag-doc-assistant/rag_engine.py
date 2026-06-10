from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from config import settings
import os
import json
from logger import logger


class RAGEngine:
    def __init__(self):
        # 1. 创建 Embedding
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_base=settings.API_BASE_URL,
            check_embedding_ctx_length=False
        )
        # 2. 创建 LLM
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            openai_api_base=settings.API_BASE_URL,
        )
        # 3. 初始化变量
        self.vectorstore = None
        self.qa_chain = None

        # 4. 检查有没有旧的 chroma_db，有就加载
        if os.path.exists(settings.CHROMA_DIR) and os.listdir(settings.CHROMA_DIR):
            self.vectorstore = Chroma(
                persist_directory=settings.CHROMA_DIR,
                embedding_function=self.embeddings
            )
            self._build_qa_chain()
            logger.info("已加载之前的索引")

    def _get_ocr_reader(self):
        """懒加载 EasyOCR Reader（首次调用时才加载模型，避免启动变慢）"""
        if not hasattr(self, '_ocr_reader'):
            logger.info("首次使用 OCR，正在加载 EasyOCR 模型（首次需下载约 100MB，请耐心等待）...")
            import easyocr
            self._ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
            logger.info("EasyOCR 模型加载完成")
        return self._ocr_reader

    def _ocr_pdf(self, file_path: str) -> list:
        """用 OCR 提取扫描件 PDF 的文字"""
        import fitz  # pymupdf

        reader = self._get_ocr_reader()
        docs = []
        pdf = fitz.open(file_path)
        total_pages = len(pdf)
        max_pages = min(total_pages, 20)  # 最多处理 20 页，避免超时

        if total_pages > max_pages:
            logger.warning(f"扫描件共 {total_pages} 页，OCR 仅处理前 {max_pages} 页")

        for page_num in range(max_pages):
            page = pdf[page_num]
            # 1.5 倍分辨率（平衡速度和准确率）
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img_bytes = pix.tobytes("png")

            # OCR 识别
            results = reader.readtext(img_bytes, detail=0)
            text = "\n".join(results)

            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": file_path, "page": page_num}
                ))
            logger.info(f"OCR 进度: {page_num + 1}/{max_pages} 页")

        pdf.close()
        logger.info(f"OCR 完成，共识别 {len(docs)} 页文字")
        return docs

    def _build_qa_chain(self):
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": settings.RETRIEVER_K}  # ← 从 config 读 k 值
            ),
            memory=memory,
            return_source_documents=True
        )

    def load_document(self, file_path: str) -> int:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            docs = PyPDFLoader(file_path).load()
            # 检测扫描件：如果提取的文字太少，尝试 OCR
            if docs:
                total_chars = sum(len(d.page_content.strip()) for d in docs)
                avg_chars = total_chars / len(docs)
                if avg_chars < 20:
                    logger.info(f"检测到扫描件 PDF（平均每页 {avg_chars:.0f} 字符），切换 OCR 模式")
                    ocr_docs = self._ocr_pdf(file_path)
                    if ocr_docs:
                        docs = ocr_docs
                    else:
                        logger.warning("OCR 也未识别到文字，使用原始提取结果")
        elif ext == '.txt':
            docs = TextLoader(file_path).load()
        elif ext == '.docx':
            docs = Docx2txtLoader(file_path).load()
        else:
            return -1
        splitter = RecursiveCharacterTextSplitter(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
        chunks = splitter.split_documents(docs)
        if self.vectorstore is not None:
            self.vectorstore.add_documents(chunks)
        else:
            self.vectorstore = Chroma.from_documents(chunks, self.embeddings, persist_directory=settings.CHROMA_DIR)
        self._build_qa_chain()
        return len(chunks)

    def ask(self, question: str) -> dict:
        # 调用问答链，返回 answer + sources
        if self.qa_chain is None:
            return {"error": "请先上传文档"}
        result = self.qa_chain.invoke({"question": question})
        sources = []
        for doc in result.get("source_documents", []):
            content = doc.page_content[:100].replace("\n", " ")
            page = doc.metadata.get("page", "未知")
            sources.append(f"第{page}页: {content}")
        return {
            "answer": result["answer"],
            "sources": sources
        }

    def ask_stream(self, question: str):
        """流式问答，生成器，yield SSE 格式的数据"""
        if self.vectorstore is None:
            yield f"data: {json.dumps({'error': '请先上传文档'}, ensure_ascii=False)}\n\n"
            return

        # 1. 检索相关文档
        docs = self.vectorstore.similarity_search(question, k=settings.RETRIEVER_K)
        sources = []
        for doc in docs:
            content = doc.page_content[:100].replace("\n", " ")
            page = doc.metadata.get("page", "未知")
            sources.append(f"第{page}页: {content}")

        # 先发送 sources
        yield f"data: {json.dumps({'sources': sources}, ensure_ascii=False)}\n\n"

        # 2. 拼接上下文
        context = "\n\n".join([doc.page_content for doc in docs])

        # 3. 构建 chat history
        history_messages = []
        if self.qa_chain and self.qa_chain.memory:
            memory_data = self.qa_chain.memory.load_memory_variables({})
            for msg in memory_data.get("chat_history", []):
                history_messages.append(msg)

        # 4. 构建消息列表
        messages = [
            HumanMessage(content=f"""请根据以下参考资料回答用户的问题。如果参考资料中没有相关信息，请说明无法从文档中找到答案。

参考资料：
{context}"""),
        ]
        # 添加历史对话
        for msg in history_messages:
            messages.append(msg)
        # 添加当前问题
        messages.append(HumanMessage(content=question))

        # 5. 流式生成
        full_answer = ""
        for chunk in self.llm.stream(messages):
            token = chunk.content
            if token:
                full_answer += token
                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

        # 6. 更新 chat history（手动管理）
        if self.qa_chain and self.qa_chain.memory:
            self.qa_chain.memory.chat_memory.add_user_message(question)
            self.qa_chain.memory.chat_memory.add_ai_message(full_answer)

        # 7. 发送完成信号
        yield f"data: {json.dumps({'done': True})}\n\n"

    def remove_document(self, filename: str) -> dict:
        """删除指定文档：从向量库移除对应 chunks，删除磁盘文件"""
        file_path = os.path.join(settings.UPLOAD_DIR, filename)
        removed_chunks = 0

        # 1. 从 ChromaDB 中删除该文件对应的 chunks
        if self.vectorstore is not None:
            try:
                results = self.vectorstore.get(where={"source": file_path})
                if results and results["ids"]:
                    removed_chunks = len(results["ids"])
                    self.vectorstore.delete(ids=results["ids"])
                    logger.info(f"从向量库删除了 {removed_chunks} 个 chunks: {filename}")
            except Exception as e:
                # where 条件可能因元数据结构不同而失败，尝试遍历所有文档
                logger.warning(f"按 source 查询失败，尝试全量匹配: {e}")
                all_docs = self.vectorstore.get(include=["metadatas"])
                ids_to_delete = []
                for i, meta in enumerate(all_docs["metadatas"]):
                    if meta.get("source") == file_path:
                        ids_to_delete.append(all_docs["ids"][i])
                if ids_to_delete:
                    removed_chunks = len(ids_to_delete)
                    self.vectorstore.delete(ids=ids_to_delete)
                    logger.info(f"从向量库删除了 {removed_chunks} 个 chunks: {filename}")

        # 2. 删除磁盘文件
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"已删除文件: {file_path}")

        # 3. 检查向量库是否为空
        if self.vectorstore is not None:
            remaining = self.vectorstore.get()
            if not remaining["ids"]:
                self.vectorstore = None
                self.qa_chain = None
                # 删除空的 chroma_db 目录
                import shutil
                if os.path.exists(settings.CHROMA_DIR):
                    shutil.rmtree(settings.CHROMA_DIR)
                    logger.info("向量库已清空，删除 chroma_db 目录")
        else:
            self.qa_chain = None

        return {"removed_chunks": removed_chunks}

    def remove_all_documents(self) -> dict:
        """清空所有文档和向量库"""
        import shutil

        # 1. 删除 chroma_db 目录
        if os.path.exists(settings.CHROMA_DIR):
            shutil.rmtree(settings.CHROMA_DIR)
            logger.info("已删除 chroma_db 目录")

        # 2. 清空 uploads 目录
        if os.path.exists(settings.UPLOAD_DIR):
            for f in os.listdir(settings.UPLOAD_DIR):
                os.remove(os.path.join(settings.UPLOAD_DIR, f))
            logger.info("已清空 uploads 目录")

        # 3. 重置状态
        self.vectorstore = None
        self.qa_chain = None

        return {"status": "ok"}

    def clear_history(self):
        # 清空对话历史
        if self.qa_chain and self.qa_chain.memory:
            self.qa_chain.memory.clear()
