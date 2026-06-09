from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from config import settings
import os
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

    def clear_history(self):
        # 清空对话历史
        if self.qa_chain and self.qa_chain.memory:
            self.qa_chain.memory.clear()
