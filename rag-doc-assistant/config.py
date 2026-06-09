from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    DASHSCOPE_API_KEY: str  # API 密钥
    LLM_MODEL: str = "qwen-plus"  # 用哪个模型
    EMBEDDING_MODEL: str = "text-embedding-v3"  # Embedding 模型
    API_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    UPLOAD_DIR: str = "uploads"  # 上传文件存哪里
    CHROMA_DIR: str = "chroma_db"  # 向量数据库存哪里

    CHUNK_SIZE: int = 500  # 每块多大
    CHUNK_OVERLAP: int = 50  # 重叠多少
    RETRIEVER_K: int = 3  # 检索返回几条

    class Config:
        env_file = ".env"  # 从 .env 文件读取


settings = Settings()  # 创建全局配置实例
os.environ["OPENAI_API_KEY"] = settings.DASHSCOPE_API_KEY
