from openai import OpenAI
from config import settings

client = OpenAI(
    api_key=settings.DASHSCOPE_API_KEY,
    base_url=settings.API_BASE_URL
)


def chat(question: str, role: str = "你是一个有帮助的助手") -> str:
    resp = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": role},
            {"role": "user", "content": question}
        ]
    )
    return resp.choices[0].message.content


def get_embedding(text: str) -> list[float]:
    resp = client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=text
    )
    return resp.data[0].embedding
