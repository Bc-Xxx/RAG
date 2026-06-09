from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_home():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "RAG" in resp.json()["message"]


def test_ask_without_upload():
    """没上传文档就提问，应该返回错误"""
    resp = client.get("/ask?question=测试")
    assert resp.status_code == 200
    assert "error" in resp.json()


def test_clear():
    """测试清空对话历史"""
    resp = client.post("/clear")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
