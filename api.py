# ============================================================
# 3_api.py  —  FastAPI 推理服务
# 启动: uvicorn 3_api:app --reload
# 依赖: pip install fastapi uvicorn sentence-transformers joblib
# ============================================================
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import joblib, time

app = FastAPI(title="Melatonin Review Classifier")

# ------ 启动时加载模型（只加载一次）------
encoder = SentenceTransformer("encoder_model")
clf     = joblib.load("clf_model.pkl")
mlb     = joblib.load("mlb.pkl")

# ------ 请求 / 响应结构 ------
class ReviewRequest(BaseModel):
    texts: List[str]

class ReviewResponse(BaseModel):
    predictions: List[List[str]]
    latency_ms:  float

# ------ 推理端点 ------
@app.post("/predict", response_model=ReviewResponse)
def predict(req: ReviewRequest):
    t0   = time.time()
    emb  = encoder.encode(req.texts, batch_size=32)
    raw  = clf.predict(emb)
    labs = mlb.inverse_transform(raw)
    ms   = round((time.time() - t0) * 1000, 2)
    return ReviewResponse(
        predictions=[list(l) for l in labs],
        latency_ms=ms
    )

@app.get("/health")
def health():
    return {"status": "ok", "labels": list(mlb.classes_)}


# ============================================================
# 测试命令（服务启动后在终端运行）
# ============================================================
# curl -X POST http://localhost:8000/predict \
#   -H "Content-Type: application/json" \
#   -d '{
#     "texts": [
#       "Works great, fell asleep in 30 minutes, no grogginess!",
#       "Terrible taste and gave me nightmares every night.",
#       "Does nothing for me, complete waste of money."
#     ]
#   }'
