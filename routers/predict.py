from fastapi import FastAPI, File, UploadFile
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from insightface.app import FaceAnalysis
import numpy as np
import cv2
import faiss
import pickle
import os
import uvicorn
import tempfile
from sqlalchemy import text
from db import SessionLocal

faiss_index_name="faiss_index.index"
faiss_label_name="faiss_labels.pkl"

# ✅ FastAPI 앱 생성
router = APIRouter()

# ✅ 모델 및 벡터 인덱스 로드
load_path = os.path.abspath("embedding/person")  # 실제 경로로 변경 필요

model = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0)

index = faiss.read_index(os.path.join(load_path, faiss_index_name))
with open(os.path.join(load_path, faiss_label_name), "rb") as f:
    labels = pickle.load(f)

# ✅ 얼굴 임베딩 추출 함수
def get_face_embedding_from_bytes(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = model.get(img)
    if faces:
        return faces[0].embedding
    return None

# ✅ 얼굴 예측 API
@router.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 1):
    contents = await file.read()
    embedding = get_face_embedding_from_bytes(contents)

    if embedding is None:
        return JSONResponse(status_code=400, content={"message": "❌ 얼굴 인식 실패"})

    # 정규화
    embedding = embedding.astype("float32")
    embedding /= np.linalg.norm(embedding)

    # 검색
    scores, indices = index.search(np.array([embedding]), top_k)
    results = []
    for idx, score in zip(indices[0], scores[0]):
        results.append({
            "label": labels[idx],
            "score": float(score)
        })

    return {"results": results}


@router.get("/")
def hello():
    return {"msg": "predict api"}

@router.get("/dbtest")
def db_test():
    db = SessionLocal()
    try:
        result = db.execute(text("SELECT now();"))
        now = result.fetchone()[0]
        print(f"## now: ",now)
        return {"db_time": now}
    finally:
        db.close()
"""
# ✅ 로컬에서 실행할 경우
if __name__ == "__main__":
    uvicorn.run("predict:app", host="0.0.0.0", port=8000, reload=True)
"""
