from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from insightface.app import FaceAnalysis
import numpy as np
import cv2
import faiss
import pickle
import os
import uvicorn
import tempfile

from routers import embed_v2, predict

# ✅ FastAPI 앱 생성
app = FastAPI()

app.include_router(embed_v2.router, prefix="/embed")
app.include_router(predict.router, prefix="/predict")

@app.get("/")
def hello():
    return {"msg": "Hello FastAPI!"}

# ✅ 로컬에서 실행할 경우
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
