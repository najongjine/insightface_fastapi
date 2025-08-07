from fastapi import FastAPI
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import os
import cv2
import faiss
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import insightface
import albumentations as A
from sqlalchemy import text
from db import SessionLocal

# 🔧 경로 설정 (하드코딩)
data_folder = os.path.abspath("person") 
save_path = os.path.abspath("embedding/person") 

# ✅ FastAPI 인스턴스 생성
router = APIRouter()

# 🔧 증강 설정
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.3),
])

# 🚀 모델 초기화 함수
def load_face_model(device: str = "cpu"):
    providers = ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider"]
    model = insightface.app.FaceAnalysis(name='buffalo_l', providers=providers)
    model.prepare(ctx_id=0 if device != "cpu" else -1)
    return model

# 🚀 임베딩 추출 함수
def get_face_embedding(image_path: str, model, n_augment: int = 5):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    embeddings = []

    # 원본
    faces = model.get(img)
    if faces:
        embeddings.append(faces[0].embedding)
    else:
        print(f"❌ 얼굴 인식 실패 (원본): {image_path}")

    # 증강
    for i in range(n_augment):
        augmented = augment(image=img)
        img_aug = augmented['image']
        faces = model.get(img_aug)
        if faces:
            embeddings.append(faces[0].embedding)
        else:
            print(f"❌ 얼굴 인식 실패 (증강 {i+1}): {image_path}")

    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        print(f"❌ 모든 시도 실패: {image_path}")
        return None

# 🚀 폴더 스캔 및 임베딩 추출
def process_folder(data_folder: str, model) -> pd.DataFrame:
    data = []
    data_path = Path(data_folder)
    for person_dir in data_path.iterdir():
        if not person_dir.is_dir():
            continue
        label = person_dir.name
        print(f"▶ 폴더: {label}")
        count = 0
        for image_path in person_dir.glob("*"):
            if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            emb = get_face_embedding(image_path, model)
            if emb is not None:
                data.append({
                    "label": label,
                    "image_path": str(image_path),
                    "embedding": emb
                })
                count += 1
        print(f"✅ 얼굴 인식 성공 수: {count}")
    return pd.DataFrame(data)

# 🚀 FAISS 인덱스 생성 및 저장
def build_and_save_faiss(train_df: pd.DataFrame, save_path: str):
    embeddings = np.stack(train_df['embedding'].values).astype('float32')
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, os.path.join(save_path, "faiss_index.index"))

    labels = train_df['label'].tolist()
    with open(os.path.join(save_path, "faiss_labels.pkl"), "wb") as f:
        pickle.dump(labels, f)

    train_df.to_pickle(os.path.join(save_path, "train_df.pkl"))

    print("✅ FAISS 인덱스 & 라벨 저장 완료")
    return index, labels, train_df

# 🚀 전체 실행 함수
def run_pipeline(data_folder: str, save_path: str, device: str = "cpu"):
    os.makedirs(save_path, exist_ok=True)
    print("🚀 얼굴 모델 불러오는 중...")
    model = load_face_model(device)

    print("🚀 임베딩 추출 시작...")
    train_df = process_folder(data_folder, model)

    print("🚀 FAISS 인덱스 생성 및 저장 중...")
    index, labels, df = build_and_save_faiss(train_df, save_path)

    return len(df)

# ✅ API 엔드포인트
@router.post("/train")
def train_faces():
    try:
        count = run_pipeline(data_folder, save_path)
        return {"status": "success", "count": count}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@router.get("/")
def root():
    return {"msg": "얼굴 임베딩 생성 API"}

@router.get("/dbtest")
def db_test():
    db = SessionLocal()
    try:
        result = db.execute("SELECT now();")
        now = result.fetchone()[0]
        return {"db_time": str(now)}
    finally:
        db.close()

"""
# ✅ 로컬 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("embed_v2:app", host="0.0.0.0", port=8000, reload=True)
"""
