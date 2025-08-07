# db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# PostgreSQL 연결 정보 (변경하세요)
DB_URL = "postgresql://postgres:aaaa@localhost:5432/test1"

# SQLAlchemy 엔진과 세션 팩토리
engine = create_engine(DB_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
