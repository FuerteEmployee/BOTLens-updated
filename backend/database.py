import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

# A small local SQLite file holding only the outbox (see models.py).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.getenv("DATABASE_DIR", os.path.join(BASE_DIR, "db"))
os.makedirs(db_path, exist_ok=True)
SQLALCHEMY_DATABASE_URL = f"sqlite:///{os.path.join(db_path, 'outbox.db')}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30},
    poolclass=NullPool,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
