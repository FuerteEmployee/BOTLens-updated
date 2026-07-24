import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

# Database configuration: Use Supabase (PostgreSQL) if available, else fallback to SQLite
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")

if SQLALCHEMY_DATABASE_URL and SQLALCHEMY_DATABASE_URL.startswith("postgres://"):
    # Fix for Render/Supabase postgres:// vs postgresql://
    SQLALCHEMY_DATABASE_URL = SQLALCHEMY_DATABASE_URL.replace("postgres://", "postgresql://", 1)

if not SQLALCHEMY_DATABASE_URL:
    # Default to SQLite for local development
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.getenv("DATABASE_DIR", os.path.join(BASE_DIR, "db"))
    os.makedirs(db_path, exist_ok=True)
    SQLALCHEMY_DATABASE_URL = f"sqlite:///{os.path.join(db_path, 'attendance.db')}"

print(f"--- DATABASE URL: {SQLALCHEMY_DATABASE_URL} ---")

# Create engine with appropriate arguments for SQLite vs PostgreSQL
if "sqlite" in SQLALCHEMY_DATABASE_URL:
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, 
        connect_args={"check_same_thread": False, "timeout": 30},
        poolclass=NullPool
    )
else:
    # For PostgreSQL, we don't need check_same_thread
    engine = create_engine(SQLALCHEMY_DATABASE_URL, poolclass=NullPool)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
