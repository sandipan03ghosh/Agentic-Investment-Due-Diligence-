from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config import settings

from .models import Base

_db_path = Path(settings.MEMORY_DB_PATH)
_db_path.parent.mkdir(parents=True, exist_ok=True)

_engine = create_engine(f"sqlite:///{_db_path.as_posix()}", connect_args={"check_same_thread": False})
_SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=_engine)


@contextmanager
def get_session() -> Iterator[Session]:
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
