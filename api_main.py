from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

from config import settings  # noqa: E402
from logging_config import configure_logging, get_logger  # noqa: E402

configure_logging(settings.LOG_LEVEL)
logger = get_logger(__name__)

import vector_db  # noqa: E402
from ingestion.api import router as ingestion_router  # noqa: E402
from ingestion.db import init_db as init_ingestion_db  # noqa: E402
from memory.api import router as memory_router  # noqa: E402
from memory.db import init_db as init_memory_db  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_ingestion_db()
    init_memory_db()
    try:
        vector_db.ensure_collection()
    except Exception:
        logger.warning("weaviate_unavailable_at_startup", exc_info=True)
    logger.info("api_startup_complete")
    yield
    logger.info("api_shutdown")


app = FastAPI(
    title="AI Investment Due Diligence Platform - Ingestion + Memory API",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(ingestion_router)
app.include_router(memory_router)


@app.get("/")
async def root() -> dict:
    return {"service": "ingestion-and-memory-api", "docs": "/docs"}
