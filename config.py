from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Vector DB (Weaviate)
    WEAVIATE_URL: str = Field("http://localhost:8080", env="WEAVIATE_URL")
    WEAVIATE_API_KEY: str | None = Field(None, env="WEAVIATE_API_KEY")
    WEAVIATE_COLLECTION: str = Field("due_diligence_documents", env="WEAVIATE_COLLECTION")

    # Embedding model (shared by retrieval and ingestion)
    EMBED_MODEL: str = Field("all-MiniLM-L6-v2", env="EMBED_MODEL")

    # Ingestion pipeline
    CHUNK_SIZE: int = Field(1000, env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(150, env="CHUNK_OVERLAP")
    MAX_UPLOAD_SIZE_MB: int = Field(20, env="MAX_UPLOAD_SIZE_MB")
    UPLOAD_DIR: str = Field("data/uploads", env="UPLOAD_DIR")
    INGESTION_DB_PATH: str = Field("data/ingestion.db", env="INGESTION_DB_PATH")
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    # Resource-exhaustion guards: the upload-size cap only bounds bytes on disk, not
    # what a PDF/DOCX expands to once parsed (e.g. a zip-bomb-shaped .docx, or a
    # legitimately huge page count) -- these bound the actually expensive work.
    MAX_EXTRACTED_CHARS: int = Field(3_000_000, env="MAX_EXTRACTED_CHARS")
    MAX_PDF_PAGES: int = Field(2000, env="MAX_PDF_PAGES")
    MAX_DOCX_UNCOMPRESSED_BYTES: int = Field(200 * 1024 * 1024, env="MAX_DOCX_UNCOMPRESSED_BYTES")
    MAX_CHUNKS_PER_DOCUMENT: int = Field(2000, env="MAX_CHUNKS_PER_DOCUMENT")
    # Shared-secret auth for the ingestion API (checked against the X-API-Key header).
    # Left unset by default for local/dev use; the router logs a warning at startup
    # when it's unset so an unauthenticated deployment is never silent.
    INGESTION_API_KEY: str | None = Field(None, env="INGESTION_API_KEY")

    # Shared-secret gate on the Streamlit app itself (frontend.py) -- separate from
    # INGESTION_API_KEY because the app is a different audience/entry point (a human
    # in a browser, not an API caller) and calls the ingestion service in-process,
    # bypassing the API's own key check entirely. Enforced in frontend.py immediately
    # after st.set_page_config(), before any other content renders. Left unset by
    # default for local/dev use; the app shows a visible warning when it's unset so
    # an unauthenticated deployment is never silent.
    APP_ACCESS_KEY: str | None = Field(None, env="APP_ACCESS_KEY")

    # Retrieval pipeline (query rewriting, semantic search, reranking, compression)
    RERANKER_MODEL: str = Field("cross-encoder/ms-marco-MiniLM-L-6-v2", env="RERANKER_MODEL")
    RETRIEVAL_TOP_K_PER_QUERY: int = Field(5, env="RETRIEVAL_TOP_K_PER_QUERY")
    RETRIEVAL_RERANK_TOP_N: int = Field(8, env="RETRIEVAL_RERANK_TOP_N")
    RETRIEVAL_MAX_CHARS_PER_SNIPPET: int = Field(500, env="RETRIEVAL_MAX_CHARS_PER_SNIPPET")
    RETRIEVAL_MAX_TOTAL_CHARS: int = Field(4000, env="RETRIEVAL_MAX_TOTAL_CHARS")

    # Memory system (conversation/session/long-term/research history)
    MEMORY_DB_PATH: str = Field("data/memory.db", env="MEMORY_DB_PATH")
    MEMORY_MAX_MESSAGE_CHARS: int = Field(8000, env="MEMORY_MAX_MESSAGE_CHARS")
    MEMORY_SUMMARIZATION_THRESHOLD_CHARS: int = Field(1500, env="MEMORY_SUMMARIZATION_THRESHOLD_CHARS")
    MEMORY_MAX_USER_FACTS_IN_CONTEXT: int = Field(10, env="MEMORY_MAX_USER_FACTS_IN_CONTEXT")
    MEMORY_LAUNCH_HISTORY_COLLECTION: str = Field("research_history", env="MEMORY_LAUNCH_HISTORY_COLLECTION")

    # Report generation tuning. A report fans out one LLM call per plan task, in
    # parallel, on top of router/orchestrator/reducer calls -- easily 10-15 calls,
    # concurrently. On a tight LLM rate limit (e.g. a free tier) lower MAX_PLAN_TASKS
    # and raise WORKER_STAGGER_SECONDS so the section-writer calls spread out in time
    # instead of all hitting the provider at once. STREAM_TOKENS controls whether the
    # UI attempts live token-by-token streaming -- it's the flakiest/most provider-
    # dependent path and the expensive one to retry, so it's off by default.
    MAX_PLAN_TASKS: int = Field(6, env="MAX_PLAN_TASKS")
    WORKER_STAGGER_SECONDS: float = Field(0.0, env="WORKER_STAGGER_SECONDS")
    STREAM_TOKENS: bool = Field(False, env="STREAM_TOKENS")

    # Search: caching, KB-sufficiency gating, web search hardening
    SEARCH_CACHE_TTL_SECONDS: int = Field(600, env="SEARCH_CACHE_TTL_SECONDS")
    SEARCH_CACHE_MAX_ENTRIES: int = Field(256, env="SEARCH_CACHE_MAX_ENTRIES")
    KB_SUFFICIENCY_MIN_ITEMS: int = Field(3, env="KB_SUFFICIENCY_MIN_ITEMS")
    KB_SUFFICIENCY_MIN_CONFIDENCE: float = Field(0.5, env="KB_SUFFICIENCY_MIN_CONFIDENCE")
    WEB_SEARCH_TIMEOUT_SECONDS: int = Field(10, env="WEB_SEARCH_TIMEOUT_SECONDS")
    WEB_SEARCH_MAX_QUERY_CHARS: int = Field(500, env="WEB_SEARCH_MAX_QUERY_CHARS")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
