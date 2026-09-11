import importlib
import pytest


def test_vector_db_query_importable():
    try:
        vec = importlib.import_module("vector_db")
    except Exception:
        pytest.skip("vector_db or optional deps not installed")

    # If import succeeded, calling query_kb should return a list (may be empty if Qdrant not running)
    res = vec.query_kb("test query", top_k=1)
    assert isinstance(res, list)
