import pytest

from turkish_legal_assistant import vectors


@pytest.fixture(autouse=True)
def no_embedding_model(monkeypatch):
    """Never load the 2.3 GB BGE-M3 model in tests: without an index, search falls back to keywords."""

    def missing():
        raise FileNotFoundError("no embedding index in tests")

    monkeypatch.setattr(vectors, "get_index", missing)
