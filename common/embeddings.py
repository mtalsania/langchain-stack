"""Free, local embeddings.

FastEmbed runs a small ONNX model on your CPU (about 130 MB, downloaded once).
No API key and no rate limit, so retrieval never costs you free-tier quota.

This is also a good look at how small LangChain's Embeddings interface is:
you only need embed_documents() and embed_query().
"""
from functools import lru_cache

from langchain_core.embeddings import Embeddings

MODEL_NAME = "BAAI/bge-small-en-v1.5"


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str = MODEL_NAME):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


@lru_cache
def get_embeddings() -> Embeddings:
    return LocalEmbeddings()
