"""RAG over the latest 10-K: Items 1 (Business), 1A (Risk Factors), 7 (MD&A).

Pipeline: core.filings cuts the sections -> split into chunks that remember their
item -> embed locally (free, no API key) -> InMemoryVectorStore, saved to disk per
filing so each 10-K is embedded once.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.filings import SECTION_NAMES, Filing, tenk_sections
from core.tickers import lookup

INDEX_DIR = Path(".cache/index")
ITEMS = ("1", "1A", "7")

# ~250 tokens per chunk: small enough that 8-10 retrieved chunks fit Groq's
# free 8K tokens/minute budget alongside the prompt.
_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=120)


def _embeddings() -> Embeddings:
    from common.embeddings import get_embeddings

    return get_embeddings()


def chunk_sections(filing: Filing, sections: dict[str, str]) -> list[Document]:
    docs = []
    for item, text in sections.items():
        if not text:
            continue
        for i, piece in enumerate(_splitter.split_text(text)):
            docs.append(
                Document(
                    page_content=piece,
                    metadata={
                        "item": item,
                        "section": SECTION_NAMES.get(item, f"Item {item}"),
                        "chunk": i,
                        "accession": filing.accession,
                        "filing_date": filing.filing_date,
                        "report_date": filing.report_date,
                        "url": filing.url,
                    },
                )
            )
    return docs


@lru_cache(maxsize=16)
def filing_index(ticker: str, embeddings: Embeddings | None = None) -> tuple[Filing, InMemoryVectorStore]:
    """Vector index for a company's latest 10-K, built once and cached on disk."""
    emb = embeddings or _embeddings()
    co = lookup(ticker)
    filing, sections = tenk_sections(co.cik, ITEMS)
    path = INDEX_DIR / f"{co.cik}-{filing.accession}.json"
    if path.exists():
        return filing, InMemoryVectorStore.load(str(path), embedding=emb)
    store = InMemoryVectorStore(embedding=emb)
    docs = chunk_sections(filing, sections)
    if not docs:
        raise RuntimeError(f"No Items {ITEMS} found in {filing.url}")
    store.add_documents(docs)
    path.parent.mkdir(parents=True, exist_ok=True)
    store.dump(str(path))
    return filing, store


def search(ticker: str, query: str, k: int = 4, item: str | None = None) -> list[Document]:
    _, store = filing_index(ticker.upper())
    flt = (lambda d: d.metadata["item"] == item) if item else None
    return store.similarity_search(query, k=k, filter=flt)


@tool
def search_filing(ticker: str, query: str, item: str | None = None) -> str:
    """Search the company's latest 10-K text for reasons, strategy, segments and risks.

    item optionally limits the search: "1" Business, "1A" Risk Factors, "7" MD&A.
    Use get_metric for numbers; use this for the why.
    """
    docs = search(ticker, query, k=4, item=item)
    return "\n\n".join(
        f"[10-K Item {d.metadata['item']} ({d.metadata['section']}), filed {d.metadata['filing_date']}]\n{d.page_content}"
        for d in docs
    )
