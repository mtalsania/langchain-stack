"""Build (or load) the vector index over ./data and expose a retriever.

Module 04 walks through each step of this by hand. Later modules just call
get_retriever().
"""
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from common.embeddings import get_embeddings

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_PATH = ROOT / ".cache" / "index.json"


def load_documents() -> list[Document]:
    docs = []
    for path in sorted(DATA_DIR.glob("*.md")):
        docs.append(Document(page_content=path.read_text(), metadata={"source": path.name}))
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    # First split on markdown headings so each chunk stays on one topic,
    # then cap chunk size so a retrieved chunk never blows the token budget.
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section")], strip_headers=False
    )
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=80)
    chunks = []
    for doc in docs:
        for piece in header_splitter.split_text(doc.page_content):
            piece.metadata["source"] = doc.metadata["source"]
            chunks.extend(size_splitter.split_documents([piece]))
    return chunks


def build_index(rebuild: bool = False) -> InMemoryVectorStore:
    embeddings = get_embeddings()
    if INDEX_PATH.exists() and not rebuild:
        return InMemoryVectorStore.load(str(INDEX_PATH), embedding=embeddings)
    store = InMemoryVectorStore(embedding=embeddings)
    store.add_documents(split_documents(load_documents()))
    INDEX_PATH.parent.mkdir(exist_ok=True)
    store.dump(str(INDEX_PATH))
    return store


def get_retriever(k: int = 3):
    return build_index().as_retriever(search_kwargs={"k": k})


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)
