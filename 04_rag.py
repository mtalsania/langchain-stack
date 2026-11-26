"""Module 04 - Retrieval-augmented generation (RAG).

What you'll learn
- The RAG pipeline: load -> split -> embed -> store -> retrieve -> generate.
- Why we split by markdown headings, and why chunk size matters for token budgets.
- How to make the model cite sources and admit when it doesn't know.

Run:  python 04_rag.py            (first run downloads a ~130MB embedding model)
      python 04_rag.py --rebuild  (after you edit files in ./data)
Check it worked: the return window answer says 37 days (a number only our docs have),
and the question about Mars says it doesn't know.
"""
import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from common.llm import get_llm
from common.retrieval import build_index, format_docs, load_documents, split_documents

# --- Steps 1-2: load and split ---
docs = load_documents()
chunks = split_documents(docs)
print(f"Loaded {len(docs)} files -> {len(chunks)} chunks")
print("Example chunk metadata:", chunks[0].metadata)

# --- Steps 3-4: embed and store (cached to .cache/index.json) ---
store = build_index(rebuild="--rebuild" in sys.argv)

# --- Step 5: retrieve. Look at the scores - this is all "semantic search" is.
print("\nTop matches for 'how long do I have to send a bike back?':")
for doc, score in store.similarity_search_with_score("how long do I have to send a bike back?", k=3):
    print(f"  {score:.3f}  [{doc.metadata['source']}] {doc.page_content[:70]!r}...")

# --- Step 6: generate an answer grounded in the retrieved chunks ---
retriever = store.as_retriever(search_kwargs={"k": 3})
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer using ONLY the context below. Cite sources like [products.md]. "
            "If the context doesn't contain the answer, say \"I don't know.\"\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | get_llm()
    | StrOutputParser()
)

for q in [
    "How many days do I have to return a bike?",
    "Which bike is lightest and how far does it go?",
    "Do you ship to Mars?",
]:
    print(f"\nQ: {q}\nA: {rag_chain.invoke(q)}")

# Try this: add a new file to ./data (say, a FAQ), rerun with --rebuild, and ask about it.
