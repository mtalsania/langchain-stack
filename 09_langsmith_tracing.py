"""Module 09 - LangSmith: tracing, metadata, and the Prompt Hub.

What you'll learn
- Tracing is ON just by setting LANGSMITH_TRACING=true + LANGSMITH_API_KEY in .env.
  Every module you already ran has been traced. Go look: smith.langchain.com ->
  Tracing -> project "langchain-stack-course".
- @traceable traces your own Python functions too, not just LangChain calls.
- Tags and metadata make traces filterable (by user, feature, version...).
- The Prompt Hub stores versioned prompts outside your code.

Run:  python 09_langsmith_tracing.py
Check it worked:
 1. In LangSmith, open the newest trace named "answer_customer". Expand it: you see
    the retrieval, the prompt, the model call, token counts and latency for each step.
 2. Filter traces by tag "module-09" or metadata customer_tier = "student".
 3. Open Prompts in the left sidebar: "nimbus-rag" is there with a commit.

Free plan budget: 5,000 base traces per month, kept 14 days. One script run is a few
traces. You'd have to run evals in a loop for hours to hit it, but watch Usage anyway.
"""
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client, traceable

from common.llm import get_llm
from common.retrieval import format_docs, get_retriever

if os.getenv("LANGSMITH_TRACING", "").lower() != "true":
    raise SystemExit("Set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in .env first.")

client = Client()

# --- 1. Push a prompt to the Prompt Hub (a new commit only if it changed) ---
rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are Nimbus Bikes support. Answer using ONLY the context. Cite sources "
            "like [products.md]. If unsure, say you don't know.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ]
)
try:
    url = client.push_prompt("nimbus-rag", object=rag_prompt, description="Course module 09")
    print("Pushed prompt:", url)
except Exception as exc:  # pushing an unchanged prompt raises a conflict error
    print("Prompt already up to date:", type(exc).__name__)

# --- 2. Pull it back. Edit it in the LangSmith UI, rerun, and see the change here. ---
hub_prompt = client.pull_prompt("nimbus-rag")
chain = hub_prompt | get_llm() | StrOutputParser()


# --- 3. Trace your own function, with tags and metadata ---
@traceable(name="answer_customer", tags=["module-09"])
def answer_customer(question: str) -> str:
    docs = get_retriever().invoke(question)
    return chain.invoke({"context": format_docs(docs), "question": question})


for q, tier in [
    ("Is there a student discount and does it stack with other offers?", "student"),
    ("How long is the frame warranty?", "regular"),
]:
    # langsmith_extra attaches metadata to this one trace at call time.
    answer = answer_customer(q, langsmith_extra={"metadata": {"customer_tier": tier}})
    print(f"\nQ: {q}\nA: {answer}")

print("\nNow open https://smith.langchain.com and find these traces.")
