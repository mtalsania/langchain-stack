"""Module 00 - Check your setup.

Run:  python 00_check_setup.py
Each line should say OK. Fix any FAIL before moving on.
"""
import os

from dotenv import load_dotenv

load_dotenv()


def check(name, fn):
    try:
        detail = fn()
        print(f"OK    {name}{': ' + detail if detail else ''}")
    except Exception as exc:  # noqa: BLE001 - we want to show any failure
        print(f"FAIL  {name}: {type(exc).__name__}: {exc}")


def env_keys():
    missing = [k for k in ("GROQ_API_KEY", "LANGSMITH_API_KEY") if not os.getenv(k)]
    if os.getenv("PROVIDER", "groq") != "groq":
        missing = [k for k in missing if k != "GROQ_API_KEY"]
    if missing:
        raise RuntimeError(f"missing in .env: {', '.join(missing)}")
    return "keys present"


def llm():
    from common.llm import get_llm

    reply = get_llm().invoke("Reply with exactly: pong")
    return f"{os.getenv('MODEL') or 'default model'} said {reply.content.strip()!r}"


def embeddings():
    from common.embeddings import get_embeddings

    return f"{len(get_embeddings().embed_query('hello'))}-dim vectors (downloads ~130MB once)"


def langsmith():
    from langsmith import Client

    projects = list(Client().list_projects(limit=1))
    return f"connected, tracing={os.getenv('LANGSMITH_TRACING')}, {len(projects)} project(s) visible"


check(".env keys", env_keys)
check("LLM", llm)
check("Local embeddings", embeddings)
check("LangSmith", langsmith)
