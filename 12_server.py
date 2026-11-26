"""Module 12 - Serve the agent as an API you can deploy for free.

What you'll learn
- How to put a LangGraph app behind a plain HTTP API, with per-user threads
  and the human-approval step exposed as an endpoint.
- Why the managed option (LangSmith Deployment / Agent Server) exists: it gives
  you this plus queues, streaming, cron and scaling. It is a paid feature, so here
  we hand-roll the free version.

Run locally:   uvicorn 12_server:app --reload   (then open http://127.0.0.1:8000/docs)
Check it worked: in the /docs page, POST /chat with
    {"thread_id": "t1", "message": "Open a ticket: my brakes failed"}
You get pending_approval back. POST /resume with {"thread_id": "t1", "action": "approve"}.
Traces still show up in LangSmith because the same env vars are set.
"""
import sqlite3
from typing import Literal

from fastapi import FastAPI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel

from app.graph import build_graph

# SQLite is fine for one small instance. On free hosts the disk is often wiped on
# restart, so memory resets; a free Postgres (e.g. Neon/Supabase) + PostgresSaver fixes that.
conn = sqlite3.connect("server_memory.sqlite", check_same_thread=False)
assistant = build_graph(checkpointer=SqliteSaver(conn))

app = FastAPI(title="Nimbus assistant")


class ChatIn(BaseModel):
    thread_id: str
    message: str


class ResumeIn(BaseModel):
    thread_id: str
    action: Literal["approve", "reject", "edit"]
    args: dict | None = None
    feedback: str | None = None


def _respond(result: dict) -> dict:
    if "__interrupt__" in result:
        return {"reply": None, "pending_approval": result["__interrupt__"][0].value}
    return {"reply": result["messages"][-1].content, "pending_approval": None}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat")
def chat(body: ChatIn):
    config = {"configurable": {"thread_id": body.thread_id}}
    return _respond(assistant.invoke({"messages": [("user", body.message)]}, config))


@app.post("/resume")
def resume(body: ResumeIn):
    config = {"configurable": {"thread_id": body.thread_id}}
    decision = body.model_dump(exclude={"thread_id"}, exclude_none=True)
    return _respond(assistant.invoke(Command(resume=decision), config))
