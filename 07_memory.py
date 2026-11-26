"""Module 07 - Memory with checkpointers and threads.

What you'll learn
- A graph has no memory by default. A checkpointer saves state after every step.
- thread_id picks which conversation to load. Same id = it remembers; new id = fresh.
- SqliteSaver writes to disk, so memory survives restarts. Run this file twice.
- get_state / get_state_history let you inspect (and time-travel) a conversation.

Run:  python 07_memory.py        then run it again
Check it worked: thread "mihir-1" remembers your name and bike; thread "stranger" doesn't.
On the second run, the very first answer already knows your name (loaded from disk).
"""
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from app.graph import build_graph

conn = sqlite3.connect("memory.sqlite", check_same_thread=False)
assistant = build_graph(checkpointer=SqliteSaver(conn))


def chat(thread_id: str, text: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    result = assistant.invoke({"messages": [("user", text)]}, config)
    print(f"[{thread_id}] you: {text}\n[{thread_id}] bot: {result['messages'][-1].content}\n")


chat("mihir-1", "What's my name and which bike do I own, if you know?")
chat("mihir-1", "My name is Mihir and I ride a Nimbus Fold.")
chat("mihir-1", "What's the range of my bike?")
chat("stranger", "Which bike do I own?")

# Inspect the saved state.
config = {"configurable": {"thread_id": "mihir-1"}}
state = assistant.get_state(config)
print(f"mihir-1 has {len(state.values['messages'])} messages saved.")
print(f"Checkpoints in history: {sum(1 for _ in assistant.get_state_history(config))}")

# Free-tier note: the whole history is resent to the model on every turn, so long
# threads burn tokens fast (Groq free: ~8K tokens/minute). Module 12's README section
# mentions trimming / summarizing history as the fix.
