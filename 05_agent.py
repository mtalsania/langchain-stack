"""Module 05 - Your first agent with create_agent.

What you'll learn
- An agent = a model + tools + a loop that keeps going until the model stops
  calling tools. Module 03's manual loop, automated.
- create_agent (LangChain v1) builds that loop for you, on top of LangGraph.
- Streaming the agent's steps shows you how it "thinks".

Run:  python 05_agent.py
Check it worked: you see search_docs, then calculator, then an answer of
3,695.40 USD (Cargo 3,420 x 0.92 = 3,146.40, plus a 549 USD spare battery).
If the number is different, look at which step went wrong. That is what tracing
in module 09 makes easy.
"""
from langchain.agents import create_agent

from app.graph import SYSTEM_PROMPT
from common.llm import get_llm
from common.tools import SAFE_TOOLS

agent = create_agent(model=get_llm(), tools=SAFE_TOOLS, system_prompt=SYSTEM_PROMPT)

question = (
    "I'm a student. What would a Nimbus Cargo plus one spare battery for it cost me "
    "with my student discount? Discount applies to the bike only."
)
print(f"Q: {question}\n")

# stream_mode="updates" yields one event per node (model call or tool run).
for step in agent.stream({"messages": [("user", question)]}, stream_mode="updates"):
    for node, update in step.items():
        for msg in update["messages"]:
            if getattr(msg, "tool_calls", None):
                for call in msg.tool_calls:
                    print(f"[{node}] wants {call['name']}({call['args']})")
            elif msg.type == "tool":
                print(f"[{node}] {msg.name} returned {msg.content[:120]!r}...")
            else:
                print(f"\n[{node}] FINAL: {msg.content}")

# Try this: ask "What's the weather in Paris?" - the agent has no tool for it.
# Does it admit that, or make something up? That's what evals (module 10) catch.
