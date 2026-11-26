"""Module 08 - Human in the loop.

What you'll learn
- interrupt() pauses a graph mid-run and saves everything (needs a checkpointer).
- Command(resume=...) continues from exactly that point with the human's answer.
- This is how you gate risky actions: sending emails, refunds, opening tickets.

Read app/graph.py -> human_review() alongside this file.

Run:  python 08_human_in_the_loop.py
Check it worked: the run pauses and shows the ticket the agent wants to create.
Try all three answers on different runs: approve, edit (change the priority),
and reject (the agent should tell the customer no ticket was created).
"""
import json
import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.graph import build_graph

assistant = build_graph(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

request = (
    "The battery on my Nimbus Commute is swelling and the case is cracked. "
    "Please open a support ticket for me."
)
print(f"you: {request}\n")
result = assistant.invoke({"messages": [("user", request)]}, config)

while "__interrupt__" in result:
    payload = result["__interrupt__"][0].value
    print("PAUSED. The agent wants to run:")
    print(json.dumps(payload["tool_calls"], indent=2))
    choice = input("\n[a]pprove, [e]dit, or [r]eject? ").strip().lower()

    if choice.startswith("a"):
        decision = {"action": "approve"}
    elif choice.startswith("e"):
        args = dict(payload["tool_calls"][0]["args"])
        args["priority"] = input(f"New priority (was {args.get('priority')}): ").strip() or args["priority"]
        args["summary"] = input("New summary (enter to keep): ").strip() or args["summary"]
        decision = {"action": "edit", "args": args}
    else:
        decision = {"action": "reject", "feedback": input("Reason: ").strip() or "Rejected."}

    result = assistant.invoke(Command(resume=decision), config)

print(f"\nbot: {result['messages'][-1].content}")
