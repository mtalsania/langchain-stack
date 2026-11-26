"""Module 06 - LangGraph from scratch.

What you'll learn
- State: a typed dict every node reads and updates. Reducers decide how updates merge.
- Nodes are plain functions. Edges (fixed or conditional) decide what runs next.
- create_agent from module 05 is just a prebuilt graph. Here you build one yourself,
  which is what you do when you need control the prebuilt can't give you.

Run:  python 06_langgraph_basics.py
Check it worked: part A prints the step log; part B prints a Mermaid diagram and answers
using search_docs. Paste the diagram into https://mermaid.live to see it.
"""
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

# ---------------- Part A: a tiny graph with no LLM at all ----------------
print("--- Part A: state, nodes, edges, reducers ---")


class OrderState(TypedDict):
    price: float
    is_student: bool
    # Annotated with operator.add = "append to the list" instead of "overwrite".
    log: Annotated[list[str], operator.add]


def apply_discount(state: OrderState) -> dict:
    new_price = round(state["price"] * 0.92, 2)
    return {"price": new_price, "log": [f"student discount -> {new_price}"]}


def add_shipping(state: OrderState) -> dict:
    return {"log": ["EU shipping is free"]}  # we only return what changed


def is_student(state: OrderState) -> str:
    return "apply_discount" if state["is_student"] else "add_shipping"


g = StateGraph(OrderState)
g.add_node("apply_discount", apply_discount)
g.add_node("add_shipping", add_shipping)
g.add_conditional_edges(START, is_student)  # branch
g.add_edge("apply_discount", "add_shipping")
g.add_edge("add_shipping", END)
order_graph = g.compile()

print(order_graph.invoke({"price": 2690, "is_student": True, "log": []}))
print(order_graph.invoke({"price": 2690, "is_student": False, "log": []}))

# ---------------- Part B: the real assistant graph ----------------
print("\n--- Part B: the Nimbus assistant graph (app/graph.py) ---")
from app.graph import build_graph  # noqa: E402

assistant = build_graph()
print(assistant.get_graph().draw_mermaid())

result = assistant.invoke({"messages": [("user", "What does the battery warranty cover?")]})
for m in result["messages"]:
    label = m.type.upper()
    text = m.content if m.content else f"tool_calls={[c['name'] for c in m.tool_calls]}"
    print(f"{label:>6}: {str(text)[:200]}")

# Read app/graph.py now. Every line should make sense. The human_review node is
# explained in module 08.
