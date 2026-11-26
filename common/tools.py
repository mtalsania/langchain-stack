"""The tools our assistant can use. Shared by modules 05-12."""
import ast
import operator
import uuid

from langchain_core.tools import tool

from common.retrieval import format_docs, get_retriever


@tool
def search_docs(query: str) -> str:
    """Search the Nimbus Bikes knowledge base (products, prices, returns, warranty,
    support hours, shipping). Use this for any question about Nimbus Bikes."""
    return format_docs(get_retriever().invoke(query))


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Only numbers and + - * / ** are allowed")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as '2690 * 0.92'. Use it for any math."""
    return str(round(_eval(ast.parse(expression, mode="eval").body), 2))


@tool
def create_support_ticket(summary: str, priority: str) -> str:
    """Open a support ticket for the customer. priority must be P1, P2 or P3
    (P1 = safety issue). Only call this when the customer asks for a ticket."""
    # In a real app this would call a helpdesk API. That side effect is why
    # module 08 makes a human approve this tool before it runs.
    ticket_id = f"NB-{uuid.uuid4().hex[:6].upper()}"
    return f"Created ticket {ticket_id} ({priority}): {summary}"


SAFE_TOOLS = [search_docs, calculator]
SENSITIVE_TOOLS = [create_support_ticket]
ALL_TOOLS = SAFE_TOOLS + SENSITIVE_TOOLS
