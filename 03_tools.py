"""Module 03 - Tools and tool calling, by hand.

What you'll learn
- @tool turns a Python function into something a model can call. The docstring and
  type hints become the tool's description and schema, so write them well.
- The model does NOT run tools. It returns tool_calls; YOUR code runs them and sends
  back ToolMessages. Seeing this loop by hand makes agents (module 05) obvious.

Run:  python 03_tools.py
Check it worked: you see the model request `calculator`, the result, and a final answer.
"""
from langchain_core.messages import HumanMessage, ToolMessage

from common.llm import get_llm
from common.tools import calculator

print("Tool schema the model sees:\n", calculator.tool_call_schema.model_json_schema(), "\n")

llm_with_tools = get_llm().bind_tools([calculator])
messages = [HumanMessage("A Nimbus Trail costs 2690 USD. What is it with an 8% student discount?")]

# Step 1: the model decides to call a tool.
ai_msg = llm_with_tools.invoke(messages)
print("Model asked for:", ai_msg.tool_calls)
messages.append(ai_msg)

# Step 2: we run each requested tool and reply with a ToolMessage (matched by id).
for call in ai_msg.tool_calls:
    result = calculator.invoke(call["args"])
    print(f"Ran {call['name']}({call['args']}) -> {result}")
    messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

# Step 3: the model reads the tool result and answers.
final = llm_with_tools.invoke(messages)
print("\nFinal answer:", final.content)

# Try this: ask a question with no math ("What colour is the sky?") and check
# that tool_calls comes back empty. The model chooses when to use tools.
