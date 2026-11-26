"""Module 01 - Chat models, messages, prompts, and chains (LCEL).

What you'll learn
- A chat model takes a list of messages and returns an AIMessage.
- Prompt templates turn variables into messages.
- The `|` operator chains Runnables: prompt | model | parser.
- Every Runnable has the same methods: invoke, batch, stream.

Run:  python 01_models_and_prompts.py
Check it worked: you see three answers, a batch of two, then tokens streaming in.
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from common.llm import get_llm

llm = get_llm()

# 1) Raw messages in, AIMessage out.
print("\n--- 1. Messages ---")
ai_msg = llm.invoke(
    [
        SystemMessage("You answer in one short sentence."),
        HumanMessage("What is an e-bike?"),
    ]
)
print(type(ai_msg).__name__, "->", ai_msg.content)
# Look at what else comes back: token counts are what your free tier is metered on.
print("usage:", ai_msg.usage_metadata)

# 2) A prompt template with variables.
print("\n--- 2. Prompt template ---")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a {persona}. Answer in at most {n} words."),
        ("human", "{question}"),
    ]
)
print(prompt.invoke({"persona": "bike mechanic", "n": 20, "question": "Why do chains squeak?"}))

# 3) A chain: prompt -> model -> plain string.
print("\n--- 3. Chain (LCEL) ---")
chain = prompt | llm | StrOutputParser()
print(chain.invoke({"persona": "bike mechanic", "n": 20, "question": "Why do chains squeak?"}))

# 4) Same chain, many inputs at once.
print("\n--- 4. Batch ---")
answers = chain.batch(
    [
        {"persona": "pirate", "n": 12, "question": "Describe a bicycle."},
        {"persona": "lawyer", "n": 12, "question": "Describe a bicycle."},
    ]
)
for a in answers:
    print("-", a)

# 5) Stream tokens as they arrive (this is how chat UIs feel fast).
print("\n--- 5. Stream ---")
for chunk in chain.stream({"persona": "poet", "n": 40, "question": "Describe a rainy commute."}):
    print(chunk, end="", flush=True)
print()

# Try this: change MODEL in .env to openai/gpt-oss-20b and rerun. Same code, new model.
