"""Module 10 - Datasets and evaluations in LangSmith.

What you'll learn
- A dataset = example inputs + reference outputs, stored in LangSmith.
- An experiment runs your app on every example and scores it with evaluators.
- Two kinds of evaluators: plain code (cheap, exact) and LLM-as-judge (flexible).
- Compare experiments side by side after you change a prompt or model.

Run:  python 10_evals.py
Check it worked: the script prints an experiment link. Open it: each row has your app's
answer, the reference, and two scores. Then change MODEL in .env (e.g. to
openai/gpt-oss-20b), rerun, and use "Compare" in LangSmith on the two experiments.

Free-tier note: each example costs ~3 Groq calls (agent x2 + judge). We run one at a
time (max_concurrency=1) so the 30 requests/min limit is never hit. Expect ~1-2 min.
"""
from pydantic import BaseModel, Field
from langsmith import Client

from app.graph import build_graph
from common.llm import get_llm

client = Client()
DATASET = "nimbus-support-qa"

EXAMPLES = [
    ("How many days do I have to return an unused bike?", "37 days from delivery.", ["37"]),
    ("How long is the frame warranty?", "9 years against manufacturing defects.", ["9"]),
    ("What's the cheapest bike and its price?", "The Nimbus Fold at 1,299 USD.", ["Fold", "1,299"]),
    ("When is chat support open?", "Monday to Saturday, 07:30-19:00 CET.", ["07:30", "19:00"]),
    ("Do you ship to Canada?", "No, shipping is only available inside the EU and UK.", ["EU"]),
    (
        "Student price for a Nimbus Trail?",
        "2,690 USD minus 8% = 2,474.80 USD.",
        ["2,474.8", "2474.8"],
    ),
]

# --- 1. Create the dataset once ---
if not client.has_dataset(dataset_name=DATASET):
    ds = client.create_dataset(DATASET, description="Nimbus support questions with known answers")
    client.create_examples(
        dataset_id=ds.id,
        examples=[
            {
                "inputs": {"question": q},
                "outputs": {"answer": a, "must_include": keys},
            }
            for q, a, keys in EXAMPLES
        ],
    )
    print(f"Created dataset {DATASET!r} with {len(EXAMPLES)} examples")

# --- 2. The target: what we're testing (our full LangGraph assistant) ---
assistant = build_graph()


def target(inputs: dict) -> dict:
    result = assistant.invoke({"messages": [("user", inputs["question"])]})
    return {"answer": result["messages"][-1].content}


# --- 3a. Code evaluator: did the answer contain a required fact? ---
def contains_key_fact(outputs: dict, reference_outputs: dict) -> bool:
    answer = outputs["answer"].replace(" ", "")
    return any(k.replace(" ", "") in answer for k in reference_outputs["must_include"])


# --- 3b. LLM-as-judge evaluator ---
class Grade(BaseModel):
    reasoning: str = Field(description="One sentence on why")
    correct: bool = Field(description="True if the answer matches the reference in substance")


judge = get_llm().with_structured_output(Grade)


def judged_correct(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    grade = judge.invoke(
        "You grade a support bot. Is the ANSWER correct given the REFERENCE? "
        "Extra helpful detail is fine; wrong or missing facts are not.\n\n"
        f"QUESTION: {inputs['question']}\nREFERENCE: {reference_outputs['answer']}\n"
        f"ANSWER: {outputs['answer']}"
    )
    return {"key": "judged_correct", "score": grade.correct, "comment": grade.reasoning}


# --- 4. Run the experiment ---
results = client.evaluate(
    target,
    data=DATASET,
    evaluators=[contains_key_fact, judged_correct],
    experiment_prefix="nimbus-assistant",
    max_concurrency=1,
)
print("\nDone. Open the experiment in LangSmith (Datasets & Experiments ->", DATASET, ")")
