"""Module 02 - Structured output.

What you'll learn
- with_structured_output() makes the model return a validated Pydantic object,
  not free text. This is how you extract data reliably.

Run:  python 02_structured_output.py
Check it worked: you get Ticket objects with the right priority (the brake one is P1).
"""
from typing import Literal

from pydantic import BaseModel, Field

from common.llm import get_llm


class Ticket(BaseModel):
    """A support ticket extracted from a customer message."""

    summary: str = Field(description="One-line summary of the problem")
    bike_model: Literal["Commute", "Trail", "Cargo", "Fold", "unknown"]
    priority: Literal["P1", "P2", "P3"] = Field(
        description="P1 = rider safety (brakes, frame, battery swelling), "
        "P2 = bike unusable but safe, P3 = questions/billing/accessories"
    )
    customer_mood: Literal["calm", "annoyed", "angry"]


extractor = get_llm().with_structured_output(Ticket)

messages = [
    "My Trail's rear brake barely works since yesterday, I nearly hit a car!!",
    "hey, can I get an invoice for my Fold order? need it for taxes",
    "The Cargo won't turn on at all. Battery shows full. Kids are waiting for school run. Ugh.",
]

for text in messages:
    ticket = extractor.invoke(text)
    print(f"\n{text}\n -> {ticket!r}")

# Try this: add a field `needs_callback: bool` and see the model fill it in.
