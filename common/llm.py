"""One place to create the chat model, so every module uses the same provider.

Switch providers by changing PROVIDER / MODEL in .env. Nothing else changes:
that is the point of LangChain's common chat-model interface.
"""
import os

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

load_dotenv()

DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "gemini": "gemini-2.5-flash",
    "ollama": "qwen3:8b",
}

# Groq's free tier allows ~30 requests/minute. Staying at 1 request every ~2.5s
# keeps agents and eval runs from tripping 429 errors.
_rate_limiter = InMemoryRateLimiter(requests_per_second=0.4, max_bucket_size=3)


def get_llm(temperature: float = 0.0, **kwargs) -> BaseChatModel:
    provider = os.getenv("PROVIDER", "groq").lower()
    model = os.getenv("MODEL") or DEFAULT_MODELS[provider]

    if provider == "groq":
        from langchain_groq import ChatGroq

        extra = {}
        if model.startswith("openai/gpt-oss"):
            # gpt-oss "thinks" before answering, and thinking tokens count against
            # your 8K tokens/minute free limit. Low effort keeps runs cheap.
            extra = {"reasoning_effort": "low"}
        return ChatGroq(
            model=model,
            temperature=temperature,
            rate_limiter=_rate_limiter,
            max_retries=3,
            **extra,
            **kwargs,
        )
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model, temperature=temperature, rate_limiter=_rate_limiter, **kwargs
        )
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, temperature=temperature, **kwargs)
    raise ValueError(f"Unknown PROVIDER={provider!r}. Use groq, gemini or ollama.")
