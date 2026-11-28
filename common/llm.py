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


SMALL_DEFAULTS = {
    "groq": "openai/gpt-oss-20b",
    "gemini": "gemini-2.5-flash-lite",
    "ollama": "qwen3:8b",
}


def get_llm(temperature: float = 0.0, small: bool = False, **kwargs) -> BaseChatModel:
    """The strong model by default; small=True for cheap steps (planning, extraction, eval QA).

    On Groq each model has its own daily token quota, so routing cheap steps to the
    small model also stretches the free tier.
    """
    provider = os.getenv("PROVIDER", "groq").lower()
    if small:
        model = os.getenv("SMALL_MODEL") or SMALL_DEFAULTS[provider]
    else:
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


# ------------------------------------------------------- structured output ----
def structured(llm: BaseChatModel, schema):
    """llm.with_structured_output(schema), made reliable on Groq.

    The default (function calling) lets gpt-oss emit fields in the wrong shape, e.g. a
    list where the schema says string, and Groq rejects the whole response with 400
    "tool call validation failed". On gpt-oss, Groq's strict json_schema mode constrains
    decoding to the schema, so a wrong shape cannot be generated. If Groq ever refuses
    the strict request, fall back to function calling with a couple of retries.
    """
    if getattr(llm, "model_name", "") in ("openai/gpt-oss-20b", "openai/gpt-oss-120b"):
        strict = llm.with_structured_output(schema, method="json_schema", strict=True)
        loose = llm.with_structured_output(schema).with_retry(stop_after_attempt=3)
        return strict.with_fallbacks([loose])
    return llm.with_structured_output(schema)


# ---------------------------------------------------------------- resilience ----
def is_retryable(exc: Exception) -> bool:
    """Errors worth another try on the free tier.

    - Groq "output_parse_failed": gpt-oss sometimes emits a malformed tool call
      (its reasoning text, e.g. "Return value.", instead of the call). A retry usually works.
    - Rate limits and 5xx.
    """
    text = str(exc)
    status = getattr(exc, "status_code", None)
    return "output_parse_failed" in text or status == 429 or (status is not None and status >= 500)


def resilient_middleware(fallback_to_large: bool = True) -> list:
    """Agent middleware: retry flaky calls, then fall back to the large model.

    This is LangChain middleware doing what you'd otherwise hand-write around every
    model call. Order matters: the first entry wraps the rest, so the fallback only
    kicks in after the retries are used up.
    """
    from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware

    retry = ModelRetryMiddleware(max_retries=2, retry_on=is_retryable, on_failure="error", initial_delay=2.0)
    return [ModelFallbackMiddleware(get_llm()), retry] if fallback_to_large else [retry]
