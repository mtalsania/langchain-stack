"""Polite HTTP client for SEC EDGAR (data.sec.gov and www.sec.gov).

The SEC's fair-access rules, which this client follows:
- every request carries a User-Agent with a name and contact email
  (read from the SEC_USER_AGENT env var, never hard-coded);
- no more than 10 requests per second;
- back off when the SEC answers 429 or 503.

Responses are cached on disk so a re-run costs zero requests. Filing documents
under /Archives/ never change once filed, so they are cached forever; the JSON
APIs (tickers, submissions, companyfacts) are refreshed after `ttl` seconds.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

DAY = 24 * 60 * 60
FOREVER = None  # ttl=None means "never expires"

DEFAULT_CACHE_DIR = Path(".cache/sec")
SEC_MAX_RPS = 10  # SEC limit: https://www.sec.gov/os/webmaster-faq#developers


class SECConfigError(RuntimeError):
    """SEC_USER_AGENT is missing or does not look like 'Name email@domain'."""


class SECHTTPError(RuntimeError):
    def __init__(self, url: str, status: int, body: str = ""):
        super().__init__(f"SEC request failed ({status}) for {url}: {body[:200]}")
        self.url = url
        self.status = status


_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


def user_agent_from_env() -> str:
    load_dotenv()
    ua = (os.getenv("SEC_USER_AGENT") or "").strip().strip('"').strip("'")
    if not ua or not _EMAIL.search(ua) or "example.com" in ua:
        raise SECConfigError(
            "Set SEC_USER_AGENT in .env to your name and email, e.g. "
            'SEC_USER_AGENT="Jane Doe jane@yourdomain.com". The SEC rejects requests without it.'
        )
    return ua


class RateLimiter:
    """Sliding window: at most `max_calls` in any `period` seconds. Thread-safe."""

    def __init__(self, max_calls: int = SEC_MAX_RPS, period: float = 1.0, clock=time.monotonic, sleep=time.sleep):
        self.max_calls = max_calls
        self.period = period
        self._clock = clock
        self._sleep = sleep
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            while True:
                now = self._clock()
                while self._calls and now - self._calls[0] >= self.period:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                self._sleep(self.period - (now - self._calls[0]))


class SECClient:
    def __init__(
        self,
        user_agent: str | None = None,
        cache_dir: Path | str | None = None,
        max_rps: int | None = None,
        session: requests.Session | None = None,
        max_retries: int = 4,
    ):
        self.user_agent = user_agent or user_agent_from_env()
        self.cache_dir = Path(cache_dir or os.getenv("SEC_CACHE_DIR") or DEFAULT_CACHE_DIR)
        self.limiter = RateLimiter(max_calls=min(max_rps or SEC_MAX_RPS, SEC_MAX_RPS))
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"})
        self.max_retries = max_retries
        self.requests_made = 0  # network calls, not cache hits; handy for tests and demos

    # -- cache -------------------------------------------------------------
    def _cache_path(self, url: str) -> Path:
        return self.cache_dir / hashlib.sha256(url.encode()).hexdigest()[:32]

    def _read_cache(self, url: str, ttl: float | None) -> bytes | None:
        path = self._cache_path(url)
        if not path.exists():
            return None
        if ttl is not None and time.time() - path.stat().st_mtime > ttl:
            return None
        return path.read_bytes()

    def _write_cache(self, url: str, body: bytes) -> None:
        path = self._cache_path(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(body)
        tmp.replace(path)

    # -- network -----------------------------------------------------------
    def get_bytes(self, url: str, ttl: float | None = DAY) -> bytes:
        cached = self._read_cache(url, ttl)
        if cached is not None:
            return cached

        delay = 1.0
        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            self.requests_made += 1
            resp = self.session.get(url, timeout=30)
            if resp.status_code == 200:
                self._write_cache(url, resp.content)
                return resp.content
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                retry_after = resp.headers.get("Retry-After", "")
                time.sleep(float(retry_after) if retry_after.isdigit() else delay)
                delay *= 2
                continue
            raise SECHTTPError(url, resp.status_code, resp.text)
        raise SECHTTPError(url, -1, "retries exhausted")  # pragma: no cover

    def get_json(self, url: str, ttl: float | None = DAY) -> Any:
        return json.loads(self.get_bytes(url, ttl))

    def get_text(self, url: str, ttl: float | None = FOREVER) -> str:
        return self.get_bytes(url, ttl).decode("utf-8", errors="replace")


_default: SECClient | None = None


def default_client() -> SECClient:
    """One shared client per process, so the rate limit is shared too."""
    global _default
    if _default is None:
        _default = SECClient()
    return _default
