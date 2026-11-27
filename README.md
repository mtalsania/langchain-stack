# Watchlist Intelligence Agent

A LangChain + LangGraph + LangSmith project, built on free tiers only, using real public data.

**Main outcome:** you keep a watchlist of public companies. Each week, or when one of them files
something new with the SEC, the agent reads the filings, updates the numbers (checked against the
SEC's official XBRL data), compares the companies with each other, flags new or dropped risk factors,
and drafts an investment-style memo that waits for your review before it is final.

Company research is one step inside that. Later add-ons reuse the same core: a fact-checker,
a screener, a credit/supplier risk brief, and executive pay analysis.

## Why this project shows off LangChain

| Need | LangChain piece |
|---|---|
| Research several companies at once, then merge | LangGraph `Send` fan-out + subgraphs |
| Check every number, redraft if wrong (bounded) | LangGraph conditional edges |
| Pause for your markup, resume the next day | `interrupt()` + persistent checkpointer |
| Small model for extraction, strong one for writing, fallback when a free cap hits | one chat-model interface across Groq and Gemini |
| Find which step got a number wrong | LangSmith tracing |
| Prove accuracy with no hand labeling | LangSmith datasets + evals generated from XBRL |
| See and step through the graph | LangGraph Studio (`langgraph dev`) |

## Layout

```
core/     plain Python, no LLM: SEC client, ticker->CIK, XBRL facts, metric mapping, filing sections
tools/    LangChain tools that wrap core/
graphs/   one LangGraph graph per use case, built from shared subgraphs (research, verify, review)
evals/    datasets generated from XBRL + evaluators, so every milestone has a pass/fail bar
common/   model factory (Groq primary, free Gemini fallback) and local embeddings
```

## Milestones

Each milestone ends in something you can demo, and the next starts only after it passes.
Issues live under [GitHub milestones](https://github.com/mtalsania/langchain-stack/milestones).

| # | Milestone | Done when |
|---|---|---|
| M0 | Data core | Exact revenue, margins and R&D for any ticker from XBRL, tested against live SEC data |
| M1 | Research one company | Single-company brief scores at least 90% numeric accuracy on 20 auto-generated questions |
| M2 | Risk-factor changes | Year-over-year Item 1A diff for 5 companies passes your spot-check |
| M3 | Peer comparison | 3-company comparison table plus narrative, within free-tier limits |
| M4 | Memo with review | Memo survives an overnight pause for your markup; all evals pass |
| M5 | Watchlist monitor | Weekly brief runs on its own for 2 weeks. **Main outcome done.** |
| + | Add-ons | Fact-checker, screener, credit/supplier risk, executive pay |

## Setup checklist (you do these)

- [ ] Python 3.11 to 3.13
- [ ] **Groq API key** (free): <https://console.groq.com/keys>
- [ ] **LangSmith API key** (free Developer plan): <https://smith.langchain.com> → Settings → API Keys
- [ ] **SEC contact string**: the SEC requires a User-Agent with a name and email on every request,
      for example `Mihir Research you@example.com`. It goes in `.env` as `SEC_USER_AGENT`, never in code.
- [ ] **Gemini API key** (free tier, needed from M3): <https://aistudio.google.com/apikey>.
      A Gemini app subscription does not cover the API; the API free tier is separate.
- [ ] Install:
  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  cp .env.example .env   # fill in the keys
  ```

## M0: the data core (no LLM)

```bash
python -m core.demo NVDA              # 3-year table of ~15 metrics, each with its 10-K accession
python -m core.demo AAPL --sections   # also cut Items 1, 1A, 7 out of the latest 10-K
pytest                                # offline logic tests
pytest --live                         # checks exact figures against the live SEC APIs
```

| Module | What it does |
|---|---|
| `core/sec_client.py` | User-Agent from `SEC_USER_AGENT`, at most 10 requests/second, retries on 429/5xx, disk cache in `.cache/sec` |
| `core/tickers.py` | Ticker to CIK from the SEC's `company_tickers.json` |
| `core/xbrl.py` | Companyfacts loader. One value per fiscal year: full-year periods only, labelled by the company's own fiscal calendar, latest filing wins, accession kept |
| `core/metrics.py` | ~15 metrics with tag variants (e.g. `Revenues` vs `RevenueFromContractWithCustomerExcludingAssessedTax`), plus margins, FCF and total debt |
| `core/filings.py` | Latest 10-K from the submissions API; extracts Item 1, 1A, 7 and skips the table of contents |

GitHub Actions runs the offline tests on every push. Add a repository secret named `SEC_USER_AGENT`
and it runs the live tests too.

## Data sources

All public, all free:
- **SEC EDGAR** filings (10-K, 10-Q, 8-K, DEF 14A) and the **XBRL** financial data APIs
  (`data.sec.gov`). US government data, public domain. Limit: 10 requests per second.

## Free-tier rules

Free tiers only, no paid APIs. Known limits (checked 2026-11-26):
- Groq gpt-oss models: about 30 requests/min, 8K tokens/min, 200K tokens/day.
  That is roughly 2 to 3 three-company memos per day, so filings are cached and extraction uses a small model.
- Gemini API free tier: low and frequently changed limits; prompts may be used by Google to improve
  its products, which is acceptable here because SEC data is public. Re-check limits before M3.
- LangSmith Developer: 1 seat, 5,000 traces/month, 14-day retention.
- LangSmith Deployment and LangSmith cron jobs are paid, so scheduling (M5) runs locally.

## History

The earlier e-bike demo course (fictional data) is in commit `2e5e9e7` if you want to look back at it.
