# Learn the LangChain stack by building one app

You'll build **one support assistant for a fictional e-bike company (Nimbus Bikes)**, one
piece at a time. By the end it answers from your documents (RAG), uses tools, remembers
conversations, asks a human before risky actions, is traced and evaluated in LangSmith,
runs in LangGraph Studio, and is served as an API you can deploy for free.

Everything here runs on free tiers. Where a free tier runs out, or a product is paid-only,
this README says so.

| Layer | What it is | Free? |
|---|---|---|
| **LangChain** | Building blocks: models, prompts, tools, retrievers, `create_agent` | Open source |
| **LangGraph** | Runtime for agents as graphs: state, memory, interrupts | Open source |
| **LangSmith** | Hosted tracing, datasets, evals, Prompt Hub | Free "Developer" plan |
| **LangGraph Studio** | Visual debugger for your graph | Free with `langgraph dev` locally |
| **LangSmith Deployment** (Agent Server, formerly LangGraph Platform) | Managed hosting for agents | **Paid (Plus, $39/seat/mo)** |
| **LLM** | Groq API, `openai/gpt-oss-120b` | Free tier with caps |
| **Embeddings** | FastEmbed `bge-small-en`, runs on your CPU | Free, no key |

---

## Part 1. Setup checklist (you do these, about 15 minutes)

- [ ] **Python 3.11–3.13** installed. Check with `python3 --version`.
- [ ] **Groq API key.** Sign up at <https://console.groq.com>, then **API Keys → Create**. Copy the `gsk_...` key.
- [ ] **LangSmith account.** Sign up at <https://smith.langchain.com> (free Developer plan, no card).
      Then **Settings → API Keys → Create API Key** (personal access token). Copy the `lsv2_...` key.
      If you chose the EU region at signup, note it (see `.env.example`).
- [ ] **Clone and install:**
  ```bash
  git clone https://github.com/mtalsania/langchain-stack.git
  cd langchain-stack
  python3 -m venv .venv
  source .venv/bin/activate          # Windows: .venv\Scripts\activate
  pip install -r requirements.txt
  cp .env.example .env               # then paste your two keys into .env
  ```
- [ ] **Verify:** `python 00_check_setup.py` prints four `OK` lines.

> Never commit `.env`. It's already in `.gitignore`.

---

## Part 2. The modules

Run them in order from the repo root. Each file starts with **What you'll learn / Run /
Check it worked**, so open the file before running it. Each module adds one idea to the same app.

| # | File | You learn | Stack piece |
|---|---|---|---|
| 00 | `00_check_setup.py` | Keys, model, embeddings, LangSmith all connected | Setup |
| 01 | `01_models_and_prompts.py` | Messages, prompt templates, `prompt \| model \| parser`, invoke / batch / stream | LangChain |
| 02 | `02_structured_output.py` | `with_structured_output` + Pydantic for reliable extraction | LangChain |
| 03 | `03_tools.py` | `@tool`, `bind_tools`, and the tool-call loop **by hand** | LangChain |
| 04 | `04_rag.py` | Load → split → embed → store → retrieve → answer with citations | LangChain |
| 05 | `05_agent.py` | `create_agent`: the loop from 03, automated, with RAG as a tool | LangChain on LangGraph |
| 06 | `06_langgraph_basics.py` | State, reducers, nodes, conditional edges; then the real graph in `app/graph.py` | LangGraph |
| 07 | `07_memory.py` | Checkpointers, `thread_id`, persistent memory in SQLite, `get_state` | LangGraph |
| 08 | `08_human_in_the_loop.py` | `interrupt()` + `Command(resume=...)`: approve / edit / reject a tool call | LangGraph |
| 09 | `09_langsmith_tracing.py` | Tracing, `@traceable`, tags/metadata, Prompt Hub push/pull | LangSmith |
| 10 | `10_evals.py` | Datasets, experiments, code evaluators, LLM-as-judge, comparing runs | LangSmith |
| 11 | *(below)* | Run the graph in **LangGraph Studio** via `langgraph dev` | Studio |
| 12 | `12_server.py` + `Dockerfile` | Serve the agent as an API; deploy it for free | Deployment |

`common/` holds shared code (model factory, embeddings, retriever, tools).
`app/graph.py` is the finished assistant. Modules 06–12 all use it, so read it closely.

### Module 11: LangGraph Studio (local, free)

**You'll learn** how to see your graph, step through runs, inspect state at each node, edit
state and re-run, and handle interrupts in a UI.

```bash
langgraph dev
```

This starts a local Agent Server (in-memory, hot reload) at `http://127.0.0.1:2024` using
`langgraph.json`, and opens Studio in your browser at
`https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`.
Safari and Brave block the localhost connection; use Chrome, or run `langgraph dev --tunnel`.

**Do this:**
1. Pick graph `nimbus_assistant`. Send: *"Student price for a Nimbus Cargo?"* Watch it go agent → tools → agent.
2. Click a node in the run to see the exact state there. Use **Fork / Edit state** to change a message and re-run from that point.
3. Send *"My brakes failed, open a ticket please."* The run pauses at `human_review`. Resume it with
   `{"action": "approve"}`, or `{"action": "reject", "feedback": "Call the customer instead"}`.
4. Edit `SYSTEM_PROMPT` in `app/graph.py` and save. The server reloads; try again.

**Check it worked:** you see the graph diagram, a completed run, and a paused run you resumed. The
runs also appear as traces in LangSmith.

### Module 12: deploy for free

**You'll learn** how to put the agent behind an HTTP API and what you give up versus the managed platform.

```bash
uvicorn 12_server:app --reload   # then open http://127.0.0.1:8000/docs
```

To put it online for free, build the `Dockerfile` on a free container host. Options as of Nov 2026
(check each one's current terms, they change):
- **Hugging Face Spaces** (Docker Space, free CPU). Set `GROQ_API_KEY` and `LANGSMITH_*` as Space secrets. Sleeps when idle.
- **Render** free web service. Spins down after inactivity, so the first request is slow, and the disk is wiped on redeploy.

Free hosts wipe local disk, so `server_memory.sqlite` (conversation memory) resets. The fix is a free
Postgres (Neon or Supabase) plus `langgraph-checkpoint-postgres`'s `PostgresSaver`. That's a good stretch goal.

**What's paid-only, honestly:**
- **LangSmith Deployment / Agent Server (managed).** Not on the free Developer plan. Plus ($39/seat/month)
  includes one small serverless Agent Server. This gives you the same API `langgraph dev` exposes,
  plus task queues, cron jobs, double-texting handling, and scaling.
- **Self-hosting the official Agent Server** in your own cloud is listed under Enterprise.
- `langgraph dev` gives you the full Agent Server API **locally** for free. It's for development only:
  in-memory, single process, not for production.

---

## Where the free tiers bite

| Limit | What happens | What to do |
|---|---|---|
| **Groq: ~8K tokens/minute** (gpt-oss models, free plan) | 429 errors when an agent sends long histories or several retrieved chunks | `common/llm.py` already rate-limits and retries. Keep `k=3` chunks. Use short threads. |
| **Groq: ~200K tokens/day, 1K requests/day** | Calls fail for the rest of the day | Switch `MODEL` to `openai/gpt-oss-20b` or `qwen/qwen3.8-27b` (each model has its own quota), or `PROVIDER=ollama` |
| **Groq reasoning tokens** | gpt-oss "thinks" first, and those tokens count | `reasoning_effort="low"` is set for you |
| **LangSmith: 5,000 traces/month, 14-day retention, 1 seat** | Pay-as-you-go after that (you'd need a card on file) | Set `LANGSMITH_TRACING=false` when just iterating. Evals add ~4 traces per example. |
| **Long conversations** (module 07) | Full history is resent every turn and eats token budget | Try `SummarizationMiddleware` with `create_agent`, or trim messages, as a stretch goal |

Limits checked on 2026-11-26 at <https://console.groq.com/docs/rate-limits> and
<https://www.langchain.com/pricing>. Your exact Groq limits are at <https://console.groq.com/settings/limits>.

## Switching models

Change two lines in `.env`. No code changes. That's the point of LangChain's model interface.

```
PROVIDER=gemini   MODEL=gemini-2.5-flash   GOOGLE_API_KEY=...    # pip install langchain-google-genai
PROVIDER=ollama   MODEL=qwen3:8b                                  # pip install langchain-ollama; ollama pull qwen3:8b
```

## After the course: stretch goals

1. Swap `InMemoryVectorStore` for a persistent store (Chroma or pgvector) and index a real document set you care about.
2. Add online evaluation in LangSmith: an LLM-as-judge rule that scores production traces automatically.
3. Build a multi-agent version: a router that hands off to a "sales" agent and a "support" agent.
4. Try the [Deep Agents](https://github.com/langchain-ai/deepagents) library for long-running tasks with planning and a file system.
