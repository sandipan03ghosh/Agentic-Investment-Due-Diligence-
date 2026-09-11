# Agentic Investment Due Diligence System

A multi-step AI agent system that generates investment due diligence reports using a graph-based workflow.

This project combines:
- Planning and orchestration with LangGraph
- LLM generation with Groq (LangChain integration)
- Optional web research with Tavily
- Optional image generation with Hugging Face Inference API
- A Streamlit frontend for interactive report generation and past report browsing
- LangSmith tracing support for observability

## What This Project Does

Given a research topic (for example: a company, sector, or investment thesis to evaluate), the app:
1. Routes the request into a mode (`closed_book`, `hybrid`, or `open_book`)
2. Optionally runs web research and normalizes evidence
3. Builds a detailed research plan with multiple tasks
4. Fans out tasks to worker nodes that generate section markdown
5. Reduces/merges sections into a final due diligence report
6. Optionally plans and generates supporting images
7. Saves the output as markdown and lets users download markdown/zip bundles

## Tech Stack

- Python
- Streamlit
- LangGraph
- LangChain Core
- Pydantic
- Groq via `langchain-groq`
- Tavily via `langchain-tavily`
- Hugging Face Inference API via `huggingface_hub`
- Pillow
- Pandas
- python-dotenv
- LangSmith (tracing)

## Project Structure

- `backend.py`: Core agent graph, schemas, routing, research, orchestration, worker generation, reducer/image pipeline
- `frontend.py`: Streamlit UI, run flow, progress display, preview/downloads, past report loading
- `requirement.txt`: Python dependencies
- `images/`: Generated images used in reports
- `*.md`: Generated due diligence report outputs

## Agent Architecture (High Level)

Main graph flow:

`START -> router -> (research or orchestrator) -> fanout workers -> reducer subgraph -> END`

### Nodes

- `router_node`
  - Decides whether research is needed
  - Selects mode: `closed_book`, `hybrid`, or `open_book`
  - Sets recency constraints

- `research_node`
  - Runs Tavily queries when required
  - Extracts and deduplicates evidence
  - Applies recency filtering (especially in `open_book` mode)

- `orchestrator_node`
  - Produces structured plan (`Plan`) with 6-10 tasks
  - Defines audience, tone, report type, and constraints

- `worker_node`
  - Generates one markdown section per task
  - Applies grounding/citation constraints based on mode

- `reducer` subgraph
  1. `merge_content`
  2. `decide_images`
  3. `generate_and_place_images`

## Frontend UX

The Streamlit app includes:
- Bottom chat-style input for research topic submission
- Sidebar for past report selection/loading
- Tabs for:
  - Strategy Plan
  - Market Evidence
  - Report Preview
  - Visuals
  - Logs
- Download options:
  - Markdown only
  - Markdown + images bundle zip
  - Images zip

## Setup

### 1) Create and activate virtual environment (Windows PowerShell)

```powershell
cd D:\prod_launch_agent\product_agent
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirement.txt
```

### 3) Create `.env`

Create a `.env` file in project root and set required keys.

Minimal example:

```env
# LLM
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile

# Research (optional but recommended for hybrid/open_book)
TAVILY_API_KEY=your_tavily_api_key

# Images (optional; needed for image generation)
HF_TOKEN=your_huggingface_token
# Optional override:
# HF_IMAGE_MODEL=stabilityai/stable-diffusion-xl-base-1.0

# LangSmith tracing (optional but recommended)
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=investment-dd-agent
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

Compatibility fallback (only if traces do not appear):

```env
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=investment-dd-agent
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

## Run

```powershell
streamlit run frontend.py
```

Then open the local URL shown by Streamlit (typically `http://localhost:8501`).

## Ingestion API

A production-style document ingestion pipeline lets you upload PDF/DOCX/TXT/Markdown files — annual reports, earnings transcripts, SEC filings, financial statements, research notes, and similar due diligence source material — which are chunked, embedded, and indexed into Weaviate — the same collection `research_node` queries — so uploaded knowledge becomes usable by the agent automatically.

### 1) Start Weaviate

```powershell
docker compose up -d
```

### 2) Run the ingestion API (separate process from Streamlit)

```powershell
uvicorn api_main:app --reload
```

Interactive docs: `http://localhost:8000/docs`

### 3) Example calls (PowerShell)

```powershell
# Upload a document
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/v1/ingestion/documents `
  -Form @{ file = Get-Item .\sample_data\sample_brief.md }

# Check status
Invoke-RestMethod -Uri http://localhost:8000/api/v1/ingestion/documents/<document_id>

# List documents
Invoke-RestMethod -Uri http://localhost:8000/api/v1/ingestion/documents

# Delete (soft delete + vector cleanup)
Invoke-RestMethod -Method Delete -Uri http://localhost:8000/api/v1/ingestion/documents/<document_id>

# Reindex
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/v1/ingestion/documents/<document_id>/reindex
```

Duplicate uploads (same SHA-256 content hash) are rejected with `409`. Unsupported file types or spoofed content are rejected with `400`. Oversized uploads are rejected with `413`. Uploaded originals and the ingestion status DB live under `data/` (git-ignored, never committed).

## Retrieval Demo and Evaluation

The project now includes a small vector retrieval demo path:

1. Index the sample docs:

```powershell
python -m scripts.ingest_documents --dir sample_data --recreate
```

Or run the end-to-end demo helper:

```powershell
.\demo\run_demo.ps1
```

2. Run the retrieval evaluation:

```powershell
python -m scripts.evaluate_retrieval --cases sample_data/eval_cases.json --output demo/retrieval_report.json
```

3. Review the walkthrough in [demo/demo_walkthrough.md](demo/demo_walkthrough.md).

The evaluation uses a small set of queries in [sample_data/eval_cases.json](sample_data/eval_cases.json) and reports a simple keyword-based precision proxy so you can show whether the KB is surfacing relevant concepts before generation.

## How To Use

1. Enter a company/topic to research in the bottom input bar and send.
2. Wait for graph execution and progress updates.
3. Review output in tabs:
   - Plan quality
   - Evidence coverage
   - Final report markdown with local/remote image rendering
4. Download outputs as needed.
5. Use sidebar to load past generated reports.

## Output Files

- Report markdown file: `<slugified_title>.md`
- Generated images: `images/<filename>.png`
- Optional downloads via UI:
  - Markdown
  - Bundle zip (markdown + images)
  - Images zip

## Demo Assets

- [demo/demo_walkthrough.md](demo/demo_walkthrough.md): end-to-end demo script for recording or presenting the app
- [demo/run_demo.ps1](demo/run_demo.ps1): one-command demo runner that indexes sample data and produces a retrieval report
- [demo/retrieval_report.example.json](demo/retrieval_report.example.json): example retrieval report showing the expected output shape
- [sample_data/sample_brief.md](sample_data/sample_brief.md): sample due diligence document used for ingestion and demoing retrieval
- [sample_data/eval_cases.json](sample_data/eval_cases.json): retrieval evaluation cases and expected keywords

## LangSmith: What You Can Monitor

Once tracing is enabled:
- End-to-end runs
- Node-level execution paths
- Inputs/outputs per step
- Latency and token/cost insights (when available)
- Regression comparisons after prompt/code changes

## License

Add a project license before public release (for example: MIT).
