# Demo Walkthrough

This repository now supports an end-to-end demo without Docker.

## 1) Index the sample data

```powershell
cd D:\prod_launch_agent\product_agent
.\demo\run_demo.ps1
```

If you want to continue into the UI after ingestion and evaluation, run:

```powershell
.\demo\run_demo.ps1 -LaunchUI
```

## 2) Run retrieval evaluation

The PowerShell runner already generates `demo/retrieval_report.json`.

## 3) Launch the app

```powershell
streamlit run frontend.py
```

## 4) Demo flow to record

1. Enter a topic like `due diligence on a mid-cap consumer electronics company entering India`.
2. Show the retrieval/evidence tab after the graph runs.
3. Show the strategy plan and final report preview.
4. Mention the sample retrieval evaluation report produced in `demo/retrieval_report.json`.

## 5) Suggested narration

"The agent first indexes markdown knowledge, then retrieves the most relevant passages, then turns those into a structured due diligence report. The evaluation script checks whether the right concepts are being surfaced before generation."
