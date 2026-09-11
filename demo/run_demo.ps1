param(
    [switch]$LaunchUI
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

Write-Host 'Indexing sample markdown into the vector DB...'
python -m scripts.ingest_documents --dir sample_data --recreate

Write-Host 'Running retrieval evaluation...'
python -m scripts.evaluate_retrieval --cases sample_data/eval_cases.json --output demo/retrieval_report.json

if ($LaunchUI) {
    Write-Host 'Launching Streamlit UI...'
    streamlit run frontend.py
} else {
    Write-Host 'Demo steps complete. Use -LaunchUI to open the app.'
}
