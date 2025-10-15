# AbProp Demo App

Gradio interface for showcasing AbProp's sequence analysis capabilities.

## Setup

```bash
pip install -r demo/requirements.txt
export ABPROP_DEMO_CHECKPOINT=outputs/real_data_run/checkpoints/best.pt  # optional
python demo/app.py
```

Open the printed URL to explore the demo. Provide FASTA-formatted sequences to view perplexity, predicted liabilities, CDR highlighting, attention summaries, and MC-dropout uncertainty. Results can be downloaded as CSV or PDF.

## Deployment

- **Hugging Face Spaces**: Create a new Space (SDK: Gradio). Push `demo/app.py`, `demo/requirements.txt`, and checkpoints (or configure remote loading). Set `ABPROP_DEMO_CHECKPOINT` via Space secrets.
- **Streamlit Cloud / Other PaaS**: Use `demo/app.py` as the entrypoint; convert to `gradio` hosted service or wrap in a lightweight FastAPI container.

## Comparison Mode

Supply multiple FASTA sequences in the input box. The demo returns a table with per-sequence metrics and aggregated uncertainty statistics, enabling side-by-side inspection.

## Notes

- The demo falls back to randomly initialized weights when no checkpoint is configured; predictions will be meaningless in that case.
- For production deployment, gate access with authentication and throttle throughput to protect model IP.
