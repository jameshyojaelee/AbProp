"""Public Gradio demo for AbProp."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("Install gradio or run `pip install -r demo/requirements.txt`." ) from exc

try:  # pragma: no cover - environment may lack torch
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - degrade gracefully
    torch = None
    F = None

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

from abprop.models import AbPropModel, TransformerConfig
from abprop.tokenizers.aa import ID_TO_TOKEN, TOKEN_TO_ID, collate_batch


@dataclass
class DemoConfig:
    checkpoint: Path | None
    mc_samples: int = 8


def load_model(cfg: DemoConfig) -> AbPropModel | None:
    if torch is None:
        return None
    if cfg.checkpoint and cfg.checkpoint.is_file():
        state = torch.load(cfg.checkpoint, map_location="cpu")
        config_kwargs = state.get("model_config", {})
        config = TransformerConfig(**config_kwargs) if config_kwargs else TransformerConfig()
        model = AbPropModel(config)
        model_state = state.get("model_state", state)
        model.load_state_dict(model_state, strict=False)
    else:
        model = AbPropModel()
    model.eval()
    return model


def parse_sequences(text: str) -> List[Tuple[str, str]]:
    sequences: List[Tuple[str, str]] = []
    current_name = "Sequence"
    current_seq: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">"):
            if current_seq:
                sequences.append((current_name, "".join(current_seq).upper()))
                current_seq = []
            current_name = stripped[1:].strip() or f"Sequence {len(sequences) + 1}"
        else:
            current_seq.append(stripped.upper())
    if current_seq:
        sequences.append((current_name, "".join(current_seq).upper()))
    return sequences


def compute_perplexity(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    vocab = logits.size(-1)
    loss = F.cross_entropy(logits.view(-1, vocab), targets.view(-1), reduction="none")
    masked = loss * mask.view(-1)
    denom = mask.sum().clamp_min(1.0)
    mean_loss = masked.sum() / denom
    return float(torch.exp(mean_loss))


def decode_tokens(input_ids: Sequence[int]) -> List[str]:
    return [ID_TO_TOKEN.get(int(idx), "?") for idx in input_ids]


def render_cdr_html(tokens: Sequence[str], cdr_mask: Sequence[int]) -> str:
    styled = []
    for token, mask in zip(tokens, cdr_mask):
        if token in {"<pad>", "<bos>", "<eos>"}:
            continue
        if mask == 1:
            styled.append(f"<span style='background-color:#ffcc80;padding:2px;margin:1px;border-radius:4px;'>{token}</span>")
        else:
            styled.append(f"<span style='padding:2px;margin:1px;color:#424242;'>{token}</span>")
    return "".join(styled)


def summarize_attention(attentions: Sequence[torch.Tensor], tokens: Sequence[str]) -> Dict[str, object]:
    if not attentions:
        return {}
    last = attentions[-1][0]  # shape: heads x tgt x src
    avg = last.mean(dim=0)
    cls_attention = avg[0]
    topk = torch.topk(cls_attention, k=min(5, cls_attention.size(0)))
    summary = []
    for idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
        token = tokens[idx] if idx < len(tokens) else "?"
        summary.append({"token": token, "position": idx, "weight": score})
    return {"cls_attention": summary}


def build_csv(results: List[Dict[str, object]]) -> Path:
    df = pd.DataFrame(results)
    tmp_dir = Path(tempfile.mkdtemp())
    path = tmp_dir / "abprop_demo_results.csv"
    df.to_csv(path, index=False)
    return path


def build_pdf(results: List[Dict[str, object]]) -> Path:
    tmp_dir = Path(tempfile.mkdtemp())
    path = tmp_dir / "abprop_demo_report.pdf"
    with PdfPages(path) as pdf:
        for row in results:
            fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait
            ax.axis("off")
            lines = [f"Sequence: {row['name']}"]
            for key, value in row.items():
                if key == "name":
                    continue
                lines.append(f"{key}: {value}")
            ax.text(0.05, 0.95, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")
            pdf.savefig(fig)
            plt.close(fig)
    return path


def analyze(text: str, mc_samples: int, cfg: DemoConfig, model: AbPropModel | None):
    sequences = parse_sequences(text)
    if not sequences:
        return pd.DataFrame(), "Provide at least one sequence.", json.dumps({}), None, None
    if torch is None or model is None:
        message = "PyTorch not available in this environment. Install torch to enable inference."
        return pd.DataFrame(), message, json.dumps({}), None, None

    summaries: List[Dict[str, object]] = []
    cdr_html_segments: List[str] = []
    attention_payload: Dict[str, object] = {}

    for name, sequence in sequences:
        batch = collate_batch([sequence], add_special=True)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"].to(dtype=torch.bool)
        with torch.no_grad():
            outputs = model(
                input_ids,
                attention_mask,
                tasks=("cls", "reg"),
                return_attentions=True,
            )
        hidden = outputs["hidden_states"]
        logits = model.mlm_head(hidden)
        shifted_targets = input_ids.clone()
        shifted_targets[:, :-1] = input_ids[:, 1:]
        shifted_targets[:, -1] = TOKEN_TO_ID["<pad>"]
        token_mask = (shifted_targets != TOKEN_TO_ID["<pad>"]).float()
        perplexity = compute_perplexity(logits, shifted_targets, token_mask)

        cls_logits = outputs.get("cls_logits", torch.zeros_like(input_ids, dtype=torch.float))
        cdr_mask = torch.argmax(cls_logits, dim=-1).squeeze(0).tolist()
        regression = outputs.get("regression", torch.zeros(1, len(model.config.liability_keys)))
        liability_scores = regression.squeeze(0).tolist()

        mc_std = None
        if mc_samples > 0:
            stochastic = model.stochastic_forward(
                input_ids,
                attention_mask,
                mc_samples=mc_samples,
                tasks=("reg",),
                return_attentions=False,
                enable_dropout=True,
            )
            preds = torch.stack([sample["regression"].squeeze(0) for sample in stochastic])
            mc_std = preds.std(dim=0).mean().item()

        tokens = decode_tokens(input_ids.squeeze(0).tolist())
        cdr_html_segments.append(f"<h4>{name}</h4>" + render_cdr_html(tokens, cdr_mask))
        attention_payload[name] = summarize_attention(outputs.get("attentions", []), tokens)

        summary = {
            "name": name,
            "length": len(sequence),
            "perplexity": round(perplexity, 3),
            "liability_mean": round(float(torch.tensor(liability_scores).mean().item()), 3),
            "liability_max": round(max(liability_scores), 3) if liability_scores else 0.0,
        }
        if mc_std is not None:
            summary["uncertainty_std"] = round(mc_std, 4)
        summaries.append(summary)

    csv_path = build_csv(summaries)
    pdf_path = build_pdf(summaries)
    summary_df = pd.DataFrame(summaries)
    return summary_df, "".join(cdr_html_segments), json.dumps(attention_payload, indent=2), str(csv_path), str(pdf_path)


def build_app():
    checkpoint = os.environ.get("ABPROP_DEMO_CHECKPOINT")
    cfg = DemoConfig(checkpoint=Path(checkpoint) if checkpoint else None)
    model = load_model(cfg)

    examples = [
        (Path("examples/attention_success.fa").read_text() if Path("examples/attention_success.fa").exists() else ""),
        (Path("examples/attention_failure.fa").read_text() if Path("examples/attention_failure.fa").exists() else ""),
    ]

    with gr.Blocks(title="AbProp Demo") as demo:
        gr.Markdown("""
        # AbProp Interactive Demo
        Paste FASTA-formatted sequences to inspect perplexity, liabilities, attention, and uncertainty.
        """)
        with gr.Row():
            sequence_input = gr.Textbox(label="Sequences (FASTA or newline-separated)", lines=10)
        mc_slider = gr.Slider(0, 16, value=cfg.mc_samples, step=1, label="MC samples for uncertainty")
        submit = gr.Button("Analyze")

        summary_output = gr.Dataframe(label="Summary Metrics")
        cdr_output = gr.HTML(label="CDR Highlighting")
        attention_output = gr.JSON(label="Attention Summary")
        csv_file = gr.File(label="Download CSV")
        pdf_file = gr.File(label="Download PDF")

        submit.click(
            lambda text, mc: analyze(text, int(mc), cfg, model),
            inputs=[sequence_input, mc_slider],
            outputs=[summary_output, cdr_output, attention_output, csv_file, pdf_file],
        )

        if all(examples):
            gr.Examples(examples, inputs=sequence_input, label="Curated Examples")

        gr.Markdown("""This demo loads `ABPROP_DEMO_CHECKPOINT` if set; otherwise it uses randomly initialized weights.""")

    return demo


if __name__ == "__main__":
    build_app().launch()

