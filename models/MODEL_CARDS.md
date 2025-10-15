# AbProp Model Cards

This file centralizes the key model records tracked in the lightweight registry. Update it whenever you register a new checkpoint with `scripts/registry.py`.

---

## best-val-2024-07-01

- **Checkpoint**: `outputs/real_data_run/checkpoints/best.pt`
- **Created**: 2024-07-01T00:00:00Z
- **Tags**: `prod`, `ensemble`

**Metrics**

| Metric | Value |
|--------|-------|
| Perplexity ↓ | 1.95 |
| Macro F1 ↑ | 0.89 |
| Liability RMSE ↓ | 0.27 |

**Configuration**

```json
{
  "d_model": 384,
  "nhead": 6,
  "num_layers": 3,
  "dropout": 0.1
}
```

**Notes**

Baseline production checkpoint captured after running `scripts/reproduce_all.sh`; MC-dropout instrumentation enabled for liability regression monitoring.

---

## repro-run

- **Checkpoint**: `outputs/real_data_run/checkpoints/best.pt`
- **Created**: 2024-07-02T00:00:00Z
- **Tags**: `reproducibility`

**Metrics**

| Metric | Value |
|--------|-------|
| Perplexity ↓ | 1.95 |
| Liability RMSE ↓ | 0.27 |

**Configuration**

```json
{
  "d_model": 384,
  "nhead": 6,
  "num_layers": 3,
  "dropout": 0.1
}
```

**Notes**

Generated automatically by `scripts/reproduce_all.sh`; serves as the reference checkpoint for documentation screenshots and sanity checks in the demo.

---

Add a new section per model (`## model-id`) and keep the metrics in sync with the registry JSON (`models/registry.json`).
