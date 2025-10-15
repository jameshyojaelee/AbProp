import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for embedding tests.")
pd = pytest.importorskip("pandas", reason="Pandas required for embedding tests.")

from abprop.viz.embeddings import bucketize_liabilities  # noqa: E402


def test_bucketize_liabilities_creates_buckets():
    frame = pd.DataFrame(
        {
            "liability_nglyc": [0.0, 0.5, 1.0, float("nan")],
            "liability_deamidation": [0.2, 0.3, 0.4, 0.5],
        }
    )
    enriched = bucketize_liabilities(frame, quantiles=(0.25, 0.75))
    assert "liability_nglyc_bucket" in enriched.columns
    assert "liability_deamidation_bucket" in enriched.columns
    buckets = enriched["liability_nglyc_bucket"].tolist()
    assert buckets[0] == "low"
    assert buckets[2] == "high"
    assert buckets[-1] == "missing"
