import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for tokenizer tests.")

from abprop.tokenizers import TOKEN_TO_ID, collate_batch, decode, encode


def test_encode_decode_roundtrip():
    sequence = "ACDX"
    token_ids = encode(sequence)
    decoded = decode(token_ids)
    assert decoded == sequence


def test_encode_maps_unknown_to_x():
    ids = encode("AZ")
    decoded = decode(ids)
    assert decoded == "AX"


def test_collate_batch_padding_and_mask():
    sequences = ["ACD", "A", "ACDEFG"]
    batch = collate_batch(sequences)

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(attention_mask, torch.Tensor)

    assert input_ids.shape == attention_mask.shape
    assert input_ids.shape[0] == len(sequences)

    pad_id = TOKEN_TO_ID["<pad>"]
    # Shortest sequence should be padded after first actual token (<bos>, 'A', <eos>)
    assert input_ids[1, -1].item() == pad_id
    # Mask should be zero at padded positions.
    assert attention_mask[1, -1].item() == 0
    expected_mask = input_ids != pad_id
    assert torch.equal(attention_mask, expected_mask)
