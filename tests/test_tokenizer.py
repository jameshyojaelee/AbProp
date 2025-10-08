from abprop.tokenizers import AminoAcidTokenizer


def test_encode_decode_roundtrip():
    tokenizer = AminoAcidTokenizer.build_default()
    sequence = "ACDX"
    token_ids = tokenizer.encode(sequence)
    decoded = tokenizer.decode(token_ids)
    assert decoded == sequence
