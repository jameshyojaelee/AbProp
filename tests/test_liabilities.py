from abprop.utils import find_motifs, normalize_by_length


def test_find_motifs_detects_overlapping_patterns():
    sequence = "NNSTNGNGDGM"
    counts = find_motifs(sequence)
    assert counts["nglyc"] == 2  # NNS and NST
    assert counts["deamidation"] == 2  # NG, NG
    assert counts["isomerization"] == 1  # DG
    assert counts["oxidation"] == 1  # single M


def test_free_cysteine_estimation_pairs_within_window():
    sequence = "C" + "A" * 5 + "C" + "B" * 15 + "C" + "AA" + "C"
    counts = find_motifs(sequence)
    # First pair (positions 0 and 6) and second pair (positions 22 and 25) -> zero free
    assert counts["free_cysteines"] == 0


def test_free_cysteine_estimation_marks_unpaired_cysteine():
    sequence = "C" + "A" * 5 + "C" + "A" * 25 + "C"
    counts = find_motifs(sequence)
    assert counts["free_cysteines"] == 1


def test_extra_patterns_supported():
    sequence = "AAABBBAAA"
    counts = find_motifs(sequence, {"triple_a": r"AAA"})
    assert counts["triple_a"] == 2  # overlapping AAA at start and end


def test_normalize_by_length_handles_zero_length_and_ratios():
    counts = {"nglyc": 2, "oxidation": 1}
    normalized = normalize_by_length(counts, 10)
    assert normalized["nglyc"] == 0.2
    assert normalized["oxidation"] == 0.1

    zero = normalize_by_length(counts, 0)
    assert all(value == 0.0 for value in zero.values())
