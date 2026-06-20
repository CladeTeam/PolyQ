from htt_delta.sequences import (
    build_sequence,
    cr_htt_72q_original,
    human_htt_72q_natural_trunc,
    synthetic_htt_pure_cag,
)


def test_synthetic_htt_pure_cag_counts():
    record = synthetic_htt_pure_cag(72)
    assert record.metadata["q_count"] == 72
    assert record.metadata["cag_count"] == 72
    assert record.metadata["caa_count"] == 0
    assert record.metadata["dna_length"] == len(record.sequence)


def test_human_htt_72q_natural_anchor_counts():
    record = human_htt_72q_natural_trunc(72)
    assert record.metadata["q_count"] == 72
    assert record.metadata["cag_count"] == 36
    assert record.metadata["caa_count"] == 36
    assert record.metadata["dna_length"] == 414
    assert record.sequence.endswith("CCGCTGCACCGA")


def test_human_htt_72q_natural_truncates_from_end():
    q10 = human_htt_72q_natural_trunc(10)
    q11 = human_htt_72q_natural_trunc(11)
    q72 = human_htt_72q_natural_trunc(72)
    assert q10.sequence[:51] == q72.sequence[:51]
    assert q10.sequence[-147:] == q72.sequence[-147:]
    assert len(q11.sequence) - len(q10.sequence) == 3


def test_cr_htt_72q_original_anchor_counts():
    record = cr_htt_72q_original(72)
    assert record.metadata["q_count"] == 72
    assert record.metadata["cag_count"] == 62
    assert record.metadata["caa_count"] == 10
    assert record.metadata["dna_length"] == 429


def test_build_sequence_dispatch():
    assert build_sequence("human_htt_72q_natural_trunc", 10).metadata["q_count"] == 10
