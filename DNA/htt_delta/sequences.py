"""Sequence constructors for HTT/polyQ scoring experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SequenceRecord:
    sequence_id: str
    sequence: str
    metadata: dict[str, int | str]


SYNTHETIC_HTT_PREFIX = (
    "ATGGCGACCCTGGAAAAGCTGATGAAGGCCTTCGAGTCCCTCAAAAGCTTCCAA"
)
SYNTHETIC_HTT_SUFFIX = (
    "CGCCACCACCTCCCCCTCCACCCCCACCTCCTCAACTTCCTCAACCTCCTCCACAGG"
    "CACAGCCTCTGCTGCCTCAGCCACAACCTCCTCCACCTCCACCTCCACCTCCTCCAGG"
    "CCCAGCTGTGGCTGAGGAGCCGCTGCACCGA"
)

CR_HTT_72Q = (
    "atggttaacatggccaccctggagaagctgatgaaggccttcgagtcgctcaagagcttccagcagcagcag"
    "cagcagcagcagcagcaacagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcagcag"
    "cagcaacaacagcagcagcagcagcagcaacagcagcagcagcaacagcagcagcagcagcagcagcagcag"
    "cagcagcagcagcagcagcagcagcaacaacaacagcagcagcagcaacagcaacagcagccgcctcctccc"
    "ccacccccgccccctccaccacagctgccgcagccgccgccgcaggcgcagcccttgctgccccagccgcag"
    "ccaccgccccctccgccgccgccgccccccggccccgcagtggcggaggagccgctgcaccgcgttaac"
).upper()
CR_HTT_PREFIX_LEN = 60
CR_HTT_72Q_POLYQ_LEN = 72

HUMAN_HTT_72Q_NATURAL = (
    "ATGGCGACCCTGGAAAAGCTGATGAAGGCCTTCGAGTCCCTCAAAAGCTTCCAACAG"
    "CAGCAACAGCAACAACAGCAGCAACAGCAACAACAGCAGCAACAGCAACAACAGCAG"
    "CAACAACAGCAGCAACAGCAACAACAGCAGCAACAGCAACAACAGCAGCAGCAGCAA"
    "CAACAGCAGCAACAGCAACAACAACAGCAGCAACAGCAACAACAGCAGCAACAGCAA"
    "CAACAGCAGCAACAGCAACAACAGCAGCAACAGCAACAACCGCCACCACCTCCCCCT"
    "CCACCCCCACCTCCTCAACTTCCTCAACCTCCTCCACAGGCACAGCCTCTGCTGCCT"
    "CAGCCACAACCTCCTCCACCTCCACCTCCACCTCCTCCAGGCCCAGCTGTGGCTGAG"
    "GAGCCGCTGCACCGA"
)
HUMAN_HTT_NATURAL_PREFIX = (
    "ATGGCGACCCTGGAAAAGCTGATGAAGGCCTTCGAGTCCCTCAAAAGCTTC"
)


def codons(sequence: str) -> list[str]:
    if len(sequence) % 3 != 0:
        raise ValueError(f"Sequence length is not divisible by 3: {len(sequence)}")
    return [sequence[i:i + 3].upper() for i in range(0, len(sequence), 3)]


def _metadata(sequence_family: str, q_count: int, sequence: str, polyq_codons: list[str]) -> dict[str, int | str]:
    return {
        "sequence_family": sequence_family,
        "q_count": q_count,
        "dna_length": len(sequence),
        "cag_count": sum(1 for codon in polyq_codons if codon == "CAG"),
        "caa_count": sum(1 for codon in polyq_codons if codon == "CAA"),
    }


def synthetic_htt_pure_cag(q_count: int) -> SequenceRecord:
    """Synthetic human HTT exon1 prefix/suffix with pure CAG x Q."""
    polyq_codons = ["CAG"] * q_count
    sequence = SYNTHETIC_HTT_PREFIX + "".join(polyq_codons) + SYNTHETIC_HTT_SUFFIX
    return SequenceRecord(
        sequence_id=f"synthetic_htt_pure_cag_Q{q_count}",
        sequence=sequence,
        metadata=_metadata("synthetic_htt_pure_cag", q_count, sequence, polyq_codons),
    )


def _cr_htt_72q_parts() -> tuple[str, list[str], str]:
    prefix = CR_HTT_72Q[:CR_HTT_PREFIX_LEN]
    polyq = CR_HTT_72Q[CR_HTT_PREFIX_LEN:CR_HTT_PREFIX_LEN + CR_HTT_72Q_POLYQ_LEN * 3]
    suffix = CR_HTT_72Q[CR_HTT_PREFIX_LEN + CR_HTT_72Q_POLYQ_LEN * 3:]
    polyq_codons = codons(polyq)
    if len(polyq_codons) != 72:
        raise ValueError(f"Expected 72 Cr_HTT Q codons, got {len(polyq_codons)}")
    if any(codon not in {"CAG", "CAA"} for codon in polyq_codons):
        raise ValueError("Cr_HTT polyQ tract contains non-Q codons")
    return prefix, polyq_codons, suffix


def cr_htt_72q_original(q_count: int) -> SequenceRecord:
    """Cr_HTT 72Q series, natural CAG/CAA layout, truncated from the polyQ end."""
    prefix, anchor_polyq, suffix = _cr_htt_72q_parts()
    polyq_codons = anchor_polyq[:q_count]
    sequence = prefix + "".join(polyq_codons) + suffix
    return SequenceRecord(
        sequence_id=f"cr_htt_72q_original_Q{q_count}",
        sequence=sequence,
        metadata=_metadata("cr_htt_72q_original", q_count, sequence, polyq_codons),
    )


def _human_htt_72q_natural_parts() -> tuple[str, list[str], str]:
    if not HUMAN_HTT_72Q_NATURAL.startswith(HUMAN_HTT_NATURAL_PREFIX):
        raise ValueError("Human HTT 72Q natural sequence does not start with expected prefix")
    remainder = HUMAN_HTT_72Q_NATURAL[len(HUMAN_HTT_NATURAL_PREFIX):]
    remainder_codons = codons(remainder)
    polyq_codons = []
    for codon in remainder_codons:
        if codon in {"CAG", "CAA"}:
            polyq_codons.append(codon)
        else:
            break
    if len(polyq_codons) != 72:
        raise ValueError(f"Expected 72 human HTT Q codons, got {len(polyq_codons)}")
    suffix = HUMAN_HTT_72Q_NATURAL[
        len(HUMAN_HTT_NATURAL_PREFIX) + len(polyq_codons) * 3:
    ]
    if not suffix.startswith("CCG"):
        raise ValueError(f"Unexpected human HTT suffix start: {suffix[:12]}")
    return HUMAN_HTT_NATURAL_PREFIX, polyq_codons, suffix


def human_htt_72q_natural_trunc(q_count: int) -> SequenceRecord:
    """Human HTT 72Q natural CAG/CAA anchor, truncated from the polyQ end."""
    prefix, anchor_polyq, suffix = _human_htt_72q_natural_parts()
    polyq_codons = anchor_polyq[:q_count]
    sequence = prefix + "".join(polyq_codons) + suffix
    return SequenceRecord(
        sequence_id=f"human_htt_72q_natural_trunc_Q{q_count}",
        sequence=sequence,
        metadata=_metadata("human_htt_72q_natural_trunc", q_count, sequence, polyq_codons),
    )


SEQUENCE_BUILDERS = {
    "cr_htt_72q_original": cr_htt_72q_original,
    "human_htt_72q_natural_trunc": human_htt_72q_natural_trunc,
    "synthetic_htt_pure_cag": synthetic_htt_pure_cag,
}


def build_sequence(sequence_family: str, q_count: int) -> SequenceRecord:
    try:
        builder = SEQUENCE_BUILDERS[sequence_family]
    except KeyError as exc:
        valid = ", ".join(sorted(SEQUENCE_BUILDERS))
        raise ValueError(f"Unknown sequence family {sequence_family!r}. Valid: {valid}") from exc
    return builder(q_count)


def q_sweep(sequence_family: str, q_min: int = 10, q_max: int = 72) -> list[SequenceRecord]:
    return [build_sequence(sequence_family, q_count) for q_count in range(q_min, q_max + 1)]
