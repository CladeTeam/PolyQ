"""FASTA helpers for CDS preparation and sequence export."""

from __future__ import annotations

from pathlib import Path

VALID_DNA = set("ACGTNacgtn")


def read_fasta(path: Path, min_len: int = 1, max_len: int | None = None) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    parts: list[str] = []
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    sequence = "".join(parts).upper()
                    if _keep_sequence(sequence, min_len, max_len):
                        records.append((header, sequence))
                header = line[1:]
                parts = []
            else:
                parts.append(line)
    if header is not None:
        sequence = "".join(parts).upper()
        if _keep_sequence(sequence, min_len, max_len):
            records.append((header, sequence))
    return records


def _keep_sequence(sequence: str, min_len: int, max_len: int | None) -> bool:
    if len(sequence) < min_len:
        return False
    if max_len is not None and len(sequence) > max_len:
        return False
    return all(base in VALID_DNA for base in sequence)


def write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n")
            for i in range(0, len(sequence), 60):
                handle.write(sequence[i:i + 60] + "\n")

