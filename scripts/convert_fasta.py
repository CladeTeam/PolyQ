#!/usr/bin/env python3
"""
Convert Ensembl protein FASTA files to JSONL format with parsed metadata.
Each line is a JSON object: {id, seq, species, gene, length, desc}

Usage:
    python convert_fasta.py --fasta_dir /path/to/fasta \
        --species human rice chlamydomonas \
        --output_dir data/training
"""
import argparse, json, re, sys, gzip
from pathlib import Path

SPECIES_CONFIG = {
    "human": {
        "pep_file": "human/pep.fa",
        "name": "Homo sapiens",
        "common": "human",
    },
    "rice": {
        "pep_file": "rice/pep.fa",
        "name": "Oryza sativa",
        "common": "rice",
    },
    "chlamydomonas": {
        "pep_file": "chlamydomonas/pep.fa",
        "name": "Chlamydomonas reinhardtii",
        "common": "chlamydomonas",
    },
}


def parse_ensembl_header(header: str, species_key: str) -> dict:
    """Parse an Ensembl-style FASTA header into structured metadata."""
    # Remove leading '>'
    line = header.lstrip(">").strip()

    # Protein ID is the first token
    parts = line.split()
    protein_id = parts[0]

    # Extract gene_symbol using regex
    gene = None
    gene_match = re.search(r"gene_symbol:(\S+)", line)
    if gene_match:
        gene = gene_match.group(1)

    # Extract description
    desc = None
    desc_match = re.search(r"description:(.+)$", line)
    if desc_match:
        desc = desc_match.group(1)

    # Extract gene_biotype
    biotype = None
    biotype_match = re.search(r"gene_biotype:(\S+)", line)
    if biotype_match:
        biotype = biotype_match.group(1)

    # Extract chromosome/coordinate info
    chrom = None
    chrom_match = re.search(r"chromosome:([^:]+):(\d+):(\d+):(-?\d+)", line)
    if not chrom_match:
        chrom_match = re.search(r"scaffold:([^:]+):(\d+):(\d+):(-?\d+)", line)
    if chrom_match:
        chrom = {
            "name": chrom_match.group(1),
            "start": int(chrom_match.group(2)),
            "end": int(chrom_match.group(3)),
            "strand": int(chrom_match.group(4)),
        }

    return {
        "id": protein_id,
        "gene": gene,
        "biotype": biotype,
        "desc": desc,
        "chrom": chrom,
    }


def fasta_to_jsonl(fasta_path: Path, species_key: str, output_path: Path):
    """Convert a protein FASTA to JSONL, one JSON per line."""
    species_info = SPECIES_CONFIG[species_key]
    count = 0
    total_aa = 0
    min_len = float("inf")
    max_len = 0

    with open(fasta_path) as f_in, open(output_path, "w") as f_out:
        header = None
        seq_parts = []
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # Write previous record
                if header is not None:
                    seq = "".join(seq_parts)
                    meta = parse_ensembl_header(header, species_key)
                    record = {
                        "id": meta["id"],
                        "seq": seq,
                        "species": species_key,
                        "species_name": species_info["name"],
                        "gene": meta["gene"],
                        "length": len(seq),
                        "desc": meta["desc"],
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                    total_aa += len(seq)
                    min_len = min(min_len, len(seq))
                    max_len = max(max_len, len(seq))
                header = line
                seq_parts = []
            else:
                seq_parts.append(line)
        # Write final record
        if header is not None:
            seq = "".join(seq_parts)
            meta = parse_ensembl_header(header, species_key)
            record = {
                "id": meta["id"],
                "seq": seq,
                "species": species_key,
                "species_name": species_info["name"],
                "gene": meta["gene"],
                "length": len(seq),
                "desc": meta["desc"],
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            total_aa += len(seq)
            min_len = min(min_len, len(seq))
            max_len = max(max_len, len(seq))

    stats = {
        "species": species_key,
        "species_name": species_info["name"],
        "sequences": count,
        "total_aa": total_aa,
        "mean_length": round(total_aa / count) if count else 0,
        "min_length": min_len,
        "max_length": max_len,
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert Ensembl FASTA to JSONL")
    parser.add_argument("--fasta_dir", required=True,
                        help="Directory containing species subdirectories with pep.fa files")
    parser.add_argument("--output_dir", default="data/training",
                        help="Output directory for JSONL files")
    parser.add_argument("--species", nargs="+", default=["human", "rice", "chlamydomonas"],
                        help="Species to convert")
    args = parser.parse_args()

    fasta_base = Path(args.fasta_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    total_seqs = 0

    for species_key in args.species:
        cfg = SPECIES_CONFIG[species_key]
        fasta_path = fasta_base / cfg["pep_file"]
        if not fasta_path.exists():
            print(f"SKIP {species_key}: {fasta_path} not found")
            continue

        output_path = out_dir / f"{species_key}_proteins.jsonl"
        print(f"Converting {species_key} ({cfg['name']}): {fasta_path} ...")
        stats = fasta_to_jsonl(fasta_path, species_key, output_path)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  -> {stats['sequences']:,} sequences, {size_mb:.1f} MB")
        print(f"     lengths: min={stats['min_length']}, mean={stats['mean_length']}, max={stats['max_length']}")
        all_stats[species_key] = stats
        total_seqs += stats["sequences"]

    # Write stats summary
    stats_path = out_dir / "dataset_stats.json"
    summary = {
        "total_species": len(all_stats),
        "total_sequences": total_seqs,
        "per_species": all_stats,
    }
    with open(stats_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nStats written to {stats_path}")
    print(f"Total: {total_seqs:,} sequences across {len(all_stats)} species")


if __name__ == "__main__":
    main()
