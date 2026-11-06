"""Streaming JSONL -> frozen tokenizer -> compact-int shards (no silent skips)."""
import argparse
from collections import Counter
import json
from pathlib import Path
import struct

from moltoken.tokenizer import MolTokenizer, DOMAINS
from cli.data.build_compact_int_index import HEADER_STRUCT, MAGIC, VERSION


def encode_corpus(source, output):
    """Input records: {id, domain, sequence}; small molecules may use smiles."""
    tokenizer = MolTokenizer()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = output.with_suffix(output.suffix + ".records.jsonl")
    nseq = ntokens = 0
    counts = Counter()
    max_token = -1
    with Path(source).open() as inputs, output.open("xb+") as dest, sidecar.open("x") as provenance:
        dest.write(HEADER_STRUCT.pack(MAGIC, VERSION, 0, 0, -1))
        for line_no, line in enumerate(inputs, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            domain = row["domain"]
            if domain not in DOMAINS:
                raise ValueError(f"line {line_no}: unknown domain {domain!r}")
            sequence = row.get("smiles") if domain == "small_molecule" and row.get("smiles") else row["sequence"]
            ids = tokenizer.encode(sequence, domain)
            dest.write(struct.pack("<I", len(ids)))
            dest.write(struct.pack(f"<{len(ids)}i", *ids))
            provenance.write(json.dumps({"row": nseq, "id": row.get("id", str(line_no)),
                                         "domain": domain, "domain_id": DOMAINS.index(domain),
                                         "length": len(ids)}, ensure_ascii=False) + "\n")
            nseq += 1
            ntokens += len(ids)
            max_token = max(max_token, max(ids))
            counts[domain] += 1
        if not nseq:
            raise ValueError("empty corpus")
        dest.seek(0)
        dest.write(HEADER_STRUCT.pack(MAGIC, VERSION, nseq, ntokens, max_token))
    report = {"records": nseq, "raw_bpe_tokens": ntokens, "domain_counts": counts,
              "token_offset_on_disk": 0, "max_token": max_token}
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(encode_corpus(args.input, args.output)))


if __name__ == "__main__":
    main()
