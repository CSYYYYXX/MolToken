"""Stream conservative AA source references into reusable length buckets.

The sources include upstream graph failures and pretraining validation records;
this is NOT proof of individual gradient exposure. No AA alphabet filtering,
sequence repair or deduplication is applied. Lengths above the storage limit
are counted, not silently treated as absent from the reference.
"""
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess


def source_rows(path):
    path = Path(path)
    if path.suffix == ".zst":
        command = ["zstd", "-dc", "--", str(path)]
        with subprocess.Popen(command, stdout=subprocess.PIPE, text=True) as process:
            for line in process.stdout:
                yield json.loads(line)
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
    else:
        with path.open() as handle:
            for line in handle:
                yield json.loads(line)


def build_index(catalog, output, *, maximum_length=400, progress_every=1000000):
    if maximum_length < 1 or progress_every < 1:
        raise ValueError("positive storage length and progress interval required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sources, lengths = [], Counter()
    total = 0
    with ExitStack() as stack:
        buckets = {}
        for source in catalog["sources"]:
            source_lengths = Counter()
            count = 0
            for row in source_rows(source["source_path"]):
                seq = row["normalized"]
                if not isinstance(seq, str) or not seq or not seq.isascii() or any(c in seq for c in " \t\n\r\v\f"):
                    raise ValueError(f"{source['source_path']} row {count + 1}: invalid normalized sequence")
                length = len(seq)
                if row["length"] != length:
                    raise ValueError(f"{source['source_path']} row {count + 1}: inconsistent length")
                count += 1
                source_lengths[length] += 1
                if length <= maximum_length:
                    if length not in buckets:
                        buckets[length] = stack.enter_context(
                            (output / f"length_{length:04d}.txt").open("x", buffering=1024 * 1024))
                    buckets[length].write(seq + "\n")
                if count % progress_every == 0:
                    print(f"SOURCE_PROGRESS path={source['source_path']} records={count}", flush=True)
            if count != source["records"]:
                raise ValueError(f"{source['source_path']}: read {count}, catalog expected {source['records']}")
            total += count
            lengths.update(source_lengths)
            sources.append({"source_path": source["source_path"], "records": count,
                            "stored_records": sum(v for k, v in source_lengths.items() if k <= maximum_length)})
            print(f"SOURCE_COMPLETE path={source['source_path']} records={count}", flush=True)
    if total != catalog["source_records_total"]:
        raise ValueError("source total differs from catalog")
    omitted = {k: v for k, v in lengths.items() if k > maximum_length}
    result = {
        "complete": True, "source_records_total": total, "sources": sources,
        "maximum_stored_length": maximum_length,
        "length_counts": dict(sorted(lengths.items())),
        "stored_records": sum(v for k, v in lengths.items() if k <= maximum_length),
        "omitted_records": sum(omitted.values()),
        "minimum_omitted_length": min(omitted) if omitted else None,
        "reference_scope": catalog["reference_scope"],
        "per_record_gradient_exposure_proven": False,
        "includes_upstream_graph_failures_and_pretraining_validation": True,
        "includes_small_molecule_graph_equivalents": False,
        "exact_membership_scope": "complete listed AA sources for query lengths <= maximum_stored_length",
        "nearest_distance_scope": "not computed; omitted lengths require a proven edit-distance lower-bound exclusion, not silent truncation",
    }
    with (output / "summary.json").open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return result


def exact_source_matches(sequences, index):
    index = Path(index)
    summary = json.loads((index / "summary.json").read_text())
    if not summary["complete"]:
        raise ValueError("source index is incomplete")
    queries = set(sequences)
    if any(not isinstance(s, str) or not s or len(s) > summary["maximum_stored_length"] for s in queries):
        raise ValueError("query falls outside exact membership index coverage")
    by_length = {}
    for seq in queries:
        by_length.setdefault(len(seq), set()).add(seq)
    matched = set()
    for length, wanted in sorted(by_length.items()):
        expected = summary["length_counts"].get(str(length), 0)
        if not expected:
            continue
        count = 0
        with (index / f"length_{length:04d}.txt").open() as handle:
            for line in handle:
                sequence = line.rstrip("\n")
                count += 1
                if sequence in wanted:
                    matched.add(sequence)
        if count != expected:
            raise ValueError(f"length {length}: index has {count} rows, expected {expected}")
    return matched, summary
