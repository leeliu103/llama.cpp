#!/usr/bin/env python3
"""Check measured source and sanitized TG128 evidence without using a GPU."""

import hashlib
import json
from pathlib import Path
import statistics
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    root = here.parent.parent
    evidence = here / "evidence"
    provenance = json.loads((evidence / "provenance.json").read_text())
    manifest = json.loads((evidence / "source-manifest.json").read_text())
    excluded = provenance["omitted_snapshot_files"]
    errors = []
    checked = 0

    for name, record in manifest["files"].items():
        if name in excluded:
            if (root / name).exists():
                errors.append(f"Excluded operational notes present: {name}")
            continue
        path = root / name
        if not path.is_file() or digest(path) != record["sha256"]:
            errors.append(f"Source missing or changed: {name}")
        elif bool(path.stat().st_mode & 0o111) != bool(record["mode"] & 0o111):
            errors.append(f"Executable permission differs: {name}")
        checked += 1

    for name, expected in provenance["evidence_sha256"].items():
        path = evidence / name
        if not path.is_file() or digest(path) != expected:
            errors.append(f"Evidence missing or changed: {name}")
    if checked != provenance["retained_snapshot_file_count"]:
        errors.append("Retained source file count differs")
    if digest(evidence / "source-manifest.json") != provenance["source_manifest_sha256"]:
        errors.append("Source inventory differs from measured build")

    runtime = json.loads((evidence / "runtime-manifest.json").read_text())
    runtime_hash = hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest()
    if runtime_hash != provenance["sanitization"]["published_runtime_manifest_canonical_sha256"]:
        errors.append("Sanitized runtime manifest differs")

    result = json.loads((evidence / "result.json").read_text())
    row = result["rows"][0]
    rates = [row["n_gen"] * 1e9 / ns for ns in row["samples_ns"]]
    mean, deviation = statistics.mean(rates), statistics.stdev(rates)
    if round(mean, 6) != row["avg_ts"] or round(deviation, 6) != row["stddev_ts"]:
        errors.append("Recorded timing statistics differ from raw durations")

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Verified {checked} retained snapshot files and sanitized published evidence.")
    print(f"TG128: {mean:.6f} tokens/s, sample SD {deviation:.6f}, {len(rates)} samples.")
    print("No build, model execution, or network access performed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
