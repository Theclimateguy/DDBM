"""Record a SHA-256 for every benchmark series, so regeneration can be checked.

The referee is right that a manifest plus a build script is a recipe, not a
frozen benchmark. This distinguishes the two halves honestly:

  * synthetic series are byte-reproducible from fixed seeds, and their hashes
    are a hard check;
  * real series are downloaded from live sources and are NOT reproducible ---
    Yahoo appends new trading days, HadCET and the NOAA indices are extended
    monthly. Their hashes pin the snapshot actually used here, and a mismatch
    on a later download is expected rather than an error.

Usage:
    python hash_manifest.py write   # after build_dataset.py
    python hash_manifest.py check   # verify a regenerated tree
"""
import csv
import hashlib
import pathlib
import sys

SERIES = pathlib.Path("data_bench/series")
OUT = pathlib.Path("data_bench/checksums.csv")

REAL_GROUPS = {"finance", "climate", "physiology", "eeg"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_groups():
    return {r["name"]: r["group"]
            for r in csv.DictReader(open("data_bench/manifest.csv"))}


def write():
    groups = load_groups()
    rows = []
    for name, group in sorted(groups.items()):
        p = SERIES / f"{name}.csv"
        if not p.exists():
            print(f"  missing: {name}")
            continue
        rows.append(dict(name=name, group=group,
                         reproducible=("no" if group in REAL_GROUPS else "yes"),
                         sha256=sha256(p), bytes=p.stat().st_size))
    with open(OUT, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["name", "group", "reproducible",
                                           "sha256", "bytes"])
        w.writeheader()
        w.writerows(rows)
    n_repro = sum(r["reproducible"] == "yes" for r in rows)
    print(f"wrote {OUT}: {len(rows)} series "
          f"({n_repro} byte-reproducible, {len(rows) - n_repro} live-source)")


def check():
    if not OUT.exists():
        print(f"{OUT} not found; run `python hash_manifest.py write` first")
        return 1
    ok = bad = missing = drift = 0
    for r in csv.DictReader(open(OUT)):
        p = SERIES / f"{r['name']}.csv"
        if not p.exists():
            print(f"  MISSING  {r['name']}")
            missing += 1
            continue
        if sha256(p) == r["sha256"]:
            ok += 1
        elif r["reproducible"] == "no":
            drift += 1
        else:
            print(f"  MISMATCH {r['name']}  (synthetic: this is a real failure)")
            bad += 1
    print(f"\nmatched {ok}; synthetic mismatches {bad}; "
          f"live-source drift {drift} (expected); missing {missing}")
    return 1 if bad else 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "write"
    sys.exit(check() if cmd == "check" else (write() or 0))
