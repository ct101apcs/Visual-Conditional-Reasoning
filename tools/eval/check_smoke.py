"""Gate a full run on its smoke run: exit 1 if any job kind parses badly, truncates, or skips marks.

  python3 tools/eval/check_smoke.py data/eval/runs/<name>-smoke
"""
import sys

import score

MIN_PARSED, MAX_TRUNC, MAX_MISSING = 0.95, 0.02, 0.05


def main(run_dir):
    health, _ = score.report(run_dir)
    bad = []
    for kind, h in sorted(health.items()):
        n = h["n"]
        if h["parsed"] / n < MIN_PARSED:
            bad.append(f"{kind}: parsed {h['parsed'] / n:.0%}")
        if h["truncated"] / n > MAX_TRUNC:
            bad.append(f"{kind}: truncated {h['truncated'] / n:.0%}")
        if h["missing_cand"] / n > MAX_MISSING:
            bad.append(f"{kind}: probes missing a candidate {h['missing_cand'] / n:.0%}")
    if not health:
        bad.append("no responses")
    print("SMOKE " + ("FAIL: " + "; ".join(bad) if bad else f"OK ({len(health)} kinds)"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1])
