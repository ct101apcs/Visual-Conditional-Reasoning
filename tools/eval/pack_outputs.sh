#!/usr/bin/env bash
# Bundle what the analysis reads besides the files in git, for a GitHub release / Zenodo:
#   the model jobs, each run's outputs, and the per-image GT records of the Test images (slim_records.py).
#   tools/eval/pack_outputs.sh                     # default runs below -> dist/anchor-eval-bundle.tar.gz
#   tools/eval/pack_outputs.sh run-a run-b         # other runs
# Extract at the root of a fresh clone (tar xzf ...), then run consistency.py / analyze.py as in the README.
# Pack the SAM 3 Agent run only after it has finished: a line being written can be cut off.
set -euo pipefail
cd "$(dirname "$0")/../.."
RUNS=("$@")
[ ${#RUNS[@]} -eq 0 ] && RUNS=(qwen3-vl-4b-instruct qwen3-vl-8b-instruct internvl3-8b-hf gemma-3-4b-it qwen3-vl-8b-thinking)
FILES=(data/eval/jobs.jsonl)
for r in "${RUNS[@]}"; do
  for f in responses.jsonl progress.json run.log; do
    [ -f "data/eval/runs/$r/$f" ] && FILES+=("data/eval/runs/$r/$f")
  done
done
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
python3 tools/eval/slim_records.py "$TMP"
mkdir -p dist
tar czf dist/anchor-eval-bundle.tar.gz "${FILES[@]}" -C "$TMP" data/anchor_gt/test_records
ls -lh dist/anchor-eval-bundle.tar.gz
