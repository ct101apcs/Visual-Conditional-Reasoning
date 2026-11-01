#!/usr/bin/env bash
# All non-thinking runs, one model at a time on the single GPU. Every new model first runs a smoke
# test (24 jobs per kind, rows 1-5); a failed gate skips that model and the queue moves on. Nothing
# is downloaded: every model below is already on disk.
#   setsid nohup tools/eval/queue.sh > data/eval/logs/queue.out 2>&1 &      # track data/eval/logs/queue.log
cd "$(dirname "$0")/../.."
LOG=data/eval/logs/queue.log
mkdir -p data/eval/logs
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

run() {   # run <name> <model> <rows> [extra run.py args...]; stdout goes to runs/<name>/stdout.log
  local name=$1 model=$2 rows=$3; shift 3
  mkdir -p "data/eval/runs/$name"
  say "START $name rows $rows"
  tools/eval/vllm.sh tools/eval/run.py --model "$model" --name "$name" --rows "$rows" "$@" \
    >> "data/eval/runs/$name/stdout.log" 2>&1
  local rc=$?
  say "END   $name rows $rows exit $rc"
  return $rc
}

model() {  # model <name> <path> [extra run.py args...]: smoke, gate, then rows 1-5
  local name=$1 path=$2; shift 2
  if [ ! -f "$path/config.json" ]; then say "SKIP  $name: no checkpoint at $path"; return; fi
  run "$name" "$path" 1,2,3,4,5 --smoke 24 "$@" || { say "SKIP  $name: smoke run failed"; return; }
  python3 tools/eval/check_smoke.py "data/eval/runs/$name-smoke" | tee -a "$LOG" | grep -q "SMOKE OK" \
    || { say "SKIP  $name: smoke gate failed"; return; }
  run "$name" "$path" 1,2,3,4,5 "$@"
}

say "QUEUE start"
while pgrep -f "run.py --model Qwen/Qwen3-VL-8B-Instruct --rows 1,2,3" > /dev/null; do sleep 30; done
say "8B rows 1-3 finished: $(tail -1 data/eval/runs/qwen3-vl-8b-instruct/run.log)"

Q8=/home/stackops/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
run qwen3-vl-8b-instruct "$Q8" 4,5 --smoke 24 \
  && python3 tools/eval/check_smoke.py data/eval/runs/qwen3-vl-8b-instruct-smoke | tee -a "$LOG" | grep -q "SMOKE OK" \
  && run qwen3-vl-8b-instruct "$Q8" 4,5 \
  || say "SKIP  qwen3-vl-8b-instruct rows 4-5"

model qwen3-vl-4b-instruct /home/stackops/.cache/modelscope/hub/models/Qwen/Qwen3-VL-4B-Instruct
# bf16 checkpoint (63 GB) quantized to FP8 while loading; FP8 KV cache. If loading runs out of memory the
# smoke step fails and 32B is skipped; the fallback (download Qwen/Qwen3-VL-32B-Instruct-FP8) needs a go-ahead.
model qwen3-vl-32b-instruct /home/stackops/SAIH-SADL/Qwen3-VL-32B-Instruct \
  --quantization fp8 --kv-cache-dtype fp8 --gpu-mem 0.92 --max-model-len 6144
model internvl3-8b-hf /home/stackops/.cache/modelscope/hub/models/OpenGVLab/InternVL3-8B-hf
say "QUEUE done"
