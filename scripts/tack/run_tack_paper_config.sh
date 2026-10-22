#!/bin/bash
set -u
project_dir="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$project_dir" || exit 1
output_dir="${1:-results/tack_paper_config_benchmark}"
if [ "$#" -gt 0 ]; then shift; fi
mkdir -p "$output_dir"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR=/private/tmp/tack-mpl
export DYLD_LIBRARY_PATH="$project_dir/.pixi/envs/default/lib/python3.10/site-packages/torch/lib"
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=42
printf '%s\n' "$$" > "$output_dir/supervisor.pid"
.pixi/envs/default/bin/python -u scripts/tack/tack_paper_config_benchmark.py \
  --output-dir "$output_dir" "$@" &
training_pid=$!
printf '%s\n' "$training_pid" > "$output_dir/training.pid"
wait "$training_pid"
training_exit_code=$?
printf '%s\n' "$training_exit_code" > "$output_dir/exit_code.txt"
exit "$training_exit_code"
