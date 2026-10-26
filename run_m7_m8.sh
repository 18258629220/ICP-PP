#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLCONFIGDIR="$PWD/output/.mplconfig" XDG_CACHE_HOME="$PWD/output/.cache"
run_dir="${1:-output/m7_m8/run_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
planning_config="${2:-planning.example.json}"
mkdir -p "$run_dir"
.venv/bin/python 06_validate_m7.py --run-dir "$run_dir" --planning-config "$planning_config" | tee "$run_dir/m7_console.log"
.venv/bin/python 07_generate_m8.py --run-dir "$run_dir" | tee "$run_dir/m8_console.log"
