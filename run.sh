#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR="$PWD/output/.mplconfig"
export XDG_CACHE_HOME="$PWD/output/.cache"
.venv/bin/python 01_convert_step.py
.venv/bin/python 02_register_and_analyze.py --stage all
.venv/bin/python 04_registration_sensitivity.py | tee output/sensitivity_console.log
.venv/bin/python -m pytest -q tests | tee output/tests.log
.venv/bin/python 03_validate_and_report.py
