"""M7: validate numerical deviations and compare the mesh with original STEP."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
from pathlib import Path
os.environ.setdefault('XDG_CACHE_HOME', str(Path(__file__).resolve().parent/'output/.cache'))
import argparse
import json
from pipeline_common import ROOT, config
from planning.validation import run_m7

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--planning-config', type=Path, default=ROOT/'planning.example.json')
    args = parser.parse_args()
    pcfg = json.loads(args.planning_config.read_text())
    report = run_m7(ROOT, config(), pcfg, args.run_dir.resolve())
    print('M7 numerical passed:', report['numerical_passed'], 'physical gate passed:', report['physical_gate']['passed'])
    if not report['numerical_passed']:
        raise SystemExit(2)
