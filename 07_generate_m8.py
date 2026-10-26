"""M8: generate CAD-frame deburring candidate paths and provisional tool poses."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent/'output/.mplconfig'))
import argparse
import json
from pipeline_common import ROOT, config
from planning.paths import run_m8
from planning.presentation import write_deliverables

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    pcfg = json.loads((run_dir/'planning_config.json').read_text())
    cfg = config()
    classified, poses, strokes, manifest = run_m8(ROOT, cfg, pcfg, run_dir)
    write_deliverables(ROOT/cfg['output_dir'], run_dir, classified, poses, strokes, manifest)
    print('M8 preview complete:', len(strokes), 'strokes;', len(poses), 'poses; execution_allowed=False')
