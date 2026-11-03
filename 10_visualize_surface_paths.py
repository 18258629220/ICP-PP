"""Render focused CAD/observed-surface path and full-pose comparisons."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent/'output/.mplconfig'))
import argparse
from planning.path_visualization import write_path_details


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='Saved ROBOT_BASE surface workflow')
    parser.add_argument('--output-dir', type=Path, required=True, help='New empty local visualization directory')
    parser.add_argument('--path-id', action='append', help='Repeat for one to three specific CAD path IDs')
    parser.add_argument('--axis-length-mm', type=float, help='Arrow display length; geometry remains unscaled')
    args = parser.parse_args()
    result = write_path_details(args.run_dir, args.output_dir, args.path_id,
                                axis_length_mm=args.axis_length_mm)
    print(f'Path visualization: {result}')
