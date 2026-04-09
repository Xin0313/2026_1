"""
BOP official evaluation script wrapper for SCHPose.

Usage:
    python tools/eval_bop.py --results_path outputs/ycbv/bop_results.csv \
                              --dataset ycbv --split test
"""

import os
import sys
import argparse
import subprocess
import logging

logger = logging.getLogger("schpose.eval_bop")
logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="BOP evaluation wrapper")
    parser.add_argument("--results_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, help="ycbv/lmo/tless")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--bop_path", type=str, default=None,
                        help="Path to BOP toolkit (if installed separately)")
    args = parser.parse_args()

    if not os.path.isfile(args.results_path):
        logger.error(f"Results file not found: {args.results_path}")
        sys.exit(1)

    # Try to use installed bop_toolkit_lib
    try:
        import bop_toolkit_lib
        bop_available = True
    except ImportError:
        bop_available = False

    if bop_available:
        logger.info("Using installed bop_toolkit_lib for evaluation")
        from bop_toolkit_lib import config as bop_config
        from bop_toolkit_lib import score

        # Run BOP evaluation
        cmd = [
            sys.executable, "-m", "bop_toolkit_lib.scripts.eval_bop19",
            "--renderer_type", "python",
            "--results_path", args.results_path,
            "--eval_path", os.path.dirname(args.results_path),
            "--datasets_path", f"data/{args.dataset}",
        ]
        subprocess.run(cmd, check=False)
    else:
        logger.warning(
            "bop_toolkit_lib not found. Install it from: "
            "https://github.com/thodan/bop_toolkit\n"
            "  pip install git+https://github.com/thodan/bop_toolkit.git\n"
            "\nAlternatively, use the metrics computed by test.py (ADD/ADD-S/AUC)."
        )
        logger.info(f"Results file is ready for manual submission: {args.results_path}")


if __name__ == "__main__":
    main()
