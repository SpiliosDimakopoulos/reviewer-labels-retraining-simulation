"""Reproduce the results: experiments, analysis, figures and tests.
Run from this folder: python run_all.py   (use --skip-experiments to rebuild from the stored raw output)"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run([sys.executable] + cmd, cwd=HERE, check=True)


if __name__ == "__main__":
    if "--skip-experiments" not in sys.argv:
        run(["experiments.py"])
    run(["analyze.py"])
    run(["figures.py"])
    run(["-m", "pytest", "tests", "-q"])
