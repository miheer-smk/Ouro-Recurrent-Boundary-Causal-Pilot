"""Compute full statistics from immutable NPZ files; output to stdout by default."""
import json
import argparse
from pathlib import Path
from ouro_boundary_pilot.artifacts import read_raw
from ouro_boundary_pilot.statistics import calculate

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, help='Optional JSON output path; never overwrites raw input files')
    args = p.parse_args()
    result = calculate(*read_raw())
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding='utf-8')
    else: print(payload)
if __name__ == '__main__': main()
