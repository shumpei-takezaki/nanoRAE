"""Inception Score of uint8 image arrays (.npz with arr_0), with torch-fidelity (the library behind rFID here).

Usage: PYTHONPATH=src uv run python scripts/eval_is.py --out docs/figures/<exp>/inception_score.json samples/stage2/<name>.npz ...
"""

import argparse
import json
from pathlib import Path

import numpy as np
from torch_fidelity import calculate_metrics

from eval.fid import ImgArrDataset

p = argparse.ArgumentParser()
p.add_argument("samples", nargs="+")
p.add_argument("--out", required=True)
args = p.parse_args()

out = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
for f in args.samples:
    m = calculate_metrics(input1=ImgArrDataset(np.load(f)["arr_0"]), isc=True, batch_size=500, cuda=True, verbose=False)
    out[Path(f).stem] = dict(is_mean=m["inception_score_mean"], is_std=m["inception_score_std"])
    print(Path(f).stem, round(out[Path(f).stem]["is_mean"], 3), round(out[Path(f).stem]["is_std"], 3), flush=True)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(out, indent=1))
