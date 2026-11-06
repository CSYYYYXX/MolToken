#!/usr/bin/env python3
import inspect
from pathlib import Path
import sys

PIPE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPE))
from pipeline.runtime_host import require_compute_host
require_compute_host('MOSES evaluator dependency preflight')
from cli.evaluate.evaluate_axis_b_moses_official import load_official_api
api = load_official_api({})
import fcd_torch
import fcd_torch.fcd as fcd
from transformers import StaticCache
print('STATIC_CACHE', inspect.signature(StaticCache), flush=True)
print('FCD_CLASS', inspect.signature(fcd.FCD), flush=True)
root = Path(fcd_torch.__file__).parent
print('FCD_PACKAGE', root, flush=True)
print('FCD_WEIGHTS', [str(p) for p in root.rglob('*.pt')], flush=True)
print('MOSES_IMPORTS_PASS', api.provenance, flush=True)
