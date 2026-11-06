#!/usr/bin/env python3
"""Audit completed predictions and select a clearly incomplete result summary."""
import argparse
import json
from pathlib import Path
import sys

PIPE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPE))
from pipeline.runtime_host import require_compute_host
from experiments.rebuttal.common import write_json
from experiments.rebuttal.transfer_model import metrics


def main():
    import numpy as np
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    require_compute_host('transfer saved prediction audit')
    reports = []
    for result_file in sorted((args.run / 'models').glob('*/result.json')):
        directory = result_file.parent
        result = json.loads(result_file.read_text())
        data = args.run / 'data' / result['task']
        expected_indices = np.flatnonzero(np.load(data / 'splits.npy') == 2)
        source_targets = np.load(data / 'targets.npy')
        source_ids = json.loads((data / 'ids.json').read_text())
        test_ids = json.loads((directory / 'test_ids.json').read_text())
        with np.load(directory / 'test_predictions.npz') as saved:
            assert np.array_equal(saved['indices'], expected_indices)
            assert np.array_equal(saved['targets'], source_targets[expected_indices])
            assert test_ids == [source_ids[i] for i in expected_indices]
            assert len(test_ids) == len(set(test_ids)) == result['test_records']
            assert np.isfinite(saved['predictions']).all()
            fresh = metrics(saved['targets'], saved['predictions'], result['task'] == 'promoter')
        for key, value in fresh.items():
            if isinstance(value, bool) or value is None:
                assert value == result['metrics'][key], (directory.name, key)
            else:
                assert np.isclose(value, result['metrics'][key], rtol=1e-10, atol=1e-12), (directory.name, key)
        a = json.loads((directory / 'alignment.json').read_text())
        f = json.loads((directory / 'full.json').read_text())
        assert f['alignment_best_epoch'] == a['best_epoch'] == result['alignment_best_epoch']
        assert f['best_epoch'] == result['full_best_epoch']
        for stage in (a, f):
            assert stage['max_epochs'] == 100
            for epoch in stage['history']:
                assert epoch['audit']['duplicates'] == epoch['audit']['omissions'] == epoch['audit']['unexpected'] == 0
                assert epoch['audit']['records'] == epoch['audit']['unique_records'] == result['config']['train_records']
        assert (directory / 'alignment_best.pt').stat().st_size > 0
        assert (directory / 'full_best.pt').stat().st_size > 0
        reports.append(dict(name=directory.name, test_records=len(test_ids), metrics_recomputed=True,
                            unique_test_ids=True, train_epoch_accounting=True, alignment_full_lineage=True))
    out = dict(verified_completed=len(reports), planned=36, reports=reports)
    write_json(args.run / 'runtime' / 'completed_prediction_audit_20261018.json', out)
    print('TRANSFER_COMPLETED_PREDICTION_AUDIT', len(reports), flush=True)


if __name__ == '__main__':
    main()
