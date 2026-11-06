#!/usr/bin/env python3
"""Worker-only, checkpoint-free exploratory comparison of scaling frontiers.

Read scalar exports and training logs only. Never change a training campaign,
its published fits, its inclusion rules, or any checkpoint. All exclusions from
a derived frontier are explicit; rejected vertices and every raw point remain.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import socket

import numpy as np

PROJECT = Path(__file__).resolve().parents[4]
WORKSPACE = PROJECT.parent
STORAGE = Path('./storage/molscaling')
DOMAINS = ('small_molecule', 'peptide', 'protein', 'dna', 'rna')
NAMES = dict(zip(DOMAINS, ('Small molecule', 'Peptide', 'Protein', 'DNA', 'RNA')))
ZH = dict(zip(DOMAINS, ('小分子', '多肽', '蛋白质', 'DNA', 'RNA')))
COLORS = {'single': '#2166ac', 'multi': '#d95f02', 'target_normalized': '#16805d'}
GROUP_NAMES = {'single': 'Single -> single', 'multi': 'Multi -> single (total C)',
               'target_normalized': 'Multi -> single (C_target = r C)'}
PLOT_SETS = (
    ('recorded_objectives', '', 'EXPLORATORY: single validation vs multi training proxy | all observations'),
    ('train_proxy_same_tail', 'train_proxy_same_tail', 'TRAINING PROXY: last 10% progress | local single logs only'),
    ('single_validation_same_log_subset', 'single_validation_same_log_subset',
     'CONTROL: same local single subset (validation) vs multi training'),
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'Cannot publish an unspecified empty CSV: {path}')
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def positive(value, context):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'Invalid positive scalar {context}: {value}')
    return value


def train_events(path):
    events = []
    previous = 0
    with path.open(encoding='utf-8') as stream:
        for lineno, line in enumerate(stream, 1):
            event = json.loads(line)
            if event.get('event') != 'train':
                continue
            step = int(event['step'])
            if step <= previous:
                raise ValueError(f'Non-monotone train steps: {path}:{lineno}')
            event['_previous_step'] = previous
            previous = step
            events.append(event)
    if not events:
        raise ValueError(f'No training events: {path}')
    return events


def window_loss(events, domain=None, *, last_windows=None, final_step=None, tail_fraction=None):
    """Weight by supervised tokens; fractional edge windows assume uniformity.

    The last-fraction diagnostic uses optimizer progress, not an unlogged exact
    endpoint. Missing post-log tail is reported separately, never invented.
    """
    if (last_windows is None) == (tail_fraction is None):
        raise ValueError('Choose exactly one tail definition')
    selected = events[-last_windows:] if last_windows is not None else events
    start = final_step * (1 - tail_fraction) if tail_fraction is not None else None
    total = weighted = 0.0
    used = []
    for event in selected:
        fraction = 1.0
        if start is not None:
            span = event['step'] - event['_previous_step']
            fraction = max(0., min(1., (event['step'] - start) / span))
        if not fraction:
            continue
        if domain is None:
            tokens = positive(event['loss_tokens'], 'loss_tokens')
            loss_sum = positive(event['loss'], 'train loss') * tokens
        else:
            tokens = positive(event['domain_loss_tokens'][domain], f'{domain} tokens')
            loss_sum = positive(event['domain_loss_sum'][domain], f'{domain} loss_sum')
        total += fraction * tokens
        weighted += fraction * loss_sum
        used.append({'start_step': event['_previous_step'], 'end_step': event['step'], 'weight': fraction})
    if not used:
        return {'status': 'no_logged_window_in_requested_tail', 'loss': None, 'tokens': 0, 'windows': []}
    return {'status': 'available', 'loss': weighted / total, 'tokens': total, 'windows': used,
            'boundary_window_uniformity_approximation': any(w['weight'] != 1 for w in used)}


def single_attempt(row):
    if row.get('attempt_dir'):
        return Path(row['attempt_dir'])
    value = row.get('summary_relative_to_workspace')
    if not value:
        raise ValueError(f'No summary provenance: {row["run_id"]}')
    path = Path(value)
    return (path if path.is_absolute() else WORKSPACE / path).parent


def collect_single(snapshot):
    points, audits = [], []
    rows = snapshot['completed_rows']
    if len({r['run_id'] for r in rows}) != len(rows) or snapshot['pending_rows']:
        raise ValueError('Expected unique completed rows and no pending rows')
    for row in rows:
        domain = row['domain']
        if domain not in DOMAINS:
            raise ValueError(domain)
        attempt = single_attempt(row)
        point = {'run_id': row['run_id'], 'model_id': row['model_id'], 'domain': domain,
                 'budget': row['budget'], 'C': positive(row['target_flops_6nd'], 'single C'),
                 'N': int(row['parameter_count']), 'loss': positive(row['validation_loss'], 'validation loss'),
                 'loss_kind': 'full_validation', 'attempt': str(attempt),
                 'epochs': float(row['actual_epoch_fraction']),
                 'actual_C': float(row['actual_flops_6nd']),
                 'actual_input_tokens': int(row['actual_input_tokens'])}
        points.append(point)
        audit = {'run_id': row['run_id'], 'domain': domain, 'attempt': str(attempt),
                 'train_log': str(attempt / 'train_log.jsonl'), 'status': 'missing_offhost_log'}
        if not attempt.exists():
            if row.get('attempt_dir'):
                raise FileNotFoundError(f'Previously local completed attempt is missing: {attempt}')
            audits.append(audit)
            continue
        summary = read_json(attempt / 'summary.json')
        if int(summary['parameter_count']) != point['N']:
            raise ValueError(f'Parameter count mismatch: {attempt}')
        if not math.isclose(summary['final_validation']['loss'], point['loss'], abs_tol=1e-8, rel_tol=1e-8):
            raise ValueError(f'Validation export differs from native summary: {attempt}')
        events = train_events(attempt / 'train_log.jsonl')
        final_step = int(summary['global_step'])
        if events[-1]['step'] > final_step:
            raise ValueError(f'Train log beyond final step: {attempt}')
        audit.update(status='available', final_step=final_step, last_logged_step=events[-1]['step'],
                     train_tail10=window_loss(events, last_windows=10),
                     train_tail_fraction10=window_loss(events, final_step=final_step, tail_fraction=.1))
        if audit['train_tail_fraction10']['status'] != 'available':
            raise ValueError(f'No end-of-training proxy for completed single run: {attempt}')
        audits.append(audit)
    return points, audits


def collect_multi(source):
    with source.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    points, audits = [], []
    seen = set()
    for row in rows:
        key = (row['target_label'], row['model_id'])
        if key in seen:
            raise ValueError(f'Duplicate multi run: {key}')
        seen.add(key)
        root = Path(row['out_dir'])
        config, summary = read_json(root / 'run_config.json'), read_json(root / 'summary.json')
        args = config['args']
        required = {'dataset_mode': 'online-pad', 'model_implementation': 'packed-xformers',
                    'online_pad_length': 512, 'bpe_unfold_policy': 'none', 'vocab_size': 17207,
                    'token_offset': 3, 'pad_token_id': 0, 'eos_token_id': 2, 'log_domain_loss': True,
                    'online_shuffle': True, 'split_strategy': 'stride', 'val_fraction': .001}
        for k, v in required.items():
            if args.get(k) != v:
                raise ValueError(f'Multi protocol mismatch {root}: {k}={args.get(k)!r}, expected {v!r}')
        n = int(config['parameter_count'])
        if n != int(summary['parameter_count']):
            raise ValueError(f'Native N mismatch: {root}')
        events = train_events(root / 'train_log.jsonl')
        final_step = int(summary['global_step'])
        if final_step != int(args['max_steps']) or events[-1]['step'] > final_step:
            raise ValueError(f'Incomplete multi training: {root}')
        input_counts, loss_counts, sample_counts = Counter(), Counter(), Counter()
        rank_ratio = []
        for event in events:
            for field in ('domain_input_tokens', 'domain_loss_tokens', 'domain_samples', 'domain_loss_sum'):
                if set(event[field]) != set(DOMAINS):
                    raise ValueError(f'Domain keys changed: {root}, {field}')
            for domain in DOMAINS:
                # No BOS, independent causal samples: one input token per sample
                # has no supervised successor prediction. Counts already sum DDP ranks.
                if event['domain_input_tokens'][domain] - event['domain_loss_tokens'][domain] != event['domain_samples'][domain]:
                    raise ValueError(f'Input/loss/sample accounting mismatch: {root}, {domain}')
            input_counts.update(event['domain_input_tokens'])
            loss_counts.update(event['domain_loss_tokens'])
            sample_counts.update(event['domain_samples'])
            rank_ratio.append(sum(event['domain_loss_tokens'].values()) / event['loss_tokens'])
        total = sum(input_counts.values())
        logged_fraction = events[-1]['step'] / final_step
        audit = {'run_id': '/'.join(key), 'model_id': row['model_id'], 'out_dir': str(root),
                 'N': n, 'old_csv_N': int(row['parameter_count']), 'world_size': config['world_size'],
                 'C': float(row['target_flops_6nd']), 'final_step': final_step,
                 'last_logged_step': events[-1]['step'], 'logged_step_fraction': logged_fraction,
                 'missing_tail_steps': final_step - events[-1]['step'],
                 'logged_input_tokens': dict(input_counts), 'logged_loss_tokens': dict(loss_counts),
                 'logged_samples': dict(sample_counts),
                 'input_token_ratios': {d: input_counts[d] / total for d in DOMAINS},
                 'global_domain_vs_rank0_loss_token_ratio_min': min(rank_ratio),
                 'global_domain_vs_rank0_loss_token_ratio_max': max(rank_ratio),
                 'logged_input_flops_vs_nominal': 6 * n * total / float(row['target_flops_6nd']),
                 'legacy_65_scope': row['model_id'] not in {'1p70b', '4p00b'}}
        audits.append(audit)
        for domain in DOMAINS:
            tail10 = window_loss(events, domain, last_windows=10)
            fractional = window_loss(events, domain, final_step=final_step, tail_fraction=.1)
            if fractional['status'] != 'available':
                raise ValueError(f'No last 10 percent logged for {root}/{domain}')
            points.append({'run_id': '/'.join(key), 'model_id': row['model_id'], 'domain': domain,
                           'budget': row['target_label'], 'C': audit['C'], 'N': n,
                           'loss': tail10['loss'], 'loss_kind': 'train_last10_logged_windows',
                           'train_tail10': tail10, 'train_tail_fraction10': fractional,
                           'r_observed': audit['input_token_ratios'][domain], 'out_dir': str(root),
                           'legacy_65_scope': audit['legacy_65_scope']})
    return points, audits


def ratio_report(audits):
    denominator = sum(sum(r['logged_input_tokens'].values()) for r in audits)
    result = {}
    for domain in DOMAINS:
        values = np.array([r['input_token_ratios'][domain] for r in audits])
        pooled = sum(r['logged_input_tokens'][domain] for r in audits) / denominator
        result[domain] = {'r': pooled, 'run_min': float(values.min()), 'run_max': float(values.max()),
                          'run_mean': float(values.mean()), 'run_std': float(values.std()),
                          'max_relative_deviation_from_pooled': float(np.max(np.abs(values / pooled - 1))),
                          'definition': 'pooled observed input tokens incl EOS, without PAD; logged windows only'}
    return result


def fit_quadratic(points):
    ordered = sorted(points, key=lambda r: (r['N'], r['run_id']))
    if len({p['N'] for p in ordered}) != len(ordered):
        raise ValueError('Repeated model size needs an explicit architecture/seed analysis')
    if len({p['C'] for p in ordered}) != 1:
        raise ValueError('Cannot mix compute budgets in an IsoFLOP curve')
    x, y = np.log10([p['N'] for p in ordered]), np.array([p['loss'] for p in ordered])
    best = int(np.argmin(y))
    result = {'domain': ordered[0]['domain'], 'budget': ordered[0]['budget'], 'C': ordered[0]['C'],
              'count': len(ordered), 'run_ids': [p['run_id'] for p in ordered],
              'observed_N': ordered[best]['N'], 'observed_loss': float(y[best]),
              'observed_min_sides': [best, len(ordered) - best - 1],
              'status': 'too_few_points', 'vertex': None, 'strong_support': False}
    if len(ordered) < 3:
        return result
    center = float(x.mean())
    coefficients = np.polyfit(x - center, y, 2)
    predicted = np.polyval(coefficients, x - center)
    residual_ss, total_ss = float(np.sum((y - predicted)**2)), float(np.sum((y-y.mean())**2))
    r2 = 1 - residual_ss / total_ss if total_ss > 0 else None
    result.update(coefficients_centered=coefficients.tolist(), log10_N_center=center,
                  r2=r2, rmse=math.sqrt(residual_ss/len(y)), residuals=(y-predicted).tolist())
    if coefficients[0] <= 0:
        result['status'] = 'nonconvex'
        return result
    vx = float(center - coefficients[1] / (2 * coefficients[0]))
    result['candidate_log10_N'] = vx
    if not x.min() < vx < x.max():
        result['status'] = 'vertex_outside_observed_N'
        return result
    if best == 0 or best == len(y)-1:
        result['status'] = 'observed_best_at_boundary'
        return result
    n, loss = 10**vx, float(np.polyval(coefficients, vx-center))
    if loss <= 0:
        result['status'] = 'nonpositive_fitted_loss'
        return result
    sides = [int(np.sum(x < vx)), int(np.sum(x > vx))]
    agreement = abs(int(np.argmin(predicted)) - best) <= 1
    strong = len(y) >= 5 and r2 is not None and r2 >= .8 and min(sides) >= 2 and min(result['observed_min_sides']) >= 2 and agreement
    result.update(status='interior', strong_support=strong, vertex_min_sides=sides,
                  fitted_observed_grid_min_agreement=agreement,
                  vertex={'C': result['C'], 'N': n, 'D': result['C']/(6*n), 'L': loss,
                          'strong_support': strong, 'budget': result['budget']})
    return result


def linear_fit(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or len(set(x)) < 3:
        return None
    center = float(x.mean())
    slope, value = np.polyfit(x-center, y, 1)
    predicted = slope*(x-center)+value
    total = float(np.sum((y-y.mean())**2))
    r2 = 1-float(np.sum((y-predicted)**2))/total if total > 0 else None
    loo = [float(np.polyfit(np.delete(x, i)-center, np.delete(y, i), 1)[0]) for i in range(len(x))]
    return {'slope': float(slope), 'intercept': float(value-slope*center),
            'log10_C_pivot': center, 'value_at_pivot': float(value), 'r2': r2,
            'leave_one_budget_out_slope_range': [min(loo), max(loo)], 'count': len(x)}


def fit_frontier(curves):
    vertices = sorted([r['vertex'] for r in curves if r['vertex'] is not None], key=lambda r: r['C'])
    result = {'vertices': vertices, 'vertex_count': len(vertices),
              'strong_vertex_count': sum(v['strong_support'] for v in vertices),
              'curve_status_counts': dict(Counter(r['status'] for r in curves)), 'fits': {}}
    if len(vertices) < 3:
        result['status'] = 'fewer_than_three_interior_vertices_no_line'
        return result
    x = np.log10([v['C'] for v in vertices])
    for field in ('N', 'D', 'L'):
        values = [v[field] for v in vertices]
        result['fits'][field] = linear_fit(x, values if field == 'L' else np.log10(values))
    result.update(status='exploratory_fit', C_range=[vertices[0]['C'], vertices[-1]['C']],
                  alpha_plus_beta=result['fits']['N']['slope']+result['fits']['D']['slope'])
    strong = [v for v in vertices if v['strong_support']]
    result['strong_only_fits'] = {field: linear_fit(np.log10([v['C'] for v in strong]),
                                [v[field] for v in strong] if field == 'L' else np.log10([v[field] for v in strong]))
                                for field in ('N', 'D', 'L')} if len(strong) >= 3 else None
    return result


def normalize_frontier(curves, ratio):
    converted = []
    for curve in curves:
        result = dict(curve)
        if curve['vertex']:
            v = curve['vertex']
            result['vertex'] = {**v, 'C': ratio*v['C'], 'D': ratio*v['D']}
        converted.append(result)
    return fit_frontier(converted)


def evaluate_line(frontier, field, compute):
    fit = frontier['fits'][field]
    y = fit['slope']*(math.log10(compute)-fit['log10_C_pivot'])+fit['value_at_pivot']
    return y if field == 'L' else 10**y


def compare_loss(a, b):
    if not a['fits'] or not b['fits']:
        return {'status': 'insufficient_fitted_vertices'}
    low = max(a['C_range'][0], b['C_range'][0])
    high = min(a['C_range'][1], b['C_range'][1])
    if high <= low:
        return {'status': 'no_common_supported_compute_interval'}
    rows = []
    for compute in (low, math.sqrt(low*high), high):
        av, bv = evaluate_line(a, 'L', compute), evaluate_line(b, 'L', compute)
        rows.append({'C_axis': compute, 'single_loss_fit': av, 'multi_loss_fit': bv,
                     'delta_multi_minus_single': bv-av,
                     'single_N_fit': evaluate_line(a, 'N', compute),
                     'multi_N_fit': evaluate_line(b, 'N', compute),
                     'single_D_fit': evaluate_line(a, 'D', compute),
                     'multi_D_fit': evaluate_line(b, 'D', compute)})
    return {'status': 'within_fitted_vertex_support_not_causal', 'C_range': [low, high], 'comparisons': rows,
            'delta_slope': b['fits']['L']['slope']-a['fits']['L']['slope'],
            'warning': 'Different fitted N optima; target-accounting comparison does not hold N or D_target fixed.'}


def analyze(single, multi, ratios):
    groups = defaultdict(list)
    for regime, rows in (('single', single), ('multi', multi)):
        for point in rows:
            groups[(point['domain'], regime, point['C'])].append(point)
    curves = {key: fit_quadratic(points) for key, points in groups.items()}
    domains = {}
    for domain in DOMAINS:
        sc = [v for (d, g, c), v in curves.items() if d == domain and g == 'single']
        mc = [v for (d, g, c), v in curves.items() if d == domain and g == 'multi']
        s, m, norm = fit_frontier(sc), fit_frontier(mc), normalize_frontier(mc, ratios[domain]['r'])
        if m['fits'] and norm['fits']:
            for field in ('N', 'D', 'L'):
                if not math.isclose(m['fits'][field]['slope'], norm['fits'][field]['slope'], abs_tol=1e-9):
                    raise ValueError('A constant-r coordinate change must not change slopes')
        domains[domain] = {'single': s, 'multi': m, 'target_normalized': norm,
                           'same_total_C': compare_loss(s, m), 'same_target_accounted_C': compare_loss(s, norm),
                           'single_curves': sc, 'multi_curves': mc}
    return domains


def refit_protein_without_budget(record, budget):
    """Named sensitivity fit from saved vertices, without changing source results."""
    excluded = [curve for curve in record['multi_curves'] if curve['budget'] == budget]
    if len(excluded) != 1 or excluded[0]['vertex'] is None:
        raise ValueError(f'Expected one existing interior protein vertex for refit exclusion: {budget}')
    retained = [curve for curve in record['multi_curves'] if curve['budget'] != budget]
    multi = fit_frontier(retained)
    target = normalize_frontier(retained, record['ratio'])
    if not multi['fits'] or not target['fits']:
        raise ValueError(f'Not enough retained protein vertices after excluding {budget}')
    for field in ('N', 'D', 'L'):
        if not math.isclose(multi['fits'][field]['slope'], target['fits'][field]['slope'], abs_tol=1e-9):
            raise ValueError('Target-normalized refit must keep the multi slope')
    return {**record, 'multi': multi, 'target_normalized': target, 'multi_curves': retained,
            'same_total_C': compare_loss(record['single'], multi),
            'same_target_accounted_C': compare_loss(record['single'], target),
            'refit_exclusion': {'budget': budget, 'total_C': excluded[0]['C'],
                               'excluded_curves': excluded,
                               'reason': 'User-requested point-exclusion sensitivity; not a finding of invalid experimental data.',
                               'single_unchanged': True, 'new_quadratic_fit': False}}


def save_figure(fig, root, name):
    for extension in ('png', 'pdf'):
        fig.savefig(root / f'{name}.{extension}', dpi=170, bbox_inches='tight')


def extrapolation_segments(frontier, field, xlim, ylim, scale):
    """Clip the fitted straight line to a fixed viewport in plotting coordinates.

    Return only intervals outside the fitted-vertex C range. This affects the
    display, never the frontier coefficients or supported numerical comparisons.
    """
    if not frontier['fits']:
        return []
    fit = frontier['fits'][field]
    low, high = map(math.log10, xlim)
    ylow, yhigh = (tuple(ylim) if field == 'L' else
                   tuple(math.log10(y * scale) for y in ylim))
    slope, pivot, value = fit['slope'], fit['log10_C_pivot'], fit['value_at_pivot']
    if slope == 0:
        if not ylow <= value <= yhigh:
            return []
    else:
        yclip = sorted(pivot + (y - value) / slope for y in (ylow, yhigh))
        low, high = max(low, yclip[0]), min(high, yclip[1])
    support_low, support_high = map(math.log10, frontier['C_range'])
    intervals = ((low, min(high, support_low)), (max(low, support_high), high))
    segments = []
    for start, end in intervals:
        if end > start:
            xs = [10**start, 10**end]
            ys = [evaluate_line(frontier, field, x) / scale for x in xs]
            segments.append((xs, ys))
    return segments


def extend_frontier_lines(ax, record, field, scale, linewidth):
    # Freeze the original data-supported viewport before adding extrapolation.
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    for group in COLORS:
        for xs, ys in extrapolation_segments(record[group], field, xlim, ylim, scale):
            ax.plot(xs, ys, color=COLORS[group], lw=linewidth, linestyle='--',
                    zorder=1.9, scalex=False, scaley=False, dash_capstyle='butt')


def vertices_for_display(frontier, domain, group, hidden_protein_budget=None):
    """User-requested display omission only; all source vertices and fits survive."""
    vertices = frontier['vertices']
    if domain != 'protein' or group == 'single' or hidden_protein_budget is None:
        return vertices
    hidden = [v for v in vertices if v['budget'] == hidden_protein_budget]
    if len(hidden) != 1:
        raise ValueError(f'Expected exactly one protein {group} display point for '
                         f'{hidden_protein_budget}, found {len(hidden)}')
    return [v for v in vertices if v['budget'] != hidden_protein_budget]


def add_peptide_nd_inset(ax, record):
    """Keep the full linear panel, and magnify its low-D blue/green series."""
    from matplotlib.ticker import MaxNLocator
    groups = ('single', 'target_normalized')
    if not all(record[group]['vertices'] for group in groups):
        return None
    # Empty area below the orange curve; parent axes retain all original points.
    inset = ax.inset_axes((.36, .13, .60, .31), facecolor='#fcfcfc')
    for group in groups:
        vertices = sorted(record[group]['vertices'], key=lambda vertex: vertex['C'])
        xs = [vertex['N'] / 1e6 for vertex in vertices]
        ys = [vertex['D'] / 1e6 for vertex in vertices]
        inset.plot(xs, ys, color=COLORS[group], lw=1.3)
        for vertex, x, y in zip(vertices, xs, ys):
            inset.scatter(x, y, s=18, edgecolor=COLORS[group],
                          facecolor=COLORS[group] if vertex['strong_support'] else 'white', zorder=3)
    inset.set_xscale('linear')
    inset.set_yscale('linear')
    inset.set_xlim(left=0)
    inset.set_ylim(bottom=0)
    inset.set_title('Blue + green detail (linear)', fontsize=8, pad=4)
    inset.set_xlabel('N* / M', fontsize=7, labelpad=1)
    inset.set_ylabel('D* / M tokens', fontsize=7, labelpad=2)
    inset.tick_params(labelsize=7, length=2, pad=1)
    inset.xaxis.set_major_locator(MaxNLocator(nbins=4))
    inset.yaxis.set_major_locator(MaxNLocator(nbins=3))
    inset.grid(alpha=.15)
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_color('#999999')
        spine.set_linewidth(.7)
    return inset


def plot_nd_vertices(ax, record, domain, show_legend=False, hidden_protein_budget=None):
    """Pair the saved per-budget N*, D* vertices; never evaluate a fitted line.

    The points are the same quadratic vertices used in the existing columns,
    not individual training runs. Unit conversions do not change linear axes.
    """
    for group in COLORS:
        vertices = sorted(vertices_for_display(record[group], domain, group, hidden_protein_budget),
                          key=lambda vertex: vertex['C'])
        if not vertices:
            continue
        xs = [vertex['N'] / 1e6 for vertex in vertices]
        ys = [vertex['D'] / 1e9 for vertex in vertices]
        ax.plot(xs, ys, color=COLORS[group], lw=1.5, label=GROUP_NAMES[group])
        for vertex, x, y in zip(vertices, xs, ys):
            ax.scatter(x, y, s=27, edgecolor=COLORS[group],
                       facecolor=COLORS[group] if vertex['strong_support'] else 'white', zorder=3)
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Parameters N* / M (linear)')
    ax.set_ylabel('Input tokens D* / B (linear)')
    ax.set_title(f'{NAMES[domain]}: N* vs D* (linear)')
    ax.grid(alpha=.17)
    if show_legend:
        ax.legend(fontsize=7, loc='best')
    if domain == 'peptide':
        add_peptide_nd_inset(ax, record)


def plot_frontiers(domains, root, title, hidden_protein_budget=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    refit_exclusion = domains['protein'].get('refit_exclusion')
    if hidden_protein_budget:
        title = title.replace('all observations', 'all observations in fits')
    elif refit_exclusion:
        title = title.replace('all observations', 'protein point-exclusion refit')
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    legend = [Line2D([], [], color=COLORS[g], lw=2, label=GROUP_NAMES[g]) for g in COLORS]
    legend += [Line2D([], [], color='gray', marker='o', mfc='white', lw=0, label='Weak-fit vertex (exploratory)')]
    legend += [Line2D([], [], color='gray', lw=1.7, linestyle='--', label='Extrapolation beyond fitted C range')]
    fig, axes = plt.subplots(5, 4, figsize=(21, 19))
    for domain, axs in zip(DOMAINS, axes):
        record = domains[domain]
        for field, ax in zip(('N', 'D', 'L'), axs):
            for group in COLORS:
                frontier = record[group]
                scale = 1e6 if field == 'N' else 1e9 if field == 'D' else 1
                for v in vertices_for_display(frontier, domain, group, hidden_protein_budget):
                    ax.scatter(v['C'], v[field]/scale, s=27, edgecolor=COLORS[group],
                               facecolor=COLORS[group] if v['strong_support'] else 'white', zorder=3)
                if frontier['fits']:
                    grid = np.geomspace(*frontier['C_range'], 150)
                    ax.plot(grid, [evaluate_line(frontier, field, c)/scale for c in grid], color=COLORS[group], lw=1.7)
            ax.set_xscale('log')
            if field != 'L':
                ax.set_yscale('log')
            extend_frontier_lines(ax, record, field, scale, 1.7)
            ax.grid(alpha=.17)
            ax.set_title(f'{NAMES[domain]}: {field}*')
            ax.set_xlabel('C or target-accounted r C (FLOPs)')
            ax.set_ylabel({'N': 'Optimal parameters / M', 'D': 'Optimal input tokens / B', 'L': 'Fitted optimal NLL'}[field])
            if field == 'L':
                ax.text(.98, .96, f'r = {100*record["ratio"]:.4f}%', ha='right', va='top', transform=ax.transAxes, fontsize=8)
        if not record['single']['fits']:
            axs[0].text(.03, .03, 'Single: insufficient interior vertices', transform=axs[0].transAxes, fontsize=8)
        plot_nd_vertices(axs[3], record, domain, hidden_protein_budget=hidden_protein_budget)
    fig.suptitle(title, fontsize=14, y=.995)
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .975), ncol=2, frameon=False)
    fig.text(.5, .018, 'Columns 1-3: solid = fitted-vertex C range; dashed = extrapolation. Column 4: saved N*, D* vertices, linear axes, connected in C order; no fit or extrapolation.', ha='center')
    display_note = 'Peptide inset: blue/green detail, D in millions of tokens; main panel retains all points.'
    if hidden_protein_budget:
        display_note += f' Protein: multi/target pair at {hidden_protein_budget} hidden for display; fits unchanged.'
    elif refit_exclusion:
        display_note += f' Protein: C={refit_exclusion["total_C"]:.2g} multi/target pair excluded; orange/green refitted.'
    fig.text(.5, .006, display_note, ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .94), h_pad=2.0)
    save_figure(fig, root, 'three_frontiers')
    plt.close(fig)
    # Separate five-domain pages keep all three fits readable without a giant overview.
    for domain in DOMAINS:
        fig, axs = plt.subplots(1, 4, figsize=(20, 4.9))
        record = domains[domain]
        for field, ax in zip(('N', 'D', 'L'), axs):
            scale = 1e6 if field == 'N' else 1e9 if field == 'D' else 1
            for group in COLORS:
                frontier = record[group]
                vertices = vertices_for_display(frontier, domain, group, hidden_protein_budget)
                ax.scatter([v['C'] for v in vertices], [v[field]/scale for v in vertices], color=COLORS[group], s=22)
                if frontier['fits']:
                    grid = np.geomspace(*frontier['C_range'], 150)
                    ax.plot(grid, [evaluate_line(frontier, field, c)/scale for c in grid], color=COLORS[group], lw=1.8,
                            label=f'{group}: slope={frontier["fits"][field]["slope"]:.3f}')
            ax.set_xscale('log')
            if field != 'L':
                ax.set_yscale('log')
            extend_frontier_lines(ax, record, field, scale, 1.8)
            ax.set_xlabel('C / target-accounted r C (FLOPs)')
            ax.set_ylabel({'N': 'N* / M', 'D': 'D* / B input tokens', 'L': 'L* (NLL; semilog fit)'}[field])
            ax.grid(alpha=.2)
            ax.legend(fontsize=7, loc='best')
        plot_nd_vertices(axs[3], record, domain, show_legend=True,
                         hidden_protein_budget=hidden_protein_budget)
        fig.suptitle(f'{NAMES[domain]} | {title}\nTarget input-token ratio r={record["ratio"]:.6f}; hollow/weak status in overview', fontsize=11)
        fig.text(.5, .030, 'Columns 1-3: solid = fitted-vertex C range; dashed = extrapolation. Column 4: saved N*, D* vertices, linear axes, connected in C order; no fit or extrapolation.', ha='center')
        if domain == 'peptide':
            fig.text(.5, .006, 'Inset magnifies blue/green points using D in millions of tokens. Both axes remain linear; the full panel retains all points.', ha='center', fontsize=8)
        elif domain == 'protein' and hidden_protein_budget:
            fig.text(.5, .006, f'Display-only omission: multi/target pair at {hidden_protein_budget}. Source data and all fitted coefficients unchanged.', ha='center', fontsize=8)
        elif domain == 'protein' and refit_exclusion:
            fig.text(.5, .006, f'Point-exclusion sensitivity: C={refit_exclusion["total_C"]:.2g} multi/target pair removed; orange/green refitted. Blue unchanged; original full-point results preserved.', ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .065, 1, 1))
        save_figure(fig, root, domain)
        plt.close(fig)


def redraw_saved_frontiers(output, hidden_protein_budget=None, refit_protein_budget=None):
    """Render saved results, optionally with an explicitly requested protein refit."""
    report = read_json(output / 'summary.json')
    analyses = dict(report['analyses'])
    refits = {}
    if refit_protein_budget:
        for key, _, _ in PLOT_SETS:
            refits[key] = refit_protein_without_budget(report['analyses'][key]['protein'], refit_protein_budget)
            analyses[key] = {**report['analyses'][key], 'protein': refits[key]}
    previous = [output / subdir / f'{name}.{extension}'
                for _, subdir, _ in PLOT_SETS
                for name in ('three_frontiers', *DOMAINS)
                for extension in ('png', 'pdf')]
    for path in previous:
        if not path.is_file():
            raise FileNotFoundError(f'Expected existing plot before redraw: {path}')
    archive = output / ('archive_before_protein_refit_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    archive.mkdir()
    for path in previous:
        target = archive / path.relative_to(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    if (output / 'plot_rendering.json').is_file():
        shutil.copy2(output / 'plot_rendering.json', archive / 'plot_rendering.json')
    for filename in ('protein_point_exclusion_refit.json', 'protein_point_exclusion_coefficients.csv'):
        if (output / filename).is_file():
            shutil.copy2(output / filename, archive / filename)
    print(f'ARCHIVED {len(previous)} plots: {archive}', flush=True)
    if refits:
        write_json(output / 'protein_point_exclusion_refit.json', {
            'created_at': datetime.now(timezone.utc).isoformat(), 'worker': socket.gethostname(),
            'source': str(output / 'summary.json'), 'excluded_budget': refit_protein_budget,
            'kind': 'User-requested point-exclusion sensitivity; source summary and full-point coefficients unchanged.',
            'analyses': refits})
        write_csv(output / 'protein_point_exclusion_coefficients.csv', [
            {'analysis': key, 'domain': 'protein', 'group': group, 'field': field, **fit}
            for key, record in refits.items() for group in COLORS
            for field, fit in record[group]['fits'].items()])
        for key, record in refits.items():
            old = report['analyses'][key]['protein']['multi']
            new = record['multi']
            print(f'REFIT {key}: protein multi vertices {old["vertex_count"]}->{new["vertex_count"]}; '
                  f'alpha={new["fits"]["N"]["slope"]:.6f}, beta={new["fits"]["D"]["slope"]:.6f}, '
                  f'L_slope={new["fits"]["L"]["slope"]:.6f}', flush=True)
    for key, subdir, title in PLOT_SETS:
        print(f'RENDERING {key}; protein_refit={bool(refits)}', flush=True)
        plot_frontiers(analyses[key], output / subdir, title, hidden_protein_budget)
    exclusions = []
    if hidden_protein_budget:
        for key, _, _ in PLOT_SETS:
            for group in ('multi', 'target_normalized'):
                for vertex in report['analyses'][key]['protein'][group]['vertices']:
                    if vertex['budget'] == hidden_protein_budget:
                        exclusions.append({'analysis': key, 'domain': 'protein', 'group': group, **vertex,
                                           'reason': 'User-requested display-only omission; source data and fits retained.'})
    write_json(output / 'plot_rendering.json', {
        'created_at': datetime.now(timezone.utc).isoformat(), 'worker': socket.gethostname(),
        'source': str(output / 'summary.json'), 'archive': str(archive),
        'plots_rendered': len(previous), 'refitted': bool(refits),
        'refit_scope': 'Protein multi and target_normalized N/D/L frontiers only.' if refits else None,
        'refit_results': str(output / 'protein_point_exclusion_refit.json') if refits else None,
        'display': 'Columns 1-3 use solid fits within retained-vertex ranges and dashed extensions. Column 4 connects the same vertices on linear axes without fitting or extrapolation.',
        'nd_column': {'source': 'Saved per-budget quadratic vertices, not individual training runs or frontier-line predictions.',
                      'x': 'N / 1e6', 'y': 'D / 1e9', 'axes': 'linear', 'connection_order': 'ascending C',
                      'target_normalized': 'Same multi N; D_target = r D.'},
        'peptide_inset': {'groups': ['single', 'target_normalized'], 'x': 'N / 1e6',
                           'y': 'D / 1e6', 'axes': 'linear', 'main_panel_keeps_all_points': True},
        'display_exclusions': exclusions,
        'fit_excluded_budget': refit_protein_budget,
        'numeric_comparisons': ('Protein comparisons recomputed in the separate refit JSON; original summary unchanged.'
                                if refits else 'Unchanged; restricted to common fitted-vertex support.'),
    })
    print(f'REDRAW COMPLETE: {len(previous)} plots; protein_refit={bool(refits)}; no checkpoint evaluation', flush=True)


def report_markdown(primary, ratios, inventory, output):
    lines = ['# 单对单 / 多对单 / 目标域折算：三组 scaling 拟合', '',
             '这是利用现有日志的探索性拟合，不是新测的统一 test-loss 对照；没有读取或下载 checkpoint。', '',
             '## 数据与口径', '',
             f'- 单领域：{inventory["single_points"]} 个已完成点，完整 validation loss；多领域：{inventory["multi_runs"]} 个运行，各有五个领域的末尾 10 个 train 日志窗口 token 加权 loss。',
             '- 主图使用全部原始点，未沿用历史多肽五档的展示性排除。原来旧图省略的 1p70b、4p00b 也纳入；另给 legacy65_scope 敏感性结果，不把训练不足当作删除依据。',
             '- 参数量使用原生 run_config / summary 中的实际 N；C 沿用计划 IsoFLOP 档位，D*=C/(6N*) 是该档位推导值，不伪称逐运行精确消费量。',
             '- r 来自累计已记录的实际 input tokens（包含 EOS、不含 PAD），不是样本占比，也不是 loss-token 占比；DDP 域统计已跨卡求和，不再乘 world_size。未记录的训练末尾不补造，覆盖率在 multi_run_audit.json。',
             f'- 本地可读训练日志 {inventory["single_logs_available"]}/{inventory["single_points"]}，缺少的 {inventory["single_logs_missing"]} 个仅列入 missing_training_logs.csv，不从主图 validation 点中移除。', '',
             '## 拟合定义', '',
             '每个域、每个预算使用全部 N 点，在 log10(N) 上做二次最小二乘拟合。仅正曲率、顶点与实测最低点均在观测 N 范围内部的曲线提供顶点；边界/非凸/不足三点一律保留诊断但不伪造最优点。', '',
             '主线是探索性全内部顶点拟合；R²<0.8、不足双侧 2+2 支持或拟合/观测谷底不一致的顶点以空心标注。strong_only_fits 给出较强支持子集敏感性，不能用它改变原实验验收状态。', '',
             '- log10(N*) = α log10(C) + a',
             '- log10(D*) = β log10(C) + b',
             '- L* = s log10(C) + t（第三项沿用旧项目半对数定义，不是 log(loss) 幂律）', '',
             '目标域折算组取 C_target=rC、D_target=rD，N 和 L 不变。固定 r 必须保持 α、β、s 不变，只改截距；α+β=1 来自 D=C/(6N)，不是独立证据。第三组是同一批 multi 模型的重标坐标，不是第三次训练实验。', '',
             '![三组拟合](three_frontiers.png)', '',
             '实线表示拟合顶点覆盖的 C 区间；同色虚线将相同拟合延伸至图框边界，仅用于展示外推，不扩大数值比较的共同支撑范围。', '',
             '第四列使用前三列相同的每档预算 N*、D* 顶点，横轴参数量（百万）、纵轴输入 token 数（十亿），均为线性刻度，按 C 递增逐点相连，不拟合、不外推。点来自已保存的抛物线顶点，不是幂律直线预测值，也不是全部训练配置；绿色仍采用 D_target=rD。', '',
             '## 系数表', '',
             '| 层级 | 训练方式 | 内部顶点 / 较强支持 | α | β | s（每十倍 C） | L* 直线 R² |',
             '|---|---|---:|---:|---:|---:|---:|']
    for d in DOMAINS:
        for g, label in (('single', '单对单，validation'), ('multi', '多对单，training proxy')):
            f = primary[d][g]
            values = f['fits']
            suffix = (f'{values["N"]["slope"]:.4f} | {values["D"]["slope"]:.4f} | {values["L"]["slope"]:.4f} | {values["L"]["r2"]:.4f}'
                      if values and values['L']['r2'] is not None else '— | — | — | —')
            lines.append(f'| {ZH[d]} | {label} | {f["vertex_count"]} / {f["strong_vertex_count"]} | {suffix} |')
    lines += ['', '目标域折算组斜率与多对单完全相同，截距与全部精确参数在 coefficients.csv / summary.json。', '',
              '## 实际输入 token 占比', '', '| 层级 | 合并占比 r | 各运行占比最小值 | 最大值 |', '|---|---:|---:|---:|']
    for d, r in ratios.items():
        lines.append(f'| {ZH[d]} | {100*r["r"]:.5f}% | {100*r["run_min"]:.5f}% | {100*r["run_max"]:.5f}% |')
    lines += ['', '## 支撑范围内的拟合差值', '',
              'Δ=多对单拟合 loss−单对单拟合 loss，负值只表示当前记录口径下多对单数值较低。由于 train/validation 不统一，不据此下正迁移/负迁移定论；不在共同拟合顶点 C 范围内比较。', '',
              '| 层级 | 对齐方式 | 共同 C 轴范围 | Δ（低端 → 高端） |', '|---|---|---|---:|']
    for d in DOMAINS:
        for key, label in (('same_total_C', '实际总预算'), ('same_target_accounted_C', '目标域账面预算')):
            comp = primary[d][key]
            if 'comparisons' in comp:
                lo, hi = comp['C_range']
                a, b = comp['comparisons'][0], comp['comparisons'][-1]
                result = f'{lo:.3g}–{hi:.3g} | {a["delta_multi_minus_single"]:+.4f} → {b["delta_multi_minus_single"]:+.4f}'
            else:
                result = '没有足够共同支撑 | —'
            lines.append(f'| {ZH[d]} | {label} | {result} |')
    lines += ['', '## 不能省略的限制与补测需求', '',
              '1. 第一张主图直接复用两批实验各自已记录的目标，不能把它标成统一 test loss。辅助目录 train_proxy_same_tail 使用双方训练进度最后 10% 的 token 加权日志；边界窗口按步数占比加权是假定窗口内均匀，未记录末尾仍缺失。single_validation_same_log_subset 使用相同可读日志子集的 validation，可检查子集覆盖与 train/validation 差异。',
              '2. log_every、有效 batch、训练重复次数及模型宽深分布并不完全一致。多肽只有三档 C，部分新增模型超过四个 token-epoch；重复数据上的 training loss 尤其不能自动代表泛化误差。所有 epoch 值随单点导出。',
              '3. 相同 C_target=6ND_target 并不等于同时固定 N 与 D_target；两条前沿的 N* 可以不同。本次只回答用户后续提出的前沿比较，不冒充 need.md 中严格同 N、同 D_target 的因果控制。',
              '4. 旧 multi 使用 stride 0.1% 留出，新 single 使用独立哈希 0.5% validation / 0.5% test。新 test 大部分在旧训练候选池中；统一补测前必须验证共同未训练集合或独立去重测试集。不能简单拿新 test 跑旧模型后宣称严格可比。',
              '5. 本次拟合不需要任何权重。若要先补齐训练代理口径，最小所需是 missing_training_logs.csv 列出的原生 train_log.jsonl 与 summary.json，不是所有 checkpoint。若要统一 held-out loss，再按每档谷底及两侧代表 N 选择 checkpoint；不能只测一个最优 checkpoint 就宣称重建整条新评测前沿。',
              '6. 无独立种子重复；R² 与 leave-one-budget-out 范围只是拟合诊断，不是统计显著性或置信区间。所有源点及无效顶点保留，原有图与训练状态完全不改。', '',
              '文件：summary.json（所有诊断与敏感性）、coefficients.csv（三组线系数）、single_points.csv、multi_points.csv、multi_run_audit.json、missing_training_logs.csv。', '']
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-host', required=True)
    parser.add_argument('--snapshot', type=Path, default=PROJECT/'doc/results/domainwise145_visualization_20260906/snapshot.json')
    parser.add_argument('--multi-csv', type=Path, default=STORAGE/'runs/scaling_fddfs_mf80000_fixed_flops_extension_domainlog_v1/plots/all_fixed_flops_10curves/all_fixed_flops_results.csv')
    parser.add_argument('--output', type=Path, default=PROJECT/'doc/results/analysis/scaling_frontier_comparison_20260910')
    parser.add_argument('--redraw-only', action='store_true',
                        help='Use saved summary.json without reading training logs; refitting requires an explicit option.')
    exclusion = parser.add_mutually_exclusive_group()
    exclusion.add_argument('--hide-protein-budget',
                        help='Display only: hide the named protein multi/target vertex pair; retain data and fitted lines.')
    exclusion.add_argument('--refit-protein-without-budget',
                           help='With --redraw-only, omit this protein multi budget and refit its N/D/L frontiers into separate results.')
    args = parser.parse_args()
    if args.refit_protein_without_budget and not args.redraw_only:
        parser.error('--refit-protein-without-budget requires --redraw-only')
    if socket.gethostname() != args.expected_host or '-worker-' not in socket.gethostname():
        raise RuntimeError(f'Refusing workload outside explicitly named allocated worker: {socket.gethostname()}')
    if args.redraw_only:
        redraw_saved_frontiers(args.output, args.hide_protein_budget, args.refit_protein_without_budget)
        return
    if args.output.exists():
        raise FileExistsError(f'Use a fresh output directory; refusing to overwrite: {args.output}')
    single, sa = collect_single(read_json(args.snapshot))
    print(f'SINGLE: {len(single)} scalar points, {sum(r["status"]=="available" for r in sa)} native logs', flush=True)
    multi, ma = collect_multi(args.multi_csv)
    ratios = ratio_report(ma)
    print(f'MULTI: {len(ma)} runs, {len(multi)} domain points; ratios={ratios}', flush=True)
    available = {a['run_id']: a for a in sa if a['status'] == 'available'}
    proxy_single = [{**p, 'loss': available[p['run_id']]['train_tail_fraction10']['loss'],
                     'loss_kind': 'train_last10percent_steps_logged'} for p in single if p['run_id'] in available]
    proxy_multi = [{**p, 'loss': p['train_tail_fraction10']['loss'],
                    'loss_kind': 'train_last10percent_steps_logged'} for p in multi]
    datasets = {
        'recorded_objectives': (single, multi),
        'legacy65_scope': (single, [p for p in multi if p['legacy_65_scope']]),
        'train_proxy_same_tail': (proxy_single, proxy_multi),
        'single_validation_same_log_subset': ([p for p in single if p['run_id'] in available], proxy_multi),
    }
    analyses = {}
    for label, (sp, mp) in datasets.items():
        analyses[label] = analyze(sp, mp, ratios)
        for domain in DOMAINS:
            analyses[label][domain]['ratio'] = ratios[domain]['r']
    args.output.mkdir(parents=True)
    inventory = {'single_points': len(single), 'multi_runs': len(ma), 'single_logs_available': len(available),
                 'single_logs_missing': len(single)-len(available),
                 'single_domain_counts': dict(Counter(p['domain'] for p in single)),
                 'available_log_domain_counts': dict(Counter(p['domain'] for p in proxy_single)),
                 'weights_loaded': False, 'new_evaluation': False, 'hash_gate_used': False}
    report = {'schema': 'domain_scaling_frontier_comparison_v1', 'created_at': datetime.now(timezone.utc).isoformat(),
              'worker': socket.gethostname(), 'inputs': {'snapshot': str(args.snapshot), 'multi_csv': str(args.multi_csv)},
              'inventory': inventory, 'ratios': ratios, 'analyses': analyses}
    write_json(args.output/'summary.json', report)
    write_json(args.output/'single_log_audit.json', sa)
    write_json(args.output/'multi_run_audit.json', ma)
    write_csv(args.output/'single_points.csv', single)
    write_csv(args.output/'multi_points.csv', multi)
    missing = [a for a in sa if a['status'] != 'available']
    if missing:
        write_csv(args.output/'missing_training_logs.csv', missing)
    coefficients = []
    for analysis_name, data in analyses.items():
        for domain, record in data.items():
            for group in COLORS:
                for field, fit in record[group]['fits'].items():
                    coefficients.append({'analysis': analysis_name, 'domain': domain, 'group': group,
                                         'field': field, **fit})
    write_csv(args.output/'coefficients.csv', coefficients)
    report_markdown(analyses['recorded_objectives'], ratios, inventory, args.output)
    for key, subdir, title in PLOT_SETS:
        root = args.output / subdir
        root.mkdir(exist_ok=True)
        plot_frontiers(analyses[key], root, title, args.hide_protein_budget)
    print('COMPLETE', args.output, flush=True)
    print(json.dumps(inventory, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
