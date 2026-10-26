"""Independent CPU recomputation from raw H2 score/timing CSVs and paired metadata.

Does not import the experiment's finalizer, run inference, or change raw results.
"""
import csv
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/hya_h2_20261019'
OUT = ROOT / 'Result/HYA_H2_20261019'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def csvrows(name):
    with (OUT / name).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


protocol, stats, controller = [read(p) for p in [RUN / 'protocol.json', OUT / 'statistics.json', RUN / 'controller.json']]
complete = read(OUT / 'COMPLETE.json')
for name, expected in complete['sha256'].items():
    assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == expected
raw = csvrows('raw_scores.csv')
assert len(raw) == 224
values = {(r['prompt_id'], r['metric'], r['arm']): float(r['score']) for r in raw}
expected = {(p['id'], m, a) for p in protocol['prompts'] for m in p['metrics'] for a in ['T50', 'S5']}
assert len(values) == 224 and set(values) == expected
for f in (RUN / 'scores').glob('*.json'):
    rec = read(f)
    for row in rec['rows']:
        assert values[(row['prompt_id'], row['metric'], row['arm'])] == row['score']
paired = csvrows('paired_scores.csv')
assert len(paired) == 112 and len({(r['prompt_id'], r['metric']) for r in paired}) == 112
for r in paired:
    t, s = [values[(r['prompt_id'], r['metric'], a)] for a in ['T50', 'S5']]
    assert (float(r['T50']), float(r['S5']), float(r['T50_minus_S5'])) == (t, s, t-s)

comparisons = []
def match(name, actual, expected):
    error = abs(float(actual) - float(expected))
    assert error <= 1e-12, (name, actual, expected, error)
    comparisons.append({'name': name, 'absolute_error': error})


summaries = []
for official in stats['metrics']:
    metric = official['metric']
    ids = [p['id'] for p in protocol['prompts'] if metric in p['metrics']]
    t = np.array([values[(pid, metric, 'T50')] for pid in ids], dtype=np.float64)
    s = np.array([values[(pid, metric, 'S5')] for pid in ids], dtype=np.float64)
    differences = t - s
    rng = np.random.Generator(np.random.PCG64(20261019))
    # Chunked resampling is independent of the finalizer's implementation.
    sampled_means = np.concatenate([
        differences[rng.integers(len(ids), size=(2000, len(ids)))].mean(axis=1)
        for _ in range(10)])
    quantiles = [0.025, 0.975] if metric == 'dynamic_degree' else [0.00625, 0.99375]
    lo, hi = np.quantile(sampled_means, quantiles, method='linear')
    result = {'metric': metric, 'prompt_count': len(ids), 'T50_mean': float(t.mean()),
              'S5_mean': float(s.mean()), 'T50_minus_S5': float(differences.mean()),
              'ci_low': float(lo), 'ci_high': float(hi),
              'teacher_higher': int(np.count_nonzero(differences > 0)),
              'equal': int(np.count_nonzero(differences == 0)),
              'student_higher': int(np.count_nonzero(differences < 0)),
              'T50_min': float(t.min()), 'T50_max': float(t.max()),
              'S5_min': float(s.min()), 'S5_max': float(s.max())}
    for k, v in result.items():
        if k != 'metric': match(metric + '.' + k, v, official[k])
    decision = None if metric == 'dynamic_degree' else bool(lo > 0)
    assert decision == official['teacher_advantage_candidate']
    result['teacher_advantage_candidate'] = decision
    result['ci_level'] = official['ci_level']
    if metric == 'dynamic_degree':
        assert np.isin(t, [0, 1]).all() and np.isin(s, [0, 1]).all()
        counts = {'T50': int(t.sum()), 'S5': int(s.sum()),
                  'both_dynamic': int(((t == 1) & (s == 1)).sum()),
                  'both_static': int(((t == 0) & (s == 0)).sum()),
                  'T_dynamic_S_static': int(((t == 1) & (s == 0)).sum()),
                  'T_static_S_dynamic': int(((t == 0) & (s == 1)).sum())}
        for k, v in counts.items():match('dynamic_counts.' + k, v, official['dynamic_counts'][k])
        result['dynamic_counts'] = counts
    summaries.append(result)

timing_rows = csvrows('timing_per_video.csv')
assert len(timing_rows) == 128
times = {(r['prompt_id'], r['arm']): r for r in timing_rows}
assert len(times) == 128
prompt_ids = [p['id'] for p in protocol['prompts']]
matrix = np.array([[float(times[(p, a)]['hot_denoise_decode_seconds']) for a in ['T50', 'S5']] for p in prompt_ids])
cost_fields = ['denoising_seconds', 'vae_decode_seconds', 'hot_denoise_decode_seconds',
               'warmup_seconds', 'model_to_gpu_seconds', 'model_offload_seconds',
               'vae_to_gpu_seconds', 'vae_offload_seconds', 'export_seconds', 'video_wall_seconds']
for p in protocol['prompts']:
    rec = read(RUN / 'pairs' / (p['id'] + '.json'))['records']
    for arm in ['T50', 'S5']:
        for k in cost_fields:
            assert float(times[(p['id'], arm)][k]) == rec[arm][k]
        assert np.isclose(rec[arm]['hot_denoise_decode_seconds'], rec[arm]['denoising_seconds'] + rec[arm]['vae_decode_seconds'], atol=1e-12, rtol=0)
timing = {}
costs = {}
for j, arm in enumerate(['T50', 'S5']):
    q1, median, q3 = np.quantile(matrix[:, j], [0.25, 0.5, 0.75], method='linear')
    timing[arm] = {'median_seconds': float(median), 'q1': float(q1), 'q3': float(q3),
                   'iqr': float(q3-q1), 'total_seconds': float(matrix[:, j].sum()), 'n': 64}
    for k, v in timing[arm].items():match('timing.' + arm + '.' + k, v, stats['timing'][arm][k])
    costs[arm] = {k: {'median_seconds': float(np.median([float(r[k]) for r in timing_rows if r['arm'] == arm])),
                      'total_seconds': float(sum(float(r[k]) for r in timing_rows if r['arm'] == arm))}
                  for k in cost_fields}
rng = np.random.Generator(np.random.PCG64(20261019))
groups = list(dict.fromkeys(p['library'] for p in protocol['prompts']))
indices = []
for group in groups:
    ix = np.array([i for i, p in enumerate(protocol['prompts']) if p['library'] == group])
    indices.append(rng.choice(ix, size=(20000, len(ix)), replace=True))
resampled = matrix[np.concatenate(indices, axis=1)]
ratios = np.median(resampled[:, :, 0], axis=1) / np.median(resampled[:, :, 1], axis=1)
interval = np.quantile(ratios, [0.025, 0.975], method='linear')
timing['ratio_of_medians'] = float(np.median(matrix[:, 0]) / np.median(matrix[:, 1]))
timing['median_paired_ratio'] = float(np.median(matrix[:, 0] / matrix[:, 1]))
timing['ratio_of_medians_95ci'] = interval.tolist()
for k in ['ratio_of_medians', 'median_paired_ratio']:match('timing.' + k, timing[k], stats['timing'][k])
for i in range(2):match('timing.ratio_ci.' + str(i), interval[i], stats['timing']['ratio_of_medians_95ci'][i])

attempts = [{'task_id': tid, **a} for tid, t in controller['tasks'].items() for a in t['attempts']]
assert len(attempts) == len(stats['execution_attempts'])
by_outcome = {}
for code in [0, 75, -15, None]:
    rows = [a for a in attempts if a.get('returncode') == code]
    by_outcome[str(code)] = {'count': len(rows), 'summed_worker_wall_seconds': sum(a['wall_seconds'] for a in rows),
                             'generation': sum(a['task_id'].startswith('gen_') for a in rows),
                             'evaluation': sum(a['task_id'].startswith('eval_') for a in rows)}
final_paths = {v['video_path'].removesuffix('.mp4') + '.json' for p in protocol['prompts']
               for v in read(RUN / 'pairs' / (p['id'] + '.json'))['records'].values()}
offpath = [r for r in stats['all_completed_video_attempts'] if r['path'] not in final_paths]
assert len(stats['all_completed_video_attempts']) - len(offpath) == 128
inputs = [read(RUN / 'inputs' / (p['id'] + '.json')) for p in protocol['prompts']]
load_seconds = {'T50': 0.0, 'S5': 0.0}
successful_blocks = list((RUN / 'generation').glob('*/attempt*/result.json'))
for file in successful_blocks:
    rec = read(file)
    for arm in load_seconds:load_seconds[arm] += rec['model_cpu_load_seconds'][arm]
score_seconds = sum(read(p)['load_scoring_seconds'] for p in (RUN / 'scores').glob('*.json'))
tz = timezone(timedelta(hours=8))
def stamp(value): return datetime.fromtimestamp(value, tz).isoformat(timespec='seconds')
report = {
    'status': 'passed', 'independent_of_finalizer_import': True,
    'python': sys.version, 'numpy_version': np.__version__, 'bootstrap_seed': 20261019,
    'bootstrap_repetitions': 20000, 'numeric_comparisons': len(comparisons),
    'maximum_absolute_error': max(r['absolute_error'] for r in comparisons),
    'raw_scores': 224, 'paired_metric_rows': 112, 'timing_rows': 128,
    'metrics': summaries, 'timing': timing, 'final_video_cost_fields': costs,
    'attempt_outcomes': by_outcome,
    'offpanel_completed_video_attempts': {'count': len(offpath),
        'hot_denoise_decode_seconds': sum(r['hot_denoise_decode_seconds'] for r in offpath),
        'video_wall_seconds': sum(r['video_wall_seconds'] for r in offpath),
        'paths': [r['path'] for r in offpath]},
    'shared_text_inputs_load_encode_total_seconds': sum(r['text_encoder_load_encode_seconds'] for r in inputs),
    'completed_generation_block_load_totals_seconds': load_seconds,
    'completed_generation_block_result_files': len(successful_blocks),
    'committed_evaluation_load_scoring_total_seconds': score_seconds,
    'first_dispatch_beijing': stamp(controller['started_unix']),
    'completed_beijing': stamp(complete['finished_unix']),
    'elapsed_seconds_including_waits_and_recovery': complete['finished_unix'] - controller['started_unix'],
    'original_interval_rule_candidates': [m['metric'] for m in summaries if m['teacher_advantage_candidate']],
    'current_researcher_mean_rule_winners': [m['metric'] for m in summaries
        if m['metric'] != 'dynamic_degree' and m['T50_minus_S5'] > 0],
    'decision': 'Researcher updated qualification during result review: higher mean wins the tested dimension. Teacher wins subject_consistency and multiple_objects. S3 remains unstarted.',
    'limits': ['Summed worker wall time includes CPU loading, lock waits and scoring; it is not GPU kernel time or pure retry waste.',
               'Offpanel complete video costs are part of attempt wall costs, not additive to them.',
               'Load totals from completed block results omit failed blocks without result.json; not a full cold-start benchmark.',
               'Fixed seed, selected prompt panel, approximate simultaneous bootstrap intervals.'],
}
(OUT / 'INDEPENDENT_RECOMPUTATION.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in report.items() if k not in ['metrics', 'python', 'final_video_cost_fields', 'offpanel_completed_video_attempts']}, ensure_ascii=False, indent=2))
print(json.dumps({'final_video_wall_medians': {a: costs[a]['video_wall_seconds']['median_seconds'] for a in costs},
                  'offpanel': {k: v for k, v in report['offpanel_completed_video_attempts'].items() if k != 'paths'}}, indent=2))
