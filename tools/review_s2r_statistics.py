"""Independent CPU review from raw CSVs; does not import frozen analysis helpers."""
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'Result/S2R_20261017'


def read(rel):
    return json.loads((ROOT / rel).read_text(encoding='utf-8'))


def csv_read(name):
    with (OUT / name).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def main():
    q = read('runs/s2r_20261017/protocol.json')
    saved = read('Result/S2R_20261017/statistics.json')
    generation = read('Result/S2R_20261017/generation_records.json')
    score_rows = csv_read('metrics_per_video.csv')
    assert len(score_rows) == 576
    scores = {(r['prompt_id'], int(r['seed']), r['arm'], r['metric']): float(r['normalized_score']) for r in score_rows}
    assert len(scores) == 576 and all(math.isfinite(v) for v in scores.values())
    times = {(r['prompt_id'], r['seed'], r['arm']): r['denoise_plus_vae_seconds'] for r in generation}
    timing_csv = csv_read('timing_per_video.csv')
    assert len(times) == len(timing_csv) == 192
    for r in timing_csv:
        assert times[r['prompt_id'], int(r['seed']), r['arm']] == float(r['denoise_plus_vae_seconds'])
    seeds = q['screen']['seed_slots']
    draws = np.random.Generator(np.random.PCG64(20261017)).integers(0, 16, (10000, 2, 16))
    deltas = []
    reconstructed = []
    per_seed = []
    sampled_times = []
    all_times = []

    def check(a, b):
        error = abs(float(a) - float(b))
        assert math.isfinite(error) and error < 1e-10, (a, b)
        deltas.append(error)

    for domain_i, (domain, ids) in enumerate(q['screen']['active_prompt_ids_by_domain'].items()):
        weights = np.array([np.bincount(row, minlength=16) for row in draws[:, domain_i]]) / 16.0
        for metric in q['metric_domains'][domain]:
            prompt_means = []
            teacher = []
            student = []
            for pid in ids:
                t = [scores[pid, s, 'T50', metric] for s in seeds]
                u = [scores[pid, s, 'S4', metric] for s in seeds]
                teacher.extend(t)
                student.extend(u)
                prompt_means.append(math.fsum(a - b for a, b in zip(t, u)) / 3)
            boot = weights @ np.array(prompt_means)
            lo, hi = np.quantile(boot, [.025, .975] if metric == 'dynamic_degree' else [.005, .995])
            row = {'metric': metric, 'T50_mean': math.fsum(teacher) / 48,
                   'S4_mean': math.fsum(student) / 48, 'T50_minus_S4': math.fsum(prompt_means) / 16,
                   'ci_low': float(lo), 'ci_high': float(hi)}
            original = next(x for x in saved['metric_summaries'] if x['metric'] == metric)
            for key in row:
                if key != 'metric':
                    check(row[key], original[key])
            if metric != 'dynamic_degree':
                assert (lo > 0) == original['teacher_advantage_candidate']
            reconstructed.append(row)
            for seed in seeds:
                per_seed.append({'metric': metric, 'domain': domain, 'seed': seed, 'prompts': 16,
                    'T50_minus_S4': math.fsum(scores[p, seed, 'T50', metric] - scores[p, seed, 'S4', metric] for p in ids) / 16})
        t = np.array([[[times[pid, seed, arm] for arm in ('T50', 'S4')] for seed in seeds] for pid in ids])
        sample = t[draws[:, domain_i]].reshape(10000, 48, 2)
        sampled_times.append(sample)
        all_times.append(t.reshape(48, 2))
        target = saved['timing_by_domain'][domain]
        for a, arm in enumerate(('T50', 'S4')):
            check(np.median(t[:, :, a]), target['per_arm'][arm]['median'])
        check(np.median(t[:, :, 0]) / np.median(t[:, :, 1]), target['speed_ratio_medians'])
        ci = np.quantile(np.median(sample[:, :, 0], axis=1) / np.median(sample[:, :, 1], axis=1), [.025, .975])
        for a, b in zip(ci, target['speed_ratio_95ci']):
            check(a, b)
    flat = np.concatenate(all_times)
    boot = np.concatenate(sampled_times, axis=1)
    target = saved['timing_by_domain']['all_domains_equal_weight']
    check(np.median(flat[:, 0]) / np.median(flat[:, 1]), target['speed_ratio_medians'])
    ci = np.quantile(np.median(boot[:, :, 0], axis=1) / np.median(boot[:, :, 1], axis=1), [.025, .975])
    for a, b in zip(ci, target['speed_ratio_95ci']):
        check(a, b)
    keep = [p for p in q['screen']['active_prompt_ids_by_domain']['visual_semantic'] if p != 'VB0724']
    differences = [math.fsum(scores[p, s, 'T50', 'overall_consistency'] - scores[p, s, 'S4', 'overall_consistency'] for s in seeds) / 3 for p in keep]
    subdraws = np.random.Generator(np.random.PCG64(20261017)).integers(0, 15, (10000, 15))
    subweights = np.array([np.bincount(row, minlength=15) for row in subdraws]) / 15.0
    subci = np.quantile(subweights @ np.array(differences), [.005, .995])
    check(math.fsum(differences) / 15, saved['semantic_nontruncated_sensitivity']['T50_minus_S4'])
    for key, value in zip(('ci_low', 'ci_high'), subci):
        check(value, saved['semantic_nontruncated_sensitivity'][key])
    strata = {}
    for pid in q['screen']['active_prompt_ids_by_domain']['motion']:
        for seed in seeds:
            t = int(scores[pid, seed, 'T50', 'dynamic_degree'])
            s = int(scores[pid, seed, 'S4', 'dynamic_degree'])
            name = {(1, 1): 'both_dynamic', (0, 0): 'both_static', (1, 0): 'teacher_only_dynamic', (0, 1): 'student_only_dynamic'}[t, s]
            strata.setdefault(name, []).append((pid, seed))
    strata_rows = []
    for name, pairs in strata.items():
        strata_rows.append({'stratum': name, 'pairs': len(pairs),
            **{m + '_T50_minus_S4': math.fsum(scores[p, s, 'T50', m] - scores[p, s, 'S4', m] for p, s in pairs) / len(pairs)
               for m in ('subject_consistency', 'motion_smoothness')}})
    for key, name in [('both_dynamic', 'both_dynamic'), ('both_static', 'both_static'),
                      ('T50_dynamic_S4_static', 'teacher_only_dynamic'), ('T50_static_S4_dynamic', 'student_only_dynamic')]:
        assert saved['dynamic'][key] == len(strata[name])
    supervisor = read('runs/s2r_20261017/supervisor_status.json')
    final = read('runs/s2r_20261017/finalizer_status.json')
    start = min(r['unix_start'] for r in supervisor['history'])
    tz = timezone(timedelta(hours=8))
    result = {'status': 'passed', 'implementation': 'Independent raw-CSV grouping; math.fsum and bootstrap frequency-weight matrix; no frozen analysis helper imported.',
        'compared_numbers': len(deltas), 'max_absolute_difference': max(deltas), 'tolerance': 1e-10,
        'qualified_quality_dimensions': sum(r['ci_low'] > 0 for r in reconstructed if r['metric'] != 'dynamic_degree'),
        'metric_summaries_recomputed': reconstructed,
        'wall_time': {'first_worker_start_beijing': datetime.fromtimestamp(start, tz).isoformat(),
                      'archive_complete_beijing': datetime.fromtimestamp(final['unix_time'], tz).isoformat(),
                      'total_elapsed_hours': (final['unix_time'] - start) / 3600,
                      'successful_hot_generation_gpu_hours': float(flat.sum() / 3600),
                      'effective_metric_scoring_seconds': math.fsum(float(r['scoring_seconds']) for r in csv_read('evaluation_timing.csv'))},
        'descriptive_only': {'per_seed_means': per_seed, 'joint_motion_strata': strata_rows,
            'warning': 'Post-generation descriptive grouping only. No exclusions, alternative qualification tests, causal identification, or new thresholds.'},
        'sources_sha256': {n: hashlib.sha256((OUT / n).read_bytes()).hexdigest() for n in
            ['metrics_per_video.csv', 'generation_records.json', 'timing_per_video.csv', 'statistics.json']}}
    dest = OUT / 'INDEPENDENT_STATISTICS_REVIEW.json'
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if dest.exists():
        assert dest.read_text(encoding='utf-8') == text
    else:
        dest.write_text(text, encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('metric_summaries_recomputed', 'sources_sha256')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
