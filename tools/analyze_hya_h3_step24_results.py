"""Independent CPU-only reaggregation of the completed four-position panel.

Reads exported raw scoring/record metadata. Original results remain unchanged;
only REVIEW.json is written. No model, video decoding or CUDA imports.
"""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import itertools
import json
import math
from pathlib import Path
import statistics

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main(bundle_path, project):
    b = read(bundle_path)
    p, refs = b['protocol'], b['prompts']
    result = project / 'Result/HYA_H3_STEP24_V1'
    original = read(result / 'summary.json')
    configs = {**p['reused_variants'], **p['variants']}
    variants = sorted(configs, key=lambda v: (configs[v]['intervention_step'], configs[v]['strength']))
    arms = ['T50', 'S5'] + variants
    scores, official = {}, {}
    for source, rows in (('baseline', b['baseline_scores']), ('prior', b['prior_scores']), ('new', b['new_scores'])):
        for r in rows:
            arm = r['arm'] if source == 'baseline' else r['variant']
            key = (r['prompt_id'], arm, r['metric'])
            assert key not in scores and math.isfinite(r['score']) and 0 <= r['score'] <= 1
            scores[key], official[key] = r['score'], r['official_row']
    expected = {(pid, arm, m) for pid in p['prompt_ids'] for arm in arms
                for m in refs[pid]['metrics'] if m in p['metrics']}
    assert set(scores) == expected and len(scores) == 1120
    diffs = []

    def check(actual, recorded):
        if actual is None or recorded is None:
            assert actual is None and recorded is None
            return
        diff = float(np.max(np.abs(np.asarray(actual, dtype=float) - np.asarray(recorded, dtype=float))))
        assert diff <= 1e-12, (actual, recorded)
        diffs.append(diff)

    def csv_rows(name):
        with (result / name).open(encoding='utf-8', newline='') as f:
            return list(csv.DictReader(f))

    exported = csv_rows('scores.csv')
    assert len(exported) == len({(r['prompt_id'], r['group'], r['metric']) for r in exported}) == 1120
    for r in exported:
        check(scores[(r['prompt_id'], r['group'], r['metric'])], float(r['score']))
    paired = csv_rows('paired_scores.csv')
    assert len(paired) == len({(r['prompt_id'], r['variant'], r['metric']) for r in paired}) == 960
    for r in paired:
        pid, v, m = r['prompt_id'], r['variant'], r['metric']
        for field, arm in (('T50', 'T50'), ('S5', 'S5'), ('corrected', v)):
            check(scores[(pid, arm, m)], float(r[field]))
        check(scores[(pid, v, m)] - scores[(pid, 'S5', m)], float(r['corrected_minus_student']))

    summaries, subgroups, object_frames, changes = [], [], {}, {}
    for metric in p['metrics']:
        ids = [pid for pid in p['prompt_ids'] if metric in refs[pid]['metrics']]
        t, s = [np.array([scores[(pid, arm, metric)] for pid in ids]) for arm in ('T50', 'S5')]
        indices = np.random.default_rng(20261020).integers(0, len(ids), (20000, len(ids)))
        for v in variants:
            values = np.array([scores[(pid, v, metric)] for pid in ids])
            delta = values - s
            row = {'metric': metric, 'variant': v, 'n': len(ids), 'T50': float(t.mean()), 'S5': float(s.mean()),
                   'corrected': float(values.mean()), 'corrected_minus_student': float(delta.mean()),
                   'corrected_minus_teacher': float((values - t).mean()),
                   'paired_bootstrap_95_reference': np.quantile(delta[indices].mean(axis=1), [0.025, 0.975]).tolist(),
                   'positive_pairs': int(sum(delta > 0)), 'equal_pairs': int(sum(delta == 0)), 'negative_pairs': int(sum(delta < 0))}
            old = next(r for r in original['metrics'] if r['metric'] == metric and r['variant'] == v)
            for field in row:
                if field not in ('metric', 'variant'):
                    check(row[field], old[field])
            row.update(median_change=float(np.median(delta)), mean_absolute_change=float(np.abs(delta).mean()),
                       positive_contribution=float(np.maximum(delta, 0).mean()), negative_contribution=float(np.minimum(delta, 0).mean()))
            summaries.append(row)
            changes.setdefault(metric, {})[v] = [
                {'prompt_id': pid, 'prompt': refs[pid]['evaluation_prompt'], 'T50': scores[(pid, 'T50', metric)],
                 'S5': float(sv), 'corrected': float(cv), 'delta': float(cv - sv)}
                for pid, sv, cv in zip(ids, s, values) if cv != sv]
            if metric != 'dynamic_degree':
                for label, mask in (('teacher_higher', t > s), ('teacher_lower', t < s), ('equal', t == s)):
                    n = int(sum(mask)); ds = delta[mask]
                    sub = {'metric': metric, 'variant': v, 'baseline_group': label, 'n': n,
                        'T50': float(t[mask].mean()) if n else None, 'S5': float(s[mask].mean()) if n else None,
                        'corrected': float(values[mask].mean()) if n else None,
                        'corrected_minus_student': float(ds.mean()) if n else None,
                        'positive_pairs': int(sum(ds > 0)), 'equal_pairs': int(sum(ds == 0)), 'negative_pairs': int(sum(ds < 0))}
                    previous = next(r for r in original['supplementary_quality_subgroups']
                                    if all(r[k] == sub[k] for k in ('metric', 'variant', 'baseline_group')))
                    for field in sub:
                        if field not in ('metric', 'variant', 'baseline_group'):
                            check(sub[field], previous[field])
                    subgroups.append(sub)
        if metric == 'multiple_objects':
            for arm in arms:
                count = 0
                for pid in ids:
                    r = official[(pid, arm, metric)]
                    assert r['frame_count'] == 16
                    check(r['success_frame_count'] / 16, scores[(pid, arm, metric)])
                    count += r['success_frame_count']
                object_frames[arm] = {'success': count, 'sampled': len(ids) * 16}

    records = {**b['prior_records'], **b['new_records']}
    prefix, within_step = [], []
    for pid in p['prompt_ids']:
        group = [records[pid + '|' + v] for v in variants]
        reference = b['baseline_pairs'][pid]['records']['S5']['first_prediction_sha256']
        for v, r in zip(variants, group):
            if r['first_prediction_sha256'] != reference:
                prefix.append({'prompt_id': pid, 'variant': v, 'new_batch': v in p['variants'],
                    'gpu': r['admission']['index'], 'pid': r['admission']['pid']})
        assert len({r['intervention']['renoise_noise_sha256'] for r in group}) == 1
        for step in range(4):
            rows = [r for r in group if r['intervention_step'] == step]
            for field in ('query_sha256', 'student_endpoint_sha256', 'teacher_prediction_sha256'):
                if len({r['intervention'][field] for r in rows}) != 1:
                    within_step.append({'prompt_id': pid, 'intervention_step': step + 1, 'field': field})

    timing, timing_distribution, magnitudes = {}, {}, {}
    timing_csv = {(r['prompt_id'], r['group']): r for r in csv_rows('timing.csv')}
    assert len(timing_csv) == 672
    magnitude_csv = {(r['prompt_id'], r['group']): r for r in csv_rows('correction_magnitudes.csv')}
    assert len(magnitude_csv) == 576
    for arm in arms:
        values = []
        for pid in p['prompt_ids']:
            if arm in ('T50', 'S5'):
                r = b['baseline_pairs'][pid]['records'][arm]
                compute = r['hot_denoise_decode_seconds']
                inclusive = compute + sum(r[k] for k in ('model_to_gpu_seconds', 'model_offload_seconds', 'vae_to_gpu_seconds', 'vae_offload_seconds'))
            else:
                r = records[pid + '|' + arm]
                compute = sum(r[k] for k in ('student_denoise_seconds', 'teacher_forward_seconds', 'correction_algebra_seconds', 'vae_decode_seconds'))
                inclusive = compute + sum(r[k] for k in ('model_transfer_seconds', 'vae_to_gpu_seconds', 'vae_offload_seconds'))
                check(compute, r['hot_denoise_decode_seconds'])
                check(inclusive, r['inference_with_transfers_seconds'])
                for k in ('student_sigma', 'teacher_sigma', 'endpoint_difference_rms', 'applied_delta_rms', 'applied_delta_relative_rms'):
                    check(r['intervention'][k], float(magnitude_csv[(pid, arm)][k]))
            item = {'compute_seconds': compute, 'inference_with_transfers_seconds': inclusive, 'video_wall_seconds': r['video_wall_seconds']}
            for k, value in item.items():
                check(value, float(timing_csv[(pid, arm)][k]))
            values.append(item)
        timing[arm] = {k: statistics.median(r[k] for r in values) for k in values[0]}
        timing_distribution[arm] = {k: dict(zip(('min', 'p10', 'median', 'p90', 'max'),
                                                  np.quantile([r[k] for r in values], [0, .1, .5, .9, 1]).tolist())) for k in values[0]}
        for k, value in timing[arm].items():
            check(value, original['timing_medians'][arm][k])
        if arm in variants:
            magnitudes[arm] = {k: statistics.median(records[pid + '|' + arm]['intervention'][k] for pid in p['prompt_ids'])
                               for k in ('endpoint_difference_rms', 'applied_delta_rms', 'applied_delta_relative_rms')}
    ratios = {v: timing['T50']['inference_with_transfers_seconds'] / timing[v]['inference_with_transfers_seconds'] for v in variants}
    overhead = {v: timing[v]['inference_with_transfers_seconds'] / timing['S5']['inference_with_transfers_seconds'] - 1 for v in variants}
    for v in variants:
        check(ratios[v], original['teacher_over_corrected_inference_median_ratio'][v])

    lookup = {(cfg['intervention_step'] + 1, cfg['strength']): v for v, cfg in configs.items()}
    contrasts = []
    for metric in p['metrics']:
        ids = [pid for pid in p['prompt_ids'] if metric in refs[pid]['metrics']]
        for first, second in itertools.combinations(range(1, 5), 2):
            for strength in (.1, .3, .5):
                va, vb = lookup[(first, strength)], lookup[(second, strength)]
                aa, bb = [np.array([scores[(pid, arm, metric)] for pid in ids]) for arm in (va, vb)]
                delta = bb - aa
                row = {'metric': metric, 'first_step': first, 'second_step': second, 'strength': strength,
                       'n': len(ids), 'first_mean': float(aa.mean()), 'second_mean': float(bb.mean()),
                       'second_minus_first': float(delta.mean()), 'positive_pairs': int(sum(delta > 0)),
                       'equal_pairs': int(sum(delta == 0)), 'negative_pairs': int(sum(delta < 0))}
                old = next(r for r in original['position_comparisons']
                           if all(r[k] == row[k] for k in ('metric', 'first_step', 'second_step', 'strength')))
                for k in row:
                    if k != 'metric': check(row[k], old[k])
                contrasts.append(row)

    exception_ids = {r['prompt_id'] for r in prefix}
    sensitivity = {}
    for metric in p['metrics']:
        ids = [pid for pid in p['prompt_ids'] if metric in refs[pid]['metrics'] and pid not in exception_ids]
        sensitivity[metric] = {'n': len(ids), 'excluded_ids': sorted(exception_ids & {pid for pid in p['prompt_ids'] if metric in refs[pid]['metrics']}),
            'corrected_minus_student': {v: statistics.mean(scores[(pid, v, metric)] - scores[(pid, 'S5', metric)] for pid in ids) for v in variants}}
    by = {(r['metric'], r['variant']): r for r in summaries}
    both = [v for v in variants if all(by[(m, v)]['corrected_minus_student'] > 0 for m in ('subject_consistency', 'multiple_objects'))]
    unchanged = [v for v in variants if not changes['dynamic_degree'][v]]
    dual_unchanged = [v for v in both if v in unchanged]
    tz = timezone(timedelta(hours=8))
    date = lambda t: datetime.fromtimestamp(t, tz).isoformat(timespec='seconds')
    review = {'verification': b['audit'], 'started_beijing': date(b['audit']['started_unix']),
        'finished_beijing': date(b['audit']['finished_unix']),
        'independent_recalculation': {'checked_numeric_items': len(diffs), 'max_abs_difference': max(diffs),
                                     'score_rows': len(scores), 'paired_rows': len(paired), 'timing_rows': len(timing_csv)},
        'prediction_prefix_check': {'exceptions': prefix, 'new_batch_exception_count': sum(r['new_batch'] for r in prefix),
                                    'within_step_exceptions': within_step},
        'metrics': summaries, 'supplementary_quality_subgroups': subgroups,
        'dynamic_changes': changes['dynamic_degree'], 'object_changes': changes['multiple_objects'],
        'subject_changes': changes['subject_consistency'], 'multiple_object_success_frames': object_frames,
        'position_comparisons': contrasts, 'timing_medians': timing, 'timing_distributions': timing_distribution,
        'teacher_speed_ratios': ratios, 'student_time_overhead': overhead, 'correction_magnitude_medians': magnitudes,
        'sensitivity_excluding_prefix_exception_from_all_arms': sensitivity,
        'interpretation': {'criterion': 'Full-library means; no confidence-interval-crossing gate',
            'both_quality_means_higher': both, 'dynamic_unchanged_per_prompt': unchanged,
            'both_quality_means_higher_and_dynamic_unchanged': dual_unchanged,
            'scope': 'Fixed 48 prompts, one seed, four applicable intervention positions; no claim of universal improvement',
            'no_new_experiment_started': True}}
    (result / 'REVIEW.json').write_text(json.dumps(review, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    compact = {k: review[k] for k in ('independent_recalculation', 'prediction_prefix_check', 'dynamic_changes',
        'multiple_object_success_frames', 'interpretation', 'object_changes', 'teacher_speed_ratios', 'student_time_overhead')}
    compact['new_metric_summaries'] = [r for r in summaries if r['variant'] in p['variants']]
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--project', type=Path, required=True)
    args = parser.parse_args()
    main(args.bundle, args.project)
