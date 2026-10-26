"""Verify and collect the 12 CP01 scores; descriptive differences only."""
import csv
import json
import math
from pathlib import Path
import time

from evaluate_complex_prompt_probe import BATCH, CONFIG, inputs, validate_frozen
from s1_common import PROJECT, METRICS, file_hash, json_write, gpu_snapshot


def main():
    pre = validate_frozen()
    cfg, pair, videos = inputs()
    controller = json.loads((BATCH/'controller.json').read_text())
    assert controller['status'] == 'completed'
    assert [r['metric'] for r in controller['history']] == METRICS
    assert all(r['exit_code'] == 0 for r in controller['history'])
    assert not list(BATCH.glob('evaluation/*/attempt*/failure.json'))
    reports, rows, deltas, occupancy, timing = [], [], [], [], []
    digest = file_hash(BATCH/'preflight.json')
    expected = {r['arm']: r for r in videos}
    sources = {}
    for metric in METRICS:
        path = BATCH/'evaluation'/metric/'attempt01/result.json'
        r = json.loads(path.read_text())
        assert r['status'] == 'passed' and r['metric'] == metric
        assert r['input_preflight_sha256'] == digest
        assert r['vbench_commit'] == cfg['vbench_commit'] and r['network_attempts'] == []
        assert r['admission']['physical_index'] == 6
        assert len(r['rows']) == 2 and {v['arm'] for v in r['rows']} == {'T50', 'S4'}
        by_arm = {}
        for row in r['rows']:
            assert row['metric'] == metric and row['eval_prompt'] == pair['eval_prompt']
            assert row['seed'] == pair['seed'] and row['prompt_id'] == pair['prompt_id']
            for key in ('video_path', 'video_sha256'):
                assert row[key] == expected[row['arm']][key]
            value, n = row['source_returned_score'], row['normalized_score']
            assert math.isfinite(value) and n == (value/100 if metric == 'imaging_quality' else value)
            assert row['raw_score'] == (value*10 if metric == 'aesthetic_quality' else value)
            assert (-1 <= n <= 1) if metric == 'overall_consistency' else (0 <= n <= 1)
            if metric == 'dynamic_degree':
                assert value in (0, 1)
            by_arm[row['arm']] = n
            rows.append(row)
        assert abs(r['aggregate']-(by_arm['T50']+by_arm['S4'])/2) < 1e-6
        deltas.append({'metric': metric, 'T50': by_arm['T50'], 'S4': by_arm['S4'],
            'T50_minus_S4': by_arm['T50']-by_arm['S4'],
            'meaning': 'binary_motion_flag_not_quality' if metric == 'dynamic_degree' else
                       ('native_truncated_prompt_similarity' if metric == 'overall_consistency' else 'quality_proxy')})
        monitor = json.loads((path.parent/'occupancy.json').read_text())
        assert monitor['error'] is None
        assert all(not any(int(v[1]) != r['admission']['pid'] for v in s['processes'])
                   for s in monitor['observations'])
        occupancy.append({'metric': metric, 'observations': len(monitor['observations']), 'error': None})
        timing.append({'metric': metric, **{k:r[k] for k in ('model_load_seconds','scoring_seconds','peak_allocated_gib')}})
        sources[str(path.relative_to(PROJECT))] = file_hash(path)
        reports.append(r)
    assert len(rows) == 12 and len({(r['arm'], r['metric']) for r in rows}) == 12
    text = next(r['text_metadata'] for r in reports if r['metric'] == 'overall_consistency')
    assert text['original_text'] == pair['eval_prompt'] and text['context_length'] == 32
    assert text['truncated'] and text['tokens_with_special_before_truncation'] == 261
    assert len(text['effective_token_ids']) == 32
    output = PROJECT/'Result/COMPLEX_PROMPT_PROBE_20261018/SIX_METRICS_20261019'
    assert output.resolve() == output
    output.mkdir(exist_ok=False)
    json_write(output/'raw_metric_reports.json', reports)
    json_write(output/'summary.json', {'scope': cfg['scope'], 'prompt_id': pair['prompt_id'],
        'seed': pair['seed'], 'prompt_count': 1, 'seeds_per_prompt': 1, 'video_count': 2,
        'metric_values': 12, 'metric_differences': deltas, 'text_metadata': text,
        'evaluation_timing': timing, 'confidence_intervals': None, 'qualification_conclusion': None,
        's3_started': False})
    for name, data in [('metrics_per_video.csv', rows), ('paired_differences.csv', deltas),
                       ('evaluation_timing.csv', timing)]:
        with (output/name).open('x', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    integrity = {'status': 'passed', 'unix_time': time.time(), 'video_hashes_verified_now': videos,
        'source_files_verified': len(pre['source_sha256']), 'evaluation_resources_rehashed_before_run': 8,
        'resource_size_mtime_unchanged_after_run': True, 'metric_count': 6, 'score_count': 12,
        'all_worker_exit_codes': [r['exit_code'] for r in controller['history']],
        'occupancy': occupancy, 'network_attempts': [], 'raw_result_sha256': sources,
        'preflight_sha256': digest, 'config_sha256': file_hash(CONFIG),
        'finalizer_sha256': file_hash(Path(__file__)), 'gpu_snapshot_after': gpu_snapshot(),
        'preflight_note': 'Initial preflight exited 1 because Git discovered an ancestor repository instead of an archive checkout. No GPU work began. Repaired only new validation code to verify fixed source archive bytes; subsequent preflight exited 0.',
        'output_sha256': {p.name:file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}}
    json_write(output/'integrity.json', integrity)
    print(json.dumps({'status': 'passed', 'output': str(output), 'differences': deltas,
                      'text_metadata': text, 'timing': timing}), flush=True)


if __name__ == '__main__':
    main()
