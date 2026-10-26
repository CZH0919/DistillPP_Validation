"""S2-R frozen official-domain input and complete-pair validation."""
import hashlib
import json
from s1_common import (PROJECT, SHAPE, METRICS, NETWORK_ATTEMPTS, json_write,
                       file_hash, tensor_hash, tensor_stats, gpu_snapshot,
                       ensure_no_other_process)
import s1_common as accepted
from s2_common import OccupancyMonitor

BATCH = PROJECT / 'runs/s2r_20261017'


def configure():
    accepted.BATCH = BATCH
    accepted.configure()


def gpu_admission(index, min_free_gib=120):
    return accepted.gpu_admission(index, min_free_gib)


def validate_protocol(q):
    assert q['screen']['prompt_count'] == 32
    ordered = []
    for domain, ids in q['screen']['active_prompt_ids_by_domain'].items():
        assert len(ids) == 16 and len(set(ids)) == 16
        rows = {r['id']: r for r in q['prompts'] if r['screen_selected'] and r['domain'] == domain}
        assert set(rows) == set(ids)
        for identifier in ids:
            r = rows[identifier]
            assert len(r['primary_metrics']) == 3 and r['primary_metrics'] == q['metric_domains'][domain]
            assert not r['used_in_s2_v1'] and r['token_counts']['t5_generation_tokens'] <= 512
            ordered.append(r)
    assert len(ordered) == len({r['cluster_id'] for r in ordered}) == 32
    payload = {'protocol_id': q['protocol_id'], 'seed_slots': q['screen']['seed_slots'],
        'metric_domains': q['metric_domains'], 'prompts': [{k: r[k] for k in
        ('id','domain','cluster_id','eval_prompt','generation_prompt','primary_metrics')} for r in ordered]}
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode('utf-8')).hexdigest()
    assert digest == q['screen']['active_panel_sha256'] == q['migration']['panel_sha256']
    assert q['screen']['seed_slots'] == [1529987743, 1993624635, 1454077509]
    assert q['screen']['video_count'] == 192 and q['screen']['metric_value_count'] == 576
    assert q['score_policy']['compute_only_domain_metrics'] and not q['score_policy']['cross_domain_evaluation']
    return ordered


def load_frozen():
    execution = json.loads((BATCH / 'execution.json').read_text())
    for rel, expected in execution['input_sha256'].items():
        assert file_hash(PROJECT / rel) == expected, 'Frozen input changed: ' + rel
    q = json.loads((BATCH / 'protocol.json').read_text())
    assert file_hash(BATCH / 'protocol.json') == execution['input_sha256']['Project_Plan/QUALIFICATION_V2.json']
    validate_protocol(q)
    assert len(execution['pairs']) == 96 and execution['stage'] == 'S2_R_ONLY'
    return execution, q['baseline']


def complete_generations(execution, verify_video=True, require_all=True):
    result = []
    for pair in execution['pairs']:
        candidates = sorted((BATCH / 'pairs' / pair['pair_id']).glob('attempt*/pair_complete.json'))
        assert len(candidates) <= 1, 'Duplicate accepted pair: ' + pair['pair_id']
        if not candidates and not require_all:
            continue
        assert len(candidates) == 1, 'Missing complete pair: ' + pair['pair_id']
        complete = json.loads(candidates[0].read_text())
        assert complete['status'] == 'passed'
        pair_rows = []
        for arm in ['T50','S4']:
            path = candidates[0].parent / arm / 'result.json'
            row = json.loads(path.read_text())
            for key in ('pair_id','prompt_id','domain','prompt','eval_prompt','seed','primary_metrics','cluster_id'):
                assert row[key] == pair[key], (pair['pair_id'], key)
            assert row['status'] == 'passed' and row['arm'] == arm and row['warmup'] is False
            assert row['forward_count'] == (100 if arm == 'T50' else 4)
            video = path.parent / 'video.mp4'
            if verify_video:
                assert file_hash(video) == row['video']['sha256']
            row.update(video_path=str(video), result_path=str(path))
            pair_rows.append(row)
        for key in ('initial_latent_sha256','positive_context_sha256','gpu_uuid','worker_id'):
            assert pair_rows[0][key] == pair_rows[1][key], (pair['pair_id'], key)
        result.extend(pair_rows)
    if require_all:
        assert len(result) == 192
    return result
