"""Read-only H2 completion audit; standard library only, no CUDA or mutation.

Run from stdin on h200; save JSON stdout locally. Does not rerun inference/scoring.
"""
import hashlib
import json
import math
from pathlib import Path
import time
from collections import Counter

ROOT = Path('/data/chenjiayu/changzihe/DistillPP_Validation')
RUN = ROOT / 'runs/hya_h2_20261019'
RESULT = ROOT / 'Result/HYA_H2_20261019'
assert ROOT.resolve() == ROOT


def read(p):
    return json.loads(Path(p).read_text())


hash_cache = {}
def sha(p):
    p = Path(p)
    assert p.resolve() == p and p.is_relative_to(ROOT), str(p)
    if str(p) not in hash_cache:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(4 * 1024**2), b''):
                h.update(chunk)
        hash_cache[str(p)] = h.hexdigest()
    return hash_cache[str(p)]


started = time.time()
manifest = read(RUN / 'manifest.json')
protocol = read(RUN / 'protocol.json')
state = read(RUN / 'controller.json')
complete = read(RESULT / 'COMPLETE.json')
assert state['status'] == 'completed' and not state['s3_started']
assert complete['status'] == 'complete' and not complete['S3_started']
assert all(t['status'] == 'passed' for t in state['tasks'].values())
for rel, expected in complete['sha256'].items():
    assert sha(RESULT / rel) == expected, rel
for rel, expected in manifest['frozen_files'].items():
    assert sha(ROOT / rel) == expected, rel
assert sha(RUN / 'resource_verification.json') == manifest['resource_verification_sha256']
assets = read(RUN / 'resource_verification.json')['assets']
for a in assets:
    s = Path(a['path']).stat()
    assert s.st_size == a['bytes'] and s.st_mtime_ns == a['mtime_ns'], a['path']

policies = [
    ('resume_20261019', 'hya_h2_resume_20261019.py', 'resume_script_sha256'),
    ('resume_20261019_sixgpu', 'hya_h2_resume_parallel_20261019.py', 'script_sha256'),
    ('resume_stable_20261019', 'hya_h2_stable_controller_20261019.py', 'script_sha256'),
]
for folder, script, field in policies:
    pol = read(RUN / folder / 'policy.json')
    assert sha(ROOT / 'tools' / script) == pol[field]
    assert sha(RUN / 'manifest.json') == pol['original_manifest_sha256']
preserved = read(RUN / 'resume_20261019/preserved_files.json')
for rel, expected in preserved.items():
    assert (ROOT / rel).stat().st_size == expected['bytes']
    assert sha(ROOT / rel) == expected['sha256'], rel

ph = sha(RUN / 'protocol.json')
assert ph == 'ee18595f1a9cf868995cc47938aa3671ded8dce76b15d49398c1a97937472c59'
scores = {}
for file in sorted((RUN / 'scores').glob('*.json')):
    rec = read(file)
    assert rec['status'] == 'passed' and rec['protocol_sha256'] == ph
    assert not rec['network_attempts']
    for row in rec['rows']:
        key = (row['prompt_id'], row['arm'], row['metric'])
        assert key not in scores
        assert math.isfinite(row['score']) and 0 <= row['score'] <= 1
        assert row['seed'] == 2026101901
        scores[key] = row
expected = {(p['id'], a, m) for p in protocol['prompts'] for a in ['T50', 'S5'] for m in p['metrics']}
assert set(scores) == expected and len(scores) == 224
assert {p.stem for p in (RUN / 'pairs').glob('*.json')} == {p['id'] for p in protocol['prompts']}

archive_counts = {}
generation_result_dirs = {}
for p in protocol['prompts']:
    pair = read(RUN / 'pairs' / (p['id'] + '.json'))
    assert pair['status'] == 'passed' and pair['prompt_id'] == p['id']
    t, s = [pair['records'][a] for a in ['T50', 'S5']]
    assert t['inputs']['hashes'] == s['inputs']['hashes']
    assert t['actual_initial_latent_sha256'] == s['actual_initial_latent_sha256']
    assert t['admission']['uuid'] == s['admission']['uuid']
    assert t['video']['sha256'] != s['video']['sha256']
    inp = read(RUN / 'inputs' / (p['id'] + '.json'))
    assert inp['hashes'] == t['inputs']['hashes'] and inp['prompt'] == p['generation_prompt']
    assert sha(inp['tensor_file']) == inp['file_sha256']
    for a, rec in pair['records'].items():
        assert rec['status'] == 'passed' and rec['protocol_sha256'] == ph
        assert rec['prompt_id'] == p['id'] and rec['prompt'] == p['generation_prompt']
        assert rec['seed'] == 2026101901 and not rec['network_attempts']
        n = 50 if a == 'T50' else 5
        assert rec['formal_path_dit_calls'] == len(rec['steps']) == n
        assert rec['actual_initial_latent_dtype'] == 'torch.float16'
        assert all(x['prediction_dtype'] == 'torch.bfloat16' and x['state_dtype'] == 'torch.float32' for x in rec['steps'])
        assert [x['step'] for x in rec['steps']] == list(range(n))
        assert rec['steps'][-1]['next_sigma'] == 0
        if rec['extra_warmup_dit_calls']:
            assert rec['warmup_repeat_exact'] is True
        v = rec['video']
        assert (v['frames_decord'], v['frames_opencv'], v['width'], v['height'], v['fps']) == (129, 129, 1280, 720, 24)
        assert sha(rec['video_path']) == sha(rec['archive_path']) == v['sha256']
        assert Path(rec['video_path']).stat().st_size == v['bytes']
        assert Path(rec['archive_path']) == ROOT / 'Evaluation_Videos/hya_h2_20261019' / a / (p['id'] + '.mp4')
        for metric in p['metrics']:
            row = scores[(p['id'], a, metric)]
            assert row['video_sha256'] == v['sha256'] and row['video_path'] == rec['archive_path']
        d = Path(rec['video_path']).parent.parent
        generation_result_dirs.setdefault(d, {})[p['id']] = pair['committed_unix']
for a in ['T50', 'S5']:
    paths = list((ROOT / 'Evaluation_Videos/hya_h2_20261019' / a).glob('*.mp4'))
    assert {x.stem for x in paths} == {p['id'] for p in protocol['prompts']}
    archive_counts[a] = len(paths)
partially_completed_blocks = []
for d, committed in generation_result_dirs.items():
    if (d / 'result.json').exists():
        assert read(d / 'result.json')['status'] == 'passed'
    else:
        # A block can commit its first full pair before the next pair conflicts.
        failure = read(d / 'failure.json')
        assert failure['status'] == 'conflict'
        assert all(t < failure['time_unix'] for t in committed.values())
        partially_completed_blocks.append({'block_attempt': str(d),
                                           'valid_committed_prompts': list(committed),
                                           'later_failure_unix': failure['time_unix']})
    teacher = read(d / 'T50_weight_mapping.json')
    student = read(d / 'S5_weight_mapping.json')
    assert teacher['all_tensors_bitwise_roundtrip'] is True
    assert teacher['native_tensors'] == student['native_tensors'] == 856 and teacher['source_tensors'] == 1264

# Restrict process inspection to recorded PIDs and /proc/stat birth tokens.
identities = [(state['controller_pid'], state['controller_start_ticks'], 'controller')]
attempts = []
for tid, task in state['tasks'].items():
    for a in task['attempts']:
        identities.append((a['pid'], a['start_ticks'], tid))
        attempts.append({'task_id': tid, **a})
alive = []
for pid, birth, name in identities:
    stat = Path('/proc') / str(pid) / 'stat'
    try:
        parts = stat.read_text().rsplit(')', 1)[1].split()
        if parts[19] == str(birth) and parts[0] != 'Z':
            alive.append({'pid': pid, 'task': name})
    except FileNotFoundError:
        pass
assert not alive, alive
failures = [read(p) for folder in ['generation', 'evaluation'] for p in (RUN / folder).rglob('failure.json')]
assert len(failures) == 20 and all(f['status'] == 'conflict' for f in failures)
assert sum(a.get('returncode') == 75 for a in attempts) == 20
assert all(a.get('authorized_reshard') is True for a in attempts if a.get('returncode') == -15)
adopted = [a for a in attempts if a.get('returncode') is None]
for a in adopted:
    assert a['task_id'].startswith('gen_r_')
    d = RUN / 'generation' / a['task_id'][4:] / ('attempt%02d' % a['attempt'])
    assert read(d / 'result.json')['status'] == 'passed'
assert len(adopted) == 4

report = {
    'status': 'passed', 'checked_unix': time.time(), 'audit_seconds': time.time() - started,
    'protocol_sha256': ph, 'complete_sha256': sha(RESULT / 'COMPLETE.json'),
    'original_result_files_hash_checked': len(complete['sha256']),
    'frozen_files_hash_checked': len(manifest['frozen_files']),
    'resources_size_mtime_unchanged_since_startup_full_hash': len(assets),
    'scheduling_adapters_hash_checked': len(policies),
    'preserved_recovery_files_hash_checked': len(preserved),
    'pair_count': 64, 'scores': 224, 'archived_videos': archive_counts,
    'raw_and_archive_video_hash_checks': 256, 'shared_input_tensor_files_hash_checked': 64,
    'same_device_and_input_pairs': 64, 'generation_decode_geometry_records_checked': 128,
    'generation_weight_mapping_proofs': len(generation_result_dirs),
    'completed_pairs_retained_from_later_conflicted_block': partially_completed_blocks,
    'unique_files_hashed': len(hash_cache), 'active_recorded_processes': alive,
    'attempts': len(attempts), 'attempt_returncodes': dict(Counter(str(a.get('returncode')) for a in attempts)),
    'resource_conflict_failure_records': len(failures),
    'adopted_success_returncode_unavailable_but_result_verified': len(adopted),
    'S3_started': False,
    'limits': ['Current audit rehashed videos and checked saved double-decoder acceptance; no new decoding or GPU scoring.',
               'Resource identity was fully hashed at startup; this audit checks its recorded manifest digest and unchanged size/mtime, without rereading large weights.',
               'Successful worker exit codes adopted across supervisors remain null, not fabricated as zero.'],
}
print(json.dumps(report, ensure_ascii=False, indent=2))
