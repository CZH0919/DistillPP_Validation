"""Six inherited VBench metrics for the existing CP01 pair; no generation."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback

import s1_common as safety
from s1_common import PROJECT, METRICS, file_hash, json_write, NETWORK_ATTEMPTS
from s2_common import OccupancyMonitor
from evaluate_s2r import load_metric

BATCH = PROJECT / 'runs/complex_prompt_eval_20261019'
CONFIG = PROJECT / 'Project_Plan/COMPLEX_PROMPT_EVALUATION_20261019.json'
EVAL = '/data/chenjiayu/miniconda3/envs/changzihe_eval/bin/python'


def configure():
    safety.BATCH = BATCH
    safety.configure()
    sys.path[:0] = [str(PROJECT/'src'/x) for x in ('CLIP', 'VBench', 'DINO')]


def inputs():
    cfg = json.loads(CONFIG.read_text())
    assert cfg['metrics'] == METRICS and cfg['allowed_gpu_indexes'] == [6]
    for rel, digest in cfg['input_sha256'].items():
        assert file_hash(PROJECT/rel) == digest, rel
    generation = json.loads((PROJECT/cfg['generation_config']).read_text())
    accepted = json.loads((PROJECT/cfg['generation_acceptance']).read_text())
    assert accepted['status'] == 'passed'
    assert generation['pair']['prompt'] == generation['pair']['eval_prompt'] == accepted['prompt']
    rows = []
    for arm in ('T50', 'S4'):
        video = accepted['videos'][arm]
        path = Path(video['path'])
        assert path.is_relative_to(PROJECT) and path.resolve() == path
        assert file_hash(path) == video['sha256']
        rows.append({'arm': arm, 'video_path': str(path), 'video_sha256': video['sha256']})
    for key in ('initial_latent_sha256', 'positive_context_sha256'):
        assert accepted['videos']['T50'][key] == accepted['videos']['S4'][key]
    return cfg, generation['pair'], rows


def preflight():
    assert not BATCH.exists(), 'Keep previous evaluation attempts intact'
    cfg, pair, rows = inputs()
    source = {rel: file_hash(PROJECT/rel) for rel in cfg['input_sha256']}
    source['tools/evaluate_complex_prompt_probe.py'] = file_hash(Path(__file__))
    source[str(CONFIG.relative_to(PROJECT))] = file_hash(CONFIG)
    resources = []
    manifest = json.loads((PROJECT/'Project_Plan/RESOURCE_MANIFEST.json').read_text())
    for asset in manifest['assets']:
        if asset['group'] != 'evaluation':
            continue
        p = PROJECT/asset['path']
        assert p.resolve() == p and p.stat().st_size == asset['bytes']
        digest = file_hash(p)
        assert digest == asset['sha256'], asset['id']
        resources.append({'path': str(p), 'sha256': digest, 'bytes': p.stat().st_size,
                          'mtime_ns': p.stat().st_mtime_ns})
    assert len(resources) == 8
    repo = PROJECT/'src/VBench'
    # These are verified source archives, not standalone Git checkouts.
    # Never let Git discover a repository above the authorized project root.
    archives = json.loads((PROJECT/'Project_Plan/SOURCE_ARCHIVES.json').read_text())
    commit = next(a['revision'] for a in archives if a['name'] == 'VBench')
    assert commit == cfg['vbench_commit']
    for info in archives:
        if info['group'] != 'evaluation':
            continue
        archive = PROJECT/'Project_Plan'/info['archive']
        assert archive.resolve() == archive and file_hash(archive) == info['sha256']
        assert archive.stat().st_size == info['bytes']
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                if member.isdir():
                    continue
                assert member.isfile() and Path(member.name).parts[0] == info['name']
                p = PROJECT/'src'/member.name
                assert p.is_relative_to(PROJECT/'src') and p.resolve() == p
                expected = hashlib.sha256(tf.extractfile(member).read()).hexdigest()
                assert file_hash(p) == expected, str(p)
                source[str(p.relative_to(PROJECT))] = expected
    bpe = PROJECT/'src/CLIP/clip/bpe_simple_vocab_16e6.txt.gz'
    assert file_hash(bpe) == '924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a'
    source[str(bpe.relative_to(PROJECT))] = file_hash(bpe)
    import decord
    for row in rows:
        reader = decord.VideoReader(row['video_path'], ctx=decord.cpu(0), num_threads=1)
        assert len(reader) == 81 and abs(reader.get_avg_fps()-16) < 1e-6
        frames = reader.get_batch(list(range(len(reader)))).asnumpy()
        assert frames.shape == (81, 480, 832, 3) and float(frames.std()) > 0
        row.update(decoded_frames=81, fps=16, size=[480, 832])
    from vbench.third_party.ViCLIP.simple_tokenizer import SimpleTokenizer
    tokenizer = SimpleTokenizer(str(PROJECT/'resources/evaluation/ViCLIP/bpe_simple_vocab_16e6.txt.gz'))
    ids = tokenizer.encode(pair['eval_prompt'])
    text = {'tokens_with_special_before_truncation': len(ids)+2, 'context_length': 32,
            'truncated': len(ids)+2 > 32, 'decoded_first_30_content_tokens': tokenizer.decode(ids[:30])}
    import torch
    assert not torch.cuda.is_initialized() and not NETWORK_ATTEMPTS
    BATCH.mkdir()
    report = {'status': 'passed', 'unix_time': time.time(), 'source_sha256': source,
              'evaluation_resources': resources, 'videos': rows, 'text_metadata': text,
              'vbench_commit': commit, 'cuda_initialized': False, 'gpu_snapshot': safety.gpu_snapshot(),
              'dependencies': {x: importlib.metadata.version(x) for x in
                               ('torch', 'numpy', 'transformers', 'decord', 'pyiqa', 'timm')}}
    json_write(BATCH/'preflight.json', report)
    print(json.dumps(report), flush=True)


def validate_frozen():
    pre = json.loads((BATCH/'preflight.json').read_text())
    assert pre['status'] == 'passed'
    for rel, digest in pre['source_sha256'].items():
        assert file_hash(PROJECT/rel) == digest, rel
    for r in pre['evaluation_resources']:
        p = Path(r['path'])
        assert p.resolve() == p and (p.stat().st_size, p.stat().st_mtime_ns) == (r['bytes'], r['mtime_ns'])
    return pre


def evaluate(metric):
    pre = validate_frozen()
    cfg, pair, rows = inputs()
    output = BATCH/'evaluation'/metric/'attempt01'
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'starting', 'metric': metric, 'rows': [],
              'scope': 'One exploratory pair; no confidence interval or qualification claim',
              'vbench_commit': cfg['vbench_commit'], 'input_preflight_sha256': file_hash(BATCH/'preflight.json')}
    try:
        with safety.gpu_admission(6, min_free_gib=20) as admission:
            report['admission'] = admission
            import torch
            import cv2
            torch.set_num_threads(4)
            torch.set_num_interop_threads(4)
            torch.set_grad_enabled(False)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.manual_seed(20261017)
            cv2.setNumThreads(1)
            state = {'phase': 'loading_metric', 'metric': metric, 'completed_videos': 0}
            with OccupancyMonitor(admission, output, state) as monitor:
                start = time.perf_counter()
                score, metadata = load_metric(metric, torch)
                torch.cuda.synchronize()
                report['model_load_seconds'] = time.perf_counter()-start
                meta = metadata(pair['eval_prompt']) if metadata else None
                if meta:
                    from vbench.third_party.ViCLIP.simple_tokenizer import SimpleTokenizer
                    tokenizer = SimpleTokenizer(str(PROJECT/'resources/evaluation/ViCLIP/bpe_simple_vocab_16e6.txt.gz'))
                    effective = meta['effective_token_ids']
                    eot = tokenizer.encoder['<|endoftext|>']
                    meta['effective_text_decoded'] = tokenizer.decode(effective[1:effective.index(eot)])
                    assert meta['tokens_with_special_before_truncation'] == pre['text_metadata']['tokens_with_special_before_truncation']
                    assert meta['effective_text_decoded'] == pre['text_metadata']['decoded_first_30_content_tokens']
                monitor.check()
                safety.ensure_no_other_process(admission['uuid'])
                state['phase'] = 'scoring'
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.no_grad():
                    aggregate, returned = score([r['video_path'] for r in rows], pair['eval_prompt'])
                torch.cuda.synchronize()
                report['scoring_seconds'] = time.perf_counter()-start
                assert len(returned) == 2 and {r['video_path'] for r in returned} == {r['video_path'] for r in rows}
                for row in rows:
                    v = float(next(r['video_results'] for r in returned if r['video_path'] == row['video_path']))
                    n = v/100 if metric == 'imaging_quality' else v
                    assert math.isfinite(v)
                    assert (-1 <= n <= 1) if metric == 'overall_consistency' else (0 <= n <= 1)
                    if metric == 'dynamic_degree':
                        assert v in (0, 1)
                    report['rows'].append(dict(row, metric=metric, prompt_id=pair['prompt_id'], seed=pair['seed'],
                        source_returned_score=v, raw_score=v*10 if metric == 'aesthetic_quality' else v,
                        normalized_score=n, eval_prompt=pair['eval_prompt']))
                assert abs(float(aggregate)-sum(r['normalized_score'] for r in report['rows'])/2) < 1e-6
                monitor.check()
                safety.ensure_no_other_process(admission['uuid'])
                assert not NETWORK_ATTEMPTS
                report.update(aggregate=float(aggregate), text_metadata=meta,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30, network_attempts=NETWORK_ATTEMPTS)
                state.update(phase='completed', completed_videos=2)
            assert monitor.error is None
            validate_frozen()
            inputs()
            report['status'] = 'passed'
            json_write(output/'result.json', report)
            print('CP01_METRIC_PASS', metric, json.dumps(report['rows']), flush=True)
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc(), network_attempts=NETWORK_ATTEMPTS)
        json_write(output/'failure.json', report)
        raise


def run_all():
    validate_frozen()
    assert not (BATCH/'controller.json').exists(), 'Do not launch duplicate evaluation'
    history = []
    for metric in METRICS:
        command = [EVAL, '-B', '-u', str(Path(__file__).resolve()), '--metric', metric]
        with (BATCH/(metric+'.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=PROJECT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            json_write(BATCH/'controller.json', {'status': 'running', 'controller_pid': os.getpid(),
                       'active_metric': metric, 'worker_pid': child.pid, 'history': history, 'unix_time': time.time()})
            try:
                rc = child.wait(timeout=1200)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                rc = 124
        history.append({'metric': metric, 'exit_code': rc, 'worker_pid': child.pid, 'unix_end': time.time()})
        if rc:
            json_write(BATCH/'controller.json', {'status': 'failed', 'history': history})
            raise RuntimeError('Evaluation stopped; inspect own log: '+metric)
    json_write(BATCH/'controller.json', {'status': 'completed', 'history': history,
               'unix_time': time.time(), 'expected_scores': 12, 's3_started': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--run-all', action='store_true')
    mode.add_argument('--metric', choices=METRICS)
    args = parser.parse_args()
    configure()
    if args.preflight:
        preflight()
    elif args.run_all:
        run_all()
    else:
        evaluate(args.metric)
