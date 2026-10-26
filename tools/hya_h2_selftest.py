"""CPU-only scheduling, transaction and statistics tests with synthetic data."""
import ast
import copy
import os
from pathlib import Path
import tempfile
import json

import hya_common as h1
import hya_h2_common as c
from hya_h2_controller import task_definitions, process_alive, process_ticks
from hya_h2_finalize import metric_summary, timing_summary
from hya_h2_evaluate import evaluation_info


def main():
    import numpy as np
    protocol = c.read(c.ROOT/'Project_Plan/HUNYUAN_ACCVIDEO_PROTOCOL.json')
    for p in (c.ROOT/'tools').glob('hya_h2_*.py'): ast.parse(p.read_text())
    for n, counts in [(4, [16, 16, 16, 16]), (3, [22, 21, 21])]:
        blocks = c.make_blocks(protocol['prompts'], n)
        seen = [pid for b in blocks for pid in b['prompt_ids']]
        assert len(seen) == len(set(seen)) == 64
        assert [sum(len(b['prompt_ids']) for b in blocks if b['shard'] == i) for i in range(n)] == counts
    blocks = c.make_blocks(protocol['prompts'])
    defs = task_definitions({'blocks': blocks}, protocol)
    assert len(blocks) == 32 and len(defs) == 88 and len({t['id'] for t in defs}) == 88
    assert sum(t['kind'] == 'evaluation' for t in defs) == 56
    assert sum(2*len([p for p in protocol['prompts'] if p['id'] in b['prompt_ids'] and m in p['metrics']])
               for b in blocks for m in set(m for p in protocol['prompts'] if p['id'] in b['prompt_ids'] for m in p['metrics'])) == 224
    for row in protocol['prompts']:
        if 'scene' in row['metrics']: assert isinstance(row['auxiliary_info']['scene']['scene']['scene'], str)
        if 'multiple_objects' in row['metrics']: assert len(row['auxiliary_info']['multiple_objects']['object'].split(' and ')) == 2
    snap = {'gpus': [['0', 'g0', '0', '143771', '0'], ['1', 'g1', '1', '143771', '0'],
                     ['2', 'g2', '0', '143771', '10'], ['3', 'g3', '80', '143771', '0']],
            'processes': [['g1', '99999', '1']]}
    assert c.free_indices(snap) == [0], 'Utilization zero alone must not admit a resident process'
    assert process_alive(os.getpid(), process_ticks(os.getpid()))
    assert not process_alive(os.getpid(), '0')
    # Pair commit must reject mixed device/input identity and preserve orphaned prior copies.
    original_run, original_archive = c.RUN, c.ARCHIVE
    (c.ROOT/'runs').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='hya_h2_selftest_', dir=c.ROOT/'runs') as tmp:
        c.RUN = Path(tmp); c.ARCHIVE = Path(tmp)/'archive'
        utils = ast.parse((c.ROOT/'src/VBench/vbench/utils.py').read_text())
        loader = next(x for x in utils.body if isinstance(x, ast.FunctionDef) and x.name == 'load_dimension_info')
        namespace = {'load_json': c.read}
        exec(compile(ast.Module(body=[loader], type_ignores=[]), '<official VBench load_dimension_info>', 'exec'), namespace)
        fixture = Path(tmp)/'official_labels.json'
        for metric in ['scene', 'multiple_objects']:
            row = next(p for p in protocol['prompts'] if metric in p['metrics'])
            fixture.write_text(json.dumps([{'prompt_en': row['evaluation_prompt'], 'dimension': row['metrics'],
                                           'video_list': ['synthetic.mp4'], 'auxiliary_info': row['auxiliary_info']}]))
            _, official = namespace['load_dimension_info'](str(fixture), metric, 'en')
            assert [evaluation_info(row, metric, 'synthetic.mp4')] == official
        records = {}
        for arm in c.ARMS:
            f = Path(tmp)/(arm+'.mp4'); f.write_bytes(('synthetic-test-'+arm).encode())
            records[arm] = {'status': 'passed', 'prompt_id': 'TEST', 'seed': 1, 'protocol_sha256': 'p',
                            'actual_initial_latent_sha256': 'n', 'inputs': {'hashes': {'context': 'c'}},
                            'admission': {'uuid': 'gpu0'}, 'video_path': str(f), 'video': {'sha256': c.digest(f)}}
        mixed = copy.deepcopy(records); mixed['S5']['admission']['uuid'] = 'gpu1'
        try: c.commit_pair('TEST', mixed)
        except AssertionError: pass
        else: raise AssertionError('Mixed-GPU pair was accepted')
        orphan = c.ARCHIVE/'T50/TEST.mp4'; orphan.parent.mkdir(parents=True); orphan.write_bytes(b'orphan')
        c.commit_pair('TEST', records); c.commit_pair('TEST', records)
        assert len(list((c.RUN/'pairs').glob('*.json'))) == 1
        assert len(list((c.ARCHIVE/'_superseded').glob('*.mp4'))) == 1
        with c.task_claim('gen_TEST'):
            try:
                with c.task_claim('gen_TEST'): pass
            except c.ResourceConflict: pass
            else: raise AssertionError('Duplicate task claim accepted')
    c.RUN, c.ARCHIVE = original_run, original_archive
    positive = metric_summary('scene', [[.7, .5]]*16)
    assert positive['teacher_advantage_candidate'] and abs(positive['ci_low']-.2) < 1e-12
    null = metric_summary('scene', [[.5, .5]]*16)
    assert null['ci_low'] == null['ci_high'] == 0 and not null['teacher_advantage_candidate']
    dynamic = metric_summary('dynamic_degree', [[1, 0]]*32)
    assert dynamic['teacher_advantage_candidate'] is None and dynamic['dynamic_counts']['T_dynamic_S_static'] == 32
    opposed = [[.8, .2]]*8+[[.2, .8]]*8
    assert metric_summary('scene', opposed)['ci_low'] < 0 < metric_summary('scene', opposed)['ci_high']
    timing = timing_summary([[10, 1]]*64, ['motion']*32+['scene']*16+['objects']*16)
    assert timing['ratio_of_medians_95ci'] == [10, 10]
    assert h1.scheduler_contract() == c.read(c.ROOT/'Project_Plan/HYA_RUNTIME_CONTRACT.json')['schedule']
    import torch
    assert not torch.cuda.is_initialized()
    c.RUN.mkdir(exist_ok=True)
    c.write_json(c.RUN/'selftest.json', {'status': 'passed', 'CUDA_initialized': False,
        'checks': ['AST', 'sharding_3_and_4_gpu', '88_unique_tasks_224_scores', 'official_auxiliary_label_adapter', 'resident_pid_rejection',
                   'PID_birth_identity', 'same_device_pair_rejection', 'orphan_copy_recovery', 'idempotent_commit',
                   'duplicate_task_lock', 'paired_statistics_null_and_signed_cases', 'stratified_timing', 'H1_scheduler_identity']}, replace=True)
    print('HYA_H2_CPU_SELFTEST_PASSED', flush=True)


if __name__ == '__main__':
    main()
