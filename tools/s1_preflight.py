"""CPU-only S0 completion and S1 admission evidence; no model inference."""
import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from s1_common import PROJECT, BATCH, PROMPT, METRICS, configure, file_hash, json_write, gpu_snapshot


def main():
    configure()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    stage = PROJECT / 'Project_Progress/Stage_Progress'
    inventory = json.loads((stage / 'resource_recovery_inventory.json').read_text())
    assert not Path(inventory['root']).exists()
    clean = sorted(stage.glob('duplicate_cleanup_*.jsonl'))[-1]
    assert json.loads(clean.read_text().splitlines()[-1])['event'] == 'completed'
    current = []
    for entry in inventory['model_inventory']:
        stat = Path(entry['path']).stat()
        assert stat.st_size == entry['bytes'] and stat.st_mtime_ns == entry['mtime_ns'], entry['path']
        current.append({'path': entry['path'], 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    previous = json.loads((stage / 'resource_check_20261016_091241_1370827.json').read_text())
    assert previous['ok'] and len(previous['records']) == 24
    evaluation = []
    for item in json.loads((PROJECT / 'Project_Plan/RESOURCE_MANIFEST.json').read_text())['assets']:
        if item['group'] != 'evaluation':
            continue
        path = PROJECT / item['path']
        assert path.resolve() == path and path.stat().st_size == item['bytes']
        digest = file_hash(path)
        assert digest == item['sha256'], item['id']
        evaluation.append({'id': item['id'], 'path': str(path), 'sha256': digest})
    source_checks = []
    for group in ['generation', 'evaluation']:
        r = subprocess.run([sys.executable, '-B', str(PROJECT / 'Project_Plan/prepare_sources.py'), '--group', group],
                            capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr
        source_checks.append({'group': group, 'stdout': r.stdout})
    deps = {}
    for name in ['changzihe', 'changzihe_eval']:
        py = '/data/chenjiayu/miniconda3/envs/' + name + '/bin/python'
        result = subprocess.run([py, '-B', '-m', 'pip', 'check'], capture_output=True, text=True, timeout=30)
        known = result.returncode == 1 and result.stdout.strip() == 'decord 0.6.0 is not supported on this platform' and not result.stderr.strip()
        assert result.returncode == 0 or known, result.stdout + result.stderr
        deps[name] = {'pip_check_exit': result.returncode, 'known_decord_tag_warning': known, 'output': result.stdout}
    env = dict(os.environ, PYTHONPATH=':'.join(str(PROJECT/'src'/p) for p in ['CLIP','VBench','DINO']))
    evalpy = '/data/chenjiayu/miniconda3/envs/changzihe_eval/bin/python'
    code = '''import json, torch, clip, pyiqa, decord, importlib
from pathlib import Path
root=Path('/data/chenjiayu/changzihe/DistillPP_Validation')
from vbench.third_party.ViCLIP.simple_tokenizer import SimpleTokenizer
tok=SimpleTokenizer(str(root/'resources/evaluation/ViCLIP/bpe_simple_vocab_16e6.txt.gz'))
panel=json.loads((root/'Project_Plan/PROMPTS_QUALIFICATION.json').read_text())['prompts']
text=%r
lengths={p['id']:len(tok.encode(p['prompt']))+2 for p in panel}
lengths['S1_SMOKE']=len(tok.encode(text))+2
assert all(n<=32 for n in lengths.values()), lengths
for dimension in %r: importlib.import_module('vbench.'+dimension)
assert not torch.cuda.is_initialized()
print(json.dumps({'metric_imports':True,'viclip_tokens_including_special':lengths,'cuda_initialized':False,'clip_source':clip.__file__,'pyiqa':pyiqa.__version__}))
''' % (PROMPT, METRICS)
    result = subprocess.run([evalpy, '-B', '-c', code], env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    report = {'status': 'passed', 'cleanup_log': str(clean),
              'generation_evidence': 'previous full hashes + unchanged sizes/mtimes',
              'generation_files': current, 'evaluation_rehashed_now': evaluation,
              'source_checks': source_checks, 'dependency_checks': deps,
              'eval_cpu_check': json.loads(result.stdout.splitlines()[-1]), 'gpu_snapshot': gpu_snapshot(),
              'formal_qualification_started': False}
    json_write(stage / 'S1_PREFLIGHT.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
