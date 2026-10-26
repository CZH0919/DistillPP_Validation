"""Freeze balanced scheduling before generation. Standard library only."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
panel = json.loads((ROOT/'Project_Plan/PROMPTS_QUALIFICATION.json').read_text(encoding='utf-8'))
assert len(panel['prompts']) == 16 and panel['seed_slots'] == [42, 1234, 2026]
rng = random.Random(2026101602)
groups = []
for stratum in dict.fromkeys(p['stratum'] for p in panel['prompts']):
    group = [{'pair_id': p['id']+'_seed'+str(seed), 'prompt_id':p['id'],
              'stratum':stratum, 'prompt':p['prompt'], 'seed':seed}
             for p in panel['prompts'] if p['stratum']==stratum for seed in panel['seed_slots']]
    rng.shuffle(group)
    groups.append(group)
pairs = [groups[h][i] for i in range(12) for h in range(4)]
for i,pair in enumerate(pairs):
    pair['worker_id'] = i % 7
    pair['global_sequence'] = i
for attempt in range(10000):
    for stratum in dict.fromkeys(p['stratum'] for p in pairs):
        orders = [['T50','S4']]*6 + [['S4','T50']]*6
        rng.shuffle(orders)
        for pair,order in zip([p for p in pairs if p['stratum']==stratum],orders):
            pair['order'] = order
    if all(abs(sum(p['order'][0]=='T50' for p in pairs if p['worker_id']==w) -
                   sum(p['order'][0]=='S4' for p in pairs if p['worker_id']==w)) <= 1 for w in range(7)):
        break
else:
    raise AssertionError('No balanced schedule')
assert len({p['pair_id'] for p in pairs}) == 48
assert Counter(p['order'][0] for p in pairs) == {'T50':24, 'S4':24}
for h in groups:
    assert len(h) == 12
frozen_inputs = ['Project_Plan/BASELINE_SETTINGS.json','Project_Plan/PROMPTS_QUALIFICATION.json',
    'Project_Plan/RESOURCE_MANIFEST.json','Project_Plan/WAN_KEY_RENAME_RULES.json',
    'tools/run_s1_arm.py','tools/s1_common.py','tools/run_s2_worker.py','tools/s2_common.py']
execution = {
    'batch_id':'s2_20261016', 'stage':'S2_ONLY', 'authorized_formal_videos':96,
    'worker_count':7, 'cpu_threads_per_worker':4,
    'initial_physical_gpu_candidates':[0,1,2,3,4,5,6],
    'gpu_policy':'Each worker rechecks vacancy twice before load; 10-second PID watchdog; no global reservation; stop own worker on collision.',
    'schedule_rng':'Python random.Random(2026101602); randomized strata interleaving; round-robin worker; balanced arm order in each stratum and worker',
    'order_balance_search_iterations':attempt+1,
    'warmup':'One full T50 and one full S4 trajectory + VAE on independent S1 prompt/seed per worker; no formal metrics or video export.',
    'timing':'Both DiTs/T5/VAE resident; paired same GPU; CUDA-synchronized denoise+VAE. Setup/transfer, text, load/export separate. Occupancy-thread overhead included.',
    'timing_text':'Positive embedding computed once per pair and reused by both arms; fixed negative embedding cached once per worker. Primary excludes text.',
    'memory':'Peak includes both co-resident models, not a standalone deployment minimum.',
    'cold_start':'Per-worker load and first warm-up durations recorded separately, not included in formal hot timings.',
    'speed_statistics':'median(T50 denoise+VAE)/median(S4 denoise+VAE), paired prompt-cluster stratified bootstrap 95% CI; descriptive S2 baseline only',
    'qualification':'Existing five-metric 99% CI rule, priority fixed in baseline; dynamic 95% descriptive without threshold; no S3.',
    'disk_budget_gib':5,
    'input_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in frozen_inputs},
    'pairs':pairs}
path=ROOT/'Project_Plan/S2_EXECUTION.json'
text=json.dumps(execution,ensure_ascii=False,indent=2)+'\n'
if path.exists():
    assert path.read_text(encoding='utf-8') == text, 'Do not overwrite another freeze'
else:
    path.write_text(text,encoding='utf-8',newline='\n')
print('FROZEN',len(pairs),'pairs',Counter(p['worker_id'] for p in pairs),
      Counter((p['stratum'],p['order'][0]) for p in pairs))
