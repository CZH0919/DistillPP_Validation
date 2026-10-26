"""CPU-only schedule freeze for the authorized S2-R, preserving its panel."""
import collections
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
q = json.loads((ROOT/'Project_Plan/QUALIFICATION_V2.json').read_text(encoding='utf-8'))
selected = {r['id']: r for r in q['prompts'] if r['screen_selected']}
assert len(selected) == 32
rng = random.Random(2026101702)
workers = {w: [] for w in range(3)}
for domain, ids in q['screen']['active_prompt_ids_by_domain'].items():
    assert len(ids) == 16
    pairs = []
    for pid in ids:
        p = selected[pid]
        for seed in q['screen']['seed_slots']:
            pairs.append({'pair_id':pid+'_seed'+str(seed), 'prompt_id':pid,
                'domain':domain, 'stratum':domain, 'cluster_id':p['cluster_id'],
                'prompt':p['generation_prompt'], 'eval_prompt':p['eval_prompt'],
                'primary_metrics':p['primary_metrics'], 'token_counts':p['token_counts'], 'seed':seed})
    rng.shuffle(pairs)
    for w in workers:
        subset = pairs[w::3]
        assert len(subset) == 16
        orders = [['T50','S4'] for _ in range(8)] + [['S4','T50'] for _ in range(8)]
        rng.shuffle(orders)
        for p, order in zip(subset, orders):
            p.update(worker_id=w, order=order)
        workers[w].extend(subset)
all_pairs = []
for w, pairs in workers.items():
    # Alternate domains while keeping within-domain randomized order.
    groups = [[p for p in pairs if p['domain']==d] for d in q['metric_domains']]
    ordered = [groups[j][i] for i in range(16) for j in range(2)]
    for i,p in enumerate(ordered): p['worker_sequence'] = i
    all_pairs.extend(ordered)
assert len(all_pairs)==96 and len({p['pair_id'] for p in all_pairs})==96
assert collections.Counter(p['order'][0] for p in all_pairs)=={'T50':48,'S4':48}
warm = dict(all_pairs[0], pair_id='warmup')
frozen = ['Project_Plan/QUALIFICATION_V2.json','Project_Plan/RESOURCE_MANIFEST.json',
    'Project_Plan/WAN_KEY_RENAME_RULES.json','tools/run_s1_arm.py','tools/s1_common.py',
    'tools/s2_common.py','tools/evaluate_s2.py','tools/s2r_common.py',
    'tools/run_s2r_worker.py','tools/evaluate_s2r.py','tools/prepare_s2r.py']
execution = {'batch_id':'s2r_20261017','stage':'S2_R_ONLY','authorized_formal_videos':192,
    'authorized_metric_values':576,'worker_count':3,'cpu_threads_per_worker':4,
    'initial_physical_gpu_candidates':[4,5,6],
    'gpu_policy':'Vacancy checked twice before load; 10-second PID monitoring; terminate only own work on collision; never use a card with a resident foreign process.',
    'schedule_rng':'Python random.Random(2026101702); shuffle within each domain; distribute to 3 workers; balance arm order within worker/domain; alternate domains.',
    'warmup_pair':warm,'warmup':'Full T50 and S4 warmup on a frozen-panel input; no export/scoring; excluded from formal samples.',
    'timing':'Same-GPU paired CUDA-synchronized denoise plus VAE; both DiTs/T5/VAE resident. Text, setup/transfers, load, export, warmup recorded separately.',
    'qualification':q['statistics'],'metric_domains':q['metric_domains'],
    'active_panel_sha256':q['screen']['active_panel_sha256'],
    'disk_budget_gib':10,'initial_ram_budget_gib_per_worker':100,
    'input_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in frozen},'pairs':all_pairs}
path=ROOT/'Project_Plan/S2R_EXECUTION.json'
content=json.dumps(execution,ensure_ascii=False,indent=2)+'\n'
if path.exists(): assert path.read_text(encoding='utf-8')==content, 'Do not replace a frozen schedule'
else: path.write_text(content,encoding='utf-8',newline='\n')
print(json.dumps({'status':'schedule_frozen','pairs':96,'videos':192,'metrics':576,
    'workers':dict(collections.Counter(p['worker_id'] for p in all_pairs)),
    'worker_domain_arm_order':{str(k):v for k,v in collections.Counter((p['worker_id'],p['domain'],p['order'][0]) for p in all_pairs).items()}}))
