"""Build an explicit manifest of current S2 code, documents and core results."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'Result/S2_20261016'
archive_source=ROOT.parent/'tmp/s2_sync/core_evidence.tar.gz'
archive=OUT/'core_evidence.tar.gz'
blob=archive_source.read_bytes()
assert hashlib.sha256(blob).hexdigest()=='344407b1cb7e37ac463f16fdde6b691073fc798f550088f1d4bf19b1eae2a0e6'
if archive.exists():assert archive.read_bytes()==blob
else:
    with archive.open('xb') as f:f.write(blob)
relative=[
    'Project_Plan/EXPERIMENT_PLAN.md','Project_Plan/S2_EXECUTION.json','Project_Plan/S2_EXECUTION_NOTE.md',
    'Project_Plan/BASELINE_SETTINGS.json','Project_Plan/PROMPTS_QUALIFICATION.json',
    'Project_Plan/RESOURCE_MANIFEST.json','Project_Plan/WAN_KEY_RENAME_RULES.json',
    'Project_Progress/Total_Progress.md','Project_Progress/Stage_Progress/S2_QUALIFICATION.md',
]
names=['s1_common.py','run_s1_arm.py','s2_common.py','run_s2_worker.py','prepare_s2.py',
    'evaluate_s2.py','s2_statistics.py','finalize_s2.py','run_s2_evaluation_controller.py',
    'retry_s2_evaluation.py','resume_s2_metric.py','resume_s2_evaluation.py',
    'read_s2_progress.py','s2_status.py','review_s2_results.py','plot_s2.py','pack_s2_core.py',
    'unpack_s2_core.py','build_s2_delivery_manifest.py']
relative += ['tools/'+name for name in names]
relative += [p.relative_to(ROOT).as_posix() for p in OUT.iterdir() if p.is_file() and p.name!='DELIVERY_MANIFEST.json']
assert len(relative)==len(set(relative))
records=[]
for rel in sorted(relative):
    path=ROOT/rel;assert path.resolve()==path and path.is_file() and path.stat().st_size<8*2**20
    assert path.suffix not in {'.pt','.pth','.safetensors','.mp4'}
    records.append({'path':rel,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
report=json.loads((OUT/'statistics.json').read_text(encoding='utf-8'))
assert report['integrity']['valid_metric_values']==576 and report['passing_targets_in_priority_order']==[]
manifest={'stage':'S2_COMPLETE_NO_TEACHER_QUALIFICATION_NO_S3',
    'scope':'Current small code/documents/core results. Raw evidence snapshot has its own immutable manifest.',
    'videos_remote_only':96,'metrics':576,'teacher_quality_dimensions_passed':0,'s3_started':False,
    'plot_environment':{'python':'D:/VS_code_projects/venv_daily/Scripts/python.exe','numpy':'2.4.3','matplotlib':'3.10.8'},
    'raw_snapshot_sha256':hashlib.sha256(blob).hexdigest(),'files':records,
    'excluded_from_local_sync':['videos','latent tensors','model/evaluation weights','conda environments','package caches']}
text=json.dumps(manifest,ensure_ascii=False,indent=2)+'\n'
path=OUT/'DELIVERY_MANIFEST.json'
if path.exists():assert path.read_text(encoding='utf-8')==text
else:path.write_text(text,encoding='utf-8',newline='\n')
print(json.dumps({'files':len(records),'bytes':sum(r['bytes'] for r in records),
    'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
