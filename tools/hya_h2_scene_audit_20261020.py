"""CPU-only scene audit: official prompts, labels, saved captions and aggregation.

No torch/model import, GPU scoring, new videos, or changes to the original scores.
"""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/hya_h2_20261019'
OUT = ROOT / 'Result/HYA_H2_20261019/SCENE_AUDIT_20261020'
REF = OUT / 'reference'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pure_function(file, name, globals_dict):
    tree = ast.parse(file.read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(globals_dict)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(file), 'exec'), namespace)
    return namespace[name]


protocol = read(RUN / 'protocol.json')
manifest = read(RUN / 'manifest.json')
remote = read(REF / 'remote_read_receipt.json')
remote_hashes = {e['path']: e['sha256'] for e in remote['files']}
fetch = read(REF / 'fetch_receipt.json')
assert all(e['status'] == 'fetched' and sha(REF / e['name']) == e['sha256'] for e in fetch)
source_map = {
    'scene.py': 'src/VBench/vbench/scene.py',
    'utils.py': 'src/VBench/vbench/utils.py',
    'fullinfo.json': 'src/VBench/vbench/VBench_full_info.json',
    'all_dimension.txt': 'Project_Plan/reference_inputs/hunyuan_accvideo_20261019/vbench_all_prompts.txt',
    'hunyuan_all_dimension.txt': 'Project_Plan/reference_inputs/hunyuan_accvideo_20261019/vbench_hunyuan_prompts.txt',
}
for name, rel in source_map.items():
    assert sha(REF / name) == remote_hashes[rel], rel
    if rel in manifest['frozen_files']:
        assert sha(REF / name) == manifest['frozen_files'][rel]

full = read(REF / 'fullinfo.json')
original = (REF / 'all_dimension.txt').read_text(encoding='utf-8').splitlines()
augmented = (REF / 'hunyuan_all_dimension.txt').read_text(encoding='utf-8').splitlines()
scene_original = (REF / 'scene_prompts.txt').read_text(encoding='utf-8').splitlines()
assert len(full) == len(original) == len(augmented) == 946
assert all(f['prompt_en'] == p for f, p in zip(full, original))
pool = [i for i, f in enumerate(full) if 'scene' in f['dimension']]
assert len(pool) == len(scene_original) == 86
assert [original[i] for i in pool] == scene_original
salt = protocol['selection']['salt']
rank = sorted(pool, key=lambda i: hashlib.sha256(f'{salt}|scene|{i+1}|{original[i]}'.encode()).hexdigest())
prompts = [p for p in protocol['prompts'] if p['library'] == 'scene']
assert len(prompts) == 16 and [p['source_line_1based'] for p in prompts] == [i+1 for i in rank[:16]]

official_count = pure_function(REF / 'scene.py', 'check_generate', {})
official_indices = [int(i) for i in pure_function(REF / 'utils.py', 'get_frame_indices', {'np': np})(16, 129, sample='middle')]
our_info = pure_function(ROOT / 'tools/hya_h2_evaluate.py', 'evaluation_info', {})
score_rows, captions = {}, {}
score_files = sorted((RUN / 'scores').glob('*_scene.json'))
assert len(score_files) == 8
for file in score_files:
    assert sha(file) == remote_hashes[str(file.relative_to(ROOT)).replace('\\', '/')]
    rec = read(file)
    assert rec['status'] == 'passed' and not rec['network_attempts']
    recorded_indices = [a['sampled_frame_indices'] for a in rec['audit'] if 'sampled_frame_indices' in a]
    assert recorded_indices == [official_indices]
    loading = next(a for a in rec['audit'] if 'tag2text_unexpected' in a)
    assert not loading['tag2text_unexpected']
    assert all(k.endswith(('relative_position_index', 'attn_mask', 'position_ids')) for k in loading['tag2text_missing_nonlearned_buffers'])
    for row in rec['rows']:
        key = (row['prompt_id'], row['arm'])
        assert key not in score_rows
        score_rows[key] = row
    for item in rec['audit']:
        if 'scene_frame_captions' in item:
            assert item['video_path'] not in captions
            captions[item['video_path']] = item['scene_frame_captions']
assert len(score_rows) == len(captions) == 32

table, frame_rows = [], []
token_counts = []
for p in prompts:
    i = p['source_line_1based'] - 1
    assert p['evaluation_prompt'] == original[i]
    assert p['generation_prompt'] == augmented[i]
    assert p['auxiliary_info'] == full[i]['auxiliary_info']
    assert p['official_dimensions'] == full[i]['dimension']
    pair_file = RUN / 'pairs' / (p['id'] + '.json')
    assert sha(pair_file) == remote_hashes[str(pair_file.relative_to(ROOT)).replace('\\', '/')]
    pair = read(pair_file)['records']
    label = p['auxiliary_info']['scene']['scene']['scene']
    values = {}
    for arm in ['T50', 'S5']:
        generated = pair[arm]
        assert generated['arm'] == arm and generated['prompt_id'] == p['id']
        assert generated['prompt'] == generated['inputs']['prompt'] == augmented[i]
        assert generated['seed'] == 2026101901
        tokens = generated['inputs']['tokens']
        assert tokens['llama_with_template'] <= 351 and tokens['clip_with_special'] <= 77
        token_counts.append(tokens)
        video = generated['archive_path']
        row = score_rows[(p['id'], arm)]
        assert row['video_path'] == video and row['video_sha256'] == generated['video']['sha256']
        assert row['seed'] == generated['seed']
        official_record = {**full[i], 'video_list': [video]}
        loader = pure_function(REF / 'utils.py', 'load_dimension_info', {'load_json': lambda _: [official_record]})
        _, loaded = loader('unused', dimension='scene', lang='en')
        assert loaded == [our_info(p, 'scene', video)]
        text = captions[video]
        assert len(text) == 16
        hits = [all(word in caption for word in label.split(' ')) for caption in text]
        assert official_count({'scene': label}, text) == sum(hits)
        result = sum(hits) / 16
        assert result == row['score'] == row['official_row']['video_results']
        assert row['official_row']['success_frame_count'] == sum(hits)
        assert row['official_row']['frame_count'] == 16
        values[arm] = result
        for idx, caption, hit in zip(official_indices, text, hits):
            frame_rows.append({'prompt_id': p['id'], 'arm': arm, 'frame_index': idx,
                               'label': label, 'caption': caption, 'official_hit': hit,
                               'missing_label_tokens': [w for w in label.split(' ') if w not in caption]})
    assert pair['T50']['inputs']['hashes'] == pair['S5']['inputs']['hashes']
    table.append({'prompt_id': p['id'], 'official_line_1based': i+1,
                  'original_prompt': original[i], 'generation_prompt': augmented[i], 'scene_label': label,
                  'T50_score': values['T50'], 'S5_score': values['S5'],
                  'T50_hit_frames': round(values['T50']*16), 'S5_hit_frames': round(values['S5']*16),
                  'delta': values['T50']-values['S5'],
                  'contribution_to_mean_difference_100scale': (values['T50']-values['S5'])*100/16})
assert len(frame_rows) == 512
t_hits, s_hits = [sum(r[a+'_hit_frames'] for r in table) for a in ['T50', 'S5']]
statistics = next(x for x in read(ROOT / 'Result/HYA_H2_20261019/statistics.json')['metrics'] if x['metric']=='scene')
assert t_hits/256 == statistics['T50_mean'] and s_hits/256 == statistics['S5_mean']
assert (t_hits-s_hits)/256 == statistics['T50_minus_S5']
result = {
    'status': 'passed', 'scope': 'Prompt/label/implementation and CPU recomputation of saved frame captions; no new GPU evaluation.',
    'official_prompt_pool': 86, 'selected_scene_prompts': 16, 'seeds_per_prompt': 1,
    'official_files_byte_identical_to_used_files': source_map,
    'all_dimension_alignment_rows': 946, 'selection_reproduced': True,
    'generation_prompts_and_labels_correct': 16, 'actual_video_records_and_scoring_labels_verified': 32,
    'official_loader_auxiliary_mapping_identical': True,
    'recorded_frame_indices_match_official': official_indices,
    'caption_frame_decisions_recomputed': len(frame_rows), 'video_scores_recomputed': 32,
    'maximum_score_difference': 0.0,
    'T50': {'hits': t_hits, 'frames': 256, 'mean': t_hits/256},
    'S5': {'hits': s_hits, 'frames': 256, 'mean': s_hits/256},
    'teacher_minus_student': (t_hits-s_hits)/256,
    'prompt_tokens_max': {k: max(x[k] for x in token_counts) for k in token_counts[0]},
    'score_distribution': {a: {'zero': sum(r[a+'_score']==0 for r in table),
                               'one': sum(r[a+'_score']==1 for r in table),
                               'partial': sum(0<r[a+'_score']<1 for r in table)} for a in ['T50','S5']},
    'paper_scene_100scale': {'T50': 53.88, 'S5': 50.75},
    'our_minus_paper_100scale': {'T50': t_hits/256*100-53.88, 'S5': s_hits/256*100-50.75},
    'largest_negative_prompt_contributions': sorted(table,key=lambda r:r['delta'])[:3],
    'no_changes_to_original_scores_or_qualification': True,
}
for name, value in [('VALIDATION.json',result),('PROMPT_AND_SCORE_AUDIT.json',table),('FRAME_CAPTION_AUDIT.json',frame_rows)]:
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['official_files_byte_identical_to_used_files','largest_negative_prompt_contributions']},ensure_ascii=False,indent=2))
