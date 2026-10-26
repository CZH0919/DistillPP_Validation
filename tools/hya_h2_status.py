"""Read-only H2 progress, no CUDA initialization or downloads."""
import argparse
import json
import time

from hya_h2_common import ROOT, RUN, read
from hya_h2_controller import process_alive, worker_out


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--json', action='store_true'); args = parser.parse_args()
    if not (RUN/'controller.json').exists():
        print('H2 controller has not written status yet.'); return
    s = read(RUN/'controller.json'); manifest = read(RUN/'manifest.json')
    live = process_alive(s['controller_pid'], s['controller_start_ticks'])
    videos = list((RUN/'generation').glob('*/attempt*/*/*.json'))
    verified = {(read(p)['prompt_id'], read(p)['arm']) for p in videos}
    snapshot = {'status': s['status'], 'controller_pid': s['controller_pid'], 'controller_alive': live,
                'heartbeat_age_seconds': round(time.time()-s['updated_unix'], 1),
                'verified_video_slots': len(verified), 'committed_videos': s['committed_videos'], 'total_videos': 128,
                'completed_pairs': s['completed_pairs'], 'total_pairs': 64, 'completed_scores': s['completed_scores'], 'total_scores': 224,
                'hours_since_launch': round((time.time()-s['started_unix'])/3600, 2), 'active': [], 'paused': []}
    from hya_h2_controller import task_definitions
    definitions = {t['id']: t for t in task_definitions(manifest, read(RUN/'protocol.json'))}
    for tid, state in s['tasks'].items():
        if state['status'] == 'running':
            a = state['attempts'][-1]; p = worker_out(definitions[tid], a['attempt'])/'progress.json'
            snapshot['active'].append({'task': tid, 'gpu': a['gpu'], 'pid': a['pid'], 'alive': process_alive(a['pid'], a['start_ticks']),
                                       'log_path': a['log_path'], 'progress': read(p) if p.exists() else {'stage': 'starting'}})
        elif state['status'] == 'paused': snapshot['paused'].append({'task': tid, 'error': state.get('error')})
    if args.json:
        print(json.dumps(snapshot, ensure_ascii=False, indent=2)); return
    print(f"H2: {snapshot['status']} | controller PID {snapshot['controller_pid']} alive={live} | heartbeat {snapshot['heartbeat_age_seconds']}s ago")
    print(f"视频已验收槽位 {len(verified)}/128（含尚未成对者）；成对归档 {s['committed_videos']}/128；配对 {s['completed_pairs']}/64；评分 {s['completed_scores']}/224")
    print('运行时长:', snapshot['hours_since_launch'], '小时；S3 不会自动启动。')
    for a in snapshot['active']:
        p = a['progress']; step = f" {p.get('completed','')}/{p.get('total','')}" if 'total' in p else ''
        print(f"GPU {a['gpu']} PID {a['pid']} alive={a['alive']} {a['task']} {p.get('stage')} {p.get('prompt_id','')} {p.get('arm','')}{step}")
        print('  log:', a['log_path'])
    for p in snapshot['paused']: print('需要处理:', p['task'], p['error'])
    if not live and s['status'] not in ['completed']:
        print('控制器已退出；不要重复启动或终止未知进程，请先检查 controller.log 和记录的 worker 状态。')


if __name__ == '__main__':
    main()
