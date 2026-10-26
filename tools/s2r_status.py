"""Read-only S2-R terminal status. Ctrl-C stops this viewer only."""
import argparse
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import time

B=Path('/data/chenjiayu/changzihe/DistillPP_Validation/runs/s2r_20261017')
read=lambda p:json.loads(p.read_text())

def show():
    now=time.time();status=read(B/'supervisor_status.json')
    completed=len(list((B/'pairs').glob('*/attempt*/pair_complete.json')))
    failures=list((B/'workers').glob('*/attempt*/failure.json'))+list((B/'evaluation').glob('*/attempt*/failure.json'))
    scored=0;metric_counts={}
    for metric in ('subject_consistency','motion_smoothness','dynamic_degree','imaging_quality','aesthetic_quality','overall_consistency'):
        values=set()
        for p in (B/'evaluation'/metric).glob('attempt*/VB*.json'):
            for r in read(p)['rows']:values.add((r['pair_id'],r['arm']))
        metric_counts[metric]=len(values);scored+=len(values)
    print('\n'+datetime.fromtimestamp(now,timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S 北京时间'))
    print('状态:',status.get('status'),'| 正式完整视频: %d/192 (%d/96 对) | 评分: %d/576'%(completed*2,completed,scored))
    print('调度心跳距今: %.0f 秒 | 历史失败/保护退出: %d'%(now-status.get('unix_time',now),len(failures)))
    if (B/'runtime_policy.json').exists():
        q=read(B/'runtime_policy.json');print('允许卡号:',q['allowed_gpu_indexes'],'| 最大并发:',q['max_concurrent'])
    for j in status.get('active',[]):
        kind,identifier=j['job'].split(':',1)
        directory=B/'workers'/('worker%02d'%int(identifier)) if kind=='generation' else B/'evaluation'/identifier
        p=directory/('attempt%02d'%j['attempt'])/'progress.json'
        r=read(p) if p.exists() else {}
        print('GPU %s | PID %s | %s | attempt %s | %s | %s %s | step %s | heartbeat %.0fs'%(
            j['gpu'],j['pid'],j['job'],j['attempt'],r.get('phase','starting'),
            r.get('pair_id',r.get('prompt_id','')),r.get('arm',''),r.get('step','-'),now-r.get('unix_time',now)))
    if not status.get('active'):print('无活动 worker；请结合状态查看是否等待空闲卡、冷却或需要处理。')
    print('各指标已提交:',', '.join('%s=%s/96'%x for x in metric_counts.items()))
    for p in sorted(failures,key=lambda x:x.stat().st_mtime)[-3:]:
        print('历史异常:',str(p.relative_to(B)),read(p).get('error',''))
    for issue in status.get('paused_jobs',[]):print('待处理:',issue)
    p=B/'finalizer_status.json'
    if p.exists():print('统计/归档:',read(p).get('status'))
    print('以上只读查看，不启动计算；Ctrl-C 仅退出查看器。',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch',action='store_true');parser.add_argument('--interval',type=int,default=30)
    args=parser.parse_args();assert args.interval>=10
    try:
        while True:
            show()
            if not args.watch:break
            time.sleep(args.interval)
    except KeyboardInterrupt:print('\n已退出只读查看器；远端实验继续。')
