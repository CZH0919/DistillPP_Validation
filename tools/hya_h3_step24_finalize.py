"""Full-library mean comparisons for four positions by three strengths."""
import csv
import itertools
import math
import time

import numpy as np
import hya_h2_common as h2
import hya_h3_timing_common as completed
import hya_h3_step24_common as c
from hya_h3_finalize import csv_write, baseline_inference_seconds, summarize_scores
from hya_h3_timing_finalize import baseline_group, subgroup_summaries


def position_contrasts(p, refs, scores, variants):
    lookup = {(cfg["intervention_step"],cfg["strength"]):v for v,cfg in variants.items()}
    rows = []
    for metric in p["metrics"]:
        ids = [pid for pid in p["prompt_ids"] if metric in refs[pid]["metrics"]]
        for first,second in itertools.combinations(range(4),2):
            for strength in (0.1,0.3,0.5):
                arms = [lookup[(step,strength)] for step in (first,second)]
                a,b = [np.array([scores[(pid,arm,metric)] for pid in ids]) for arm in arms]
                delta = b-a
                rows.append({"metric":metric,"first_step":first+1,"second_step":second+1,
                    "strength":strength,"n":len(ids),"first_mean":float(a.mean()),"second_mean":float(b.mean()),
                    "second_minus_first":float(delta.mean()),"positive_pairs":int(sum(delta>0)),
                    "equal_pairs":int(sum(delta==0)),"negative_pairs":int(sum(delta<0))})
    return rows


def load_records(p):
    return {(pid,v):c.read(c.reused_record_path(pid,cfg) if "source_batch" in cfg else c.record_path(pid,v))
            for pid in p["prompt_ids"] for v,cfg in c.all_variants(p).items()}


def load_scores(p, refs, records):
    scores, rows = {}, []
    def add(pid,arm,metric,value,source):
        key = (pid,arm,metric)
        assert key not in scores and math.isfinite(value) and 0<=value<=1, key
        if metric=="dynamic_degree":
            assert value in (0,1)
        scores[key] = value
        rows.append({"prompt_id":pid,"group":arm,"metric":metric,"score":value,"source":source})
    with (completed.RESULT/"scores.csv").open(encoding="utf-8",newline="") as f:
        for r in csv.DictReader(f):
            add(r["prompt_id"],r["group"],r["metric"],float(r["score"]),"completed H2/step1/step3 reused")
    assert len(scores)==640
    nnew=0
    for metric in p["metrics"]:
        raw = c.read(c.RUN/"scores"/(metric+".json"))
        assert raw["status"]=="passed"
        for r in raw["rows"]:
            key = (r["prompt_id"],r["variant"])
            assert r["variant"] in p["variants"] and r["metric"]==metric
            assert r["video_sha256"]==records[key]["video"]["sha256"]
            add(*key,metric,float(r["score"]),"new step2/step4 correction")
            nnew+=1
    arms = ["T50","S5"]+list(c.all_variants(p))
    expected = {(pid,arm,m) for pid in p["prompt_ids"] for arm in arms
                for m in refs[pid]["metrics"] if m in p["metrics"]}
    assert set(scores)==expected and nnew==p["counts"]["new_scores"]==480
    assert len(rows)==p["counts"]["combined_scores"]==1120
    return scores,rows


def timing_records(p, manifest, records):
    rows,correction = [],[]
    old_shas = {batch:c.read(module.RUN/"manifest.json")["protocol_sha256"] for batch,module in
                (("first_step",c.first_step),("timing",completed))}
    sigmas=c.read(c.ROOT/"Project_Plan/HYA_RUNTIME_CONTRACT.json")["schedule"]["5"]["sigmas"]
    for pid in p["prompt_ids"]:
        baseline=c.read(h2.RUN/"pairs"/(pid+".json"))["records"]
        for arm in ("T50","S5"):
            r=baseline[arm]
            rows.append({"prompt_id":pid,"group":arm,"gpu":r["admission"]["index"],
                "compute_seconds":r["hot_denoise_decode_seconds"],
                "inference_with_transfers_seconds":baseline_inference_seconds(r),
                "video_wall_seconds":r["video_wall_seconds"],"source":"H2 recorded"})
        noises=set()
        for v,cfg in c.all_variants(p).items():
            r=records[(pid,v)]
            source=cfg.get("source_batch")
            assert r["status"]=="passed" and r["seed"]==p["seed"]
            assert r["strength"]==cfg["strength"] and r["intervention_step"]==cfg["intervention_step"]
            assert r["student_calls"]==5 and r["teacher_queries"]==1
            assert r["protocol_sha256"]==(old_shas[source] if source else manifest["protocol_sha256"])
            assert c.digest(r["archive_path"])==r["video"]["sha256"]
            assert r["input_hashes"]==baseline["S5"]["inputs"]["hashes"]
            assert r["initial_latent_sha256"]==baseline["S5"]["actual_initial_latent_sha256"]
            trace=r["intervention"]
            assert trace["step"]==cfg["intervention_step"]
            assert trace["teacher_sigma"]==sigmas[cfg["intervention_step"]+1]
            assert trace["renoise_rng_seed"]==c.common_noise_seed(p,pid)
            noises.add(trace["renoise_noise_sha256"])
            rows.append({"prompt_id":pid,"group":v,"gpu":r["admission"]["index"],
                "compute_seconds":r["hot_denoise_decode_seconds"],
                "inference_with_transfers_seconds":r["inference_with_transfers_seconds"],
                "video_wall_seconds":r["video_wall_seconds"],"source":"prior correction" if source else "new correction"})
            correction.append({"prompt_id":pid,"group":v,"intervention_step":cfg["intervention_step"],
                "strength":cfg["strength"],"student_sigma":trace["student_sigma"],"teacher_sigma":trace["teacher_sigma"],
                "endpoint_difference_rms":trace["endpoint_difference_rms"],"applied_delta_rms":trace["applied_delta_rms"],
                "applied_delta_relative_rms":trace["applied_delta_relative_rms"]})
        assert len(noises)==1, "Re-noise sample differs: "+pid
    return rows,correction


def render_report(p, summaries, contrasts, medians, ratios):
    variants=c.all_variants(p)
    by={(r["metric"],r["variant"]):r for r in summaries}
    metrics=p["metrics"]
    arms=["T50","S5"]+list(variants)
    def score(metric,arm):
        r=by[(metric,next(iter(variants)) if arm in ("T50","S5") else arm)]
        return r[arm] if arm in ("T50","S5") else r["corrected"]
    lines=["# 第1–4步 × 三种λ：初步迁移验证汇总", "",
        "新增第二步、第四步各 λ=0.1/0.3/0.5，共288条视频、480个新分数。复用既有六个纠偏设置及H2教师/学生，合并672条视频、1120个分数。",
        "主要依据是每个指标完整任务库的均值，以及相对学生/教师的变化；在固定λ下比较介入位置。沿用原48条prompt和seed，未按教师优势筛换任务。", "",
        "| 组别 | 主体一致性 | 多物体符合度 | 动态比例 |", "|---|---:|---:|---:|"]
    for arm in arms:
        label=arm if arm in ("T50","S5") else f"第{variants[arm]['intervention_step']+1}步 λ={variants[arm]['strength']}"
        lines.append(f"| {label} | {100*score('subject_consistency',arm):.4f} | {100*score('multiple_objects',arm):.4f} | {100*score('dynamic_degree',arm):.3f}% |")
    lines += ["", "质量分数为原分×100；主体/动态n=32、多物体n=16。dynamic_degree只作运动护栏，不按越高越好判定，也不设置自动容许界。", "",
        "| 指标 | 介入步 | λ | 相对学生变化（分/百分点） | 相对教师变化（分/百分点） |",
        "|---|---:|---:|---:|---:|"]
    for r in summaries:
        cfg=variants[r["variant"]]
        lines.append(f"| {r['metric']} | {cfg['intervention_step']+1} | {cfg['strength']} | {100*r['corrected_minus_student']:+.4f} | {100*(r['corrected']-r['T50']):+.4f} |")
    lines += ["", "| 组别 | 去噪/纠偏/VAE中位秒 | 含传输中位秒 | 教师/该组耗时比 | 相对学生耗时变化 |",
        "|---|---:|---:|---:|---:|"]
    base=medians["S5"]["inference_with_transfers_seconds"]
    for arm,r in medians.items():
        ratio=f"{ratios[arm]:.3f}×" if arm in ratios else "—"
        overhead=100*(r["inference_with_transfers_seconds"]/base-1)
        lines.append(f"| {arm} | {r['compute_seconds']:.3f} | {r['inference_with_transfers_seconds']:.3f} | {ratio} | {overhead:+.2f}% |")
    lines += ["", "各纠偏设置均为5次学生预测+1次教师预测。耗时包含模型传输，加载/预热/导出另计；旧基线采集时期不同，多卡并发不计入算法加速。", "",
        "固定λ的全部位置配对差值保留于 position_contrasts.csv；同一λ不是等latent扰动，实际幅度保留于 correction_magnitudes.csv。",
        "逐prompt差值与升降数量保留；原教师优势分层仅作为补充CSV/JSON，不作为主要判定或prompt筛选依据。参考区间不增加跨零准入门槛。", "",
        "复用记录保留已披露的 HYA_MOTION_26 / step3_lambda_0p1 首预测逐位差异，详见 HYA_H3_TIMING_V1/REVIEW_AND_INTERPRETATION.md；未重新生成或替换该条。",
        "第5步之后的下一sigma为0，不满足当前重加噪查询规则，因此本次补齐的是第1–4步的适用位置。完成本批后停止，不自动扩展任务或下一阶段。", ""]
    return "\n".join(lines)


def main():
    manifest,p=c.check_frozen()
    assert not (c.RESULT/"COMPLETE.json").exists()
    refs={r["id"]:r for r in c.read(h2.RUN/"protocol.json")["prompts"]}
    variants=c.all_variants(p)
    records=load_records(p)
    scores,raw=load_scores(p,refs,records)
    summaries,paired=summarize_scores(dict(p,variants=variants),refs,scores)
    for r in summaries:
        r["corrected_minus_teacher"]=r["corrected"]-r["T50"]
    for r in paired:
        r["baseline_group"]=baseline_group(r["T50"],r["S5"]) if r["metric"]!="dynamic_degree" else "not_quality_ranked"
    groups,strata,motion=subgroup_summaries(p,refs,scores,variants)
    contrasts=position_contrasts(p,refs,scores,variants)
    timing,correction=timing_records(p,manifest,records)
    arms=["T50","S5"]+list(variants)
    medians={arm:{key:float(np.median([r[key] for r in timing if r["group"]==arm])) for key in
        ("compute_seconds","inference_with_transfers_seconds","video_wall_seconds")} for arm in arms}
    ratios={v:medians["T50"]["inference_with_transfers_seconds"]/medians[v]["inference_with_transfers_seconds"] for v in variants}
    c.RESULT.mkdir(exist_ok=True)
    for name,rows in (("scores.csv",raw),("paired_scores.csv",paired),("timing.csv",timing),
        ("subgroup_scores.csv",strata),("position_contrasts.csv",contrasts),("correction_magnitudes.csv",correction)):
        csv_write(c.RESULT/name,rows)
    summary={"status":"complete","protocol_sha256":manifest["protocol_sha256"],**p["counts"],"metrics":summaries,
        "position_comparisons":contrasts,"timing_medians":medians,"teacher_over_corrected_inference_median_ratio":ratios,
        "primary_analysis":"Full-library means and fixed-lambda intervention-position comparisons",
        "supplementary_quality_subgroup_ids":groups,"supplementary_quality_subgroups":strata,
        "supplementary_motion_in_subject_teacher_advantage_subset":motion,
        "inherited_record_note":p["inherited_record_note"],
        "scope":"Same fixed prompts and seed; exploratory initial migration validation","finished_unix":time.time()}
    c.write_json(c.RESULT/"summary.json",summary,replace=True)
    (c.RESULT/"REPORT.md").write_text(render_report(p,summaries,contrasts,medians,ratios),encoding="utf-8")
    c.write_json(c.RESULT/"COMPLETE.json",{"status":"complete","finished_unix":time.time(),
        "sha256":{f.name:c.digest(f) for f in c.RESULT.iterdir() if f.is_file()}})
    path=c.ROOT/"Project_Progress/Total_Progress.md"
    text=path.read_text(encoding="utf-8")
    start,end="<!-- HYA_H3_STEP24_AUTO_START -->","<!-- HYA_H3_STEP24_AUTO_END -->"
    if start in text and end in text:
        left,rest=text.split(start,1)
        _,right=rest.split(end,1)
        body="\n**第二/第四步六组已完成：288/288新视频、480/480新分数。**第1–4步×三个λ已合并；结果见 `Result/HYA_H3_STEP24_V1/REPORT.md`，等待阶段性分析与研究者审阅。\n"
        temporary=path.with_name("."+path.name+".step24.tmp")
        temporary.write_text(left+start+body+end+right,encoding="utf-8")
        temporary.replace(path)


if __name__=="__main__":
    main()
