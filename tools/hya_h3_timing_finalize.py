"""Combine four new arms with two existing arms, without rescoring baselines."""
import math
import time

import numpy as np
import hya_h2_common as h2
import hya_h3_common as previous
import hya_h3_timing_common as c
from hya_h3_finalize import csv_write, baseline_inference_seconds, summarize_scores


def baseline_group(teacher, student):
    return "teacher_higher" if teacher > student else "teacher_lower" if teacher < student else "equal"


def delta_statistics(teacher, student, corrected):
    n = len(student)
    if not n:
        return {"n":0, "T50":None, "S5":None, "corrected":None,
                "corrected_minus_student":None, "positive_pairs":0, "equal_pairs":0, "negative_pairs":0}
    teacher, student, corrected = [np.asarray(x, dtype=float) for x in (teacher, student, corrected)]
    delta = corrected - student
    return {"n":n, "T50":float(teacher.mean()), "S5":float(student.mean()),
            "corrected":float(corrected.mean()), "corrected_minus_student":float(delta.mean()),
            "positive_pairs":int((delta > 0).sum()), "equal_pairs":int((delta == 0).sum()),
            "negative_pairs":int((delta < 0).sum())}


def subgroup_summaries(protocol, refs, scores, variants):
    groups, rows, motion = {}, [], []
    for metric in protocol["metrics"]:
        if metric == "dynamic_degree":
            continue
        ids = [pid for pid in protocol["prompt_ids"] if metric in refs[pid]["metrics"]]
        groups[metric] = {}
        for label in ("teacher_higher", "teacher_lower", "equal"):
            selected = [pid for pid in ids if baseline_group(
                scores[(pid,"T50",metric)], scores[(pid,"S5",metric)]) == label]
            groups[metric][label] = selected
            for variant, cfg in variants.items():
                stats = delta_statistics(*[[scores[(pid,arm,metric)] for pid in selected]
                                           for arm in ("T50", "S5", variant)])
                rows.append({"metric":metric, "baseline_group":label, "variant":variant,
                             "intervention_step":cfg["intervention_step"], "strength":cfg["strength"], **stats})
        if metric == "subject_consistency":
            selected = groups[metric]["teacher_higher"]
            for variant in variants:
                stats = delta_statistics(*[[scores[(pid,arm,"dynamic_degree")] for pid in selected]
                                           for arm in ("T50", "S5", variant)])
                motion.append({"subset":"subject_consistency_teacher_higher", "variant":variant, **stats})
    return groups, rows, motion


def timing_contrasts(protocol, refs, scores, variants):
    lookup = {(v["intervention_step"], v["strength"]):name for name,v in variants.items()}
    rows = []
    for metric in protocol["metrics"]:
        ids = [pid for pid in protocol["prompt_ids"] if metric in refs[pid]["metrics"]]
        for strength in (0.1, 0.3, 0.5):
            first, third = lookup[(0,strength)], lookup[(2,strength)]
            a, b = [np.array([scores[(pid,v,metric)] for pid in ids]) for v in (first,third)]
            delta = b-a
            rows.append({"metric":metric, "strength":strength, "n":len(ids),
                "first_step_mean":float(a.mean()), "third_step_mean":float(b.mean()),
                "third_minus_first":float(delta.mean()), "positive_pairs":int((delta>0).sum()),
                "equal_pairs":int((delta==0).sum()), "negative_pairs":int((delta<0).sum())})
    return rows


def load_records(p):
    records = {}
    for pid in p["prompt_ids"]:
        for variant, cfg in c.all_variants(p).items():
            path = previous.record_path(pid,cfg["source_variant"]) if "source_variant" in cfg else c.record_path(pid,variant)
            records[(pid,variant)] = c.read(path)
    return records


def load_scores(p, refs, records):
    scores, rows = {}, []
    def add(pid, arm, metric, value, source):
        key = (pid,arm,metric)
        assert key not in scores, "Duplicate score: " + str(key)
        assert math.isfinite(value) and 0 <= value <= 1
        if metric == "dynamic_degree":
            assert value in (0,1)
        scores[key] = value
        rows.append({"prompt_id":pid,"group":arm,"metric":metric,"score":value,"source":source})
    for file in (h2.RUN/"scores").glob("*.json"):
        for row in c.read(file)["rows"]:
            if row["prompt_id"] in p["prompt_ids"] and row["metric"] in p["metrics"]:
                add(row["prompt_id"],row["arm"],row["metric"],float(row["score"]),"H2 reused")
    reuse_map = {cfg["source_variant"]:name for name,cfg in p["reused_variants"].items()}
    counts = {"new":0, "reused":0}
    for source, run in (("reused",previous.RUN),("new",c.RUN)):
        for metric in p["metrics"]:
            for row in c.read(run/"scores"/(metric+".json"))["rows"]:
                variant = reuse_map[row["variant"]] if source == "reused" else row["variant"]
                pid = row["prompt_id"]
                assert row["metric"] == metric and row["video_sha256"] == records[(pid,variant)]["video"]["sha256"]
                add(pid,variant,metric,float(row["score"]),source+" correction")
                counts[source] += 1
    arms = ["T50","S5"]+list(c.all_variants(p))
    expected = {(pid,arm,m) for pid in p["prompt_ids"] for arm in arms
                for m in refs[pid]["metrics"] if m in p["metrics"]}
    assert set(scores) == expected and counts == {"new":320,"reused":160}
    assert len(rows) == p["counts"]["combined_scores"]
    return scores, rows


def timing_records(p, manifest, records):
    rows, correction = [], []
    old_manifest = c.read(previous.RUN/"manifest.json")
    native_sigmas = c.read(c.ROOT/"Project_Plan/HYA_RUNTIME_CONTRACT.json")["schedule"]["5"]["sigmas"]
    for pid in p["prompt_ids"]:
        baseline = c.read(h2.RUN/"pairs"/(pid+".json"))["records"]
        for arm in ("T50","S5"):
            r = baseline[arm]
            rows.append({"prompt_id":pid,"group":arm,"gpu":r["admission"]["index"],
                "compute_seconds":r["hot_denoise_decode_seconds"],
                "inference_with_transfers_seconds":baseline_inference_seconds(r),
                "video_wall_seconds":r["video_wall_seconds"],"source":"H2 recorded"})
        noise_hashes = set()
        for variant,cfg in c.all_variants(p).items():
            r = records[(pid,variant)]
            old = "source_variant" in cfg
            assert r["status"] == "passed" and r["seed"] == p["seed"]
            assert r["student_calls"] == 5 and r["teacher_queries"] == 1
            assert r["strength"] == cfg["strength"] and r["intervention_step"] == cfg["intervention_step"]
            assert r["protocol_sha256"] == (old_manifest if old else manifest)["protocol_sha256"]
            assert c.digest(r["archive_path"]) == r["video"]["sha256"]
            assert r["input_hashes"] == baseline["S5"]["inputs"]["hashes"]
            assert r["initial_latent_sha256"] == baseline["S5"]["actual_initial_latent_sha256"]
            trace = r["intervention"]
            assert trace["step"] == cfg["intervention_step"]
            assert trace["teacher_sigma"] == native_sigmas[cfg["intervention_step"]+1]
            assert trace["renoise_rng_seed"] == c.common_noise_seed(p,pid)
            noise_hashes.add(trace["renoise_noise_sha256"])
            rows.append({"prompt_id":pid,"group":variant,"gpu":r["admission"]["index"],
                "compute_seconds":r["hot_denoise_decode_seconds"],
                "inference_with_transfers_seconds":r["inference_with_transfers_seconds"],
                "video_wall_seconds":r["video_wall_seconds"],"source":"prior correction" if old else "new correction"})
            correction.append({"prompt_id":pid,"group":variant,"intervention_step":cfg["intervention_step"],
                "strength":cfg["strength"],"student_sigma":trace["student_sigma"],
                "teacher_sigma":trace["teacher_sigma"],"endpoint_difference_rms":trace["endpoint_difference_rms"],
                "applied_delta_rms":trace["applied_delta_rms"],"applied_delta_relative_rms":trace["applied_delta_relative_rms"]})
        assert len(noise_hashes) == 1, "Re-noise sample differs across groups: " + pid
    return rows, correction


def render_report(p, summaries, strata, motion, contrasts, medians, ratios):
    names = {"subject_consistency":"主体一致性","multiple_objects":"多物体符合度","dynamic_degree":"动态比例"}
    variants = c.all_variants(p)
    lines = ["# 首步/第三步 × 三种强度：纠偏效果比较", "",
        "新增四组各 48 条视频、80 个分数，共 192 条视频、320 个新分数。直接复用首步 λ=0.1/0.3 与 H2 教师、学生结果；合并为 384 条视频、640 个分数。",
        "所有组沿用 48 条既有 prompt、seed=2026101901、生成文本和评测映射。以下是固定面板的探索性比较。", "",
        "| 指标（原分×100） | 介入步 | λ | n | 教师 | 学生 | 纠偏 | 纠偏−学生 | 提升/持平/下降 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in summaries:
        cfg = variants[r["variant"]]
        lines.append(f"| {names[r['metric']]} | {cfg['intervention_step']+1} | {cfg['strength']} | {r['n']} | {100*r['T50']:.4f} | {100*r['S5']:.4f} | {100*r['corrected']:.4f} | {100*r['corrected_minus_student']:+.4f} | {r['positive_pairs']}/{r['equal_pairs']}/{r['negative_pairs']} |")
    lines += ["", "dynamic_degree 的正负只是运动判定变化，不表示质量优劣；是否可接受由研究者判断。持平指未舍入原分完全相等，不设容差。参考区间不作为跨零准入门槛。", "",
        "## 原教师优势、劣势、持平子集", "",
        "子集只按 H2 同 seed 的原始 T50 与 S5 分数划分；所有纠偏组共用，不据纠偏结果筛选。n=0 显示 —。",
        "| 质量指标 | 介入步 | λ | 原教师优势 n / Δ | 原教师劣势 n / Δ | 原教师持平 n / Δ |",
        "|---|---:|---:|---|---|---|"]
    for metric in p["metrics"]:
        if metric == "dynamic_degree":
            continue
        for variant,cfg in variants.items():
            cells = []
            for label in ("teacher_higher","teacher_lower","equal"):
                r = next(r for r in strata if r["metric"]==metric and r["variant"]==variant and r["baseline_group"]==label)
                cells.append(f"{r['n']} / {100*r['corrected_minus_student']:+.4f}" if r["n"] else "0 / —")
            lines.append(f"| {names[metric]} | {cfg['intervention_step']+1} | {cfg['strength']} | " + " | ".join(cells) + " |")
    lines += ["", "Δ 均为纠偏−学生、单位为原分×100；子集的完整均值和正/平/负数量见 subgroup_scores.csv，成员 ID 与主体优势子集运动变化见 summary.json。", "",
        "## 同一 λ 的介入位置比较", "", "| 指标 | λ | 第三步−首步（原分×100） |",
        "|---|---:|---:|"]
    for r in contrasts:
        lines.append(f"| {names[r['metric']]} | {r['strength']} | {100*r['third_minus_first']:+.4f} |")
    lines += ["", "同一 λ 是相同干净端点插值权重；第三步的 (1−σ_next) 约为首步的 5.12 倍，不代表实际扰动或质量变化放大同样倍数。实际幅度见 correction_magnitudes.csv。", "",
        "## 真实耗时", "", "| 组别 | 去噪+VAE中位秒 | 含模型传输中位秒 | 教师/该组耗时比 |",
        "|---|---:|---:|---:|"]
    for arm,r in medians.items():
        ratio = f"{ratios[arm]:.3f}×" if arm in ratios else "—"
        lines.append(f"| {arm} | {r['compute_seconds']:.3f} | {r['inference_with_transfers_seconds']:.3f} | {ratio} |")
    lines += ["", "各纠偏组均为 5 次学生预测+1 次教师预测；真实计时纳入去噪、纠偏、VAE 和模型切换，加载/预热/导出另记。复用数据采集时期不同，不能称同时段重测；多卡并发不计为算法加速。",
        "本批结束后停止；不自动增加 prompt、seed、指标或介入次数。", ""]
    return "\n".join(lines)


def main():
    manifest,p = c.check_frozen()
    assert not (c.RESULT/"COMPLETE.json").exists()
    refs = {r["id"]:r for r in c.read(h2.RUN/"protocol.json")["prompts"]}
    variants = c.all_variants(p)
    records = load_records(p)
    scores,score_rows = load_scores(p,refs,records)
    summaries,paired = summarize_scores(dict(p,variants=variants),refs,scores)
    for r in paired:
        r["baseline_group"] = baseline_group(r["T50"],r["S5"]) if r["metric"]!="dynamic_degree" else "not_quality_ranked"
    groups,strata,motion = subgroup_summaries(p,refs,scores,variants)
    contrasts = timing_contrasts(p,refs,scores,variants)
    timing,correction = timing_records(p,manifest,records)
    arms = ["T50","S5"]+list(variants)
    medians = {arm:{key:float(np.median([r[key] for r in timing if r["group"]==arm]))
        for key in ("compute_seconds","inference_with_transfers_seconds","video_wall_seconds")} for arm in arms}
    ratios = {v:medians["T50"]["inference_with_transfers_seconds"]/medians[v]["inference_with_transfers_seconds"] for v in variants}
    c.RESULT.mkdir(exist_ok=True)
    for name,rows in (("scores.csv",score_rows),("paired_scores.csv",paired),("timing.csv",timing),
                      ("subgroup_scores.csv",strata),("timing_contrasts.csv",contrasts),("correction_magnitudes.csv",correction)):
        csv_write(c.RESULT/name,rows)
    result = {"status":"complete","protocol_sha256":manifest["protocol_sha256"],**p["counts"],
        "metrics":summaries,"quality_subgroup_ids":groups,"quality_subgroups":strata,
        "motion_in_subject_teacher_advantage_subset":motion,"third_minus_first_at_same_lambda":contrasts,
        "timing_medians":medians,"teacher_over_corrected_inference_median_ratio":ratios,
        "scope":"Same H2 prompts and seed; exploratory timing/strength comparison, not independent generalization",
        "finished_unix":time.time()}
    c.write_json(c.RESULT/"summary.json",result,replace=True)
    (c.RESULT/"REPORT.md").write_text(render_report(p,summaries,strata,motion,contrasts,medians,ratios),encoding="utf-8")
    c.write_json(c.RESULT/"COMPLETE.json",{"status":"complete","finished_unix":time.time(),
        "sha256":{f.name:c.digest(f) for f in c.RESULT.iterdir() if f.is_file()}})
    path = c.ROOT/"Project_Progress/Total_Progress.md"
    text = path.read_text(encoding="utf-8")
    start,end = "<!-- HYA_H3_TIMING_AUTO_START -->","<!-- HYA_H3_TIMING_AUTO_END -->"
    if start in text and end in text:
        left,rest = text.split(start,1)
        _,right = rest.split(end,1)
        body = "\n**新增四组已完成：192/192 视频、320/320 新分数。**六个纠偏组与 H2 基线已合并，结果见 `Result/HYA_H3_TIMING_V1/REPORT.md`；等待结果审阅，不自动追加实验。\n"
        temp = path.with_name("."+path.name+".timing.tmp")
        temp.write_text(left+start+body+end+right,encoding="utf-8")
        temp.replace(path)


if __name__ == "__main__":
    main()
