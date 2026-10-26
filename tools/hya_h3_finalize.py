"""Compare both correction strengths with the existing H2 baselines."""
import csv
import io
import time

import numpy as np
import hya_h2_common as h2
import hya_h3_common as c


def csv_write(path, rows):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(output.getvalue(), encoding="utf-8")


def baseline_inference_seconds(record):
    return record["hot_denoise_decode_seconds"] + sum(record[key] for key in (
        "model_to_gpu_seconds", "model_offload_seconds", "vae_to_gpu_seconds", "vae_offload_seconds"))


def summarize_scores(protocol, refs, scores):
    summaries, paired = [], []
    for metric in protocol["metrics"]:
        ids = [pid for pid in protocol["prompt_ids"] if metric in refs[pid]["metrics"]]
        teacher = np.array([scores[(pid, "T50", metric)] for pid in ids])
        student = np.array([scores[(pid, "S5", metric)] for pid in ids])
        # Paired resampling over prompts at the fixed experiment seed.
        indices = np.random.default_rng(20261020).integers(0, len(ids), size=(20000, len(ids)))
        for variant in protocol["variants"]:
            corrected = np.array([scores[(pid, variant, metric)] for pid in ids])
            delta = corrected - student
            interval = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975]).tolist()
            summaries.append({"metric":metric, "variant":variant, "n":len(ids),
                "T50":float(teacher.mean()), "S5":float(student.mean()),
                "corrected":float(corrected.mean()), "corrected_minus_student":float(delta.mean()),
                "paired_bootstrap_95_reference":interval,
                "positive_pairs":int((delta > 0).sum()), "equal_pairs":int((delta == 0).sum()),
                "negative_pairs":int((delta < 0).sum())})
            for pid, value, change in zip(ids, corrected, delta):
                paired.append({"prompt_id":pid, "metric":metric, "variant":variant,
                    "T50":scores[(pid,"T50",metric)], "S5":scores[(pid,"S5",metric)],
                    "corrected":float(value), "corrected_minus_student":float(change)})
    return summaries, paired


def main():
    manifest, p = c.check_frozen()
    assert not (c.RESULT / "COMPLETE.json").exists()
    variants = list(p["variants"])
    arms = ["T50", "S5"] + variants
    refs = {row["id"]: row for row in c.read(h2.RUN / "protocol.json")["prompts"]}
    records = {(pid, variant): c.read(c.record_path(pid, variant))
               for pid in p["prompt_ids"] for variant in variants}
    score_rows, scores = [], {}
    for file in (h2.RUN / "scores").glob("*.json"):
        for row in c.read(file)["rows"]:
            if row["prompt_id"] in p["prompt_ids"] and row["metric"] in p["metrics"]:
                key = (row["prompt_id"], row["arm"], row["metric"])
                assert key not in scores
                scores[key] = float(row["score"])
                score_rows.append({"prompt_id":key[0], "group":key[1], "metric":key[2],
                                   "score":scores[key], "source":"H2 reused"})
    nnew = 0
    for file in (c.RUN / "scores").glob("*.json"):
        for row in c.read(file)["rows"]:
            key = (row["prompt_id"], row["variant"], row["metric"])
            assert key[:2] in records and key not in scores
            assert key[2] in refs[key[0]]["metrics"] and key[2] in p["metrics"]
            assert row["video_sha256"] == records[key[:2]]["video"]["sha256"]
            scores[key] = float(row["score"])
            score_rows.append({"prompt_id":key[0], "group":key[1], "metric":key[2],
                               "score":scores[key], "source":"correction objective evaluation"})
            nnew += 1
    expected = {(pid, arm, metric) for pid in p["prompt_ids"] for arm in arms
                for metric in refs[pid]["metrics"] if metric in p["metrics"]}
    assert set(scores) == expected and nnew == p["counts"]["new_scores"]
    summaries, paired = summarize_scores(p, refs, scores)
    timing = []
    for pid in p["prompt_ids"]:
        old = c.read(h2.RUN / "pairs" / (pid + ".json"))["records"]
        for arm in ("T50", "S5"):
            base = old[arm]
            timing.append({"prompt_id":pid, "group":arm, "gpu":base["admission"]["index"],
                "compute_seconds":base["hot_denoise_decode_seconds"],
                "inference_with_transfers_seconds":baseline_inference_seconds(base),
                "video_wall_seconds":base["video_wall_seconds"], "source":"H2 recorded"})
        for variant in variants:
            record = records[(pid, variant)]
            assert record["status"] == "passed" and record["seed"] == p["seed"]
            assert record["student_calls"] == 5 and record["teacher_queries"] == 1
            assert record["strength"] == p["variants"][variant]["strength"]
            assert record["protocol_sha256"] == manifest["protocol_sha256"]
            assert c.digest(record["archive_path"]) == record["video"]["sha256"]
            assert record["input_hashes"] == old["S5"]["inputs"]["hashes"]
            assert record["initial_latent_sha256"] == old["S5"]["actual_initial_latent_sha256"]
            timing.append({"prompt_id":pid, "group":variant, "gpu":record["admission"]["index"],
                "compute_seconds":record["hot_denoise_decode_seconds"],
                "inference_with_transfers_seconds":record["inference_with_transfers_seconds"],
                "video_wall_seconds":record["video_wall_seconds"],
                "source":"retained development sample" if record.get("reused_from") else "new correction"})
        first, second = [records[(pid, v)]["intervention"] for v in variants]
        assert first["renoise_noise_sha256"] == second["renoise_noise_sha256"]
    medians = {arm:{key:float(np.median([row[key] for row in timing if row["group"] == arm]))
                   for key in ("compute_seconds", "inference_with_transfers_seconds", "video_wall_seconds")}
               for arm in arms}
    speed_ratios = {v:medians["T50"]["inference_with_transfers_seconds"] /
                      medians[v]["inference_with_transfers_seconds"] for v in variants}
    c.RESULT.mkdir(exist_ok=True)
    csv_write(c.RESULT / "scores.csv", score_rows)
    csv_write(c.RESULT / "paired_scores.csv", paired)
    csv_write(c.RESULT / "timing.csv", timing)
    result = {"status":"complete", "protocol_sha256":manifest["protocol_sha256"],
        "corrected_videos":len(records), "new_scores":nnew, "metrics":summaries,
        "timing_medians":medians, "teacher_over_corrected_inference_median_ratio":speed_ratios,
        "scope":"Same H2 prompts and seed; target-specific comparison, not independent generalization",
        "next_intervention_times_or_prompt_expansion_started":False, "finished_unix":time.time()}
    c.write_json(c.RESULT / "summary.json", result)
    lines = ["# 首步介入：λ=0.1 与 λ=0.3 纠偏实验", "",
        "两个强度各 48 条视频、80 个分数；H2 教师和学生基线直接复用。", "",
        "| 指标 | 强度 | 样本数 | 教师 | 学生 | 纠偏学生 | 纠偏−学生 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        strength = p["variants"][row["variant"]]["strength"]
        lines.append(f"| {row['metric']} | {strength} | {row['n']} | {100*row['T50']:.4f} | {100*row['S5']:.4f} | {100*row['corrected']:.4f} | {100*row['corrected_minus_student']:+.4f} |")
    lines += ["", "分数乘以 100 展示；dynamic_degree 为动态比例，不按越高越好判定。运动取舍由研究者决定。",
        "质量按配对均值差和逐样本结果报告；95% 配对 bootstrap 区间仅供参考，不作为跨零准入门槛。", "",
        "| 组别 | 去噪及 VAE 中位秒 | 含模型传输中位秒 |", "|---|---:|---:|"]
    for arm, row in medians.items():
        lines.append(f"| {arm} | {row['compute_seconds']:.3f} | {row['inference_with_transfers_seconds']:.3f} |")
    for variant, ratio in speed_ratios.items():
        lines.append(f"\nλ={p['variants'][variant]['strength']}：含模型传输的教师/纠偏学生中位耗时比 {ratio:.3f}×。")
    lines += ["", "耗时包含学生去噪、教师预测、纠偏计算、VAE 及模型切换传输；CPU 加载、首次预热、文本编码和视频编码另计。",
        "H2 基线为既有记录，采集时期不同；这里不是同时段重测或完整冷启动加速比。多卡并发不计入算法速度收益。",
        "λ=0.1 的 HYA_MOTION_01 复用先前已完成的匹配视频；两个强度使用相同初始噪声和独立重加噪种子。",
        "完成本轮后停止。是否改变介入时机或补充 prompt/指标，等待研究者决定。", ""]
    (c.RESULT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    c.write_json(c.RESULT / "COMPLETE.json", {"status":"complete", "finished_unix":time.time(),
        "sha256":{file.name:c.digest(file) for file in c.RESULT.iterdir() if file.is_file()}})
    progress = c.ROOT / "Project_Progress/Total_Progress.md"
    text = progress.read_text(encoding="utf-8")
    start, end = "<!-- HYA_H3_AUTO_START -->", "<!-- HYA_H3_AUTO_END -->"
    if start in text and end in text:
        left, rest = text.split(start, 1)
        _, right = rest.split(end, 1)
        body = "\n**本轮已完成：96/96 纠偏视频、160/160 新分数；两强度结果已整理。**\n结果：`Result/HYA_H3_FLOW_V2/REPORT.md`；未启动其他介入时机或 prompt 扩展。\n"
        temporary = progress.with_name("." + progress.name + ".h3.tmp")
        temporary.write_text(left + start + body + end + right, encoding="utf-8")
        temporary.replace(progress)


if __name__ == "__main__":
    main()
