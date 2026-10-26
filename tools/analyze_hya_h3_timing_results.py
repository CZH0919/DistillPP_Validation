"""Independent CPU reaggregation of exported H3 timing metadata, without models.

Input is the bounded metadata bundle collected by the completion audit. Original
scores and completion artifacts are read-only; this writes only REVIEW.json.
"""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import statistics

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(bundle_path, project):
    b = read(bundle_path)
    p, refs = b["protocol"], b["prompts"]
    result_dir = project / "Result/HYA_H3_TIMING_V1"
    original = read(result_dir / "summary.json")
    configs = {**p["reused_variants"], **p["variants"]}
    variants = sorted(configs, key=lambda v:(configs[v]["intervention_step"],configs[v]["strength"]))
    arms = ["T50", "S5"] + variants
    scores, originals = {}, {}
    mapping = {cfg["source_variant"]:v for v,cfg in p["reused_variants"].items()}
    for source, rows in (("H2",b["baseline_scores"]),("prior",b["prior_scores"]),("new",b["new_scores"])):
        for r in rows:
            arm = r["arm"] if source=="H2" else mapping[r["variant"]] if source=="prior" else r["variant"]
            key = (r["prompt_id"],arm,r["metric"])
            assert key not in scores
            assert math.isfinite(r["score"]) and 0<=r["score"]<=1
            scores[key], originals[key] = r["score"], r["official_row"]
    expected = {(pid,arm,m) for pid in p["prompt_ids"] for arm in arms
                for m in refs[pid]["metrics"] if m in p["metrics"]}
    assert set(scores)==expected and len(scores)==640
    differences = []
    def check(actual, recorded):
        if actual is None or recorded is None:
            assert actual is None and recorded is None
            return
        a,r = np.asarray(actual,dtype=float), np.asarray(recorded,dtype=float)
        difference = float(np.max(np.abs(a-r)))
        assert difference <= 1e-12, (actual,recorded)
        differences.append(difference)
    with (result_dir/"scores.csv").open(encoding="utf-8",newline="") as f:
        exported = list(csv.DictReader(f))
    assert len(exported)==640 and len({(r["prompt_id"],r["group"],r["metric"]) for r in exported})==640
    for r in exported:
        check(scores[(r["prompt_id"],r["group"],r["metric"])],float(r["score"]))

    summaries, subgroups, members, object_frames = [], [], {}, {}
    for metric in p["metrics"]:
        ids = [pid for pid in p["prompt_ids"] if metric in refs[pid]["metrics"]]
        t,s = [np.array([scores[(pid,a,metric)] for pid in ids]) for a in ("T50","S5")]
        bootstrap = np.random.default_rng(20261020).integers(0,len(ids),(20000,len(ids)))
        members[metric] = {label:[pid for pid,keep in zip(ids,mask) if keep]
            for label,mask in (("teacher_higher",t>s),("teacher_lower",t<s),("equal",t==s))}
        for variant in variants:
            value = np.array([scores[(pid,variant,metric)] for pid in ids])
            delta = value-s
            row = {"metric":metric,"variant":variant,"n":len(ids),
                "T50":float(t.mean()),"S5":float(s.mean()),"corrected":float(value.mean()),
                "corrected_minus_student":float(delta.mean()),
                "paired_bootstrap_95_reference":np.quantile(delta[bootstrap].mean(axis=1),[0.025,0.975]).tolist(),
                "positive_pairs":int(sum(delta>0)),"equal_pairs":int(sum(delta==0)),"negative_pairs":int(sum(delta<0))}
            old = next(x for x in original["metrics"] if x["metric"]==metric and x["variant"]==variant)
            for field in row:
                if field not in ("metric","variant"):
                    check(row[field],old[field])
            summaries.append(row)
            if metric != "dynamic_degree":
                for label,mask in (("teacher_higher",t>s),("teacher_lower",t<s),("equal",t==s)):
                    n = int(sum(mask))
                    ds = delta[mask]
                    sub = {"metric":metric,"variant":variant,"baseline_group":label,"n":n,
                        "T50":float(t[mask].mean()) if n else None,"S5":float(s[mask].mean()) if n else None,
                        "corrected":float(value[mask].mean()) if n else None,
                        "corrected_minus_student":float(ds.mean()) if n else None,
                        "positive_pairs":int(sum(ds>0)),"equal_pairs":int(sum(ds==0)),"negative_pairs":int(sum(ds<0))}
                    old = next(x for x in original["quality_subgroups"] if all(x[k]==sub[k] for k in ("metric","variant","baseline_group")))
                    for field in sub:
                        if field not in ("metric","variant","baseline_group"):
                            check(sub[field],old[field])
                    subgroups.append(sub)
        if metric == "multiple_objects":
            for arm in arms:
                count = 0
                for pid in ids:
                    r = originals[(pid,arm,metric)]
                    assert r["frame_count"]==16
                    check(r["success_frame_count"]/r["frame_count"],scores[(pid,arm,metric)])
                    count += r["success_frame_count"]
                object_frames[arm] = {"success":count,"sampled":len(ids)*16}

    records = {**b["prior_records"],**b["new_records"]}
    prefix_checks, prefix_exceptions, within_step_exceptions = [], [], []
    for pid in p["prompt_ids"]:
        group = [records[pid+"|"+v] for v in variants]
        first_hashes = {r["first_prediction_sha256"] for r in group}
        if len(first_hashes)==1:
            prefix_checks.append(pid)
        reference_hash = b["baseline_pairs"][pid]["records"]["S5"]["first_prediction_sha256"]
        for variant,r in zip(variants,group):
            if r["first_prediction_sha256"] != reference_hash:
                prefix_exceptions.append({"prompt_id":pid,"variant":variant,
                    "first_prediction_sha256":r["first_prediction_sha256"],"baseline_first_prediction_sha256":reference_hash,
                    "identical_initial_latent":r["initial_latent_sha256"]==b["baseline_pairs"][pid]["records"]["S5"]["actual_initial_latent_sha256"],
                    "input_hashes_match":r["input_hashes"]==b["baseline_pairs"][pid]["records"]["S5"]["inputs"]["hashes"],
                    "gpu":r["admission"]["index"],"pid":r["admission"]["pid"]})
        assert len({r["intervention"]["renoise_noise_sha256"] for r in group})==1
        for step in (0,2):
            rows = [r for r in group if r["intervention_step"]==step]
            for key in ("query_sha256","student_endpoint_sha256","teacher_prediction_sha256"):
                if len({r["intervention"][key] for r in rows})!=1:
                    within_step_exceptions.append({"prompt_id":pid,"step":step,"field":key})

    timing, magnitudes = {}, {}
    with (result_dir/"timing.csv").open(encoding="utf-8",newline="") as f:
        raw_timing = {(r["prompt_id"],r["group"]):r for r in csv.DictReader(f)}
    assert len(raw_timing)==384
    for arm in arms:
        per_video = []
        for pid in p["prompt_ids"]:
            if arm in ("T50","S5"):
                r = b["baseline_pairs"][pid]["records"][arm]
                compute = r["hot_denoise_decode_seconds"]
                inclusive = compute+sum(r[k] for k in ("model_to_gpu_seconds","model_offload_seconds","vae_to_gpu_seconds","vae_offload_seconds"))
            else:
                r = records[pid+"|"+arm]
                compute = sum(r[k] for k in ("student_denoise_seconds","teacher_forward_seconds","correction_algebra_seconds","vae_decode_seconds"))
                inclusive = compute+sum(r[k] for k in ("model_transfer_seconds","vae_to_gpu_seconds","vae_offload_seconds"))
                check(compute,r["hot_denoise_decode_seconds"])
                check(inclusive,r["inference_with_transfers_seconds"])
            values = {"compute_seconds":compute,"inference_with_transfers_seconds":inclusive,"video_wall_seconds":r["video_wall_seconds"]}
            for key,value in values.items():
                check(value,float(raw_timing[(pid,arm)][key]))
            per_video.append(values)
        timing[arm] = {k:statistics.median(r[k] for r in per_video) for k in per_video[0]}
        for key,value in timing[arm].items():
            check(value,original["timing_medians"][arm][key])
        if arm in variants:
            trace = [records[pid+"|"+arm]["intervention"] for pid in p["prompt_ids"]]
            magnitudes[arm] = {k:statistics.median(r[k] for r in trace) for k in
                ("endpoint_difference_rms","applied_delta_rms","applied_delta_relative_rms")}
    ratios = {v:timing["T50"]["inference_with_transfers_seconds"]/timing[v]["inference_with_transfers_seconds"] for v in variants}
    overhead = {v:timing[v]["inference_with_transfers_seconds"]/timing["S5"]["inference_with_transfers_seconds"]-1 for v in variants}

    dynamic_changes, object_changes, same_lambda = {}, {}, []
    for variant in variants:
        dynamic_changes[variant] = [{"prompt_id":pid,"student":scores[(pid,"S5","dynamic_degree")],"corrected":scores[(pid,variant,"dynamic_degree")]}
            for pid in p["prompt_ids"] if "dynamic_degree" in refs[pid]["metrics"]
            and scores[(pid,variant,"dynamic_degree")] != scores[(pid,"S5","dynamic_degree")]]
        object_changes[variant] = []
        for pid in p["prompt_ids"]:
            if "multiple_objects" not in refs[pid]["metrics"]:
                continue
            t,s,corr = [scores[(pid,a,"multiple_objects")] for a in ("T50","S5",variant)]
            if corr != s:
                object_changes[variant].append({"prompt_id":pid,"prompt":refs[pid]["evaluation_prompt"],
                    "T50":t,"S5":s,"corrected":corr,"delta":corr-s})
    for old in original["third_minus_first_at_same_lambda"]:
        m,lam = old["metric"],old["strength"]
        a,bv = [next(v for v in variants if configs[v]["intervention_step"]==step and configs[v]["strength"]==lam) for step in (0,2)]
        ids=[pid for pid in p["prompt_ids"] if m in refs[pid]["metrics"]]
        delta=np.array([scores[(pid,bv,m)]-scores[(pid,a,m)] for pid in ids])
        r={"metric":m,"strength":lam,"n":len(ids),"third_minus_first":float(delta.mean()),
            "positive_pairs":int(sum(delta>0)),"equal_pairs":int(sum(delta==0)),"negative_pairs":int(sum(delta<0))}
        for key in r:
            if key!="metric":check(r[key],old[key])
        same_lambda.append(r)
    subject_adv = members["subject_consistency"]["teacher_higher"]
    motion_subset = {v:{"n":len(subject_adv),"dynamic_count":int(sum(scores[(pid,v,"dynamic_degree")] for pid in subject_adv)),
        "changed_pairs":sum(scores[(pid,v,"dynamic_degree")]!=scores[(pid,"S5","dynamic_degree")] for pid in subject_adv)} for v in variants}
    exception_ids = {r["prompt_id"] for r in prefix_exceptions}
    sensitivity = {}
    for metric in p["metrics"]:
        ids = [pid for pid in p["prompt_ids"] if metric in refs[pid]["metrics"] and pid not in exception_ids]
        sensitivity[metric] = {"n":len(ids),"excluded_ids":sorted(exception_ids & {pid for pid in p["prompt_ids"] if metric in refs[pid]["metrics"]}),
            "means":{arm:statistics.mean(scores[(pid,arm,metric)] for pid in ids) for arm in arms},
            "corrected_minus_student":{v:statistics.mean(scores[(pid,v,metric)]-scores[(pid,"S5",metric)] for pid in ids) for v in variants}}
    tz = timezone(timedelta(hours=8))
    def date(value):return datetime.fromtimestamp(value,tz).isoformat(timespec="seconds")
    audit = b["audit"]
    output = {"verification":audit,"started_beijing":date(audit["started_unix"]),"finished_beijing":date(audit["finished_unix"]),
        "independent_recalculation":{"checked_numeric_items":len(differences),"max_abs_difference":max(differences),
            "score_rows":len(scores),"timing_rows":len(raw_timing),"matching_prefix_prompt_count":len(prefix_checks)},
        "prediction_prefix_check":{"all_bitwise_equal":not prefix_exceptions,"exceptions":prefix_exceptions,
            "within_step_exceptions":within_step_exceptions,
            "interpretation":"Identical inputs and complete records; one bitwise prediction discrepancy needs disclosure. Cause and elementwise magnitude are not established by hashes; no automatic invalidation or deletion.",
            "candidate_step3_lambda_0p3_affected":any(r["variant"]=="step3_lambda_0p3" for r in prefix_exceptions)},
        "sensitivity_excluding_prefix_exception_from_all_arms":sensitivity,
        "metrics":summaries,"quality_subgroup_ids":{k:v for k,v in members.items() if k!="dynamic_degree"},
        "quality_subgroups":subgroups,"dynamic_changes":dynamic_changes,"motion_in_teacher_advantage_subject_subset":motion_subset,
        "multiple_object_success_frames":object_frames,"object_changes":object_changes,
        "same_lambda_position_comparisons":same_lambda,"timing_medians":timing,
        "teacher_speed_ratios":ratios,"student_time_overhead":overhead,"correction_magnitude_medians":magnitudes,
        "interpretation":{"criterion":"Mean improvements on both quality targets; dynamic changes reported without an automatic threshold",
            "candidate":"step3_lambda_0p3","scope":"Existing fixed 48 prompts and one seed; small positive mean effects, not a generalization claim",
            "no_new_experiment_started":True}}
    (result_dir/"REVIEW.json").write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:output[k] for k in ("started_beijing","finished_beijing","independent_recalculation", "prediction_prefix_check", "sensitivity_excluding_prefix_exception_from_all_arms", "dynamic_changes",
        "multiple_object_success_frames","object_changes","teacher_speed_ratios","student_time_overhead","correction_magnitude_medians")},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",required=True,type=Path)
    parser.add_argument("--project",required=True,type=Path)
    args=parser.parse_args()
    main(args.bundle,args.project)
