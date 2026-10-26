"""Distillation++ experiment bookkeeping; H2 teacher/student baselines are reused."""
import json
import os
from pathlib import Path
import time

import hya_common as h1
import hya_h2_common as h2
from hya_common import ROOT, digest, write_json

RUN = ROOT / "runs/hya_h3_flow_v2"
RESULT = ROOT / "Result/HYA_H3_FLOW_V2"
ARCHIVE = ROOT / "Evaluation_Videos/hya_h3_flow_v2"
PLAN = ROOT / "Project_Plan/HYA_H3_PROTOCOL.json"
GEN_PY = h2.GEN_PY
EVAL_PY = h2.EVAL_PY


def read(path):
    return json.loads(Path(path).read_text())


def check_frozen():
    if not read(PLAN)["authorization"]["execution"]:
        raise RuntimeError("Plan is under user review; generation and evaluation are not authorized")
    h2.check_frozen()
    manifest = read(RUN / "manifest.json")
    for rel, expected in manifest["frozen_files"].items():
        assert digest(ROOT / rel) == expected, "H3 frozen file changed: " + rel
    assert digest(RUN / "protocol.json") == manifest["protocol_sha256"]
    assert digest(h2.RESULT / "COMPLETE.json") == manifest["h2_complete_sha256"]
    for rel, expected in manifest["baseline_files"].items():
        assert digest(ROOT / rel) == expected, "Baseline changed: " + rel
    return manifest, read(RUN / "protocol.json")


def record_path(prompt_id, variant):
    return RUN / "records" / variant / (prompt_id + ".json")


def progress(path, stage, **kwargs):
    write_json(path, {"stage": stage, "time_unix": time.time(), "pid": os.getpid(), **kwargs}, replace=True)
    print("H3_PROGRESS", json.dumps({"stage": stage, **kwargs}, ensure_ascii=False), flush=True)


def ticks(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def alive(pid, birth):
    try:
        parts = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return parts[19] == str(birth) and parts[0] != "Z"
    except FileNotFoundError:
        return False


def variants_for(protocol, prompt_id):
    names = list(protocol["variants"])
    return names if protocol["prompt_ids"].index(prompt_id) % 2 == 0 else names[::-1]


def generation_complete(protocol, prompt_id):
    return all(record_path(prompt_id, v).exists() for v in variants_for(protocol, prompt_id))


def prepare():
    assert not RUN.exists(), "H3 already prepared; do not create a duplicate run"
    assert ROOT.resolve() == ROOT
    h2.check_frozen()
    p = read(PLAN)
    assert p["authorization"]["execution"] is True, "Await user review and execution authorization"
    assert p["teacher_query"] == "next_student_sigma" and p["intervention_steps"] == [0]
    assert len(set(p["allowed_gpu_indices"])) == len(p["allowed_gpu_indices"])
    assert 0 < p["max_workers"] <= len(p["allowed_gpu_indices"])
    h2p = read(h2.RUN / "protocol.json")
    assert p["parent_protocol_sha256"] == digest(h2.RUN / "protocol.json")
    by_id = {r["id"]: r for r in h2p["prompts"]}
    expected_ids = [r["id"] for lib in ["motion", "objects"]
                    for r in h2p["prompts"] if r["library"] == lib]
    assert p["prompt_ids"] == expected_ids and len(expected_ids) == 48
    assert p["counts"]["corrected_videos"] == len(expected_ids) * len(p["variants"])
    assert p["counts"]["new_scores"] == len(p["variants"]) * sum(
        len(set(by_id[pid]["metrics"]) & set(p["metrics"])) for pid in expected_ids)
    frozen = {str(PLAN.relative_to(ROOT)): digest(PLAN)}
    for file in sorted((ROOT / "tools").glob("hya_h3_*.py")):
        frozen[str(file.relative_to(ROOT))] = digest(file)
    base = {}
    for pid in p["prompt_ids"]:
        inp_path = h2.RUN / "inputs" / (pid + ".json")
        inp = read(inp_path)
        assert inp["prompt"] == by_id[pid]["generation_prompt"] and inp["seed"] == p["seed"]
        assert digest(inp["tensor_file"]) == inp["file_sha256"]
        for file in [inp_path, h2.RUN / "pairs" / (pid + ".json")]:
            base[str(file.relative_to(ROOT))] = digest(file)
    for file in (h2.RUN / "scores").glob("*.json"):
        if any(r["prompt_id"] in p["prompt_ids"] for r in read(file)["rows"]):
            base[str(file.relative_to(ROOT))] = digest(file)
    RUN.mkdir()
    (RUN / "logs").mkdir()
    write_json(RUN / "protocol.json", p)
    for rel in p.get("reuse_development_records", []):
        old = read(ROOT / rel)
        pid = old["prompt_id"]
        step = p["intervention_steps"][0]
        assert pid in p["prompt_ids"] and old["seed"] == p["seed"]
        assert old["intervention_step"] == step
        matches = [name for name, cfg in p["variants"].items() if cfg["strength"] == old["strength"]]
        assert len(matches) == 1
        variant = matches[0]
        baseline = read(h2.RUN / "pairs" / (pid + ".json"))["records"]["S5"]
        assert old["input_hashes"] == baseline["inputs"]["hashes"]
        assert old["initial_latent_sha256"] == baseline["actual_initial_latent_sha256"]
        assert old["intervention"]["teacher_sigma"] == old["steps"][step]["next_sigma"]
        assert digest(old["archive_path"]) == old["video"]["sha256"]
        base[rel] = digest(ROOT / rel)
        archive_path = ARCHIVE / variant / (pid + ".mp4")
        h2.commit_copy(Path(old["archive_path"]), archive_path, old["video"]["sha256"])
        reused = dict(old, variant=variant, original_variant=old["variant"],
                      original_protocol_sha256=old["protocol_sha256"],
                      protocol_sha256=digest(RUN / "protocol.json"),
                      video_path=old["archive_path"], archive_path=str(archive_path), reused_from=rel)
        write_json(record_path(pid, variant), reused)
    manifest = {"stage": "Distillation++ paired effect validation", "prepared_unix": time.time(),
                "protocol_sha256": digest(RUN / "protocol.json"), "frozen_files": frozen,
                "baseline_files": base, "h2_complete_sha256": digest(h2.RESULT / "COMPLETE.json"),
                "authorization": p["authorization"]}
    write_json(RUN / "manifest.json", manifest)
    import importlib.metadata as md
    write_json(RUN / "generation_environment.json", {n: md.version(n) for n in
        ["torch", "diffusers", "transformers", "flash-attn", "numpy"]})
    print(json.dumps({"status": "prepared", "prompts": p["prompt_ids"],
                      "expected_videos": p["counts"]["corrected_videos"],
                      "new_generations": p["counts"]["new_videos"],
                      "expected_new_scores": p["counts"]["new_scores"]}))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.status:
        if not (RUN / "controller.json").exists():
            print(json.dumps({"status": "not_started", "execution_authorized": read(PLAN)["authorization"]["execution"]}))
            raise SystemExit(0)
        s = read(RUN / "controller.json")
        p = read(RUN / "protocol.json")
        videos_by_group = {v: len(list((RUN / "records" / v).glob("*.json"))) for v in p["variants"]}
        scores_by_metric = {m: len(read(RUN / "scores" / (m + ".json"))["rows"])
                            if (RUN / "scores" / (m + ".json")).exists() else 0 for m in p["metrics"]}
        active = []
        for tid, task in s["tasks"].items():
            if task["status"] == "running":
                a = task["attempts"][-1]
                key = tid[4:] if tid.startswith("gen_") else tid[5:]
                progress_path = RUN / ("generation" if tid.startswith("gen_") else "evaluation") / key / f"attempt{a['attempt']:02d}" / "progress.json"
                active.append({"task": tid, "gpu": a["gpu"], "pid": a["pid"], "alive": alive(a["pid"], a["birth"]),
                               "progress": read(progress_path) if progress_path.exists() else None})
        print(json.dumps({"status": s["status"], "controller_alive": alive(s["pid"], s["birth"]),
                          "committed_videos": len(list((RUN / "records").glob("*/*.json"))),
                          "total_videos": p["counts"]["corrected_videos"],
                          "videos_by_group": videos_by_group, "videos_per_group": len(p["prompt_ids"]),
                          "new_scores": sum(len(read(x)["rows"]) for x in (RUN / "scores").glob("*.json")),
                          "total_new_scores": p["counts"]["new_scores"],
                          "scores_by_metric": scores_by_metric,
                          "allowed_gpus": p["allowed_gpu_indices"], "max_workers": p["max_workers"],
                          "active": active, "errors": {k:v.get("error") for k,v in s["tasks"].items() if v["status"]=="paused"}
                          }, ensure_ascii=False, indent=2))
