"""Four new timing/strength arms; reuse the completed H2 and first-step runs."""
import argparse
import contextlib
import fcntl
import json
from pathlib import Path
import time

import hya_h2_common as h2
import hya_h3_common as previous
from hya_common import ROOT, digest, write_json
from hya_h3_common import read, progress, ticks, alive
from hya_h3_math import renoise_seed

RUN = ROOT / "runs/hya_h3_timing_v1"
RESULT = ROOT / "Result/HYA_H3_TIMING_V1"
ARCHIVE = ROOT / "Evaluation_Videos/hya_h3_timing_v1"
PLAN = ROOT / "Project_Plan/HYA_H3_TIMING_PROTOCOL.json"
GEN_PY, EVAL_PY = h2.GEN_PY, h2.EVAL_PY


def all_variants(protocol):
    combined = {**protocol["reused_variants"], **protocol["variants"]}
    return dict(sorted(combined.items(), key=lambda item: (
        item[1]["intervention_step"], item[1]["strength"])))


def intervention_settings(protocol, variant):
    cfg = protocol["variants"][variant]
    return cfg["intervention_step"], cfg["strength"]


def common_noise_seed(protocol, prompt_id):
    # Preserve the first-step run's stream, independently of the new query step.
    return renoise_seed(prompt_id, protocol["seed"], protocol["renoise_reference_step"])


def variants_for(protocol, prompt_id):
    names = list(protocol["variants"])
    offset = protocol["prompt_ids"].index(prompt_id) % len(names)
    return names[offset:] + names[:offset]


def record_path(prompt_id, variant):
    return RUN / "records" / variant / (prompt_id + ".json")


def generation_complete(protocol, prompt_id):
    return all(record_path(prompt_id, v).exists() for v in protocol["variants"])


@contextlib.contextmanager
def task_claim(task_id):
    assert task_id.replace("_", "").isalnum()
    folder = RUN / "claims"
    folder.mkdir(exist_ok=True)
    with (folder / (task_id + ".lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise h2.ResourceConflict("Task already claimed: " + task_id) from exc
        yield


def validate_protocol(p, baseline, prior):
    expected_ids = [r["id"] for lib in ("motion", "objects")
                    for r in baseline["prompts"] if r["library"] == lib]
    refs = {r["id"]: r for r in baseline["prompts"]}
    assert p["prompt_ids"] == expected_ids == prior["prompt_ids"]
    assert len(expected_ids) == len(set(expected_ids)) == 48
    assert p["seed"] == prior["seed"] == 2026101901
    assert p["teacher_query"] == "next_student_sigma" and p["renoise_reference_step"] == 0
    assert p["metrics"] == prior["metrics"]
    assert "allowed_gpu_indices" not in p, "GPU indices must be discovered at dispatch"
    assert p["max_workers"] in (3, 4)
    new = p["variants"]
    assert {(v["intervention_step"], v["strength"]) for v in new.values()} == {
        (0, 0.5), (2, 0.1), (2, 0.3), (2, 0.5)}
    assert len(new) == 4 and not set(new) & set(p["reused_variants"])
    assert len(p["reused_variants"]) == 2
    for name, cfg in p["reused_variants"].items():
        assert cfg["source_variant"] in prior["variants"]
        assert cfg["intervention_step"] == prior["intervention_steps"][0] == 0
        assert cfg["strength"] == prior["variants"][cfg["source_variant"]]["strength"]
    assert {(v["intervention_step"], v["strength"]) for v in all_variants(p).values()} == {
        (step, strength) for step in (0, 2) for strength in (0.1, 0.3, 0.5)}
    scores_per_group = sum(len(set(refs[pid]["metrics"]) & set(p["metrics"])) for pid in expected_ids)
    assert scores_per_group == 80
    assert p["counts"] == {"prompts":48, "new_videos":192, "new_scores":320,
        "reused_H2_videos":96, "reused_H2_scores":160,
        "reused_correction_videos":96, "reused_correction_scores":160,
        "combined_videos":384, "combined_scores":640}
    return refs


def validate_sources(p):
    _, prior = previous.check_frozen()
    baseline = read(h2.RUN / "protocol.json")
    validate_protocol(p, baseline, prior)
    assert p["parent_protocol_sha256"] == digest(h2.RUN / "protocol.json")
    assert p["prior_correction_protocol_sha256"] == digest(previous.RUN / "protocol.json")
    assert p["prior_complete_sha256"] == digest(previous.RESULT / "COMPLETE.json")
    assert read(previous.RESULT / "COMPLETE.json")["status"] == "complete"
    files = {"Result/HYA_H3_FLOW_V2/COMPLETE.json": digest(previous.RESULT / "COMPLETE.json")}
    for pid in p["prompt_ids"]:
        for cfg in p["reused_variants"].values():
            path = previous.record_path(pid, cfg["source_variant"])
            rec = read(path)
            assert rec["status"] == "passed" and rec["seed"] == p["seed"]
            assert rec["intervention_step"] == 0 and rec["strength"] == cfg["strength"]
            assert rec["intervention"]["renoise_rng_seed"] == common_noise_seed(p, pid)
            files[str(path.relative_to(ROOT))] = digest(path)
    for metric in p["metrics"]:
        path = previous.RUN / "scores" / (metric + ".json")
        files[str(path.relative_to(ROOT))] = digest(path)
    return files


def check_frozen():
    if not read(PLAN)["authorization"]["execution"]:
        raise RuntimeError("Implementation is ready; this batch has not been started/authorized for execution")
    previous.check_frozen()
    manifest = read(RUN / "manifest.json")
    for rel, expected in {**manifest["frozen_files"], **manifest["reused_files"]}.items():
        assert digest(ROOT / rel) == expected, "Timing batch input changed: " + rel
    assert digest(RUN / "protocol.json") == manifest["protocol_sha256"]
    return manifest, read(RUN / "protocol.json")


def prepare():
    p = read(PLAN)
    assert p["authorization"]["execution"], "Only prepare after execution is requested"
    assert ROOT.resolve() == ROOT and not RUN.exists(), "Batch already prepared or unsafe root"
    reused = validate_sources(p)
    frozen = {str(PLAN.relative_to(ROOT)): digest(PLAN)}
    for file in sorted((ROOT / "tools").glob("hya_h3_timing_*.py")):
        frozen[str(file.relative_to(ROOT))] = digest(file)
    RUN.mkdir()
    (RUN / "logs").mkdir()
    write_json(RUN / "protocol.json", p)
    write_json(RUN / "manifest.json", {"prepared_unix":time.time(),
        "protocol_sha256":digest(RUN / "protocol.json"), "frozen_files":frozen,
        "reused_files":reused, "authorization":p["authorization"]})
    import importlib.metadata as md
    write_json(RUN / "generation_environment.json", {n:md.version(n) for n in
        ("torch", "diffusers", "transformers", "flash-attn", "numpy")})
    print(json.dumps({"status":"prepared", "new_videos":192, "new_scores":320,
                      "max_workers":p["max_workers"], "gpu_selection":"currently idle, dynamic"}))


def status():
    p = read(RUN / "protocol.json") if (RUN / "protocol.json").exists() else read(PLAN)
    if not (RUN / "controller.json").exists():
        return {"status":"prepared" if RUN.exists() else "not_started",
                "execution_authorized":p["authorization"]["execution"],
                "new_videos":0, "expected_new_videos":192, "expected_new_scores":320,
                "max_workers":p["max_workers"], "gpu_selection":"currently idle, dynamic"}
    s = read(RUN / "controller.json")
    active = []
    for tid, task in s["tasks"].items():
        if task["status"] != "running":
            continue
        a = task["attempts"][-1]
        kind = "generation" if tid.startswith("gen_") else "evaluation"
        key = tid[4:] if kind == "generation" else tid[5:]
        path = RUN / kind / key / f"attempt{a['attempt']:02d}" / "progress.json"
        active.append({"task":tid, "gpu":a["gpu"], "pid":a["pid"],
            "alive":alive(a["pid"], a["birth"]), "progress":read(path) if path.exists() else None})
    videos = {v:len(list((RUN / "records" / v).glob("*.json"))) for v in p["variants"]}
    scores = {m:len(read(RUN / "scores" / (m + ".json"))["rows"])
              if (RUN / "scores" / (m + ".json")).exists() else 0 for m in p["metrics"]}
    return {"status":s["status"], "controller_alive":alive(s["pid"], s["birth"]),
            "new_videos":sum(videos.values()), "expected_new_videos":192,
            "videos_by_group":videos, "videos_per_group":48,
            "new_scores":sum(scores.values()), "expected_new_scores":320, "scores_by_metric":scores,
            "max_workers":p["max_workers"], "gpu_selection":"currently idle, dynamic",
            "active":active, "gpu_cooldowns":s.get("gpu_cooldowns", {}),
            "errors":{k:v.get("error") for k,v in s["tasks"].items() if v["status"] == "paused"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.check:
        files = validate_sources(read(PLAN))
        print(json.dumps({"status":"source_checks_passed", "reused_metadata_files":len(files),
                          "cuda_initialized":False, "run_created":RUN.exists()}))
    else:
        print(json.dumps(status(), ensure_ascii=False, indent=2))
