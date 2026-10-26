"""One H3 prompt or one metric, using the unchanged H2 models and evaluators."""
import argparse
import gc
import time
import traceback

import hya_common as h1
import hya_h2_common as h2
import hya_h3_common as c
from hya_h3_math import endpoints, renoise, correct_euler, renoise_seed
from hya_generate_smoke import export_video, check_video


def generate_one(row, variant, student, teacher, vae, inputs, info, admitted, out, log, warm_student, warm_teacher):
    import torch
    protocol = c.read(c.RUN / "protocol.json")
    cfg = protocol["variants"][variant]
    intervention_step = protocol["intervention_steps"][0]
    gpu = admitted["index"]
    begin = time.time()
    record = {"status": "starting", "variant": variant, "prompt_id": row["id"], "seed": info["seed"],
              "generation_prompt": row["generation_prompt"], "input_hashes": info["hashes"],
              "input_file_sha256": info["file_sha256"], "admission": admitted,
              "protocol_sha256": c.digest(c.RUN / "protocol.json"), "started_unix": begin,
              "strength": cfg["strength"], "intervention_step": intervention_step, "teacher_queries": 0,
              "warmup_student_seconds": 0.0, "warmup_teacher_seconds": 0.0,
              "model_transfer_seconds": 0.0, "correction_algebra_seconds": 0.0,
              "teacher_forward_seconds": 0.0, "student_denoise_seconds": 0.0}

    def guard():
        h2.ensure_exclusive(gpu)

    def transfer(model, device):
        guard()
        torch.cuda.synchronize(); t = time.perf_counter()
        model.to(device, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        record["model_transfer_seconds"] += time.perf_counter() - t
        if device == "cpu":
            gc.collect(); torch.cuda.empty_cache()

    transfer(student, "cuda")
    context = inputs["context"].to("cuda")
    mask = inputs["mask"].to("cuda")
    pooled = inputs["pooled"].to("cuda")
    noise = inputs["noise"].to("cuda", dtype=context.dtype)
    assert noise.dtype == torch.float16 and noise.shape == h1.SHAPE
    record["initial_latent_sha256"] = h1.tensor_digest(noise)
    freqs = [x.to("cuda") for x in student.get_rotary_pos_embed((33, 45, 80))]
    guidance = torch.tensor([6.0], device="cuda", dtype=torch.bfloat16) * 1000
    _, _, scheduler_cls = h1.native_modules()
    scheduler = scheduler_cls(shift=7, reverse=True, solver="euler")
    scheduler.set_timesteps(5, "cuda")

    def forward(model, z, t):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(z, t.repeat(1), text_states=context, text_mask=mask, text_states_2=pooled,
                         freqs_cos=freqs[0], freqs_sin=freqs[1], guidance=guidance, return_dict=True)["x"]

    if warm_student:
        guard(); log("student_warmup", variant=variant)
        torch.cuda.synchronize(); timer = time.perf_counter()
        pred = forward(student, noise.clone(), scheduler.timesteps[0]); torch.cuda.synchronize()
        record["warmup_student_seconds"] = time.perf_counter() - timer
        del pred
    torch.cuda.reset_peak_memory_stats()
    latent = noise.clone()
    step_records = []
    for i, t in enumerate(scheduler.timesteps):
        guard(); log("student_step", variant=variant, completed=i, total=5)
        torch.cuda.synchronize(); timer = time.perf_counter()
        pred = forward(student, latent, t)
        base_next = scheduler.step(pred, t, latent, return_dict=False)[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - timer
        record["student_denoise_seconds"] += elapsed
        assert torch.isfinite(pred).all() and torch.isfinite(base_next).all()
        assert base_next.dtype == torch.float32
        if i == 0:
            record["first_prediction_sha256"] = h1.tensor_digest(pred)
        active = i == intervention_step
        if active:
            sigma, query_sigma = scheduler.sigmas[i], scheduler.sigmas[i+1]
            query_t = scheduler.timesteps[i+1]
            torch.cuda.synchronize(); timer = time.perf_counter()
            clean_s, eps_s = endpoints(latent.float(), pred.float(), sigma)
            stream = renoise_seed(row["id"], info["seed"], i)
            # Dedicated CPU generator: never advances the initial/student RNG stream.
            xi_cpu = torch.randn(h1.SHAPE, generator=torch.Generator("cpu").manual_seed(stream), dtype=torch.float32)
            xi = xi_cpu.to("cuda")
            query = renoise(clean_s, xi, query_sigma)
            torch.cuda.synchronize()
            record["correction_algebra_seconds"] += time.perf_counter() - timer
            assert query.dtype == torch.float32 and torch.isfinite(query).all()
            trace = {"step": i, "student_sigma": float(sigma), "teacher_sigma": float(query_sigma),
                     "teacher_timestep": float(query_t), "renoise_rng_seed": stream,
                     "renoise_noise_sha256": h1.tensor_digest(xi_cpu), "query_sha256": h1.tensor_digest(query),
                     "student_endpoint_sha256": h1.tensor_digest(clean_s),
                     "preserved_student_epsilon_sha256": h1.tensor_digest(eps_s)}
            del xi_cpu, xi
            transfer(student, "cpu")
            transfer(teacher, "cuda")
            if warm_teacher:
                guard(); log("teacher_warmup", variant=variant)
                torch.cuda.synchronize(); timer = time.perf_counter()
                wp = forward(teacher, query, query_t); torch.cuda.synchronize()
                record["warmup_teacher_seconds"] = time.perf_counter() - timer
                del wp
            guard(); log("teacher_query", variant=variant, sigma=float(query_sigma))
            torch.cuda.synchronize(); timer = time.perf_counter()
            pred_t = forward(teacher, query, query_t); torch.cuda.synchronize()
            record["teacher_forward_seconds"] += time.perf_counter() - timer
            record["teacher_queries"] += 1
            assert torch.isfinite(pred_t).all()
            trace["teacher_prediction_sha256"] = h1.tensor_digest(pred_t)
            torch.cuda.synchronize(); timer = time.perf_counter()
            clean_t, _ = endpoints(query, pred_t.float(), query_sigma)
            latent_next = correct_euler(base_next, clean_s, clean_t, query_sigma, cfg["strength"])
            torch.cuda.synchronize()
            record["correction_algebra_seconds"] += time.perf_counter() - timer
            assert torch.isfinite(clean_t).all() and torch.isfinite(latent_next).all()
            rms = lambda x: float(torch.sqrt(torch.mean(x.float().square())))
            trace.update(student_endpoint_rms=rms(clean_s), teacher_endpoint_rms=rms(clean_t),
                         endpoint_difference_rms=rms(clean_t-clean_s),
                         base_next_rms=rms(base_next), applied_delta_rms=rms(latent_next-base_next),
                         applied_delta_relative_rms=rms(latent_next-base_next)/max(rms(base_next), 1e-12))
            record["intervention"] = trace
            del clean_s, clean_t, eps_s, query, pred_t
            transfer(teacher, "cpu")
            transfer(student, "cuda")
        else:
            latent_next = base_next
        latent = latent_next
        step_records.append({"step": i, "timestep": float(t), "sigma": float(scheduler.sigmas[i]),
                             "next_sigma": float(scheduler.sigmas[i+1]), "student_seconds": elapsed,
                             "prediction_dtype": str(pred.dtype), "state_dtype": str(latent.dtype)})
        del pred, base_next, latent_next
        log("step_complete", variant=variant, completed=i+1, total=5)
    record.update(steps=step_records, student_calls=5, final_latent_sha256=h1.tensor_digest(latent),
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    assert record["teacher_queries"] == 1
    latent_cpu = latent.cpu()
    del latent, noise, context, mask, pooled, freqs, guidance
    transfer(student, "cpu")
    guard(); log("decode", variant=variant)
    torch.cuda.synchronize(); timer = time.perf_counter()
    vae.to("cuda"); torch.cuda.synchronize()
    record["vae_to_gpu_seconds"] = time.perf_counter()-timer
    z = latent_cpu.to("cuda") / vae.config.scaling_factor
    torch.cuda.synchronize(); timer = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.float16):
        decoded = vae.decode(z, return_dict=False)[0]
    torch.cuda.synchronize(); record["vae_decode_seconds"] = time.perf_counter()-timer
    assert decoded.shape == (1,3,129,720,1280) and torch.isfinite(decoded).all()
    decoded = decoded.cpu(); del z, latent_cpu
    torch.cuda.synchronize(); timer = time.perf_counter()
    vae.to("cpu"); torch.cuda.synchronize()
    record["vae_offload_seconds"] = time.perf_counter()-timer
    gc.collect(); torch.cuda.empty_cache()
    video = out / variant / (row["id"]+".mp4")
    video.parent.mkdir(exist_ok=True)
    guard(); log("export", variant=variant)
    timer = time.perf_counter(); record["encoding"] = export_video(decoded, video)
    record["export_seconds"] = time.perf_counter()-timer; del decoded
    record["video"] = check_video(video)
    record["video_path"] = str(video)
    record["hot_denoise_decode_seconds"] = sum(record[k] for k in
        ["student_denoise_seconds", "teacher_forward_seconds", "correction_algebra_seconds", "vae_decode_seconds"])
    record["inference_with_transfers_seconds"] = record["hot_denoise_decode_seconds"] + sum(record[k] for k in
        ["model_transfer_seconds", "vae_to_gpu_seconds", "vae_offload_seconds"])
    record["video_wall_seconds"] = time.time()-begin
    record["network_attempts"] = h1.NETWORK_ATTEMPTS
    assert not h1.NETWORK_ATTEMPTS
    guard()
    record.update(status="passed", finished_unix=time.time())
    h1.write_json(video.with_suffix(".json"), record)
    target = c.ARCHIVE / variant / video.name
    h2.commit_copy(video, target, record["video"]["sha256"])
    record["archive_path"] = str(target)
    h1.write_json(c.record_path(row["id"], variant), record)
    return record


def generation(prompt_id, gpu, attempt):
    manifest, protocol = c.check_frozen()
    assert prompt_id in protocol["prompt_ids"]
    assert gpu in protocol["allowed_gpu_indices"]
    assert len(protocol["intervention_steps"]) == 1 and 0 <= protocol["intervention_steps"][0] < 4
    row = next(p for p in c.read(h2.RUN/"protocol.json")["prompts"] if p["id"] == prompt_id)
    out = c.RUN / "generation" / prompt_id / f"attempt{attempt:02d}"
    out.mkdir(parents=True, exist_ok=False)
    log = lambda stage, **kw: c.progress(out/"progress.json", stage, prompt_id=prompt_id, **kw)
    try:
        h1.configure()
        with h2.task_claim("h3_gen_"+prompt_id), h2.admission(gpu) as admitted:
            import torch
            torch.set_num_threads(4); torch.set_num_interop_threads(4); torch.set_grad_enabled(False)
            torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False; torch.manual_seed(protocol["seed"])
            assert h1.scheduler_contract() == c.read(h1.ROOT/"Project_Plan/HYA_RUNTIME_CONTRACT.json")["schedule"]
            info = c.read(h2.RUN/"inputs"/(prompt_id+".json"))
            assert info["prompt"] == row["generation_prompt"] and info["seed"] == protocol["seed"]
            assert c.digest(info["tensor_file"]) == info["file_sha256"]
            inputs = torch.load(info["tensor_file"], map_location="cpu", weights_only=True)
            assert all(h1.tensor_digest(v) == info["hashes"][k] for k,v in inputs.items())
            log("load_models")
            loads = {}
            with h2.loading_slot():
                timer=time.perf_counter();student,proof=h1.load_native("S5");loads["student_cpu_seconds"]=time.perf_counter()-timer
                h1.write_json(out/"S5_weight_mapping.json",proof)
                timer=time.perf_counter();teacher,proof=h1.load_native("T50");loads["teacher_cpu_seconds"]=time.perf_counter()-timer
                h1.write_json(out/"T50_weight_mapping.json",proof)
                from diffusers import AutoencoderKLHunyuanVideo
                timer=time.perf_counter()
                vae=AutoencoderKLHunyuanVideo.from_pretrained(str(h1.MODEL/"vae"),torch_dtype=torch.float16,local_files_only=True)
                loads["vae_cpu_seconds"]=time.perf_counter()-timer
            vae.eval().requires_grad_(False);vae.enable_tiling()
            assert all(getattr(vae,k)==v for k,v in c.read(h1.ROOT/"Project_Plan/HYA_RUNTIME_CONTRACT.json")["vae_tiling"].items())
            h1.write_json(out/"load_timing.json", loads)
            warmed_s=False;warmed_t=False
            for variant in c.variants_for(protocol,prompt_id):
                pointer=c.record_path(prompt_id,variant)
                if pointer.exists():
                    existing=c.read(pointer)
                    assert c.digest(existing["archive_path"])==existing["video"]["sha256"]
                    continue
                cfg=protocol["variants"][variant]
                assert 0 < cfg["strength"] <= 1
                generate_one(row,variant,student,teacher,vae,inputs,info,admitted,out,log,not warmed_s,not warmed_t)
                warmed_s=True;warmed_t=True
            h1.write_json(out/"result.json",{"status":"passed","prompt_id":prompt_id,"admission":admitted,
                                           "cpu_load_seconds":loads,"finished_unix":time.time()})
            log("passed")
    except BaseException as exc:
        conflict=isinstance(exc,h2.ResourceConflict)
        h1.write_json(out/"failure.json",{"status":"conflict" if conflict else "failed","error":repr(exc),
                                         "traceback":traceback.format_exc(),"time_unix":time.time()})
        raise SystemExit(75 if conflict else 1)


def evaluation(metric,gpu,attempt):
    _,protocol=c.check_frozen()
    assert metric in protocol["metrics"]
    assert gpu in protocol["allowed_gpu_indices"]
    out=c.RUN/"evaluation"/metric/f"attempt{attempt:02d}";out.mkdir(parents=True,exist_ok=False)
    try:
        h1.configure()
        source=c.read(h2.RUN/"protocol.json")
        rows=[];videos=[];ids=[]
        for pid in protocol["prompt_ids"]:
            row=next(p for p in source["prompts"] if p["id"]==pid)
            if metric not in row["metrics"]:continue
            for variant in c.variants_for(protocol,pid):
                rec=c.read(c.record_path(pid,variant));video=rec["archive_path"]
                assert c.digest(video)==rec["video"]["sha256"]
                rows.append(row);videos.append(video);ids.append((pid,variant,rec["video"]["sha256"]))
        with h2.task_claim("h3_eval_"+metric),h2.admission(gpu) as admitted:
            import hya_evaluate_smoke as acceptance
            from hya_h2_evaluate import evaluate
            acceptance.ensure_exclusive=lambda:h2.ensure_exclusive(gpu)
            torch=acceptance.prepare_imports()
            torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.set_grad_enabled(False)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False;torch.manual_seed(20261019)
            import cv2
            cv2.setNumThreads(1)
            c.progress(out/"progress.json","scoring",metric=metric,videos=len(videos))
            start=time.perf_counter()
            aggregate,raw=evaluate(metric,rows,videos,torch.device("cuda"),torch)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            by_path={r["video_path"]:r for r in raw};assert set(by_path)==set(videos)
            values=[]
            import math
            for video,(pid,variant,sha) in zip(videos,ids):
                value=float(by_path[video]["video_results"]);assert math.isfinite(value) and 0<=value<=1
                if metric=="dynamic_degree":assert value in (0,1)
                values.append({"prompt_id":pid,"variant":variant,"metric":metric,"score":value,
                               "video_sha256":sha,"video_path":video,"official_row":by_path[video]})
            assert abs(float(aggregate)-sum(r["score"] for r in values)/len(values))<1e-6
            h2.ensure_exclusive(gpu);assert not h1.NETWORK_ATTEMPTS
            result={"status":"passed","metric":metric,"admission":admitted,"rows":values,
                    "load_scoring_seconds":elapsed,"audit":acceptance.AUDIT,
                    "network_attempts":h1.NETWORK_ATTEMPTS,"finished_unix":time.time()}
            h1.write_json(out/"result.json",result)
            h1.write_json(c.RUN/"scores"/(metric+".json"),result)
            c.progress(out/"progress.json","passed",metric=metric,scores=len(values))
    except BaseException as exc:
        conflict=isinstance(exc,h2.ResourceConflict)
        h1.write_json(out/"failure.json",{"status":"conflict" if conflict else "failed","error":repr(exc),
                                         "traceback":traceback.format_exc(),"time_unix":time.time()})
        raise SystemExit(75 if conflict else 1)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt");group.add_argument("--metric")
    parser.add_argument("--gpu",type=int,required=True);parser.add_argument("--attempt",type=int,required=True)
    args=parser.parse_args()
    if args.prompt:generation(args.prompt,args.gpu,args.attempt)
    else:evaluation(args.metric,args.gpu,args.attempt)
