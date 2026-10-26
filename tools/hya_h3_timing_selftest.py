"""CPU-only checks for the timing extension; no model load or real dispatch."""
import contextlib
import copy
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import hya_h2_common as h2
import hya_h3_common as previous
import hya_h3_timing_common as c
import hya_h3_timing_controller as controller
import hya_h3_timing_finalize as report
from hya_h3_math import endpoints, renoise, correct_euler, renoise_seed
from hya_h3_finalize import summarize_scores


class TimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = c.read(c.PLAN)
        cls.base = c.read(h2.RUN/"protocol.json")
        cls.prior = c.read(previous.RUN/"protocol.json")
        cls.refs = {r["id"]:r for r in cls.base["prompts"]}

    def test_exact_matrix_and_balanced_order(self):
        c.validate_protocol(self.p,self.base,self.prior)
        first = [c.variants_for(self.p,pid)[0] for pid in self.p["prompt_ids"]]
        self.assertEqual({v:first.count(v) for v in self.p["variants"]},
                         {v:12 for v in self.p["variants"]})
        self.assertEqual(c.intervention_settings(self.p,"step3_lambda_0p5"),(2,0.5))
        self.assertNotIn("allowed_gpu_indices",self.p)

    def test_endpoint_identity_at_both_positions(self):
        sigmas = c.read(c.ROOT/"Project_Plan/HYA_RUNTIME_CONTRACT.json")["schedule"]["5"]["sigmas"]
        rng = np.random.default_rng(17)
        z,v,xi,vt = rng.normal(size=(4,64))
        for step in (0,2):
            sigma,r = sigmas[step:step+2]
            xs,eps = endpoints(z,v,sigma)
            base = z+(r-sigma)*v
            query = renoise(xs,xi,r)
            xt,_ = endpoints(query,vt,r)
            for strength in (0.1,0.3,0.5):
                actual = correct_euler(base,xs,xt,r,strength)
                expected = (1-r)*((1-strength)*xs+strength*xt)+r*eps
                np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=1e-13)
            self.assertIs(correct_euler(base,xs,xt,r,0),base)
        self.assertAlmostEqual((1-sigmas[3])/(1-sigmas[1]),5.117645737,places=8)

    def test_existing_random_stream_compatibility(self):
        for pid in self.p["prompt_ids"]:
            expected = renoise_seed(pid,self.p["seed"],0)
            self.assertEqual(c.common_noise_seed(self.p,pid),expected)
            for cfg in self.p["reused_variants"].values():
                r = c.read(previous.record_path(pid,cfg["source_variant"]))
                self.assertEqual(r["intervention"]["renoise_rng_seed"],expected)
        self.assertNotEqual(c.common_noise_seed(self.p,self.p["prompt_ids"][0]),
                            c.common_noise_seed(self.p,self.p["prompt_ids"][1]))

    def test_idle_selection_is_dynamic_and_excludes_residents(self):
        snap = {"gpus":[[str(i),f"GPU-{i}","0","143771","0"] for i in range(8)],
                "processes":[["GPU-0","123","1000"]]}
        snap["gpus"][5][2] = "2000"
        snap["gpus"][6][4] = "50"
        self.assertEqual(controller.select_idle_gpus(snap,{2},{"3":200},{"4":1},100),[1,7,4])
        self.assertIn(3,controller.select_idle_gpus(snap,set(),{"3":200},{},201))

    def test_subgroups_use_unrounded_baseline_and_keep_empty_groups(self):
        self.assertEqual(report.baseline_group(0.5+1e-12,0.5),"teacher_higher")
        self.assertEqual(report.baseline_group(0.5,0.5),"equal")
        empty = report.delta_statistics([],[],[])
        self.assertEqual(empty["n"],0)
        self.assertIsNone(empty["corrected_minus_student"])
        r = report.delta_statistics([1,1,1],[0.5,0.5,0.5],[0.6,0.5,0.4])
        self.assertEqual((r["positive_pairs"],r["equal_pairs"],r["negative_pairs"]),(1,1,1))

    def test_reused_results_reproduce_recorded_statistics(self):
        scores = {}
        for path in (h2.RUN/"scores").glob("*.json"):
            for r in c.read(path)["rows"]:
                if r["prompt_id"] in self.p["prompt_ids"] and r["metric"] in self.p["metrics"]:
                    scores[(r["prompt_id"],r["arm"],r["metric"])] = float(r["score"])
        mapping = {cfg["source_variant"]:v for v,cfg in self.p["reused_variants"].items()}
        for metric in self.p["metrics"]:
            for r in c.read(previous.RUN/"scores"/(metric+".json"))["rows"]:
                scores[(r["prompt_id"],mapping[r["variant"]],metric)] = float(r["score"])
        variants = self.p["reused_variants"]
        actual,_ = summarize_scores(dict(self.p,variants=variants),self.refs,scores)
        original = c.read(previous.RESULT/"summary.json")["metrics"]
        for r in actual:
            old = next(x for x in original if x["metric"]==r["metric"] and mapping[x["variant"]]==r["variant"])
            for key in ("T50","S5","corrected","corrected_minus_student","paired_bootstrap_95_reference",
                        "positive_pairs","equal_pairs","negative_pairs"):
                np.testing.assert_allclose(r[key],old[key],rtol=0,atol=1e-14)
        groups,rows,motion = report.subgroup_summaries(self.p,self.refs,scores,variants)
        self.assertEqual([len(x) for x in groups["subject_consistency"].values()],[21,11,0])
        self.assertEqual([len(x) for x in groups["multiple_objects"].values()],[8,3,5])
        delta = [r["corrected_minus_student"] for r in rows if r["metric"]=="multiple_objects" and r["baseline_group"]=="teacher_higher"]
        np.testing.assert_allclose(delta,[0.03125,0.0390625],rtol=0,atol=1e-14)
        self.assertTrue(all(r["n"]==21 and r["corrected"]==20/21 for r in motion))

    def test_matched_lambda_position_contrast(self):
        variants = c.all_variants(self.p)
        scores = {}
        for pid in self.p["prompt_ids"]:
            for m in self.refs[pid]["metrics"]:
                for v,cfg in variants.items():
                    scores[(pid,v,m)] = cfg["strength"]+0.01*cfg["intervention_step"]
        contrasts = report.timing_contrasts(self.p,self.refs,scores,variants)
        self.assertEqual(len(contrasts),9)
        for r in contrasts:
            self.assertAlmostEqual(r["third_minus_first"],0.02,places=13)
            self.assertEqual(r["n"],16 if r["metric"]=="multiple_objects" else 32)

    def test_controller_four_worker_limit_and_conflict_resume_without_processes(self):
        # Exercise the real controller loop with in-memory subprocesses/files only.
        plan = copy.deepcopy(self.p)
        plan["prompt_ids"] = plan["prompt_ids"][:8]
        now, completed, launched, saved = [1000.0], set(), [], {}
        snap = {"gpus":[[str(i),f"GPU-{i}","0","143771","0"] for i in range(8)],
                "processes":[["GPU-0","123","1000"]]}
        peak = [0]
        class FakeProcess:
            def __init__(self,command,**kwargs):
                self.pid = 10000+len(launched)
                self.returncode = None
                flag = "--prompt" if "--prompt" in command else "--metric"
                self.key = ("gen_" if flag=="--prompt" else "eval_")+command[command.index(flag)+1]
                self.gpu = int(command[command.index("--gpu")+1])
                self.start = now[0]
                self.conflict = len(launched)==0
                launched.append(self)
                peak[0] = max(peak[0],sum(x.returncode is None for x in launched))
            def poll(self):
                if self.returncode is None and now[0]-self.start >= 30:
                    self.returncode = 75 if self.conflict else 0
                    if not self.conflict:
                        completed.add(self.key)
                return self.returncode
        def exists(path):
            if path.parent == c.RUN/"scores":
                return "eval_"+path.stem in completed
            return str(path) in saved
        def save(path,obj,**kwargs):
            saved[str(path)] = copy.deepcopy(obj)
        with contextlib.ExitStack() as stack:
            for target,name,value in (
                (c,"check_frozen",lambda:({},plan)),
                (c,"generation_complete",lambda p,pid:"gen_"+pid in completed),
                (c,"write_json",save),(c,"read",lambda path:saved[str(path)]),
                (c,"ticks",lambda pid:"1"),(c,"alive",lambda pid,birth:False),
                (h2,"gpu_snapshot",lambda:snap),(h2,"host_available_gib",lambda:2048),
                (controller.time,"time",lambda:now[0]),
                (controller.time,"sleep",lambda seconds:now.__setitem__(0,now[0]+seconds)),
                (controller.subprocess,"Popen",FakeProcess),
                (controller.subprocess,"call",lambda *a,**k:0)):
                stack.enter_context(patch.object(target,name,value))
            stack.enter_context(patch.object(Path,"exists",exists))
            stack.enter_context(patch.object(Path,"open",lambda *a,**k:io.BytesIO()))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            controller.main()
        state = saved[str(c.RUN/"controller.json")]
        self.assertEqual(state["status"],"completed")
        self.assertEqual(peak[0],4)
        self.assertEqual(len(completed),11)
        retried = [x for x in launched if x.key==launched[0].key]
        self.assertEqual(len(retried),2)
        self.assertNotEqual(retried[0].gpu,retried[1].gpu)
        self.assertTrue(all(x.gpu!=0 for x in launched))


if __name__ == "__main__":
    result = unittest.main(verbosity=2,exit=False).result
    assert "torch" not in sys.modules, "These checks must not import Torch or initialize CUDA"
    print("CPU_ONLY: no Torch import, model loading, GPU computation, or real worker dispatch")
    raise SystemExit(0 if result.wasSuccessful() else 1)
