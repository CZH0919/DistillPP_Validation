"""CPU-only tests; no Torch import, model load, GPU query or actual worker."""
import contextlib
import copy
import csv
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import hya_h2_common as h2
import hya_h3_timing_common as completed
import hya_h3_step24_common as c
import hya_h3_step24_controller as controller
import hya_h3_step24_finalize as report
from hya_h3_math import endpoints,renoise,correct_euler
from hya_h3_finalize import summarize_scores


class Step24Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=c.read(c.PLAN)
        cls.base=c.read(h2.RUN/"protocol.json")
        cls.prior=c.read(completed.RUN/"protocol.json")
        cls.refs={r["id"]:r for r in cls.base["prompts"]}

    def test_exact_remaining_matrix_and_rotation(self):
        c.validate_protocol(self.p,self.base,self.prior)
        self.assertEqual(len(c.all_variants(self.p)),12)
        first=[c.variants_for(self.p,pid)[0] for pid in self.p["prompt_ids"]]
        self.assertEqual({v:first.count(v) for v in self.p["variants"]},{v:8 for v in self.p["variants"]})
        self.assertEqual(c.intervention_settings(self.p,"step2_lambda_0p3"),(1,0.3))
        self.assertEqual(c.intervention_settings(self.p,"step4_lambda_0p5"),(3,0.5))
        self.assertIsNone(self.p["max_workers"])

    def test_all_four_positions_formula_and_terminal_boundary(self):
        sigmas=c.read(c.ROOT/"Project_Plan/HYA_RUNTIME_CONTRACT.json")["schedule"]["5"]["sigmas"]
        z,v,xi,vt=np.random.default_rng(24).normal(size=(4,64))
        for step in range(4):
            sigma,r=sigmas[step:step+2]
            xs,eps=endpoints(z,v,sigma)
            base=z+(r-sigma)*v
            query=renoise(xs,xi,r)
            xt,_=endpoints(query,vt,r)
            for strength in (0.1,0.3,0.5):
                np.testing.assert_allclose(correct_euler(base,xs,xt,r,strength),
                    (1-r)*((1-strength)*xs+strength*xt)+r*eps,rtol=1e-13,atol=1e-13)
        self.assertAlmostEqual(sigmas[2],0.9130434393882751)
        self.assertAlmostEqual(sigmas[4],0.6363636255264282)
        with self.assertRaises(ValueError):
            renoise(z,xi,sigmas[5])

    def test_launch_guards_have_no_resource_or_filesystem_effects(self):
        plan=copy.deepcopy(self.p)
        with patch.object(c,"read",lambda path:plan), \
             patch.object(Path,"mkdir",side_effect=AssertionError("must not create a run")), \
             patch.object(h2,"gpu_snapshot",side_effect=AssertionError("must not query GPUs")), \
             patch.object(controller.subprocess,"Popen",side_effect=AssertionError("must not launch")):
            plan["authorization"]["execution"]=False
            with self.assertRaises(RuntimeError):c.prepare(2)
            with self.assertRaises(RuntimeError):controller.launch_controller()
            plan["authorization"]["execution"]=True
            with self.assertRaises(ValueError):c.prepare()
        for value in (None,0,True,9):
            with self.assertRaises(ValueError):c.validate_worker_limit(value)
        self.assertEqual(c.validate_worker_limit(1),1)
        self.assertEqual(c.validate_worker_limit(5),5)

    def test_six_reused_sources_and_common_noise(self):
        for pid in self.p["prompt_ids"]:
            expected=c.common_noise_seed(self.p,pid)
            for variant,cfg in self.p["reused_variants"].items():
                r=c.read(c.reused_record_path(pid,cfg))
                self.assertEqual(r["strength"],cfg["strength"])
                self.assertEqual(r["intervention_step"],cfg["intervention_step"])
                self.assertEqual(r["intervention"]["renoise_rng_seed"],expected)

    def combined_fixture(self):
        # Only synthetic new score rows; completed 640 rows are read unchanged.
        with (completed.RESULT/"scores.csv").open(encoding="utf-8",newline="") as f:
            legacy={(r["prompt_id"],r["group"],r["metric"]):float(r["score"]) for r in csv.DictReader(f)}
        raw={m:{"status":"passed","rows":[]} for m in self.p["metrics"]}
        records={}
        for pid in self.p["prompt_ids"]:
            for variant in self.p["variants"]:
                sha="synthetic_"+pid+variant
                records[(pid,variant)]={"video":{"sha256":sha}}
                for m in self.refs[pid]["metrics"]:
                    if m in raw:
                        raw[m]["rows"].append({"prompt_id":pid,"variant":variant,"metric":m,
                            "score":legacy[(pid,"S5",m)],"video_sha256":sha})
        with patch.object(c,"read",lambda path:raw[Path(path).stem]):
            scores,rows=report.load_scores(self.p,self.refs,records)
        return scores,rows

    def test_merge_1120_scores_without_regenerating_prior_arms(self):
        scores,rows=self.combined_fixture()
        self.assertEqual(len(rows),1120)
        self.assertEqual(sum(r["source"]=="new step2/step4 correction" for r in rows),480)
        variants=c.all_variants(self.p)
        summaries,_=summarize_scores(dict(self.p,variants=variants),self.refs,scores)
        old=c.read(completed.RESULT/"summary.json")["metrics"]
        self.assertEqual(len(summaries),36)
        for r in summaries:
            if r["variant"] in self.p["variants"]:
                self.assertEqual(r["corrected_minus_student"],0)
            else:
                match=next(x for x in old if x["metric"]==r["metric"] and x["variant"]==r["variant"])
                for key in ("T50","S5","corrected","corrected_minus_student"):
                    self.assertAlmostEqual(r[key],match[key],places=14)
        arms=["T50","S5"]+list(variants)
        medians={arm:{"compute_seconds":1.,"inference_with_transfers_seconds":2.} for arm in arms}
        text=report.render_report(self.p,summaries,[],medians,{v:1. for v in variants})
        self.assertIn("288条视频、480个新分数",text)
        self.assertIn("主要依据是每个指标完整任务库的均值",text)
        for step in range(1,5):
            for strength in (0.1,0.3,0.5):self.assertIn(f"第{step}步 λ={strength}",text)

    def test_fixed_lambda_all_position_pairs(self):
        variants=c.all_variants(self.p)
        scores={(pid,v,m):cfg["strength"]+0.01*cfg["intervention_step"]
                for pid in self.p["prompt_ids"] for m in self.refs[pid]["metrics"]
                for v,cfg in variants.items()}
        rows=report.position_contrasts(self.p,self.refs,scores,variants)
        self.assertEqual(len(rows),54)
        for r in rows:
            self.assertAlmostEqual(r["second_minus_first"],0.01*(r["second_step"]-r["first_step"]),places=13)
            self.assertEqual(r["n"],16 if r["metric"]=="multiple_objects" else 32)

    def test_runtime_concurrency_is_parameterized(self):
        for limit in (2,5):
            with self.subTest(simulated_limit=limit):
                self.simulate_controller(limit)


    def simulate_controller(self, worker_limit):
        # Exercise the real controller loop with in-memory subprocesses/files only.
        plan = copy.deepcopy(self.p)
        plan["max_workers"] = worker_limit
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
        self.assertEqual(peak[0],worker_limit)
        self.assertEqual(len(completed),11)
        retried = [x for x in launched if x.key==launched[0].key]
        self.assertEqual(len(retried),2)
        self.assertNotEqual(retried[0].gpu,retried[1].gpu)
        self.assertTrue(all(x.gpu!=0 for x in launched))


if __name__=="__main__":
    result=unittest.main(verbosity=2,exit=False).result
    assert "torch" not in sys.modules
    print("CPU_ONLY: no Torch import, model loading, GPU query or real worker dispatch")
    raise SystemExit(0 if result.wasSuccessful() else 1)
