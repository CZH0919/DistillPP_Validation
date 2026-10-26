"""CPU checks of the actual fixed four-step scheduler and reproducibility contract."""
import json
import os
import ast
import typing

from s1_common import PROJECT, PROMPT, SEED, SHAPE, configure, json_write, file_hash


def main():
    configure()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(4)
    def no_cuda(*args, **kwargs):
        raise RuntimeError('CUDA prohibited in CPU contract check')
    torch.cuda._lazy_init = no_cuda
    # Top-level LightX2V import requires an available CUDA platform. Isolate
    # only these unchanged numerical source methods for the CPU unit check;
    # the full native class is separately exercised by the real student run.
    source = PROJECT / 'src/LightX2V/lightx2v/models/schedulers/wan/step_distill/scheduler.py'
    tree = ast.parse(source.read_text())
    native = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WanStepDistillScheduler')
    methods = [n for n in native.body if isinstance(n, ast.FunctionDef) and n.name in ['set_denoising_timesteps', 'step_post']]
    assert len(methods) == 2
    cls = ast.ClassDef(name='WanStepDistillScheduler', bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = {'torch': torch, 'Union': typing.Union}
    exec(compile(module, str(source), 'exec'), namespace)
    WanStepDistillScheduler = namespace['WanStepDistillScheduler']
    scheduler = object.__new__(WanStepDistillScheduler)
    scheduler.sigma_min = 0.0
    scheduler.sigma_max = 1.0
    scheduler.sample_shift = 5
    scheduler.num_train_timesteps = 1000
    scheduler.denoising_step_list = [1000, 750, 500, 250]
    scheduler.infer_steps = 4
    scheduler.set_denoising_timesteps(device='cpu')
    expected_sigmas = torch.tensor([1, 0.9375, 5/6, 0.625])
    assert torch.allclose(scheduler.sigmas, expected_sigmas, atol=1e-6, rtol=0)
    generator = torch.Generator().manual_seed(123)
    errors = []
    for i in range(4):
        z = torch.randn((2, 3, 4, 5), generator=generator)
        v = torch.randn((2, 3, 4, 5), generator=generator)
        scheduler.step_index = i
        scheduler.latents = z.clone()
        scheduler.noise_pred = v
        nxt = scheduler.sigmas[i+1] if i < 3 else 0
        expected = z + (nxt - scheduler.sigmas[i]) * v
        scheduler.step_post()
        error = float((scheduler.latents - expected).abs().max())
        assert error < 1e-6 and scheduler.latents.dtype == torch.float32
        errors.append(error)
    a = torch.randn(SHAPE, generator=torch.Generator(device='cpu').manual_seed(SEED))
    torch.randn(7)  # unrelated RNG draws cannot alter the dedicated initial generator
    b = torch.randn(SHAPE, generator=torch.Generator(device='cpu').manual_seed(SEED))
    assert torch.equal(a, b)
    # Verify the source of the previously incorrect persistent-FP32 assumption.
    base_source = PROJECT / 'src/LightX2V/lightx2v/models/schedulers/scheduler.py'
    base_tree = ast.parse(base_source.read_text())
    base_class = next(n for n in base_tree.body if isinstance(n, ast.ClassDef) and n.name == 'BaseScheduler')
    pre = next(n for n in base_class.body if isinstance(n, ast.FunctionDef) and n.name == 'step_pre')
    pre_ns = {'GET_DTYPE': lambda: torch.bfloat16, 'GET_SENSITIVE_DTYPE': lambda: torch.bfloat16}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[pre], type_ignores=[])), str(base_source), 'exec'), pre_ns)
    scheduler.latents = a.clone()
    scheduler.keep_latents_dtype_in_scheduler = False
    pre_ns['step_pre'](scheduler, 0)
    assert scheduler.latents.dtype == torch.bfloat16 and torch.equal(scheduler.latents, a.bfloat16())
    panel = json.loads((PROJECT/'Project_Plan/PROMPTS_QUALIFICATION.json').read_text())
    assert PROMPT not in [p['prompt'] for p in panel['prompts']]
    assert not torch.cuda.is_initialized()
    report = {'status': 'passed', 'actual_student_timesteps': scheduler.timesteps.tolist(),
              'test_scope': 'unchanged source methods isolated on CPU; full-class GPU test is separate',
              'source_sha256': file_hash(source),
              'native_step_pre_dtype': str(scheduler.latents.dtype),
              'base_scheduler_source_sha256': file_hash(base_source),
              'precision_note': 'Euler algebra checked in FP32; native persistent state is BF16 after step_pre and step_post',
              'actual_student_sigmas': scheduler.sigmas.tolist(), 'terminal_sigma': 0,
              'euler_equivalence_max_abs_errors': errors, 'independent_rng_passed': True,
              'smoke_prompt_outside_qualification_panel': True, 'cuda_initialized': False}
    json_write(PROJECT/'Project_Progress/Stage_Progress/S1_CPU_CONTRACT.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
