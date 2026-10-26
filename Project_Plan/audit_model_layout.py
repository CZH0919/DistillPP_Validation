#!/usr/bin/env python3
"""Read headers and meta tensors only. No inference, CUDA, downloads or weight writes."""
import json
import os
from pathlib import Path
import struct
import sys

PROJECT = Path('/data/chenjiayu/changzihe/DistillPP_Validation')
MODELS = Path('/data/chenjiayu/models')


def header(path):
    with path.open('rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        if not 0 < n < 4 * 1024 * 1024:
            raise ValueError('Unexpected header size: ' + str(path))
        return json.loads(f.read(n)), 8 + n


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    import torch
    torch.set_num_threads(1)
    # Import only model.py and its relative attention dependency. Importing the
    # top-level wan package imports T5 whose default argument queries CUDA.
    import importlib.util
    import types
    def reject_cuda_init(*args, **kwargs):
        raise RuntimeError('CUDA initialization prohibited in this metadata audit')
    torch.cuda._lazy_init = reject_cuda_init
    package = types.ModuleType('_wan_layout_audit')
    package.__path__ = [str(PROJECT / 'src/Wan2.1/wan/modules')]
    sys.modules[package.__name__] = package
    module_spec = importlib.util.spec_from_file_location(
        '_wan_layout_audit.model', PROJECT / 'src/Wan2.1/wan/modules/model.py')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    WanModel = module.WanModel
    student = MODELS / 'Wan2.1-T2V-14B-StepDistill-CfgDistill'
    teacher = MODELS / 'Wan2.1-T2V-14B-Diffusers/transformer'
    config = json.loads((student / 'config.json').read_text())
    with torch.device('meta'):
        expected = WanModel.from_config(config).state_dict()
    assert all(v.device.type == 'meta' for v in expected.values())
    # mmap avoids materializing the 28 GB payload; weights_only restricts unpickling.
    pt = torch.load(student / 'distill_model.pt', map_location='meta', weights_only=True, mmap=True)
    assert set(pt) == set(expected), 'Student key set differs from native WanModel'
    assert all(tuple(pt[k].shape) == tuple(v.shape) for k, v in expected.items())
    assert all(v.device.type == 'meta' for v in pt.values())
    spec = json.loads((Path(__file__).parent / 'WAN_KEY_RENAME_RULES.json').read_text())
    assert spec['special_keys_remap'] == {}
    mapping = {}
    for native in expected:
        key = native
        for old, new in spec['rename_pairs']:
            key = key.replace(old, new)
        assert key not in mapping, 'Rename collision'
        mapping[key] = native
    index = json.loads((teacher / 'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
    assert set(index) == set(mapping), 'Teacher index key coverage differs'
    observed, locations = {}, {}
    for name in sorted(set(index.values())):
        assert Path(name).name == name
        path = teacher / name
        h, offset = header(path)
        for key, entry in h.items():
            if key == '__metadata__':
                continue
            assert key not in observed and index[key] == name
            assert entry['shape'] == list(expected[mapping[key]].shape), key
            assert entry['dtype'] == 'F32', (key, entry['dtype'])
            observed[key] = entry
            locations[key] = (path, offset)
    assert set(observed) == set(mapping)
    # Optional bounded byte samples from the mistakenly downloaded first shard.
    # This is a spot check, never a proof of full tensor equality.
    native_path = MODELS / 'Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors'
    sample_count, sample_bytes = 0, 0
    if native_path.is_file():
        h, native_offset = header(native_path)
        inverse = {v: k for k, v in mapping.items()}
        for key in sorted(k for k in h if k != '__metadata__'):
            entry = h[key]
            mapped = inverse[key]
            other = observed[mapped]
            assert entry['dtype'] == other['dtype'] and entry['shape'] == other['shape']
            length = entry['data_offsets'][1] - entry['data_offsets'][0]
            path, offset = locations[mapped]
            for displacement in sorted(set([0, max(0, length - 64)])):
                size = min(64, length)
                with native_path.open('rb') as f:
                    f.seek(native_offset + entry['data_offsets'][0] + displacement)
                    a = f.read(size)
                with path.open('rb') as f:
                    f.seek(offset + other['data_offsets'][0] + displacement)
                    b = f.read(size)
                assert a == b, 'Byte sample mismatch: ' + key
                sample_bytes += size
            sample_count += 1
    assert not torch.cuda.is_initialized()
    print(json.dumps({'ok': True, 'teacher_shards': len(set(index.values())),
          'native_keys': len(expected), 'student_keys': len(pt), 'teacher_keys': len(observed),
          'teacher_dtype': 'F32', 'student_dtypes': sorted(set(str(v.dtype) for v in pt.values())),
          'mapping_bijective': True, 'all_shapes_match': True,
          'native_shard_spot_checked_tensors': sample_count,
          'native_shard_sample_bytes_per_copy': sample_bytes,
          'cuda_initialized': torch.cuda.is_initialized(),
          'full_weight_hashes_checked': False, 'real_model_load_or_forward_tested': False}, indent=2))


if __name__ == '__main__':
    main()
