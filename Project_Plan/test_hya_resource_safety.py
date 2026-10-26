"""Small local filesystem tests for shared-resource no-clobber boundaries."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('hya_download', HERE / 'download_hunyuan_accvideo_resources.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Safety(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='hya_safety_')
        self.root = Path(self.temp.name).resolve()
        self.roots = patch.multiple(module, MODELS=self.root, PROJECT=self.root)
        self.roots.start()
        self.item = {'id': 'synthetic', 'path': 'AccVideo/a.pt', 'target_root': 'models',
                     'bytes': 4, 'sha256': hashlib.sha256(b'good').hexdigest(), 'action': 'download_if_missing'}

    def tearDown(self):
        self.roots.stop()
        self.temp.cleanup()

    def test_missing_reuse_never_downloads_or_creates(self):
        self.item['action'] = 'reuse_only'
        with patch.object(module.subprocess, 'run', side_effect=AssertionError('Network must not run')):
            with self.assertRaisesRegex(RuntimeError, 'REUSE_ONLY'):
                module.download_one(self.item, 'https://hf-mirror.com')
        self.assertFalse((self.root / 'AccVideo').exists())

    def test_wrong_existing_hash_is_preserved(self):
        path = self.root / self.item['path']
        path.parent.mkdir()
        path.write_bytes(b'evil')
        with patch.object(module.subprocess, 'run', side_effect=AssertionError('Network must not run')):
            with self.assertRaisesRegex(RuntimeError, 'Hash mismatch'):
                module.download_one(self.item, 'https://hf-mirror.com')
        self.assertEqual(path.read_bytes(), b'evil')

    def test_existing_valid_file_does_not_download(self):
        path = self.root / self.item['path']
        path.parent.mkdir()
        path.write_bytes(b'good')
        with patch.object(module.subprocess, 'run', side_effect=AssertionError('Network must not run')):
            module.download_one(self.item, 'https://hf-mirror.com')

    def test_publication_never_clobbers(self):
        partial, target = self.root / 'sample.part', self.root / 'published.pt'
        partial.write_bytes(b'good')
        target.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):
            module.publish(partial, target)
        self.assertEqual(target.read_bytes(), b'keep')
        self.assertEqual(partial.read_bytes(), b'good')

    def test_manifest_blocks_teacher_download_and_path_escape(self):
        baseline = json.loads((HERE / 'HUNYUAN_ACCVIDEO_RESOURCES.json').read_text(encoding='utf-8'))
        module.check_manifest(baseline)
        for wrong_path, wrong_action in [('HunyuanVideo/config.json', 'download_if_missing'), ('../other/a.pt', 'download_if_missing')]:
            item = dict(baseline['assets'][0], path=wrong_path, action=wrong_action)
            with self.assertRaises(RuntimeError):
                module.check_manifest(dict(baseline, assets=[item]))

    def test_redirected_parent_rejected(self):
        # Deterministic Windows-safe test of the realpath guard, no admin symlink creation.
        with patch.object(Path, 'resolve', return_value=self.root / 'outside'):
            with self.assertRaisesRegex(RuntimeError, 'Redirected target parent'):
                module.checked_path(self.item)


if __name__ == '__main__':
    unittest.main()
