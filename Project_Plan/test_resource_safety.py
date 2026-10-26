"""Small regression checks: shared-model downloads/writes must be impossible."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('downloader', Path(__file__).with_name('download_resources.py'))
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


class ResourceSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='.resource-test-', dir=Path(__file__).resolve().parent)
        assert Path(self.tmp.name).resolve().parent == Path(__file__).resolve().parent
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        d.MODELS = self.root / 'models'
        d.PROJECT = self.root / 'project'
        d.MODELS.mkdir()
        d.PROJECT.mkdir()
        self.item = {'id': 'teacher', 'group': 'models', 'target_root': 'models',
                     'path': 'existing/model.bin', 'bytes': 3, 'sha256': hashlib.sha256(b'abc').hexdigest()}

    def test_missing_shared_model_never_downloads_or_creates_directory(self):
        # No reuse_only flag, as in the obsolete manifest: hard boundary still holds.
        with patch.object(d.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'REUSE_ONLY'):
                d.one(self.item, 'https://hf-mirror.com', False)
            run.assert_not_called()
        self.assertEqual(list(d.MODELS.iterdir()), [])

    def test_mismatch_preserves_existing_bytes(self):
        path = d.MODELS / self.item['path']
        path.parent.mkdir()
        path.write_bytes(b'bad')
        with patch.object(d.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'Checksum mismatch'):
                d.one(self.item, 'https://hf-mirror.com', False)
            run.assert_not_called()
        self.assertEqual(path.read_bytes(), b'bad')

    def test_existing_model_verified_without_network(self):
        path = d.MODELS / self.item['path']
        path.parent.mkdir()
        path.write_bytes(b'abc')
        with patch.object(d.subprocess, 'run') as run:
            self.assertEqual(d.one(self.item, '', False)['status'], 'verified_existing')
            run.assert_not_called()

    def test_evaluation_download_and_rerun_skip(self):
        item = dict(self.item, group='evaluation', target_root='project', path='resources/evaluation/file.bin', repo_id='org/repo', revision='commit', repo_file='file.bin')
        def fake_curl(cmd, **kwargs):
            Path(cmd[cmd.index('--output')+1]).write_bytes(b'abc')
            return type('Result', (), {'returncode': 0})()
        with patch.object(d.subprocess, 'run', side_effect=fake_curl) as run:
            self.assertEqual(d.one(item, 'https://hf-mirror.com', False)['status'], 'downloaded_verified')
            self.assertEqual(d.one(item, 'https://hf-mirror.com', False)['status'], 'verified_existing')
            self.assertEqual(run.call_count, 1)


if __name__ == '__main__':
    unittest.main()
