"""The fresh training workflow must not load a deleted/previous head."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_all


class RunnerTests(unittest.TestCase):
    def test_fresh_training_uses_raw_embeddings_then_new_head(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text('{}', encoding='utf-8')
            new_head = str(root / 'head.pt')
            calls = []

            def record(*args, env):
                calls.append((args, env.get('RPV_PROJECTION_CKPT')))

            with patch.dict(os.environ, {'RPV_DATA_DIR': directory,
                                         'RPV_PROJECTION_CKPT': 'deleted-head.pt',
                                         'RPV_HEAD_OUT': new_head}), \
                    patch('sys.argv', ['run_all.py', '--encoder', 'dinov2', '--train-head']), \
                    patch.object(run_all, 'run', side_effect=record), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(run_all.main(), 0)
            checkpoints = {args[2]: ckpt for args, ckpt in calls if args[:2] == ('-m', 'rpv.cli')}
            self.assertEqual(checkpoints['warm-cache'], '')
            self.assertEqual(checkpoints['train-head'], '')
            self.assertEqual(checkpoints['calibrate'], new_head)
            self.assertEqual(checkpoints['evaluate'], new_head)

    def test_missing_real_manifest_stops_before_any_pipeline_step(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {'RPV_DATA_DIR': directory}), \
                patch('sys.argv', ['run_all.py']), \
                patch.object(run_all, 'run') as run, \
                patch.object(run_all.argparse.ArgumentParser, 'error', side_effect=ValueError), \
                redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                run_all.main()
            run.assert_not_called()
