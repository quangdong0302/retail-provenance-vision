"""Keep test defaults and generated state away from the user's dataset/artifacts."""
import atexit
import os
from pathlib import Path
import tempfile

_workspace = tempfile.TemporaryDirectory(prefix='rpv_test_runtime_', ignore_cleanup_errors=True)
_root = Path(_workspace.name)
_overrides = {
    'RPV_DATA_DIR': str(_root / 'dataset'),
    'RPV_DB': str(_root / 'rpv.db'),
    'RPV_BANK': str(_root / 'prototypes.json'),
    'RPV_CALIB': str(_root / 'calibration.json'),
    'RPV_AUDIT': str(_root / 'audit.jsonl'),
    'RPV_REPORTS': str(_root / 'reports'),
    'RPV_CACHE_DIR': str(_root / 'embed_cache'),
    'RPV_HEAD_OUT': str(_root / 'head.pt'),
    'RPV_PROJECTION_CKPT': '',
    'RPV_ENCODER': 'classical',
    'RPV_DECISION_RULE': 'conformal',
    'RPV_PROTOTYPE_COLOR_WEIGHT': '0',
    'RPV_EXTERNAL_ADAPTERS': '0',
}
_previous = {key: os.environ.get(key) for key in _overrides}
os.environ.update(_overrides)

# Also cover a test runner which imported runtime modules before discovery.
from rpv.config import reload_settings
reload_settings()


def _cleanup():
    for key, value in _previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    reload_settings()
    _workspace.cleanup()


atexit.register(_cleanup)
