"""Exercise guided enrollment with real images and a copied bank; never publish into the live catalog."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

from rpv import config as app_config
from rpv.services.enrollment import EnrollmentService
from rpv.vision.encoders import DinoV2Encoder
from rpv.vision.prototypes import PrototypeBank


def main():
    encoder = DinoV2Encoder(checkpoint=app_config.settings.paths.head_path)
    bank_path = app_config.settings.paths.bank_path
    before = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    bank = PrototypeBank(encoder).load(bank_path)
    split = json.loads(Path(app_config.settings.paths.head_path).with_suffix('.splits.json').read_text(encoding='utf-8'))
    label = next(c for c in split['novel_support'] if c not in bank.entries)
    output = ROOT / 'var/thesis/enrollment_smoke' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    service = EnrollmentService(encoder, bank, output)
    session = service.create(label, label, 'smoke_only_unverified_capture_groups')['id']
    attempts = []

    def add(role, paths, negative_sku=''):
        for i, path in enumerate(paths):
            attempts.append(service.add_images(session, role, [app_config.settings.paths.data_dir / path],
                f'smoke_unverified_{role}_{negative_sku}_{i}', negative_sku))

    add('support', split['novel_support'][label][:3])
    first = service.assess(session)
    add('validation', split['novel_test'][label][:10])
    for rival in first['report']['nearest_products']:
        add('negative', split['test'][rival['sku_id']][:4], rival['sku_id'])
    service.assess(session)
    add('support', split['novel_support'][label][3:5])
    final = service.assess(session)
    assert hashlib.sha256(bank_path.read_bytes()).hexdigest() == before
    assert final['report']['operating_mode'] == 'operator_required'
    assert len(final['history']) == 3
    result = {'status': 'passed', 'scope': 'smoke_only',
              'note': 'Dataset images exercise code only. Capture groups are unverified; this is not an enrollment benchmark.',
              'live_bank_unchanged': True, 'encoder': encoder.tag, 'session_id': session,
              'accepted': sum(len(a['accepted']) for a in attempts),
              'rejected': [r for a in attempts for r in a['rejected']], 'session': final}
    (output / 'smoke.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'passed', 'report': str(output / 'smoke.json'),
                      'assessment': final['report']['status'], 'accepted': result['accepted'],
                      'rejected': len(result['rejected'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
