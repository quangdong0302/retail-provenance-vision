"""Real API/demo writers must not lose another process's registered products."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from PIL import Image

from rpv.audit.hash_chain import AuditLog
from rpv.vision.encoders import ClassicalEncoder
from rpv.vision.prototypes import PrototypeBank


ROOT = Path(__file__).resolve().parents[1]
WORKER = r'''
import sys
import time
from pathlib import Path
from rpv.audit.hash_chain import AuditLog
from rpv.provenance.repository import ProvenanceRepository
from rpv.services.checkout import CheckoutService
from rpv.vision.encoders import ClassicalEncoder
from rpv.vision.prototypes import PrototypeBank

root, worker = Path(sys.argv[1]), sys.argv[2]
encoder = ClassicalEncoder()
bank = PrototypeBank(encoder).load(root / 'bank.json')
repo = ProvenanceRepository(root / (worker + '.db'))
repo.init_schema()
service = CheckoutService(encoder, bank, repo, audit=AuditLog(root / 'audit.jsonl'),
                          bank_path=root / 'bank.json')
embed = encoder.embed_path
def slow_embed(path):
    time.sleep(.25)  # Enlarge the interval between reading and saving the bank.
    return embed(path)
encoder.embed_path = slow_embed
(root / (worker + '.ready')).touch()
sys.stdin.readline()
service.register_product(worker, [str(root / 'sample.png')])
repo.close()
'''


class BankProcessTests(unittest.TestCase):
    def test_concurrent_registrations_preserve_every_product(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            PrototypeBank(ClassicalEncoder()).save(root / 'bank.json')
            Image.new('RGB', (128, 128), 'red').save(root / 'sample.png')
            processes = []
            try:
                for worker in range(4):
                    processes.append(subprocess.Popen(
                        [sys.executable, '-B', '-c', WORKER, str(root), str(worker)],
                        cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT), RPV_EMBED_CACHE='0'),
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, encoding='utf-8',
                    ))
                deadline = time.monotonic() + 30
                while len(list(root.glob('*.ready'))) < 4:
                    if any(p.poll() is not None for p in processes) or time.monotonic() > deadline:
                        self.fail('Registration workers failed to start')
                    time.sleep(.01)
                for process in processes:
                    process.stdin.write('start\n')
                    process.stdin.flush()
                for process in processes:
                    stdout, stderr = process.communicate(timeout=45)
                    self.assertEqual(process.returncode, 0, stdout + stderr)
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                    process.communicate()
            bank = PrototypeBank(ClassicalEncoder()).load(root / 'bank.json')
            self.assertEqual(bank.sku_ids, ['0', '1', '2', '3'])
            self.assertTrue(all(e.requires_operator for e in bank.entries.values()))
            audit = AuditLog(root / 'audit.jsonl')
            self.assertTrue(audit.verify()['valid'])
            self.assertEqual(len(audit), 4)
