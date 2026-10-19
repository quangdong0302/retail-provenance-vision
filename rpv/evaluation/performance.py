"""Serial batch-one POS benchmarks, using an isolated repository/audit log."""
from __future__ import annotations

from dataclasses import replace
import platform
import tempfile
from pathlib import Path

import numpy as np

from rpv import config as app_config
from rpv.audit.hash_chain import AuditLog
from rpv.provenance.repository import ProvenanceRepository
from rpv.services.checkout import CheckoutService

BUDGETS_MS = {'vision_search': 150.0, 'policy_lookup': 50.0, 'total': 300.0}


def summarize_timings(rows):
    if not rows:
        raise ValueError('Need at least one measured checkout.')
    summary = {}
    for stage, budget in BUDGETS_MS.items():
        values = np.asarray([row[stage] for row in rows], dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Timings must be finite and nonnegative.')
        summary[stage] = {'n': len(values), 'mean_ms': float(values.mean()),
                          'p50_ms': float(np.quantile(values, .5)), 'p95_ms': float(np.quantile(values, .95)),
                          'p99_ms': float(np.quantile(values, .99)), 'budget_ms': budget,
                          'meets_budget_p95': bool(np.quantile(values, .95) < budget)}
    return summary


def benchmark_checkout(service, images, repeats=20, warmup=3, claimed_sku=None, claimed_lot=None):
    images = list(images)
    if not images or repeats < 1 or warmup < 0:
        raise ValueError('Need images, positive repeats and nonnegative warmup.')
    original = app_config.settings
    hardware = {'os': platform.platform(), 'cpu': platform.processor(),
                'logical_cpu_count': __import__('os').cpu_count(), 'encoder': service.encoder.describe(),
                'n_catalog_classes': len(service.bank),
                'n_centroids': sum(len(entry.centroids) for entry in service.bank.entries.values()),
                'use_tta': original.vision.use_tta}
    if platform.system() == 'Windows':
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                        ('total_physical', ctypes.c_ulonglong), ('available_physical', ctypes.c_ulonglong),
                        ('total_pagefile', ctypes.c_ulonglong), ('available_pagefile', ctypes.c_ulonglong),
                        ('total_virtual', ctypes.c_ulonglong), ('available_virtual', ctypes.c_ulonglong),
                        ('available_extended_virtual', ctypes.c_ulonglong)]

        memory = MemoryStatus()
        memory.length = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            hardware['ram_gib'] = memory.total_physical / (1024 ** 3)
    try:
        import torch
        hardware.update(torch_version=torch.__version__, cuda_available=torch.cuda.is_available(),
                        torch_threads=torch.get_num_threads())
        if torch.cuda.is_available():
            hardware['gpu'] = torch.cuda.get_device_name()
    except ImportError:
        pass
    modes = {}
    try:
        with tempfile.TemporaryDirectory(prefix='rpv_benchmark_') as folder:
            root = Path(folder)
            for mode, use_cache in [('uncached', False), ('cached', True)]:
                app_config.settings = replace(original, vision=replace(original.vision, use_cache=use_cache))
                with ProvenanceRepository(root / f'{mode}.db') as repo:
                    service.repo.conn.backup(repo.conn)
                    isolated = CheckoutService(service.encoder, service.bank, repo, service.calibration,
                                               policy=service.policy, audit=AuditLog(root / f'{mode}.jsonl'))
                    for i in range(warmup):
                        isolated.process(images[i % len(images)], claimed_sku, claimed_lot, operator='benchmark_warmup')
                    if use_cache:
                        service.encoder.embed_paths(images)
                    rows = [isolated.process(images[i % len(images)], claimed_sku, claimed_lot, operator='benchmark').timings_ms
                            for i in range(repeats)]
                    modes[mode] = {'summary': summarize_timings(rows), 'samples_ms': rows,
                                   'audit_valid': isolated.audit.verify()['valid']}
    finally:
        app_config.settings = original
    return {'hardware': hardware, 'modes': modes, 'warmup': warmup, 'n_images': len(images),
            'batch_size': 1, 'claimed_sku': claimed_sku, 'claimed_lot': claimed_lot,
            'scope': 'Warm model, serial CheckoutService including DB/audit writes; excludes camera, HTTP upload and model startup.',
            'target_pos': ('Proposed: x86-64 CPU >=4 cores, RAM 16 GB, SSD, NVIDIA GPU >=4 GB; '
                           f'{original.vision.encoder} {original.vision.dinov2_variant} '
                           f'{original.vision.image_size}px. Validate on the actual POS.')}
