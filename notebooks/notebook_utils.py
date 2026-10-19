"""Các thao tác dùng chung; notebook chỉ trình bày từng chức năng ngắn."""
from __future__ import annotations

import csv
import gc
import hashlib
import html
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager

# Phải đặt trước khi nạp NumPy/Torch trong kernel Conda Windows.
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
# Same working directory and .env loader as scripts/run_all.py.
os.chdir(PROJECT)
from rpv.config import reload_settings


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    rows = list(rows)
    if rows:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def show_table(rows):
    from IPython.display import HTML, display
    rows = list(rows)
    if not rows:
        print('Chưa có kết quả.'); return
    columns = list(rows[0])
    escape = lambda x: html.escape(str(x))
    header = ''.join(f'<th>{escape(c)}</th>' for c in columns)
    body = ''.join('<tr>' + ''.join(f'<td>{escape(row[c])}</td>' for c in columns) + '</tr>' for row in rows)
    display(HTML(f'<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>'))



def project_path(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def validate_dataset():
    """Run the same dataset-info command used by run_all, then lock its split."""
    run = Workflow()
    profile_path = run.data / 'model_profile.json'
    if profile_path.is_file():
        assert read_json(profile_path) == run.state['profile'], 'Dataset profile differs from .env.'
    run.cli('dataset-info', '--out', run.folder / 'splits.json', model='configured')
    run._splits = read_json(run.folder / 'splits.json')
    locked = run.data / f'splits_model_k{run.state["profile"]["k_shot"]}.json'
    if locked.is_file():
        assert run.splits == read_json(locked), 'Split differs from the prepared dataset.'
    return run


def load_workflow():
    """Use the shared .env; no notebook-specific run directory or config."""
    return Workflow()


@contextmanager
def _model_env(env):
    from rpv.config import reload_settings
    values = {k: v for k, v in env.items() if k.startswith('RPV_')}
    previous = {k: os.environ.get(k) for k in values}
    try:
        os.environ.update(values)
        reload_settings()
        yield
    finally:
        for k, value in previous.items():
            if value is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = value
        reload_settings()


class Workflow:

    def __init__(self):
        self.settings = reload_settings()
        self.data = project_path(self.settings.paths.data_dir)
        if not (self.data / 'manifest.json').is_file():
            raise FileNotFoundError('Copy the prepared dataset into data/dataset_v2 first; see data/README.md.')
        self.folder = project_path(self.settings.paths.head_path).parent
        self.head = project_path(self.settings.vision.checkpoint or self.settings.paths.head_path)
        c = self.settings.calibration
        self.state = dict(dataset_fingerprint=read_json(self.data / 'manifest.json')['fingerprint'],
                          profile=dict(k_shot=c.k_shot, class_seed=c.class_split_seed, split_seed=c.calib_test_seed))
        self._splits = None
        self._demo = None

    @property
    def splits(self):
        if self._splits is None:
            from rpv.data.managed import read_managed_dataset, build_managed_splits, split_manifest
            p = self.state['profile']
            self._splits = split_manifest(build_managed_splits(read_managed_dataset(self.data),
                                          p['k_shot'], p['class_seed'], p['split_seed']))
        return self._splits

    def metadata(self):
        with (self.data / 'metadata.csv').open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))

    def show_config(self):
        return dict(data_dir=os.path.relpath(self.data, PROJECT), profile=self.state['profile'],
                    encoder=self.settings.vision.encoder, batch_size=self.settings.vision.batch_size,
                    alpha=self.settings.calibration.alpha,
                    target_risk=self.settings.calibration.target_selective_risk,
                    decision_rule=self.settings.calibration.decision_rule,
                    prototypes_per_sku=self.settings.vision.centroids_per_sku,
                    color_weight=self.settings.vision.prototype_color_weight,
                    vector_cache=str(self.settings.paths.cache_dir),
                    head=os.path.relpath(self.head, PROJECT),
                    bank=str(self.settings.paths.bank_path), calibration=str(self.settings.paths.calib_path))

    def environment(self, model='head'):
        # Inherit the same environment loaded from .env by rpv.config.
        env = os.environ.copy()
        env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1',
                   PYTHONDONTWRITEBYTECODE='1', MKL_THREADING_LAYER='SEQUENTIAL')
        for key, field in (
            ('RPV_DATA_DIR', 'data_dir'), ('RPV_DB', 'db_path'), ('RPV_BANK', 'bank_path'),
            ('RPV_CALIB', 'calib_path'), ('RPV_AUDIT', 'audit_path'), ('RPV_REPORTS', 'reports_dir'),
            ('RPV_CACHE_DIR', 'cache_dir'), ('RPV_HEAD_OUT', 'head_path'),
        ):
            env[key] = str(getattr(self.settings.paths, field))
        if model == 'head':
            env['RPV_PROJECTION_CKPT'] = str(self.head)
        elif model in ('raw', 'classical'):
            # Baselines use the same data/protocol, but separate artifacts so the
            # operational head/bank/calibration from run_all are preserved.
            baseline = project_path(self.settings.paths.reports_dir) / 'baselines' / model
            env.update(RPV_ENCODER='classical' if model == 'classical' else 'dinov2',
                       RPV_PROJECTION_CKPT='', RPV_BANK=str(baseline / 'prototypes.json'),
                       RPV_CALIB=str(baseline / 'calibration.json'), RPV_REPORTS=str(baseline))
            env.update(RPV_CATALOG_BACKEND='prototype', RPV_CONFIDENCE_METHOD='energy',
                       RPV_CONFORMAL_SCORE='distance', RPV_RISK_CALIBRATION='known',
                       RPV_PROTOTYPE_COLOR_WEIGHT='0',RPV_DECISION_RULE='conformal')
        return env

    def cli(self, command, *args, model='raw'):
        logs = self.folder / 'logs'
        logs.mkdir(parents=True, exist_ok=True)
        label = f'{command}_{model}_{time.time_ns()}'
        stdout_path, log_path = logs / f'{label}.json', logs / f'{label}.log'
        argv = [sys.executable, '-B', '-m', 'rpv.cli', command, *map(str, args)]
        print(f'Đang chạy {command} ({model})...', flush=True)
        with stdout_path.open('w', encoding='utf-8') as output, log_path.open('w', encoding='utf-8') as log:
            process = subprocess.Popen(argv, cwd=str(PROJECT), env=self.environment(model), stdout=output,
                                       stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
            try:
                for line in process.stderr:
                    print(line, end='', flush=True)
                    log.write(line); log.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                raise
            finally:
                process.stderr.close()
        if code:
            raise RuntimeError(f'{command} lỗi (exit {code}). Log: {log_path}')
        return read_json(stdout_path)

    def check_environment(self):
        import torch
        device = self.settings.vision.device
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Đã chọn cuda nhưng PyTorch không nhận GPU.')
        return dict(python=sys.executable, torch=str(torch.__version__), cuda=torch.cuda.is_available(),
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    dataset=os.path.relpath(self.data, PROJECT), profile=self.state['profile'], output=os.path.relpath(self.folder, PROJECT))

    def dataset_statistics(self):
        by_class = defaultdict(list)
        for row in self.metadata():
            if row['pool'] == '_extra_unknown' and not (self.data / '_extra_unknown').is_dir():
                continue
            by_class[(row['pool'], row['label'])].append(row)
        rows = []
        for (pool, label), items in sorted(by_class.items()):
            row = dict(pool=pool, class_name=label, n_images=len(items), n_sessions=len({r['group_id'] for r in items}))
            if pool == 'known':
                for role in ('train', 'validation', 'support', 'calib', 'test'):
                    row[role] = len(self.splits[role][label])
            else:
                row.update({role: '' for role in ('train', 'validation', 'support', 'calib', 'test')})
            rows.append(row)
        write_csv(self.folder / 'tables' / 'dataset_counts.csv', rows)
        for pool in ('known', 'unknown', '_extra_unknown'):
            items = [r for r in rows if r['pool'] == pool]
            print(f'{pool}: {len(items)} lớp, {sum(r["n_images"] for r in items):,} ảnh')
        return rows

    def plot_distribution(self):
        import matplotlib.pyplot as plt
        rows = self.dataset_statistics()
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        for ax, pool in zip(axes, ('known', 'unknown')):
            chosen = [r for r in rows if r['pool'] == pool]
            ax.barh([r['class_name'] for r in chosen], [r['n_images'] for r in chosen])
            ax.set_title(pool); ax.set_xlabel('Số ảnh'); ax.invert_yaxis()
        fig.tight_layout(); plt.show(); plt.close(fig)

    def show_images(self, pool='known', n=6):
        import matplotlib.pyplot as plt
        from PIL import Image
        if pool == '_extra_unknown' and not (self.data / pool).is_dir():
            print('Chưa chép _extra_unknown; bỏ qua preview bổ sung.')
            return
        groups = defaultdict(list)
        for row in self.metadata():
            if row['pool'] == pool:
                groups[row['label']].append(row['path'])
        names = sorted(groups)[:n]
        if not names:
            raise ValueError(f'Pool trống: {pool}')
        fig, axes = plt.subplots((len(names) + 2) // 3, 3, figsize=(12, 4 * ((len(names) + 2) // 3)), squeeze=False)
        for ax in axes.flat:
            ax.axis('off')
        for ax, name in zip(axes.flat, names):
            with Image.open(self.data / groups[name][0]) as image:
                ax.imshow(image.convert('RGB'))
            ax.set_title(name)
        fig.tight_layout(); plt.show(); plt.close(fig)

    def require_head(self):
        training_path = self.head.with_suffix('.training.json')
        if not self.head.is_file() or not training_path.is_file():
            raise RuntimeError('Chạy ô train trong notebook 02 trước.')
        training = read_json(training_path)
        assert training['dataset_fingerprint'] == self.state['dataset_fingerprint']
        assert training['split_fingerprint'] == self.splits['fingerprint']
        assert read_json(self.head.with_suffix('.splits.json')) == self.splits
        return training

    def require_calibration(self):
        training = self.require_head()
        env = self.environment()
        if not Path(env['RPV_BANK']).is_file() or not Path(env['RPV_CALIB']).is_file():
            raise RuntimeError('Chạy notebook 03 để dựng prototype và hiệu chỉnh trước.')
        from types import SimpleNamespace
        from rpv.vision.calibration import CalibrationModel
        from rpv.vision.prototypes import PrototypeBank
        from rpv.vision.color import HueSaturationEncoder
        vision = self.settings.vision
        head_hash = hashlib.sha256(self.head.read_bytes()).hexdigest()
        tag = (f"{training['backbone_tag']}_proj{training['out_dim']}_{head_hash[:16]}"
               f"_tta{int(vision.use_tta)}")
        dim = training['out_dim']
        if vision.prototype_color_weight > 0:
            color_dim = HueSaturationEncoder.dim
            tag += f'_prototype_hs{color_dim}w{vision.prototype_color_weight:g}_v1'
            dim += color_dim
        calibration = CalibrationModel.load(env['RPV_CALIB'], tag)
        bank = PrototypeBank(SimpleNamespace(tag=tag, dim=dim)).load(env['RPV_BANK'])
        if not calibration.bank_fingerprint or calibration.bank_fingerprint != bank.fingerprint():
            raise RuntimeError('Prototype đã đổi sau calibration. Chạy lại notebook 03 rồi notebook 04.')
        return calibration


    def cache_info(self):
        """Inspect persisted backbone vectors without loading DINO or a GPU."""
        caches = []
        for path in project_path(self.settings.paths.cache_dir).glob('*.npz'):
            import numpy as np
            with np.load(path, allow_pickle=False) as data:
                caches.append(dict(path=str(path), vectors=len(data['keys']),
                                   dimensions=int(data['dim'][0]), bytes=path.stat().st_size))
        return caches

    def warm_cache(self):
        """Cache all metadata images, including extra pools; resume after interruption."""
        subprocess.run([sys.executable, '-B', '-u', str(PROJECT / 'scripts/cache_embeddings.py')],
                       cwd=str(PROJECT), env=self.environment('configured'), check=True)
        if self.settings.vision.prototype_color_weight > 0:
            subprocess.run([sys.executable, '-B', '-u', str(PROJECT / 'scripts/cache_embeddings.py'), '--color'],
                           cwd=str(PROJECT), env=self.environment('configured'), check=True)
        return self.cache_info()

    def train(self, epochs=None, episodes=None, init_head=None, lr=None, seed=0):
        # Same defaults and output as run_all --train-head; always start a fresh head.
        self.head = project_path(self.settings.paths.head_path)
        args = ['--ckpt', '', '--out', self.head, '--quiet']
        if epochs is not None:
            args.extend(['--epochs',epochs])
        if episodes is not None:
            args.extend(['--episodes',episodes])
        if init_head is not None:
            args.extend(['--init-head',project_path(init_head)])
        if lr is not None:
            args.extend(['--lr',lr])
        args.extend(['--seed',seed])
        self.cli('train-head', *args, model='configured')
        result = self.require_head()
        return dict(checkpoint=os.path.relpath(self.head, PROJECT), best_epoch=result['best_epoch'],
                    best_val_acc=result['best_val_acc'])

    def plot_training_history(self):
        import matplotlib.pyplot as plt
        history = self.require_head()['history']
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        epochs = [r['epoch'] for r in history]
        axes[0].plot(epochs, [r['train_loss'] for r in history], label='train loss')
        for field in ('train_acc', 'val_acc'):
            points = [r for r in history if field in r]
            axes[1].plot([r['epoch'] for r in points], [r[field] for r in points], label=field)
        for ax in axes:
            ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(self.folder / 'training_curves.png', dpi=150)
        plt.show(); plt.close(fig)

    def calibrate(self):
        self.require_head()
        result = self.cli('calibrate', model='head')
        assert result['test_images_embedded'] == 0
        write_json(self.folder / 'calibrate.json', result)
        return dict(calib_metrics=result['calibration'], warnings=result['warnings'], saved=result['saved'])

    def evaluate(self):
        self.require_calibration()
        result = self.cli('evaluate', '--use-catalog', '--scenarios', model='head')
        report = read_json(result['reports']['json'])
        assert report['evaluation']['config']['dataset_fingerprint'] == self.state['dataset_fingerprint']
        assert read_json(Path(result['reports']['json']).parent / 'splits.json') == self.splits
        write_json(self.folder / 'evaluate_head.json', result)
        return dict(closed_set=result['closed_set'], open_set=result['open_set'], scenarios=result['scenarios'], warnings=result['warnings'])

    def compare_baselines(self):
        self.evaluation_result()
        for model in ('classical', 'raw'):
            result = self.cli('evaluate', model=model)
            write_json(self.folder / f'evaluate_{model}.json', result)
        return self.comparison_table()

    def comparison_table(self):
        rows = []
        for model in ('classical', 'raw', 'head'):
            path = self.folder / f'evaluate_{model}.json'
            if not path.is_file():
                continue
            result = self.evaluation_result() if model == 'head' else read_json(path)
            closed, selective = result['closed_set'], result['selective']
            operation = selective['operating_point_known']
            rows.append(dict(model=model, accuracy_known=closed['accuracy_all_classes'],
                macro_f1_known=closed['macro_f1_all_classes'], auroc=result['open_set']['auroc'],
                coverage_known=operation['coverage'], risk_known=operation['selective_risk'],
                answered_known=operation['n_answered'], target_met_on_test=operation['target_met_on_test'],
                unknown_auto_accept=selective['unknown_auto_accept_rate']))
        write_csv(self.folder / 'tables' / 'model_comparison.csv', rows)
        return rows

    def evaluation_result(self):
        path = self.folder / 'evaluate_head.json'
        if not path.is_file():
            raise RuntimeError('Chạy notebook 04 để đánh giá trước.')
        calibration = self.require_calibration()
        result = read_json(path)
        if result.get('encoder', {}).get('tag') != calibration.encoder_tag:
            raise RuntimeError('Báo cáo thuộc head cũ. Chạy lại notebook 04 sau khi train/calibrate.')
        run_calibration = Path(result['reports']['json']).parent / 'calibration.json'
        from rpv.vision.calibration import CalibrationModel
        if (not run_calibration.is_file() or
                CalibrationModel.load(run_calibration, calibration.encoder_tag).to_json() != calibration.to_json()):
            raise RuntimeError('Ngưỡng hoặc prototype đã đổi sau đánh giá. Chạy lại notebook 04.')
        return result

    def plot_risk_coverage(self):
        from IPython.display import Image, display
        folder = Path(self.evaluation_result()['reports']['json']).parent
        for image in folder.glob('*.png'):
            display(Image(filename=str(image)))

    def error_analysis(self):
        result = self.evaluation_result()
        report = read_json(result['reports']['json'])
        print('Các cặp nhầm lẫn thường gặp:')
        show_table(report['evaluation']['error_analysis']['top_confusion_pairs'][:10])
        meta = {r['path']: r for r in self.metadata()}
        groups = defaultdict(list)
        with Path(result['reports']['predictions']).open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                path = Path(row['path'])
                key = path.resolve().relative_to(self.data).as_posix() if path.is_absolute() else path.as_posix()
                source = meta[key]
                groups[(source['pool'], source['packed'])].append(row)
        rows = []
        for (pool, packed), items in sorted(groups.items()):
            answered = [r for r in items if r['status'] == 'match']
            wrong = sum(r['prediction_set'][0] != r['truth'] for r in answered)
            rows.append(dict(pool=pool, packed=packed, n=len(items), coverage=len(answered) / len(items),
                             risk=wrong / len(answered) if answered else None))
        print('Có/không túi theo metadata nguồn (packed=1/0):')
        write_csv(self.folder / 'tables' / 'packaging_analysis.csv', rows)
        return rows

    def evaluate_extra_unknown(self):
        self.require_calibration()
        if not (self.data / '_extra_unknown').is_dir():
            return {'skipped': 'Chưa chép _extra_unknown vào dataset; model chính vẫn chạy được.'}
        import torch
        from rpv.vision.encoders import load_encoder
        from rpv.vision.prototypes import PrototypeBank
        from rpv.vision.calibration import CalibrationModel
        from rpv.vision.recognizer import Recognizer
        from rpv.data.extra import extra_partition
        partition = extra_partition(self.data)
        rows = partition['holdout']
        if not rows:
            raise ValueError('Không có ảnh extra unknown trong metadata.')
        assert not ({r['label'] for r in rows} & set(self.splits['support']))
        output = self.folder / 'extra_unknown'
        output.mkdir(exist_ok=True)
        counts = defaultdict(Counter)
        env = self.environment()
        with _model_env(env):
            encoder = load_encoder()
            try:
                bank = PrototypeBank(encoder).load(env['RPV_BANK'])
                calib = CalibrationModel.load(env['RPV_CALIB'], encoder.tag)
                recognizer = Recognizer(encoder, bank, calib)
                with (output / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
                    for start in range(0, len(rows), 128):
                        batch = rows[start:start + 128]
                        vectors = encoder.embed_paths([self.data / r['path'] for r in batch])
                        distances, labels = bank.distances_batch(vectors)
                        for source, values in zip(batch, distances):
                            prediction = recognizer.recognize_distances(dict(zip(labels, map(float, values))))
                            counts[source['label']][prediction.status] += 1
                            record = dict(path=source['path'], true_class=source['label'], status=prediction.status,
                                          prediction_set=prediction.prediction_set, ood_score=prediction.ood_score)
                            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                        print(f'Unknown bổ sung: {min(start + 128, len(rows))}/{len(rows)}', flush=True)
                table = []
                total = Counter()
                for label, counter in sorted(counts.items()):
                    total.update(counter)
                    n = sum(counter.values())
                    table.append(dict(class_name=label, n=n, false_auto_accept=counter['match'] / n,
                                      unknown_reject=counter['unknown'] / n, handoff=1 - counter['match'] / n))
                n = sum(total.values())
                assert n == len(rows)
                summary = dict(n=n, classes=len(counts), false_auto_accept=total['match'] / n,
                               unknown_reject=total['unknown'] / n, handoff=1 - total['match'] / n,
                               encoder_tag=encoder.tag, bank_fingerprint=bank.fingerprint(),
                               checkpoint_sha256=hashlib.sha256(self.head.read_bytes()).hexdigest())
                summary.update(evaluation_scope='14 frozen extra-holdout classes; development/reserved excluded',
                               partition_fingerprint=partition['fingerprint'])
                write_json(output / 'summary.json', summary)
                write_csv(output / 'per_class.csv', table)
                return summary
            finally:
                del encoder
                # Recognizer/bank giữ tham chiếu encoder: xóa trước khi giải phóng GPU.
                for name in ('recognizer', 'bank', 'calib'):
                    if name in locals():
                        if name == 'recognizer':
                            del recognizer
                        elif name == 'bank':
                            del bank
                        else:
                            del calib
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def evaluate_robustness(self, max_per_class=5):
        self.require_calibration()
        pairs = self.data / 'hard_pairs.csv'
        args = ['--robustness', '--stress-max-per-class', max_per_class]
        if pairs.is_file():
            args.extend(['--hard-pairs', pairs])
        result = self.cli('evaluate', *args, model='head')
        write_json(self.folder / 'robustness.json', result)
        report = read_json(result['reports']['json'])
        return report['evaluation']['robustness_on_test_split']['conditions']

    def prepare_demo(self):
        self.require_calibration()
        self.cli('init-db', model='head')
        result = self.cli('seed-scenarios', model='head')
        write_json(self.folder / 'demo' / 'scenarios.json', result)
        return dict(n_cases=result['n_cases'], stats=result['stats'])

    def benchmark(self, repeats=20, warmup=3):
        self.prepare_demo()
        cases = read_json(self.folder / 'demo' / 'scenarios.json')['cases']
        happy = next(r for r in cases if r['name'] == 'happy_path')
        images = [self.data / path for path in self.splits['test'][happy['claimed_sku']][:5]]
        result = self.cli('benchmark', *images, '--sku', happy['claimed_sku'], '--lot', happy['claimed_lot'],
                          '--repeats', repeats, '--warmup', warmup,
                          '--out', self.folder / 'performance.json', model='head')
        performance = read_json(result['report'])
        rows = [dict(mode=mode, stage=stage, **values) for mode, data in performance['modes'].items()
                for stage, values in data['summary'].items()]
        write_csv(self.folder / 'tables' / 'latency.csv', rows)
        return rows

    def verify_audit(self):
        from rpv.audit.hash_chain import AuditLog
        path = Path(self.evaluation_result()['reports']['json']).parent / 'audit.jsonl'
        assert path.is_file()
        result = AuditLog(path).verify()
        assert result['valid'], result
        write_json(self.folder / 'audit_verification.json', result)
        return result

    def output_paths(self):
        result = self.evaluation_result()
        return dict(model=str(self.head), report=result['reports']['markdown'],
                    comparison=str(self.folder / 'tables' / 'model_comparison.csv'), folder=str(self.folder))

    def start_demo(self):
        """Mở Gradio trong subprocess để vẫn chạy được ô dừng demo."""
        self.require_calibration()
        port = int(os.environ.get('GRADIO_SERVER_PORT', '7860'))
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1', port)) == 0:
                raise RuntimeError(f'Cổng {port} đang được dùng. Dừng demo cũ hoặc đổi GRADIO_SERVER_PORT.')
        logs = self.folder / 'demo' / 'server.log'
        logs.parent.mkdir(exist_ok=True)
        env = self.environment()
        env.update(GRADIO_SERVER_NAME='127.0.0.1', GRADIO_SERVER_PORT=str(port))
        with logs.open('w', encoding='utf-8') as stream:
            self._demo = subprocess.Popen([sys.executable, '-B', '-m', 'app.demo'], cwd=str(PROJECT),
                                          env=env, stdout=stream, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                if self._demo.poll() is not None:
                    raise RuntimeError(f'Demo không khởi động. Xem {logs}')
                with socket.socket() as probe:
                    probe.settimeout(.5)
                    if probe.connect_ex(('127.0.0.1', port)) == 0:
                        return f'http://127.0.0.1:{port}'
                time.sleep(.5)
            raise RuntimeError(f'Demo chưa sẵn sàng sau 45 giây. Xem {logs}')
        except BaseException:
            self.stop_demo()
            raise

    def stop_demo(self):
        if self._demo is not None and self._demo.poll() is None:
            self._demo.terminate()
            try:
                self._demo.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._demo.kill(); self._demo.wait()
        self._demo = None
        print('Đã dừng demo do notebook này mở.')
