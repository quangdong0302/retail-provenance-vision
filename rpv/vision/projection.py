"""ProjectionHead học bằng episodic prototypical training trên vector DINOv2.

Head là phần được huấn luyện; backbone giữ cố định. Vector backbone được
cache để dùng lại giữa các lượt học. Head có vài trăm nghìn tham số,
có thể huấn luyện trên CPU; thời gian phụ thuộc cấu hình và số episode.

Training và suy luận cùng dùng nearest-prototype. Danh mục lớp nằm trong
prototype bank, không nằm trong một softmax classifier cố định. Thêm SKU
tạo prototype từ 2–3 ảnh support mà không cần học lại bộ phân loại.

Module import được khi chưa cài torch; lỗi chỉ phát sinh lúc gọi phần cần
torch, nên test lõi vẫn chạy được trên máy không có thư viện này.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from rpv.util import log

_TORCH_ERROR = (
    "Cần PyTorch để dùng ProjectionHead: pip install -r requirements.txt"
)


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        return torch, nn, F
    except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
        raise RuntimeError(_TORCH_ERROR) from exc


try:  # pragma: no cover - phụ thuộc môi trường
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


if _HAS_TORCH:
    class ProjectionHead(nn.Module):
        """MLP: Linear -> GELU -> Dropout -> Linear, đầu ra chuẩn hóa L2.

        Residual từ đầu vào được dùng khi cùng số chiều, giúp giữ biểu diễn DINOv2
        ban đầu trong lúc học phần điều chỉnh. Cách này hạn chế thay đổi lớn từ
        head khởi tạo ngẫu nhiên khi dữ liệu huấn luyện ít.
        """

        def __init__(self, in_dim: int, out_dim: int = 128, dropout: float = 0.3) -> None:
            super().__init__()
            self.in_dim = int(in_dim)
            self.out_dim = int(out_dim)
            self.net = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(in_dim, out_dim),
            )
            self.residual = nn.Linear(in_dim, out_dim, bias=False)
            nn.init.eye_(self.residual.weight)
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
            self.register_buffer('metric_matrix',None)
            self.register_buffer('metric_bias',None)

        def load_state_dict(self,state_dict,strict=True,**kwargs):
            matrix = state_dict.get('metric_matrix')
            bias = state_dict.get('metric_bias')
            if (matrix is None) != (bias is None):
                raise ValueError('Metric matrix and bias must be stored together.')
            if matrix is not None:
                if (matrix.shape != (self.out_dim,self.out_dim) or bias.shape != (self.out_dim,)
                        or not torch.isfinite(matrix).all() or not torch.isfinite(bias).all()
                        or torch.linalg.vector_norm(matrix)<1e-8):
                    raise ValueError('Invalid shared prototype metric.')
                self.metric_matrix = torch.empty_like(matrix)
                self.metric_bias = torch.empty_like(bias)
            else:
                self.metric_matrix = self.metric_bias = None
            return super().load_state_dict(state_dict,strict=strict,**kwargs)

        def forward(self, x):
            z = F.normalize(self.net(x) + self.residual(x), dim=-1)
            if self.metric_matrix is not None:
                z = F.normalize(F.linear(z,self.metric_matrix,self.metric_bias),dim=-1)
            return z

    def prototypical_loss(support, query, query_labels, temperature: float = 10.0):
        """support: [n_way,k_shot,D]; query: [n_query,D]; labels: [n_query].

        Cross-entropy trên khoảng cách âm tới prototype (Snell et al., 2017).
        Hệ số nhiệt điều chỉnh độ lớn logits: vector chuẩn hóa L2 có khoảng cách
        trong [0,2], logits không scale có thể quá phẳng và làm học chậm.
        """
        protos = F.normalize(support.mean(dim=1), dim=-1)          # [n_way, D]
        dists = torch.cdist(query, protos)                         # [n_query, n_way]
        return F.cross_entropy(-dists * temperature, query_labels), dists

    def train_projection_head(
        features_by_class: dict[str, np.ndarray],
        out_dim: int = 128,
        n_way: int = 5,
        k_shot: int = 3,
        q_query: int = 8,
        episodes_per_epoch: int = 300,
        epochs: int = 20,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        noise_std: float = 0.05,
        dropout_p: float = 0.10,
        head_dropout: float = 0.30,
        temperature: float = 10.0,
        grad_clip: float = 1.0,
        patience: int = 6,
        val_classes: Sequence[str] | None = None,
        validation_features: dict[str, np.ndarray] | None = None,
        dataset_fingerprint: str = "",
        split_fingerprint: str = "",
        val_episodes: int = 100,
        backbone_tag: str = "",
        seed: int = 0,
        device: str | None = None,
        save_path: str | Path | None = None,
        log_every: int = 1,
    ) -> dict:
        """Episodic training trên embedding đã precompute.

        ``validation_features`` chứa ảnh validation độc lập của known trong v2.
        ``val_classes`` giữ tương thích cho caller cũ muốn tách theo lớp.
        Checkpoint được chọn bằng accuracy validation, không dùng unknown-test.
        """
        torch, nn, F = _require_torch()
        rng = np.random.default_rng(seed)
        val_rng = np.random.default_rng(seed + 1)
        torch.manual_seed(seed)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if epochs < 1 or episodes_per_epoch < 1 or k_shot < 1 or q_query < 1 or val_episodes < 1 or n_way < 2 or out_dim < 1:
            raise ValueError('epochs, episodes, k_shot, q_query phải >= 1.')
        for label, values in list(features_by_class.items()) + list((validation_features or {}).items()):
            if values.ndim != 2 or not np.isfinite(values).all():
                raise ValueError(f'{label}: embedding phải là ma trận hữu hạn.')
        val_classes = [c for c in (val_classes or []) if c in features_by_class]
        val_set = set(val_classes)
        train_classes = [c for c in sorted(features_by_class) if c not in val_set]
        if validation_features is not None:
            val_classes = sorted(validation_features)
        else:
            validation_features = {c: features_by_class[c] for c in val_classes}
        if len(val_classes) < 2:
            raise ValueError('Cần ít nhất 2 lớp validation độc lập để chọn checkpoint.')
        if len(train_classes) < 2:
            raise ValueError(
                f"Cần ít nhất 2 lớp train để chạy episodic training (đang có {len(train_classes)}). "
                "Kiểm tra các lớp known và số ảnh train trong manifest split."
            )
        dims = {int(v.shape[1]) for v in list(features_by_class.values()) + list(validation_features.values()) if v.size}
        if len(dims) != 1:
            raise ValueError(f"Đặc trưng các lớp không cùng số chiều: {sorted(dims)}.")
        in_dim = dims.pop()
        if any(len(v) < 2 for v in list(features_by_class.values()) + list(validation_features.values())):
            raise ValueError('Mỗi lớp train/validation cần ít nhất 2 ảnh để support/query không trùng.')

        head = ProjectionHead(in_dim, out_dim, dropout=head_dropout).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

        def sample_episode(classes: Sequence[str], augment: bool):
            episode_rng = rng if augment else val_rng
            feature_pool = features_by_class if augment else validation_features
            way = min(n_way, len(classes))
            picked = episode_rng.choice(len(classes), size=way, replace=False)
            actual_k = min(k_shot, min(len(feature_pool[classes[int(ci)]]) - 1 for ci in picked))
            sup, qry, lab = [], [], []
            for idx, ci in enumerate(picked):
                pool = feature_pool[classes[int(ci)]]
                actual_q = min(q_query, len(pool) - actual_k)
                need = actual_k + actual_q
                take = episode_rng.choice(pool.shape[0], size=need, replace=False)
                feats = pool[take].astype(np.float32)
                if augment:
                    # Nhiễu + dropout trên embedding: cách rẻ nhất để mô phỏng
                    # biến thiên ảnh mà không phải chạy lại backbone.
                    feats = feats + rng.normal(0, noise_std, feats.shape).astype(np.float32)
                    mask = (rng.random(feats.shape) > dropout_p).astype(np.float32)
                    feats = feats * mask / max(1e-6, 1.0 - dropout_p)
                    norms = np.linalg.norm(feats, axis=-1, keepdims=True)
                    feats = feats / np.maximum(norms, 1e-12)
                sup.append(feats[:actual_k])
                qry.append(feats[actual_k:need])
                lab.extend([idx] * actual_q)
            to_t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)
            return (to_t(np.stack(sup)), to_t(np.concatenate(qry)),
                    torch.as_tensor(lab, dtype=torch.long, device=device))

        def run_episode(classes, augment: bool):
            sup, qry, lab = sample_episode(classes, augment)
            sup_p = head(sup.reshape(-1, in_dim)).reshape(sup.shape[0], sup.shape[1], -1)
            return prototypical_loss(sup_p, head(qry), lab, temperature) + (lab,)

        history: list[dict] = []
        best_val, best_state, best_epoch, stale = -1.0, None, 0, 0
        for epoch in range(1, epochs + 1):
            head.train()
            losses, accs = [], []
            for _ in range(episodes_per_epoch):
                loss, dists, lab = run_episode(train_classes, augment=True)
                if not torch.isfinite(loss):
                    raise RuntimeError('Training loss không hữu hạn; dừng trước khi lưu checkpoint hỏng.')
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
                opt.step()
                losses.append(float(loss.detach()))
                accs.append(float((torch.argmin(dists, dim=1) == lab).float().mean()))
            sched.step()
            entry = {"epoch": epoch, "lr": float(opt.param_groups[0]["lr"]),
                     "train_loss": float(np.mean(losses)), "train_acc": float(np.mean(accs))}

            if val_classes and len(val_classes) >= 2:
                # Use the same validation episodes every epoch, independent of training RNG.
                val_rng = np.random.default_rng(seed + 1)
                head.eval()
                vaccs = []
                with torch.no_grad():
                    for _ in range(val_episodes):
                        _, dists, lab = run_episode(val_classes, augment=False)
                        vaccs.append(float((torch.argmin(dists, dim=1) == lab).float().mean()))
                entry["val_acc"] = float(np.mean(vaccs))
                if entry["val_acc"] > best_val + 1e-4:
                    best_val, best_epoch, stale = entry["val_acc"], epoch, 0
                    best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
                else:
                    stale += 1
            history.append(entry)
            if log_every and epoch % log_every == 0:
                log(f"[train] {entry}")
            if patience and stale >= patience:
                log(f"[train] dừng sớm ở epoch {epoch}: val không cải thiện {stale} epoch liền.")
                break

        if best_state is not None:
            head.load_state_dict(best_state)
        elif val_classes:
            log("[train] CẢNH BÁO: không có epoch nào cải thiện val; đang dùng trọng số epoch cuối.")

        result = {
            "history": history,
            "best_val_acc": best_val if best_val >= 0 else None,
            "best_epoch": best_epoch or None,
            "n_train_classes": len(train_classes),
            "n_val_classes": len(val_classes),
            "in_dim": in_dim, "out_dim": out_dim,
            "backbone_tag": backbone_tag,
            "save_path": None,
            'dataset_fingerprint': dataset_fingerprint,
            'split_fingerprint': split_fingerprint,
            'training_config': {
                'seed': seed, 'n_way': n_way, 'k_shot': k_shot, 'q_query': q_query,
                'epochs': epochs, 'episodes_per_epoch': episodes_per_epoch,
                'lr': lr, 'weight_decay': weight_decay, 'noise_std': noise_std,
                'dropout_p': dropout_p, 'head_dropout': head_dropout,
                'temperature': temperature, 'grad_clip': grad_clip,
                'patience': patience, 'val_episodes': val_episodes, 'device': str(device),
            },
            'versions': {'torch': str(torch.__version__), 'numpy': np.__version__},
            'train_images_per_class': {c: len(features_by_class[c]) for c in train_classes},
            'validation_images_per_class': {c: len(validation_features[c]) for c in val_classes},
        }
        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": head.state_dict(),
                "in_dim": in_dim,
                "out_dim": out_dim,
                # Ghi kèm tag backbone: head chỉ có nghĩa cùng đúng backbone đã
                # sinh ra đặc trưng để train nó. Không có trường này, ghép sai
                # cặp sẽ không báo lỗi mà chỉ cho khoảng cách vô nghĩa.
                "backbone_tag": backbone_tag,
                "dataset_fingerprint": dataset_fingerprint,
                "split_fingerprint": split_fingerprint,
                "train_classes": train_classes,
                "val_classes": list(val_classes),
                "best_val_acc": best_val if best_val >= 0 else None,
                'training_config': result['training_config'],
            }, path)
            result["save_path"] = str(path)
        return result

else:  # pragma: no cover - môi trường không có torch
    class ProjectionHead:  # type: ignore[no-redef]
        def __init__(self, *_a, **_kw):
            _require_torch()

    def prototypical_loss(*_a, **_kw):  # type: ignore[misc]
        _require_torch()

    def train_projection_head(*_a, **_kw):  # type: ignore[misc]
        _require_torch()
