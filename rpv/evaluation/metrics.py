"""Metric. Mỗi hàm chỉ tính đúng một thứ, để không lẫn "tốt ở đâu" với "tốt cỡ nào".

Vì sao không dùng một con số accuracy duy nhất: một hệ thống gán mọi hàng lạ vào
lớp quen vẫn có thể có closed-set accuracy rất cao, trong khi hoàn toàn không
dùng được ở quầy thật. Bộ metric dưới đây tách rạch ròi bốn câu hỏi khác nhau:
nhận đúng không, có biết từ chối không, có kiểm chứng được lời khai không, và khi
được phép im lặng thì rủi ro còn lại là bao nhiêu.
"""
from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np


def _check_lengths(first: Sequence, second: Sequence) -> None:
    if len(first) != len(second):
        raise ValueError("Paired metric inputs must have the same length.")


def _scores(values: Sequence[float]) -> np.ndarray:
    scores = np.asarray(values, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Metric scores must be a one-dimensional sequence of finite numbers.")
    return scores


def closed_set_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    _check_lengths(y_true, y_pred)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean([t == p for t, p in zip(y_true, y_pred)]))


def per_class_recall(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float]:
    _check_lengths(y_true, y_pred)
    total, hit = Counter(y_true), Counter()
    for t, p in zip(y_true, y_pred):
        if t == p:
            hit[t] += 1
    return {c: hit[c] / total[c] for c in sorted(total)}


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] | None = None) -> float:
    """Macro-F1 trên hợp của nhãn thật và nhãn dự đoán (gồm cả nhãn 'unknown').
    Macro chứ không micro: danh mục bán lẻ luôn lệch số lượng ảnh giữa các SKU,
    micro-F1 sẽ bị các SKU nhiều ảnh chi phối."""
    _check_lengths(y_true, y_pred)
    labels = sorted(set(y_true) | set(y_pred)) if labels is None else list(labels)
    if len(labels) != len(set(labels)):
        raise ValueError('Macro-F1 label list must contain distinct labels.')
    f1s = []
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


def auroc(scores_pos: Sequence[float], scores_neg: Sequence[float]) -> float:
    """AUROC bằng thống kê Mann-Whitney U (xử lý đúng trường hợp bằng nhau).
    Quy ước: điểm CAO hơn = "lạ" hơn, nên pos = unknown."""
    p, n = _scores(scores_pos), _scores(scores_neg)
    if p.size == 0 or n.size == 0:
        return float("nan")
    allv = np.concatenate([p, n])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, allv.size + 1)
    # xử lý ties: gán rank trung bình
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[: p.size].sum()
    return float((r_pos - p.size * (p.size + 1) / 2) / (p.size * n.size))


def fpr_at_tpr(known: Sequence[float], unknown: Sequence[float], tpr_target: float = 0.95) -> dict:
    """FPR@TPR: khi phải nhận đúng ``tpr_target`` hàng quen thì bao nhiêu hàng lạ
    bị nhận nhầm. Đây là con số nghiệp vụ thật, dễ hiểu hơn AUROC."""
    k, u = _scores(known), _scores(unknown)
    if not 0 < tpr_target <= 1:
        raise ValueError("tpr_target must be in (0, 1].")
    if k.size == 0 or u.size == 0:
        return {"threshold": float("nan"), "fpr": float("nan"), "tpr": float("nan")}
    # Smallest observed distance accepting at least the target known fraction.
    rank = int(np.ceil(k.size * tpr_target))
    thr = float(np.sort(k)[rank - 1])
    return {"threshold": thr, "tpr": float((k <= thr).mean()), "fpr": float((u <= thr).mean())}


def open_set_metrics(known_distances: Sequence[float], unknown_distances: Sequence[float],
                     threshold: float) -> dict:
    k, u = _scores(known_distances), _scores(unknown_distances)
    if not np.isfinite(threshold):
        raise ValueError("Open-set threshold must be finite.")
    out = {
        "threshold": float(threshold),
        "known_recall": float((k <= threshold).mean()) if k.size else float("nan"),
        "unknown_reject_rate": float((u > threshold).mean()) if u.size else float("nan"),
        "false_accept_rate": float((u <= threshold).mean()) if u.size else float("nan"),
        "auroc": auroc(u, k),
        "n_known": int(k.size),
        "n_unknown": int(u.size),
    }
    out["fpr_at_95tpr"] = fpr_at_tpr(k, u, 0.95)
    return out


def verification_metrics(records: Sequence[Mapping]) -> dict:
    """``records``: [{"genuine": bool, "accepted": bool, "score": float}, ...]

    genuine = lời khai đúng sự thật. FAR (chấp nhận lời khai sai) là chỉ số quan
    trọng nhất về mặt thất thoát; FRR là chỉ số về ma sát trải nghiệm.
    """
    gen = [r for r in records if r["genuine"]]
    imp = [r for r in records if not r["genuine"]]
    far = float(np.mean([r["accepted"] for r in imp])) if imp else float("nan")
    frr = float(np.mean([not r["accepted"] for r in gen])) if gen else float("nan")
    out = {
        "n_genuine": len(gen), "n_impostor": len(imp),
        "false_accept_rate": far, "false_reject_rate": frr,
        "accuracy": float(np.mean([r["accepted"] == r["genuine"] for r in records])) if records else float("nan"),
    }
    if gen and imp:
        gs = [r["score"] for r in gen]
        isc = [r["score"] for r in imp]
        out["auroc"] = auroc(isc, gs)     # điểm cao = giả mạo
        # EER: quét toàn bộ ngưỡng ứng viên, lấy điểm FAR ≈ FRR
        cands = np.unique(np.concatenate([np.asarray(gs, float), np.asarray(isc, float)]))
        fars = np.searchsorted(np.sort(isc), cands, side='right') / len(isc)
        frrs = 1 - np.searchsorted(np.sort(gs), cands, side='right') / len(gs)
        best = int(np.argmin(np.abs(fars - frrs)))
        out["eer"] = float((fars[best] + frrs[best]) / 2)
        out["eer_threshold"] = float(cands[best])
    return out


def risk_coverage_curve(correct: Sequence[bool], confidence: Sequence[float],
                        n_points: int = 21) -> list[dict]:
    """Đường risk-coverage: nếu hệ thống được quyền im lặng, chấp nhận trả lời
    ``coverage``% ca tự tin nhất thì tỉ lệ sai trong số đó là bao nhiêu.

    Đây là cách trung thực nhất để nói về "độ chính xác" của một hệ thống được
    phép nói không biết — và là lý do abstention không phải là gian lận số liệu.
    """
    _check_lengths(correct, confidence)
    confidence = _scores(confidence)
    if len(correct) == 0:
        return []
    order = np.argsort(-np.asarray(confidence, float))
    c = np.asarray(correct, bool)[order]
    n = c.size
    out = []
    sorted_conf = np.asarray(confidence, float)[order]
    # Whole ties must be accepted together by a deterministic threshold.
    endpoints = np.r_[np.flatnonzero(sorted_conf[1:] != sorted_conf[:-1]) + 1, n]
    errors = np.cumsum(~c)
    for k in endpoints:
        out.append({
            "coverage": float(k / n),
            "selective_risk": float(errors[k - 1] / k),
            "n_answered": int(k),
        })
    return out


def coverage_at_risk(curve: Sequence[Mapping], max_risk: float) -> float:
    ok = [p["coverage"] for p in curve if p["selective_risk"] <= max_risk]
    return float(max(ok)) if ok else 0.0


def confusion_pairs(y_true: Sequence[str], y_pred: Sequence[str], top: int = 15) -> list[dict]:
    """Cặp nhầm lẫn hay gặp nhất — dữ liệu để phân tích lỗi, không phải để trang trí."""
    _check_lengths(y_true, y_pred)
    cnt: Counter = Counter()
    for t, p in zip(y_true, y_pred):
        if t != p:
            cnt[(t, p)] += 1
    return [{"true": t, "pred": p, "count": c} for (t, p), c in cnt.most_common(top)]
