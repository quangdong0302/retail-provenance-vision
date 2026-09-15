"""CLI: chạy toàn bộ hệ thống bằng một lệnh cho mỗi bước.

    python -m rpv.cli prepare-data --source <ảnh thật> --destination data/dataset_v2
    python -m rpv.cli init-db             # tạo lược đồ SQLite
    python -m rpv.cli warm-cache          # nhúng trước toàn bộ ảnh (DINOv2 chỉ chạy 1 lần)
    python -m rpv.cli train-head          # fine-tune ProjectionHead trên đặc trưng DINOv2
    python -m rpv.cli calibrate           # đăng ký prototype + hiệu chỉnh
    python -m rpv.cli seed-scenarios      # seed hồ sơ lô/chứng từ cho 9 kịch bản
    python -m rpv.cli evaluate            # đánh giá đầy đủ + xuất báo cáo
    python -m rpv.cli register <sku> <ảnh...>
    python -m rpv.cli recognize <ảnh>
    python -m rpv.cli checkout <ảnh> --sku <sku> [--lot <lot>]
    python -m rpv.cli trace <lot_id>      # xem hồ sơ lô + xuất EPCIS
    python -m rpv.cli audit-verify        # kiểm tra toàn vẹn chuỗi audit
    python -m rpv.cli status
    python -m rpv.cli cache-info | cache-clear

Mọi lệnh nhận ``--encoder dinov2|classical`` để ghi đè cấu hình cho đúng lần chạy
đó — tiện khi cần bảng so sánh baseline mà không phải sửa file .env.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from rpv.config import reload_settings
from rpv.util import json_dumps, log


def _json(obj) -> str:
    return json_dumps(obj)


def _apply_overrides(args) -> None:
    """Ghi đè cấu hình từ tham số dòng lệnh trước khi bất kỳ module nào đọc nó."""
    changed = False
    for attr, env in (("encoder", "RPV_ENCODER"), ("variant", "RPV_DINOV2_VARIANT"),
                      ("device", "RPV_DEVICE"), ("ckpt", "RPV_PROJECTION_CKPT"),
                      ("data_dir", "RPV_DATA_DIR")):
        value = getattr(args, attr, None)
        if value is not None and (value or attr == 'ckpt'):
            os.environ[env] = str(value)
            changed = True
    if getattr(args, "no_cache", False):
        os.environ["RPV_EMBED_CACHE"] = "0"
        changed = True
    if changed:
        reload_settings()


def _settings():
    """Luôn đọc settings hiện hành (có thể đã reload sau khi ghi đè)."""
    from rpv.config import settings as current
    return current


def _load_stack(strict_calibration: bool = True):
    """Nạp encoder + bank + calibration + repo + service."""
    from rpv.audit.hash_chain import AuditLog
    from rpv.provenance.repository import ProvenanceRepository
    from rpv.services.checkout import CheckoutService
    from rpv.vision.calibration import CalibrationModel
    from rpv.vision.encoders import load_encoder
    from rpv.vision.prototypes import PrototypeBank

    st = _settings()
    st.paths.ensure()
    encoder = load_encoder()
    bank = PrototypeBank(encoder)
    if Path(st.paths.bank_path).exists():
        bank.load(st.paths.bank_path)
    calibration = None
    if Path(st.paths.calib_path).exists():
        calibration = CalibrationModel.load(
            st.paths.calib_path, encoder.tag if strict_calibration else None)
    repo = ProvenanceRepository(st.paths.db_path)
    repo.init_schema()
    service = CheckoutService(encoder, bank, repo, calibration, audit=AuditLog(st.paths.audit_path),
                              bank_path=st.paths.bank_path)
    return encoder, bank, service.calibration, repo, service


def _build_splits(k_shot: int | None = None):
    from rpv.data.dataset import build_splits, index_dataset

    st = _settings()
    cfg = st.calibration
    return build_splits(index_dataset(st.paths.data_dir), k_shot=k_shot or cfg.k_shot,
                        class_seed=cfg.class_split_seed,
                        calib_test_seed=cfg.calib_test_seed)


# --- lệnh ------------------------------------------------------------------
def cmd_prepare_data(args) -> int:
    from rpv.data.managed import prepare_dataset
    print(_json(prepare_dataset(args.source, args.destination, args.known_ratio,
                                args.seed, args.groups_csv)))
    return 0


def cmd_dataset_info(args) -> int:
    from rpv.data.managed import split_manifest
    splits = _build_splits(args.k_shot)
    payload = split_manifest(splits)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(_json(payload), encoding='utf-8')
    print(_json({'summary': splits.summary(), 'fingerprint': payload['fingerprint'],
                 'dataset_fingerprint': splits.dataset_fingerprint,
                 'known_classes': splits.train_classes, 'unknown_classes': splits.novel_classes}))
    return 0


def cmd_init_db(args) -> int:
    from rpv.provenance.repository import ProvenanceRepository
    st = _settings()
    st.paths.ensure()
    with ProvenanceRepository(st.paths.db_path) as repo:
        repo.init_schema()
        print(_json({"db": str(st.paths.db_path), "stats": repo.stats()}))
    return 0


def cmd_warm_cache(args) -> int:
    """Nhúng trước toàn bộ ảnh trong thư mục dữ liệu.

    Với DINOv2 đây là bước đáng chạy một lần: mọi lệnh sau đó (train-head,
    calibrate, evaluate, và các lần chạy lại khi đổi alpha) đọc từ cache.
    """
    from rpv.data.dataset import index_dataset
    from rpv.vision.encoders import load_encoder

    st = _settings()
    st.paths.ensure()
    encoder = load_encoder()
    index = index_dataset(st.paths.data_dir)
    total = sum(len(v) for v in index.values())
    done = 0
    for cname in sorted(index):
        encoder.embed_paths(index[cname])
        done += len(index[cname])
        log(f"[warm] {cname}: {len(index[cname])} ảnh ({done}/{total})")
    cache = encoder._cache()
    print(_json({"encoder": encoder.describe(), "n_images": total,
                 "cache": cache.info() if cache else "đang tắt (RPV_EMBED_CACHE=0)"}))
    return 0


def cmd_train_head(args) -> int:
    """Fine-tune ProjectionHead bằng episodic training trên đặc trưng DINOv2.

    Chỉ dùng ảnh train và validation riêng của known. Unknown được giữ lại
    cho đánh giá trước/sau đăng ký, không tham gia chọn checkpoint.
    """
    import numpy as np

    from rpv.vision.encoders import DinoV2Encoder, load_encoder
    from rpv.vision.projection import train_projection_head

    st = _settings()
    st.paths.ensure()
    encoder = load_encoder()
    if not isinstance(encoder, DinoV2Encoder):
        print("train-head chỉ có nghĩa với encoder dinov2 (backbone frozen + head học được). "
              f"Encoder hiện tại: {encoder.tag}. Đặt RPV_ENCODER=dinov2 hoặc --encoder dinov2.",
              file=sys.stderr)
        return 2
    if encoder.head is not None:
        log("CẢNH BÁO: RPV_PROJECTION_CKPT đang trỏ tới một head. Việc train sẽ dùng đặc trưng "
            "BACKBONE thô (bỏ qua head cũ), nên kết quả không bị chồng head lên head.")

    splits = _build_splits(args.k_shot)
    log(f"[train-head] splits={splits.summary()}")

    feats: dict[str, np.ndarray] = {}
    for cname, paths in sorted(splits.train.items()):
        if not paths:
            continue
        feats[cname] = encoder.backbone_features(paths)
    train_only = [c for c in feats if c in set(splits.train_classes)]
    if not train_only:
        print("Không có lớp train nào có đặc trưng. Kiểm tra số ảnh train trong manifest split.", file=sys.stderr)
        return 2

    out_path = Path(args.out or st.paths.head_path)
    from rpv.data.managed import split_manifest
    validation = {c: encoder.backbone_features(paths) for c, paths in splits.validation.items()}
    protocol = split_manifest(splits)
    res = train_projection_head(
        feats, out_dim=args.out_dim, n_way=args.n_way, k_shot=args.k_shot or st.calibration.k_shot,
        q_query=args.q_query, episodes_per_epoch=args.episodes, epochs=args.epochs,
        lr=args.lr, validation_features=validation, backbone_tag=encoder.backbone_tag,
        dataset_fingerprint=splits.dataset_fingerprint, split_fingerprint=protocol['fingerprint'],
        seed=args.seed, device=encoder.device, save_path=out_path,
    )
    out_path.with_suffix('.training.json').write_text(_json(res), encoding='utf-8')
    out_path.with_suffix('.splits.json').write_text(_json(protocol), encoding='utf-8')
    if args.quiet:
        res.pop("history", None)
    print(_json(res))
    log(f"\nDùng head vừa train:\n    export RPV_PROJECTION_CKPT={out_path}\n"
        "    python -m rpv.cli calibrate && python -m rpv.cli evaluate")
    return 0


def cmd_calibrate(args) -> int:
    from rpv.evaluation.pipeline import fit_catalog
    from rpv.vision.encoders import load_encoder
    st = _settings()
    st.paths.ensure()
    encoder = load_encoder()
    splits = _build_splits(args.k_shot)
    bank, calib = fit_catalog(encoder, splits, args.alpha)
    bank.save(st.paths.bank_path)
    calib.save(st.paths.calib_path)
    print(_json({
        "encoder": encoder.describe(),
        "bank": bank.summary(),
        "calibration": calib.calib_metrics,
        "warnings": calib.warnings,
        "test_images_embedded": 0,
        "saved": {"bank": str(st.paths.bank_path), "calibration": str(st.paths.calib_path)},
    }))
    return 0


def cmd_seed_scenarios(args) -> int:
    from rpv.evaluation.scenarios import build_provenance_scenarios
    from rpv.provenance.repository import ProvenanceRepository

    st = _settings()
    splits = _build_splits()
    st.paths.ensure()
    with ProvenanceRepository(st.paths.db_path) as repo:
        repo.init_schema()
        cases = build_provenance_scenarios(repo, splits.test, splits.unknown_test)
        print(_json({"n_cases": len(cases), "stats": repo.stats(),
                     "cases": [{"name": c.name, "expected": c.expected, "claimed_sku": c.claimed_sku,
                                "claimed_lot": c.claimed_lot} for c in cases]}))
    return 0


def cmd_evaluate(args) -> int:
    """Đánh giá đầy đủ: thị giác + kịch bản nghiệp vụ + báo cáo."""
    from rpv.audit.hash_chain import AuditLog
    from rpv.evaluation.pipeline import evaluate_system
    from rpv.evaluation.report import write_report
    from rpv.evaluation.scenarios import build_provenance_scenarios, evaluate_scenarios
    from rpv.provenance.repository import ProvenanceRepository
    from rpv.services.checkout import CheckoutService
    from rpv.vision.encoders import load_encoder

    st = _settings()
    st.paths.ensure()
    encoder = load_encoder()
    splits = _build_splits(args.k_shot)
    from datetime import datetime, timezone
    import uuid
    from rpv.data.managed import split_manifest
    run_dir = st.paths.reports_dir / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True)
    (run_dir / 'splits.json').write_text(_json(split_manifest(splits)), encoding='utf-8')
    out, bank, calib = evaluate_system(encoder=encoder, splits=splits, k_shot=args.k_shot,
                                       alpha=args.alpha, save_bank_to=run_dir / 'prototypes.json',
                                       save_calibration_to=run_dir / 'calibration.json')
    scen, decision_stats = None, {}
    if args.scenarios:
        with ProvenanceRepository(run_dir / 'scenarios.db') as repo:
            repo.init_schema()
            cases = build_provenance_scenarios(repo, splits.test, splits.unknown_test)
            service = CheckoutService(encoder, bank, repo, calib, audit=AuditLog(run_dir / 'audit.jsonl'))
            scen = evaluate_scenarios(service, cases)
            decision_stats = repo.decision_breakdown()
    paths = write_report(out.to_json(), scen, out_dir=run_dir,
                         stem=f"evaluation_{encoder.tag}", extra={"decision_breakdown": decision_stats})
    print(_json({
        "encoder": out.config.get("encoder"),
        "closed_set": out.closed_set,
        "open_set": {k: v for k, v in out.open_set.items() if k != "fpr_at_95tpr"},
        "selective": {k: v for k, v in out.selective.items() if k != "risk_coverage_curve"},
        "verification": out.verification,
        "novel_after_enrollment": out.novel_evaluation,
        "scenarios": ({"n_passed": scen["n_passed"], "n_cases": scen["n_cases"],
                      "failed": [c["scenario"] for c in scen["cases"] if not c["passed"]]} if scen else None),
        "warnings": out.warnings,
        "reports": {k: str(v) for k, v in paths.items()},
    }))
    return 0


def cmd_register(args) -> int:
    _, _, _, repo, service = _load_stack()
    try:
        res = service.register_product(args.sku, args.images, label=args.label or args.sku,
                                       operator=args.operator)
        print(_json(res))
    finally:
        repo.close()
    return 0


def cmd_enroll(args) -> int:
    _, _, _, repo, service = _load_stack()
    try:
        enrollment = service.enrollment
        if args.action == 'create':
            result = enrollment.create(args.sku or '', args.label, args.operator)
        elif args.action == 'list':
            result = enrollment.list_sessions()
        elif args.action == 'add':
            result = enrollment.add_images(args.session, args.role, args.images, args.group or '', args.negative_sku)
        elif args.action == 'assess':
            result = enrollment.assess(args.session)
        elif args.action == 'publish':
            result = service.publish_enrollment(args.session)
        else:
            result = enrollment.get(args.session)
        print(_json(result))
    finally:
        repo.close()
    return 0


def cmd_recognize(args) -> int:
    _, _, calib, repo, service = _load_stack()
    try:
        if calib is None:
            print("Chưa có mô hình hiệu chỉnh. Chạy `python -m rpv.cli calibrate` trước.",
                  file=sys.stderr)
            return 2
        print(_json(service.recognizer.recognize(args.image).to_json()))
    finally:
        repo.close()
    return 0


def cmd_checkout(args) -> int:
    _, _, _, repo, service = _load_stack()
    try:
        resp = service.process(args.image, claimed_sku=args.sku, claimed_lot=args.lot,
                               claimed_price=args.price, operator=args.operator, enrich=args.enrich)
        print(resp.human_summary())
        print()
        print(_json(resp.to_json()))
    finally:
        repo.close()
    return 0


def cmd_trace(args) -> int:
    from rpv.provenance.adapters import EpcisExporter
    from rpv.provenance.repository import ProvenanceRepository
    st = _settings()
    with ProvenanceRepository(st.paths.db_path) as repo:
        lot = repo.get_lot(args.lot_id)
        if lot is None:
            print(f"Không tìm thấy lô {args.lot_id!r}.", file=sys.stderr)
            return 1
        rec = repo.resolve(lot.sku_id, lot.lot_id)
        payload = rec.to_json()
        if args.epcis:
            payload["epcis_document"] = EpcisExporter.export(rec.events, lot.trace_code)
        print(_json(payload))
    return 0


def cmd_audit_verify(args) -> int:
    from rpv.audit.hash_chain import AuditLog
    res = AuditLog(_settings().paths.audit_path).verify()
    print(_json(res))
    return 0 if res["valid"] else 1


def cmd_cache_info(args) -> int:
    from rpv.vision.encoders import load_encoder
    st = _settings()
    st.paths.ensure()
    encoder = load_encoder()
    cache = encoder._cache()
    print(_json({"encoder": encoder.describe(), "cache_dir": str(st.paths.cache_dir),
                 "cache": cache.info() if cache else "đang tắt (RPV_EMBED_CACHE=0)"}))
    return 0


def cmd_cache_clear(args) -> int:
    st = _settings()
    removed = []
    if st.paths.cache_dir.exists():
        for f in sorted(st.paths.cache_dir.glob("*.npz")):
            f.unlink()
            removed.append(f.name)
    print(_json({"removed": removed, "cache_dir": str(st.paths.cache_dir)}))
    return 0


def cmd_status(args) -> int:
    from rpv.audit.hash_chain import AuditLog
    from rpv.provenance.repository import ProvenanceRepository
    from rpv.vision.encoders import load_encoder
    from rpv.vision.prototypes import PrototypeBank

    st = _settings()
    st.paths.ensure()
    try:
        encoder = load_encoder()
        enc_info = encoder.describe()
    except Exception as exc:
        # `status` phải trả lời được KỂ CẢ khi encoder không nạp được — đó chính
        # là lúc người ta cần nó nhất.
        print(_json({"encoder": {"error": str(exc)},
                     "hint": "Xem docs/SETUP.md để cài DINOv2 hoặc chạy offline."}))
        return 1
    bank = PrototypeBank(encoder)
    bank_info: dict = {"exists": False}
    if Path(st.paths.bank_path).exists():
        try:
            bank.load(st.paths.bank_path)
            bank_info = {"exists": True, **bank.summary()}
        except Exception as exc:
            bank_info = {"exists": True, "error": str(exc)}
    db_info: dict = {"exists": Path(st.paths.db_path).exists()}
    if db_info["exists"]:
        with ProvenanceRepository(st.paths.db_path) as repo:
            db_info.update(repo.stats())
            db_info["decisions_by_outcome"] = repo.decision_breakdown()
    cache = encoder._cache()
    print(_json({
        "version": __import__("rpv").__version__,
        "encoder": enc_info,
        "embedding_cache": cache.info() if cache else "đang tắt (RPV_EMBED_CACHE=0)",
        "prototype_bank": bank_info,
        "calibration_exists": Path(st.paths.calib_path).exists(),
        "database": db_info,
        "audit": AuditLog(st.paths.audit_path).verify(),
        "external_adapters_enabled": st.enable_external_adapters,
    }))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rpv", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--encoder", choices=["dinov2", "classical"], default=None,
                        help="ghi đè RPV_ENCODER cho lần chạy này")
    common.add_argument("--variant", choices=["vits14", "vitb14", "vitl14", "vitg14"], default=None,
                        help="variant DINOv2 (ghi đè RPV_DINOV2_VARIANT)")
    common.add_argument("--device", default=None, help="cuda | cpu (ghi đè RPV_DEVICE)")
    common.add_argument("--ckpt", default=None, help="đường dẫn ProjectionHead (.pt)")
    common.add_argument("--no-cache", action="store_true", help="tắt cache embedding")
    common.add_argument("--data-dir", default=None, help="dataset known/unknown đã prepare")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser('prepare-data', help='tạo hai folder known/unknown theo lớp, giữ nguyên nguồn')
    s.add_argument('--source', required=True)
    s.add_argument('--destination', default='data/dataset_v2')
    s.add_argument('--known-ratio', type=float, default=0.7)
    s.add_argument('--seed', type=int, default=1)
    s.add_argument('--groups-csv', default=None)
    s.set_defaults(func=cmd_prepare_data)

    s = sub.add_parser('dataset-info', parents=[common], help='kiểm tra dữ liệu và xuất manifest vai trò ảnh')
    s.add_argument('--k-shot', type=int, default=None)
    s.add_argument('--out', default=None)
    s.set_defaults(func=cmd_dataset_info)

    s = sub.add_parser("init-db", parents=[common], help="tạo lược đồ SQLite")
    s.set_defaults(func=cmd_init_db)

    s = sub.add_parser("warm-cache", parents=[common], help="nhúng trước toàn bộ ảnh")
    s.set_defaults(func=cmd_warm_cache)

    s = sub.add_parser("train-head", parents=[common], help="fine-tune ProjectionHead (episodic)")
    s.add_argument("--out", default=None, help="file .pt đích (mặc định RPV_HEAD_OUT: var/thesis/projection_head.pt)")
    s.add_argument("--out-dim", type=int, default=384)
    s.add_argument("--n-way", type=int, default=5)
    s.add_argument("--k-shot", type=int, default=None)
    s.add_argument("--q-query", type=int, default=8)
    s.add_argument("--episodes", type=int, default=300, help="số episode mỗi epoch")
    s.add_argument("--epochs", type=int, default=20)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--quiet", action="store_true", help="không in lịch sử từng epoch ở cuối")
    s.set_defaults(func=cmd_train_head)

    s = sub.add_parser("calibrate", parents=[common], help="đăng ký prototype + hiệu chỉnh ngưỡng")
    s.add_argument("--k-shot", type=int, default=None)
    s.add_argument("--alpha", type=float, default=None)
    s.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("seed-scenarios", parents=[common], help="seed hồ sơ lô/chứng từ cho các kịch bản")
    s.set_defaults(func=cmd_seed_scenarios)

    s = sub.add_parser("evaluate", parents=[common], help="đánh giá đầy đủ + xuất báo cáo")
    s.add_argument('--scenarios', action='store_true', help='thêm kịch bản hồ sơ MÔ PHỎNG trong DB riêng của run')
    s.add_argument("--k-shot", type=int, default=None)
    s.add_argument("--alpha", type=float, default=None)
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser('enroll', parents=[common], help='đăng ký có kiểm tra chất lượng/nhầm lẫn')
    s.add_argument('action', choices=['create', 'list', 'show', 'add', 'assess', 'publish'])
    s.add_argument('--session', default='')
    s.add_argument('--sku', default='')
    s.add_argument('--label', default='')
    s.add_argument('--operator', default='cli')
    s.add_argument('--role', choices=['support', 'validation', 'negative'], default='support')
    s.add_argument('--group', default='')
    s.add_argument('--negative-sku', default='')
    s.add_argument('--images', nargs='+', default=[])
    s.set_defaults(func=cmd_enroll)

    s = sub.add_parser("register", parents=[common], help="đăng ký SKU mới bằng vài ảnh")
    s.add_argument("sku")
    s.add_argument("images", nargs="+")
    s.add_argument("--label", default="")
    s.add_argument("--operator", default="cli")
    s.set_defaults(func=cmd_register)

    s = sub.add_parser("recognize", parents=[common], help="nhận diện tự do một ảnh")
    s.add_argument("image")
    s.set_defaults(func=cmd_recognize)

    s = sub.add_parser("checkout", parents=[common], help="một lượt quầy đầy đủ")
    s.add_argument("image")
    s.add_argument("--sku", default=None)
    s.add_argument("--lot", default=None)
    s.add_argument("--price", type=float, default=None)
    s.add_argument("--operator", default="cli")
    s.add_argument("--enrich", action="store_true", help="gọi adapter ngoài (nếu được bật)")
    s.set_defaults(func=cmd_checkout)

    s = sub.add_parser("trace", parents=[common], help="xem hồ sơ nguồn gốc của một lô")
    s.add_argument("lot_id")
    s.add_argument("--epcis", action="store_true", help="kèm EPCIS 2.0 JSON-LD")
    s.set_defaults(func=cmd_trace)

    s = sub.add_parser("audit-verify", parents=[common], help="kiểm tra toàn vẹn chuỗi audit")
    s.set_defaults(func=cmd_audit_verify)

    s = sub.add_parser("cache-info", parents=[common], help="thông tin cache embedding")
    s.set_defaults(func=cmd_cache_info)

    s = sub.add_parser("cache-clear", parents=[common], help="xoá cache embedding")
    s.set_defaults(func=cmd_cache_clear)

    s = sub.add_parser("status", parents=[common], help="tình trạng hệ thống")
    s.set_defaults(func=cmd_status)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _apply_overrides(args)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nĐã dừng theo yêu cầu.", file=sys.stderr)
        return 130
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        # Lỗi cấu hình/dữ liệu là chuyện bình thường khi dùng CLI: in gọn một
        # dòng thay vì traceback, nhưng vẫn trả exit code khác 0 cho scripts.
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
