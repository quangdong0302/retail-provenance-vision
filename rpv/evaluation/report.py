"""Xuất báo cáo:."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from rpv.util import json_dumps


def _fmt(v: Any, nd: int = 4) -> str:
    if isinstance(v, float):
        return "n/a" if v != v else f"{v:.{nd}f}"
    return str(v)


def _pct(v: Any) -> str:
    return "n/a" if not isinstance(v, (int, float)) or v != v else f"{v * 100:.2f}%"


def write_report(evaluation: Mapping, scenarios: Optional[Mapping] = None,
                 out_dir: str | Path | None = None, stem: str = "evaluation",
                 extra: Optional[Mapping] = None) -> dict[str, Path]:
    if out_dir is None:
        from rpv.config import settings
        out_dir = settings.paths.reports_dir
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluation": dict(evaluation),
        "scenarios": dict(scenarios) if scenarios else None,
        "extra": dict(extra) if extra else None,
    }
    predictions = payload['evaluation'].pop('predictions', [])
    predictions_path = out_dir / f'{stem}.predictions.jsonl'
    with predictions_path.open('w', encoding='utf-8') as stream:
        for row in predictions:
            stream.write(json_dumps(row, indent=None) + '\n')
    payload['evaluation']['predictions_file'] = predictions_path.name
    json_path = out_dir / f"{stem}.json"
    # json_dumps làm sạch NaN/Infinity: báo cáo phải là JSON hợp lệ để công cụ
    # khác đọc lại được, không chỉ để Python đọc được.
    json_path.write_text(json_dumps(payload), encoding="utf-8")

    cs = evaluation.get("closed_set_on_test_split", {})
    os_ = evaluation.get("open_set_on_test_split", {})
    sel = evaluation.get("selective_prediction_on_test_split", {})
    ver = evaluation.get("verification_on_test_split", {})
    cal = evaluation.get("calibration_on_calib_split", {})
    sp = evaluation.get("splits", {})
    cfg = evaluation.get("config", {})
    figures = {}
    if sel.get('risk_coverage_curve'):
        from rpv.evaluation.plots import write_risk_coverage
        figures = write_risk_coverage(sel['risk_coverage_curve'], sel.get('operating_point_known'), out_dir, stem)

    L: list[str] = []
    A = L.append
    A("# Báo cáo đánh giá hệ thống RPV")
    A("")
    A(f"- Thời điểm: {payload['generated_at']}")
    A(f"- Encoder: `{evaluation.get('encoder_tag', '?')}`")
    enc = cfg.get("encoder") or {}
    if enc:
        A(f"- Backbone: `{enc.get('variant', 'n/a')}` · dim {enc.get('dim')} · ảnh "
          f"{enc.get('image_size', 'n/a')}px · thiết bị `{enc.get('device', 'n/a')}`"
          + (f" · projection head: `{Path(str(enc['projection_head'])).name}`"
             if enc.get("projection_head") else " · không dùng projection head"))
    A(f"- k-shot: {cfg.get('k_shot')} · alpha: {cfg.get('alpha')} · TTA: {cfg.get('use_tta')}")
    A(f"- Dữ liệu: `{cfg.get('data_dir')}`")
    A(f"- Protocol: `{cfg.get('protocol', 'n/a')}` · dataset fingerprint: `{cfg.get('dataset_fingerprint', '')}`")
    A("")
    A("## 1. Phân chia dữ liệu")
    A("")
    A("| Hạng mục | Giá trị |")
    A("|---|---|")
    for k, v in sp.items():
        A(f"| {k} | {v} |")
    A("")
    A("> Prototype dựng từ support; ngưỡng chọn trên calib; số liệu đo trên test. "
      "Lớp novel chưa từng tham gia học biểu diễn.")
    A("")
    A("## 2. Nhận diện (closed-set, đo trên test)")
    A("")
    A("| Metric | Giá trị |")
    A("|---|---|")
    A(f"| Accuracy toàn danh mục | {_pct(cs.get('accuracy_all_classes'))} |")
    A(f"| Macro-F1 toàn danh mục | {_fmt(cs.get('macro_f1_all_classes'))} |")
    A(f"| Macro-F1 (gồm lớp 'unknown') | {_fmt(cs.get('macro_f1_with_unknown'))} |")
    A(f"| Số ảnh test | {cs.get('n_test_images')} |")
    A("")
    A("## 3. Open-set (khả năng nói \"không biết\")")
    A("")
    A("| Metric | Giá trị |")
    A("|---|---|")
    A(f"| Ngưỡng toàn cục (chọn trên calib) | {_fmt(os_.get('threshold'))} |")
    A(f"| Known recall | {_pct(os_.get('known_recall'))} |")
    A(f"| Unknown reject rate | {_pct(os_.get('unknown_reject_rate'))} |")
    A(f"| False accept rate (hàng lạ bị nhận) | {_pct(os_.get('false_accept_rate'))} |")
    A(f"| AUROC | {_fmt(os_.get('auroc'))} |")
    A(f"| Score OOD | {os_.get('score', 'distance')} |")
    fpr = os_.get("fpr_at_95tpr", {})
    A(f"| FPR @ 95% TPR | {_pct(fpr.get('fpr'))} |")
    A("")
    A("## 4. Dự đoán có kiểm soát rủi ro (conformal / selective)")
    A("")
    A("| Metric | Giá trị |")
    A("|---|---|")
    A(f"| Tỉ lệ hệ thống chủ động không kết luận | {_pct(sel.get('abstention_rate'))} |")
    A(f"| Coverage chứa nhãn đúng | {_pct(sel.get('empirical_coverage'))} |")
    A(f"| Coverage chứa nhãn đúng sau gate OOD | {_pct(sel.get('empirical_coverage_after_ood'))} |")
    A(f"| Kích thước tập trung bình | {_fmt(sel.get('mean_prediction_set_size'))} |")
    A(f"| Accuracy của câu trả lời tự động sau các gate | {_pct(sel.get('singleton_accuracy'))} |")
    A(f"| Hàng lạ bị từ chối đúng | {_pct(sel.get('unknown_items_correctly_rejected'))} |")
    A(f"| Hàng lạ được trả lời tự động | {_pct(sel.get('unknown_auto_accept_rate'))} |")
    A(f"| Hàng lạ chuyển nhân viên (OOD hoặc mơ hồ/risk) | {_pct(sel.get('unknown_handoff_rate'))} |")
    A(f"| Coverage tại rủi ro 1% | {_pct(sel.get('coverage_at_risk_1pct'))} |")
    A(f"| Coverage tại rủi ro 5% | {_pct(sel.get('coverage_at_risk_5pct'))} |")
    A(f"| Coverage tại rủi ro 10% | {_pct(sel.get('coverage_at_risk_10pct'))} |")
    A(f"| Phân bố trạng thái | `{sel.get('status_distribution')}` |")
    A("")
    if figures:
        A(f"![Risk–Coverage]({figures['risk_coverage_png'].name})")
        A('')
        A('Đường cong top-1 và điểm gate vận hành dùng hai quy tắc chọn lọc khác nhau; CSV lưu toàn bộ ngưỡng thực nghiệm.')
        A('')
    operating = sel.get('operating_point_known', {})
    mixed = sel.get('operating_point_known_and_unknown', {})
    if operating:
        A('### Ngưỡng vận hành khóa trên known-calibration')
        A('')
        A(f"- Rủi ro mục tiêu: {_pct(operating.get('target_risk'))}; coverage known-test: {_pct(operating.get('coverage'))}; risk known-test: {_pct(operating.get('selective_risk'))}.")
        A(f"- Trên known + unknown: coverage {_pct(mixed.get('coverage'))}; risk {_pct(mixed.get('selective_risk'))}.")
        A(f"- Đạt mục tiêu trên known-test: {operating.get('target_met_on_test')}. Không có câu trả lời thì risk chưa xác định, không coi là đạt.")
        A('- Coverage tại 1/5/10% trên đường cong chỉ là mô tả test; không chọn lại ngưỡng vận hành từ test. Ràng buộc calibration là thực nghiệm, chưa bảo đảm risk dân số.')
        A('')
    A("## 5. Kiểm chứng lời khai (ảnh + SKU khai báo)")
    A("")
    A("| Metric | Giá trị |")
    A("|---|---|")
    A(f"| Số cặp thật / cặp giả mạo | {ver.get('n_genuine')} / {ver.get('n_impostor')} |")
    A(f"| FAR (chấp nhận lời khai sai) | {_pct(ver.get('false_accept_rate'))} |")
    A(f"| FRR (từ chối lời khai đúng) | {_pct(ver.get('false_reject_rate'))} |")
    A(f"| EER | {_pct(ver.get('eer'))} |")
    A(f"| AUROC | {_fmt(ver.get('auroc'))} |")
    A(f"| FAR với SKU sai gần nhất | {_pct(ver.get('nearest_wrong_claim', {}).get('false_accept_rate'))} |")
    A("")
    A("FAR ở đây thuộc verifier, chưa phải tỷ lệ policy cho phép thanh toán.")
    novel = evaluation.get('novel_after_enrollment', {})
    if novel:
        A("")
        A("## Nhận diện lớp unknown sau khi đăng ký few-shot")
        A("")
        A("| k-shot | Số query | Accuracy | Macro-F1 |")
        A("|---|---|---|---|")
        for run in novel.get('runs', []):
            A(f"| {run['k_shot']} | {run.get('n_query', 'skipped')} | {_pct(run.get('accuracy'))} | {_fmt(run.get('macro_f1'))} |")
        A("")
        A(novel.get('threshold_note', ''))
    A("")
    A("## 6. Hiệu chỉnh")
    A("")
    A(f"- Ngưỡng riêng theo lớp: {cal.get('n_per_class_thresholds')} SKU; "
      f"dùng ngưỡng toàn cục: {cal.get('n_fallback_to_global')} SKU.")
    A(f"- Số liệu trên calib: `{cal.get('metrics_on_calib')}`")
    for w in cal.get("warnings", []) or []:
        A(f"- CẢNH BÁO: {w}")
    A("")

    if scenarios:
        A("## 7. Kịch bản nghiệp vụ end-to-end")
        A("")
        A(f"Đạt {scenarios.get('n_passed')}/{scenarios.get('n_cases')} kịch bản "
          f"({_pct(scenarios.get('pass_rate'))}).")
        A("")
        A("| Kịch bản | Mong đợi | Thực tế | Trạng thái quyết định | Kết quả | Mã lý do |")
        A("|---|---|---|---|---|---|")
        for c in scenarios.get("cases", []):
            mark = "PASS" if c["passed"] else "FAIL"
            A(f"| {c['scenario']} | {c['expected']} | {c['actual']} | {c.get('decision_state', '')} | {mark} | {', '.join(c['reasons'])} |")
        A("")
        comparison = scenarios.get('image_only_vs_policy')
        if comparison:
            A(f"Chấp nhận tự động sai trong kịch bản kiểm soát: image-only {comparison['n_unsafe_accept_image_only']}; full-policy {comparison['n_unsafe_accept_full_policy']}.")
            A('Đây là số ca mô phỏng, chưa phải tỷ lệ lỗi ngoài quầy.')
            A('')

    err = evaluation.get("error_analysis", {})
    robustness = evaluation.get('robustness_on_test_split', {})
    if robustness:
        A('## Nhiễu và hard negative trên test')
        A('')
        A(robustness['scope'])
        A('')
        A('| Điều kiện | Accuracy known | Coverage hỗn hợp | Risk hỗn hợp | Từ chối unknown | FAR hard negative |')
        A('|---|---|---|---|---|---|')
        for row in robustness['conditions']:
            A(f"| {row['condition']} | {_pct(row['accuracy_known'])} | {_pct(row['coverage_known_and_unknown'])} | {_pct(row['selective_risk_known_and_unknown'])} | {_pct(row['unknown_reject_rate'])} | {_pct(row['hard_negative_far'])} |")
        A('')
        A(robustness['limitations'])
        A('')
    if err:
        A("## 8. Phân tích lỗi")
        A("")
        A("Cặp nhầm lẫn hay gặp nhất:")
        A("")
        A("| Nhãn thật | Nhãn dự đoán | Số ca |")
        A("|---|---|---|")
        for p in err.get("top_confusion_pairs", [])[:10]:
            A(f"| {p['true']} | {p['pred']} | {p['count']} |")
        A("")
        A("SKU có recall thấp nhất (ưu tiên bổ sung ảnh mẫu):")
        A("")
        for r in err.get("worst_per_class_recall", [])[:8]:
            A(f"- `{r['sku']}`: {_pct(r['recall'])}")
        A("")

    warns = list(evaluation.get("warnings") or [])
    if warns:
        A("## Cảnh báo về cách đọc số liệu")
        A("")
        for w in warns:
            A(f"- {w}")
        A("")

    A("## Giới hạn của kết quả")
    A("")
    A("- Hệ thống KHÔNG suy đoán xuất xứ từ hình ảnh. Kết luận nguồn gốc luôn xuất phát từ hồ sơ lô "
      "và chứng từ; thiếu chứng từ thì trạng thái là \"chưa xác minh\".")
    A("- Số liệu trên dữ liệu mô phỏng chỉ để kiểm chứng pipeline. Khả năng sử dụng tại POS cần được đánh giá "
      "trên ảnh thật thu tại quầy, nhiều ngày và nhiều điều kiện sáng.")
    A("- Chưa xử lý chống giả mạo (ảnh in, ảnh trên màn hình điện thoại) và chưa xử lý nhiều vật thể "
      "trong một khung hình.")
    A("- Dữ liệu từ nguồn cộng đồng chỉ mang tính tham khảo, không có giá trị chứng từ.")

    md_path = out_dir / f"{stem}.md"
    md_path.write_text("\n".join(L), encoding="utf-8")
    return {"json": json_path, "markdown": md_path, 'predictions': predictions_path, **figures}
