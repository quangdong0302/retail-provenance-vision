"""Shared, offline UI styling and escaped HTML for the retail demo."""
from __future__ import annotations

from html import escape


CSS = """
.gradio-container { width: 100% !important; min-width: 0 !important; box-sizing: border-box;
  max-width: 1320px !important; margin: auto; padding: 28px 36px !important; }
.gradio-container .main { width: 100% !important; max-width: none !important; padding: 0 !important; }
.gradio-container .html-container > .prose { padding: 0 !important; }
.gradio-container .html-container { padding: 0 !important; }
body, .gradio-container { background: #f5f7f6; }
.dark .gradio-container { background: #111c20; }
.rpv-header { display: flex; align-items: center; justify-content: space-between; gap: 16px;
  padding: 0 0 24px; border-bottom: 1px solid #dbe4df; margin-bottom: 20px; }
.rpv-brand { display: flex; align-items: center; gap: 12px; color: #142e29; }
.rpv-logo { width: 44px; height: 44px; background: #16634d; border-radius: 13px;
  display: grid; place-items: center; color: white; }
.rpv-logo svg { color: #fff !important; }
.rpv-brand strong { font-size: 22px; letter-spacing: -.7px; }
.rpv-brand small { display: block; font-size: 12px; color: #687c73; margin-top: 2px; }
.rpv-header-note { color: #65786f; font-size: 12px; display: flex; align-items: center; gap: 7px; }
.rpv-dot { width: 7px; height: 7px; background: #238866; border-radius: 50%; display: inline-block; }
.rpv-hero { display: flex; justify-content: space-between; align-items: center; gap: 24px; padding: 8px 0 20px; }
.rpv-eyebrow { color: #227255; font-size: 11px; text-transform: uppercase; letter-spacing: 1.7px; font-weight: 700; }
.rpv-hero h1 { font-size: clamp(25px, 3vw, 36px); line-height: 1.25; letter-spacing: -1px;
  color: #172f29; margin: 9px 0 10px; font-weight: 650; }
.rpv-hero p { color: #63766d; font-size: 14px; line-height: 1.8; margin: 0; max-width: 680px; }
.rpv-hero-tag { padding: 10px 15px; border: 1px solid #d4e2da; border-radius: 24px;
  color: #326550; font-size: 12px; white-space: nowrap; background: #edf4ee; }
.rpv-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 0 0 20px; }
.rpv-stat { background: #fff; border: 1px solid #e0e7e2; padding: 16px 19px; border-radius: 12px; }
.rpv-stat-label { color: #6e7e76; font-size: 12px; }
.rpv-stat-value { color: #1d3c30; font-size: 24px; line-height: 1.4; font-weight: 650; margin-top: 5px; }
.rpv-stat-value.rpv-small { font-size: 17px; padding: 5px 0; }
.rpv-stat-detail { font-size: 11px; color: #74857a; margin-top: 3px; }
.rpv-notice { display: flex; gap: 10px; align-items: flex-start; background: #fff8e8;
  border: 1px solid #ecdfb9; color: #775727; padding: 13px 17px; border-radius: 10px;
  font-size: 13px; line-height: 1.65; margin: 0 0 16px; }
.rpv-notice.rpv-ready { background: #edf5f0; border-color: #d5e6da; color: #35624a; }
.rpv-tabs [role="tablist"] { gap: 8px; padding: 8px 0 16px; border-bottom: 0 !important; overflow-x: auto; }
.rpv-tabs [role="tab"] { border: 1px solid transparent !important; border-radius: 9px !important;
  padding: 11px 18px !important; min-height: 42px !important; height: auto !important;
  color: #60736a; white-space: nowrap; font-weight: 550; }
.rpv-tabs [role="tablist"] { height: auto !important; min-height: 58px; }
.rpv-tabs [role="tab"].selected { color: #fff !important; background: #17634d !important; }
.rpv-tabs > .tabitem { border: 0 !important; background: transparent !important; padding: 0 !important; }
.rpv-card { border: 1px solid #e0e7e2 !important; border-radius: 14px !important;
  background: var(--block-background-fill) !important; padding: 23px !important; }
.rpv-section { margin-bottom: 10px; }
.rpv-section h2 { margin: 0 0 6px; font-size: 20px; letter-spacing: -.5px; color: #203c31; font-weight: 650; }
.rpv-section p { margin: 0; font-size: 13px; color: #708177; line-height: 1.7; }
.rpv-step { display: inline-grid; place-items: center; width: 24px; height: 24px; background: #edf4ee;
  color: #337357; border-radius: 7px; font-size: 12px; margin-right: 7px; vertical-align: middle; }
.rpv-primary { min-height: 46px !important; font-weight: 600 !important; }
.rpv-secondary { min-height: 40px !important; }
.rpv-image { border-radius: 11px !important; }
.rpv-empty { min-height: 345px; display: flex; flex-direction: column; align-items: center;
  justify-content: center; text-align: center; padding: 30px 25px; background: #f8faf8;
  border: 1px dashed #d6e0d8; border-radius: 12px; }
.rpv-empty-icon { display: grid; place-items: center; width: 64px; height: 64px; border-radius: 19px;
  background: #eaf2eb; color: #538367; margin-bottom: 18px; }
.rpv-empty h3 { font-size: 18px; color: #335541; margin: 0 0 9px; font-weight: 600; }
.rpv-empty p { max-width: 310px; color: #7c8d81; font-size: 13px; line-height: 1.8; margin: 0; }
.rpv-flow { display: flex; flex-wrap: wrap; justify-content: center; gap: 7px; margin-top: 24px; }
.rpv-flow span { color: #6d8274; background: #eef3ee; border-radius: 6px; padding: 6px 9px; font-size: 11px; }
.rpv-result { border-radius: 12px; padding: 21px; border: 1px solid #c6e0ce; background: #f1f8f2; }
.rpv-result.warning, .rpv-result.assist { background: #fff8e9; border-color: #e8d8ae; }
.rpv-result.block { background: #fff1ee; border-color: #e9c4ba; }
.rpv-result h3 { color: #226341; margin: 8px 0; font-size: 23px; font-weight: 650; }
.rpv-result.warning h3, .rpv-result.assist h3 { color: #805d21; }
.rpv-result.block h3 { color: #a14739; }
.rpv-result-label { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #66796b; }
.rpv-result p { color: #596b5d; line-height: 1.75; font-size: 13px; }
.rpv-details { margin-top: 17px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.rpv-detail { padding: 14px; border: 1px solid #e0e7e1; border-radius: 10px; background: #fff; }
.rpv-detail small { color: #738579; font-size: 11px; display: block; margin-bottom: 6px; }
.rpv-detail strong { color: #2d4737; font-size: 13px; line-height: 1.6; overflow-wrap: anywhere; }
.rpv-reasons { margin: 16px 0 0; padding-left: 20px; color: #5f7064; font-size: 13px; line-height: 1.8; }
.rpv-candidates { margin-top: 17px; overflow-x: auto; }
.rpv-candidates table { width: 100%; border-collapse: collapse; font-size: 12px; }
.rpv-candidates th { text-align: left; color: #78887d; font-weight: 500; padding: 10px 8px; }
.rpv-candidates td { padding: 10px 8px; border-top: 1px solid #e4eae5; color: #3a5141; }
.rpv-meta { color: #87958b; font-size: 11px; line-height: 1.7; margin-top: 13px; overflow-wrap: anywhere; }
.rpv-footer { display: flex; justify-content: space-between; gap: 16px; padding: 22px 0 2px;
  margin-top: 24px; border-top: 1px solid #dde5df; color: #819087; font-size: 11px; line-height: 1.8; }
.rpv-guide { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 18px; }
.rpv-guide article { border: 1px solid #dbe5dd; border-radius: 12px; padding: 22px; background: #fff; }
.rpv-guide h3 { margin: 10px 0; color: #254b36; font-size: 17px; }
.rpv-guide p { color: #728277; font-size: 13px; line-height: 1.8; }
.rpv-card .prose { font-size: 13px; }
.dark .rpv-brand, .dark .rpv-hero h1, .dark .rpv-section h2 { color: #d8e8df; }
.dark .rpv-stat, .dark .rpv-detail, .dark .rpv-guide article { background: #1b2b2a; border-color: #344844; }
.dark .rpv-stat-value, .dark .rpv-detail strong, .dark .rpv-guide h3 { color: #d3e8db; }
.dark .rpv-empty { background: #1b2b2a; border-color: #3d5147; }
.dark .rpv-empty h3 { color: #bfdbc7; }
@media (max-width: 760px) {
  .gradio-container.gradio-container { padding: 16px !important; }
  .rpv-header { padding-bottom: 16px; margin-bottom: 12px; }
  .rpv-header-note, .rpv-hero-tag { display: none; }
  .rpv-hero { padding-bottom: 17px; }
  .rpv-stats { grid-template-columns: 1fr 1fr; gap: 8px; }
  .rpv-stat { padding: 12px; }
  .rpv-stat-value { font-size: 21px; }
  .rpv-card { padding: 16px !important; min-width: 0 !important; }
  .rpv-tabs { min-width: 0 !important; width: 100%; }
  .rpv-tabs > .tab-wrapper { min-width: 0; }
  .rpv-tabs [role="tablist"] { flex-wrap: nowrap !important; max-width: 100%; }
  .rpv-tabs [role="tab"] { padding: 10px 12px !important; flex-shrink: 0; }
  .rpv-tabs .row > .rpv-card { flex: 1 1 100% !important; }
  .rpv-empty { min-height: 245px; }
  .rpv-guide { grid-template-columns: 1fr; }
  .rpv-footer { flex-direction: column; gap: 4px; }
}
"""


def icon(name: str, size: int = 24) -> str:
    paths = {
        'leaf': '<path d="M20 4c0 12-4 16-10 16a6 6 0 0 1-6-6C4 8 8 4 20 4Z"/><path d="m4 20 10-10"/>',
        'scan': '<path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5M7 12h10"/>',
        'shield': '<path d="m12 3 8 4v5c0 5-8 9-8 9s-8-4-8-9V7Z"/><path d="m8 12 3 3 5-6"/>',
    }
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            'stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" '
            f'aria-hidden="true">{paths[name]}</svg>')


HEADER = f"""<header class="rpv-header">
  <div class="rpv-brand"><div class="rpv-logo">{icon('leaf', 26)}</div>
    <div><strong>RPV<span style="font-weight:400;color:#6b8175"> / Retail Vision</span></strong>
    <small>Nhận diện sản phẩm · Đối chiếu nguồn gốc</small></div></div>
  <div class="rpv-header-note">{icon('shield', 16)} Vision-to-Provenance</div>
</header>
<section class="rpv-hero"><div><div class="rpv-eyebrow">Không gian vận hành bán lẻ</div>
  <h1>Mỗi mặt hàng, một quyết định rõ ràng.</h1>
  <p>Nhận diện từ hình ảnh, đối chiếu hồ sơ lô và giải thích quyết định ngay tại quầy.</p>
</div><span class="rpv-hero-tag">Ảnh nhận diện · Hồ sơ xác minh</span></section>"""

FOOTER = """<footer class="rpv-footer"><span>RPV / Retail Provenance Vision</span>
  <span>Ảnh hỗ trợ nhận diện SKU. Nguồn gốc được đối chiếu từ hồ sơ lô.</span></footer>"""


def section(title: str, description: str, step: str = '') -> str:
    number = f'<span class="rpv-step">{escape(step)}</span>' if step else ''
    return (f'<div class="rpv-section"><h2>{number}{escape(title)}</h2>'
            f'<p>{escape(description)}</p></div>')


def empty_result(title='Sẵn sàng kiểm tra', description=None) -> str:
    description = description or 'Thêm ảnh và chọn mặt hàng. Kết quả đối chiếu cùng hướng xử lý sẽ xuất hiện tại đây.'
    return (f'<div class="rpv-empty"><div class="rpv-empty-icon">{icon("scan", 30)}</div>'
            f'<h3>{escape(title)}</h3><p>{escape(description)}</p>'
            '<div class="rpv-flow"><span>01 · Hình ảnh</span><span>02 · Hồ sơ lô</span>'
            '<span>03 · Quyết định</span></div></div>')


def system_overview(service) -> tuple[str, str]:
    stats = service.repo.stats()
    calibration = service.calibration
    controlled = bool(calibration and calibration.ood and calibration.risk_control.get('enabled'))
    cards = [
        (len(service.bank), 'Mặt hàng nhận diện', 'SKU trong danh mục thị giác'),
        (stats.get('lots', 0), 'Hồ sơ lô', 'Lô đã nhập trong hệ thống'),
        (stats.get('decisions', 0), 'Lượt đối chiếu', 'Quyết định đã được lưu'),
        ('Đã hiệu chỉnh' if controlled else 'Cần hiệu chỉnh', 'Cơ chế từ chối',
         'Ngưỡng thị giác đã nạp' if controlled else 'Cần nhân viên xác nhận'),
    ]
    html = '<div class="rpv-stats">'
    for value, label, detail in cards:
        small = ' rpv-small' if isinstance(value, str) else ''
        html += (f'<div class="rpv-stat"><div class="rpv-stat-label">{escape(label)}</div>'
                 f'<div class="rpv-stat-value{small}">{escape(str(value))}</div>'
                 f'<div class="rpv-stat-detail">{escape(detail)}</div></div>')
    html += '</div>'
    if not len(service.bank):
        note = 'Danh mục đang trống. Đăng ký mặt hàng, sau đó nhập hồ sơ lô để bắt đầu kiểm tra.'
    elif not controlled:
        note = 'Chưa có đủ hiệu chỉnh OOD và rủi ro cho danh mục hiện tại. Kết quả cần nhân viên xác nhận.'
    elif calibration.risk_control.get('confidence_threshold') is None:
        note = 'Chưa tìm được ngưỡng rủi ro phù hợp. Hệ thống chuyển kết quả sang nhân viên xác nhận.'
    else:
        note = 'Đã nạp ngưỡng đối chiếu. Mỗi giao dịch vẫn được kiểm tra riêng theo hình ảnh và hồ sơ lô.'
    cls = ' rpv-ready' if controlled and calibration.risk_control.get('confidence_threshold') is not None else ''
    return html, f'<div class="rpv-notice{cls}">{icon("shield", 20)}<span>{escape(note)}</span></div>'


def render_checkout(response: dict) -> str:
    """Present service facts without treating distances/p-values as probabilities."""
    outcome = response.get('outcome', 'assist')
    styles = {
        'allow': ('allow', 'Có thể tiếp tục thanh toán', 'Hình ảnh và hồ sơ đáp ứng quy tắc đối chiếu.'),
        'allow_warning': ('warning', 'Có thể thanh toán · Có cảnh báo', 'Đọc cảnh báo dưới đây trước khi tiếp tục.'),
        'assist': ('assist', 'Cần nhân viên xác nhận', 'Kiểm tra lại mặt hàng, ảnh chụp hoặc hồ sơ lô trước khi tiếp tục.'),
        'block': ('block', 'Dừng giao dịch', 'Xử lý nguyên nhân được ghi nhận trước khi kiểm tra lại.'),
    }
    css_class, title, action = styles.get(outcome, styles['assist'])
    decision_labels = {'match': 'Khớp nhận diện và hồ sơ', 'mismatch': 'Mâu thuẫn mặt hàng khai báo',
                       'abstain': 'Chưa đủ chắc chắn / hàng ngoài danh mục', 'provenance_exception': 'Lỗi hồ sơ lô'}
    rec = response.get('recognition') or {}
    provenance = response.get('provenance') or {}
    sku = provenance.get('sku') or {}
    lot = provenance.get('lot') or {}
    vision_labels = {'match': 'Đã nhận diện', 'ambiguous': 'Cần xác nhận', 'unknown': 'Ngoài danh mục / chưa nhận diện',
                     'empty_catalog': 'Danh mục chưa có ảnh mẫu'}
    fields = [
        ('Kết luận đối chiếu', decision_labels.get(response.get('decision_state'), 'Cần kiểm tra')),
        ('Nhận diện hình ảnh', vision_labels.get(rec.get('status'), 'Chưa có kết luận')),
        ('Mặt hàng trong hồ sơ', sku.get('name') or sku.get('sku_id') or 'Chưa xác định'),
        ('Lô được đối chiếu', lot.get('lot_id') or 'Chưa có hồ sơ lô'),
        ('Nguồn gốc theo hồ sơ', response.get('origin_summary') or 'Chưa có thông tin'),
        ('Thời gian xử lý', f"{response['timings_ms']['total']:.0f} ms" if response.get('timings_ms', {}).get('total') is not None else 'Chưa ghi nhận'),
    ]
    html = (f'<section class="rpv-result {css_class}" role="status" aria-live="polite">'
            f'<div class="rpv-result-label">Kết quả kiểm tra tại quầy</div><h3>{title}</h3><p>{action}</p></section>'
            '<div class="rpv-details">')
    for label, value in fields:
        html += f'<div class="rpv-detail"><small>{escape(label)}</small><strong>{escape(str(value))}</strong></div>'
    html += '</div>'
    messages = response.get('messages') or []
    if messages:
        html += '<ul class="rpv-reasons">' + ''.join(f'<li>{escape(str(m))}</li>' for m in messages) + '</ul>'
    candidates = rec.get('candidates') or []
    if candidates:
        probability_score = rec.get('score_semantics') == 'one_minus_class_probability'
        score_label = 'Điểm mô hình' if probability_score else 'Khoảng cách'
        html += ('<div class="rpv-candidates"><table><thead><tr><th>Ứng viên hình ảnh</th>'
                 f'<th>{score_label}</th><th>Trong tập chấp nhận</th></tr></thead><tbody>')
        for c in candidates:
            score = 1-c['distance'] if probability_score else c['distance']
            html += (f'<tr><td>{escape(str(c["label"]))}</td><td>{score:.4f}</td>'
                     f'<td>{"Có" if c["in_prediction_set"] else "Không"}</td></tr>')
        html += '</tbody></table></div>'
    audit = response.get('audit') or {}
    reasons = ', '.join(response.get('reasons') or []) or '—'
    html += (f'<div class="rpv-meta">Mã lý do: {escape(reasons)}<br>'
             f'Nhật ký #{escape(str(audit.get("index", "—")))} · '
             f'{escape(str(audit.get("entry_hash", ""))[:16])}</div>')
    return html
