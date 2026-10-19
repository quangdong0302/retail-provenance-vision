"""Gradio screens. All callbacks delegate to the existing shared services."""
from __future__ import annotations

import inspect
from datetime import date, timedelta

import gradio as gr

from app.enrollment import build_enrollment_tab
from app.presentation import CSS, HEADER, FOOTER, empty_result, section, system_overview
from rpv.domain.models import EvidenceType, LotStatus


def appearance() -> dict:
    return {
        'theme': gr.themes.Base(primary_hue='emerald', secondary_hue='emerald', neutral_hue='slate',
                                font=['Segoe UI', 'Arial', 'sans-serif'],
                                font_mono=['Consolas', 'monospace']).set(
            body_background_fill='#f5f7f6', block_background_fill='#ffffff',
            block_border_color='#e0e7e2', block_radius='10px',
            button_primary_background_fill='#17634d', button_primary_background_fill_hover='#124d3c',
            button_primary_text_color='#ffffff', input_background_fill='#f9fbf9',
            input_border_color='#dce5df', input_radius='8px',
        ),
        'css': CSS,
    }


def launch_options() -> dict:
    """Gradio 6 moved theme and CSS from Blocks to launch."""
    return appearance() if 'css' in inspect.signature(gr.Blocks.launch).parameters else {}


def build_demo(get_service, sku_choices, lot_choices, intake, checkout, trace, set_status, audit_view):
    def overview():
        return system_overview(get_service())

    def refresh(sku=None):
        choices = sku_choices()
        return (gr.update(choices=choices, value=sku if sku in choices else None),
                gr.update(choices=lot_choices(sku), value=None),
                gr.update(choices=lot_choices(), value=None), *overview())

    legacy = {} if 'css' in inspect.signature(gr.Blocks.launch).parameters else appearance()
    with gr.Blocks(title='RPV / Retail Vision', analytics_enabled=False, **legacy) as demo:
        gr.HTML(HEADER)
        initial_stats, initial_note = overview()
        stats = gr.HTML(initial_stats)
        note = gr.HTML(initial_note)

        with gr.Tabs(elem_classes='rpv-tabs'):
            with gr.Tab('Quầy kiểm tra', id='checkout'):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=330, elem_classes='rpv-card'):
                        gr.HTML(section('Kiểm tra mặt hàng', 'Chụp rõ một mặt hàng, chọn mã khai báo và đối chiếu hồ sơ.'))
                        co_image = gr.Image(label='Ảnh sản phẩm', type='filepath', height=270,
                                            sources=['upload', 'webcam'], elem_classes='rpv-image')
                        gr.Markdown('Ảnh nên đủ sáng và có trọn sản phẩm trong khung hình.')
                        use_claim = gr.Checkbox(label='Đối chiếu mã mặt hàng khai báo', value=True)
                        co_sku = gr.Dropdown(sku_choices(), label='Mặt hàng (SKU)',
                                             info='Chọn đúng mã trên quầy để kiểm chứng.', allow_custom_value=True)
                        co_lot = gr.Dropdown(lot_choices(), label='Lô hàng cần đối chiếu',
                                             info='Không chọn: hệ thống tìm lô đang sử dụng. Ảnh không xác định được lô vật lý.',
                                             allow_custom_value=True)
                        with gr.Accordion('Thông tin giá (tùy chọn)', open=False):
                            co_price = gr.Number(label='Giá khai báo (VND)', value=None, minimum=0)
                        co_btn = gr.Button('Kiểm tra mặt hàng →', variant='primary', elem_classes='rpv-primary')
                        with gr.Row():
                            clear_btn = gr.Button('Làm mới lượt kiểm tra', elem_classes='rpv-secondary')
                            refresh_btn = gr.Button('Cập nhật danh mục', elem_classes='rpv-secondary')
                    with gr.Column(scale=6, min_width=330, elem_classes='rpv-card'):
                        gr.HTML(section('Kết quả đối chiếu', 'Quyết định, lý do và thông tin lô được trình bày cùng nhau.'))
                        co_out = gr.HTML(empty_result())
                        with gr.Accordion('Chi tiết kỹ thuật · JSON', open=False):
                            co_json = gr.Code(value='{}', label='Phản hồi dịch vụ', language='json')

            enrollment_published = build_enrollment_tab(get_service)

            with gr.Tab('Nhập hồ sơ lô', id='intake'):
                gr.HTML(section('Tiếp nhận mặt hàng & hồ sơ lô',
                                'Liên kết mặt hàng với nhà cung cấp, nguồn gốc và chứng từ của từng lô.'))
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=320, elem_classes='rpv-card'):
                        gr.HTML(section('Thông tin mặt hàng', 'Dùng mã đã đăng ký hoặc thêm ảnh mẫu cho mã mới.', '01'))
                        sku_id = gr.Textbox(label='Mã mặt hàng (SKU)', placeholder='xoai_cat_hoa_loc')
                        sku_name = gr.Textbox(label='Tên mặt hàng', placeholder='Xoài cát Hòa Lộc')
                        with gr.Row():
                            price = gr.Number(label='Giá bán (VND/kg)', value=42000, minimum=0)
                            category = gr.Dropdown([('Nông sản tươi', 'fresh_produce'), ('Hàng nhập khẩu', 'imported'),
                                                    ('Hàng đóng gói', 'packaged'), ('Nhóm khác', 'default')],
                                                   value='fresh_produce', label='Nhóm hàng')
                        images = gr.File(label='Ảnh mẫu cho mặt hàng mới', file_count='multiple', file_types=['image'])
                        gr.Markdown('SKU đã đăng ký có kiểm tra: có thể bỏ trống ảnh để giữ nguyên mẫu nhận diện.')
                        gr.HTML(section('Nhà cung cấp', 'Thông tin này được dùng để truy ngược hồ sơ.', '02'))
                        supplier_name = gr.Textbox(label='Tên nhà cung cấp', placeholder='HTX Cao Lãnh')
                        supplier_addr = gr.Textbox(label='Địa chỉ', placeholder='Cao Lãnh, Đồng Tháp')
                    with gr.Column(scale=1, min_width=320, elem_classes='rpv-card'):
                        gr.HTML(section('Nguồn gốc & thời hạn', 'Nhập theo hồ sơ của lô, không suy từ ảnh sản phẩm.', '03'))
                        origin_country = gr.Dropdown([('Việt Nam', 'VN'), ('Hoa Kỳ', 'US'), ('Trung Quốc', 'CN'),
                                                       ('Thái Lan', 'TH'), ('Úc', 'AU'), ('New Zealand', 'NZ'),
                                                       ('Nhật Bản', 'JP'), ('Hàn Quốc', 'KR'), ('Chưa rõ', 'UNKNOWN')],
                                                      value='VN', label='Nước xuất xứ')
                        origin_region = gr.Textbox(label='Vùng trồng / sản xuất', placeholder='Cao Lãnh, Đồng Tháp')
                        farm = gr.Textbox(label='Nông trại / cơ sở đóng gói', placeholder='Tên cơ sở trên hồ sơ')
                        with gr.Row():
                            production_date = gr.Textbox(label='Sản xuất / thu hoạch', placeholder='YYYY-MM-DD',
                                                         value=str(date.today() - timedelta(days=2)))
                            expiry_date = gr.Textbox(label='Hạn sử dụng', placeholder='YYYY-MM-DD',
                                                     value=str(date.today() + timedelta(days=20)))
                        gr.HTML(section('Chứng từ kèm theo', 'Chỉ đánh dấu đã xác minh sau khi đối chiếu hồ sơ.', '04'))
                        evidence_names = {
                            'delivery_note': 'Phiếu giao hàng', 'supplier_declaration': 'Cam kết nhà cung cấp',
                            'import_customs_doc': 'Hồ sơ hải quan', 'phytosanitary_cert': 'Kiểm dịch thực vật',
                            'vietgap_cert': 'Chứng nhận VietGAP', 'globalgap_cert': 'Chứng nhận GlobalGAP',
                            'organic_cert': 'Chứng nhận hữu cơ',
                            'lab_test_report': 'Kết quả kiểm nghiệm', 'photo': 'Ảnh hồ sơ',
                            'qr_payload': 'Dữ liệu mã QR', 'other': 'Chứng từ khác',
                        }
                        evidence_types = gr.CheckboxGroup(
                            [(evidence_names.get(e.value, e.value), e.value) for e in EvidenceType],
                            label='Loại chứng từ', value=[EvidenceType.DELIVERY_NOTE.value,
                                                         EvidenceType.SUPPLIER_DECLARATION.value])
                        evidence_verified = gr.Checkbox(label='Nhân viên đã đối chiếu bản gốc', value=False)
                        gr.Markdown('Bản demo lưu bản ghi chứng từ khai báo; chưa tải hoặc kiểm chứng tệp chứng từ gốc.')
                btn = gr.Button('Lưu mặt hàng & hồ sơ lô →', variant='primary', elem_classes='rpv-primary')
                intake_out = gr.Markdown()

            with gr.Tab('Truy xuất nguồn gốc', id='trace'):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=320, elem_classes='rpv-card'):
                        gr.HTML(section('Tra cứu hồ sơ lô', 'Xem nguồn gốc, chứng từ và lịch sử sự kiện.'))
                        tr_lot = gr.Dropdown(lot_choices(), label='Chọn lô hàng')
                        tr_epcis = gr.Checkbox(label='Kèm dữ liệu trao đổi EPCIS 2.0', value=False)
                        tr_btn = gr.Button('Xem hồ sơ lô →', variant='primary', elem_classes='rpv-primary')
                        with gr.Accordion('Quản lý trạng thái lô', open=False):
                            gr.Markdown('Thay đổi sẽ được lưu và áp dụng ở lượt kiểm tra tiếp theo.')
                            status_labels = {'active': 'Đang sử dụng', 'recalled': 'Thu hồi', 'quarantined': 'Tạm giữ',
                                             'expired': 'Hết hạn', 'sold_out': 'Đã hết'}
                            st_status = gr.Dropdown([(status_labels.get(s.value, s.value), s.value) for s in LotStatus],
                                                    value=LotStatus.RECALLED.value, label='Trạng thái mới')
                            st_note = gr.Textbox(label='Lý do thay đổi', placeholder='Ghi rõ lý do để lưu lịch sử')
                            st_btn = gr.Button('Lưu trạng thái lô')
                            st_out = gr.Markdown()
                    with gr.Column(scale=7, min_width=330, elem_classes='rpv-card'):
                        gr.HTML(section('Hồ sơ nguồn gốc', 'Dữ liệu được ghi nhận khi tiếp nhận lô hàng.'))
                        tr_out = gr.Markdown('Chọn lô và bấm **Xem hồ sơ lô** để xem thông tin.')
                        with gr.Accordion('Dữ liệu hồ sơ · JSON / EPCIS', open=False):
                            tr_json = gr.Code(value='{}', label='Hồ sơ', language='json')

            with gr.Tab('Nhật ký kiểm toán', id='audit'):
                with gr.Column(elem_classes='rpv-card'):
                    gr.HTML(section('Kiểm tra lịch sử quyết định', 'Đối chiếu chuỗi hash và xem các bản ghi gần nhất.'))
                    gr.Markdown('Phép kiểm tra phát hiện sửa nội dung, đổi thứ tự hoặc xóa bản ghi giữa chuỗi. '
                                'Chưa có nơi lưu hash đầu chuỗi độc lập để phát hiện cắt đuôi hay viết lại toàn bộ log.')
                    au_btn = gr.Button('Kiểm tra toàn vẹn nhật ký →', variant='primary', elem_classes='rpv-primary')
                    au_out = gr.Markdown('Chưa chạy kiểm tra trong phiên này.')

            with gr.Tab('Hướng dẫn demo', id='guide'):
                gr.HTML(section('Một lượt demo từ đầu đến cuối', 'Chuẩn bị danh mục và hồ sơ, sau đó thử các tình huống tại quầy.'))
                gr.HTML('''<div class="rpv-guide">
                    <article><span class="rpv-step">01</span><h3>Đăng ký mặt hàng</h3>
                    <p>Tạo hồ sơ, thêm ảnh mẫu và ảnh kiểm tra riêng. Xem các mặt hàng dễ nhầm trước khi đăng ký.</p></article>
                    <article><span class="rpv-step">02</span><h3>Nhập hồ sơ lô</h3>
                    <p>Khai nhà cung cấp, xuất xứ, ngày và chứng từ. Dùng cùng mã SKU với mặt hàng vừa đăng ký.</p></article>
                    <article><span class="rpv-step">03</span><h3>Kiểm tra tại quầy</h3>
                    <p>Thêm ảnh, chọn SKU và lô. Đọc quyết định, lý do và hướng xử lý trước khi tiếp tục.</p></article>
                    </div>''')
                with gr.Column(elem_classes='rpv-card'):
                    gr.Markdown('### Các tình huống demo\n\n'
                                '- **Đúng mặt hàng, đủ hồ sơ:** kiểm tra ảnh với SKU và lô phù hợp.\n'
                                '- **Khai sai SKU:** giữ nguyên ảnh và chọn mặt hàng khác.\n'
                                '- **Hàng ngoài danh mục:** tải ảnh mặt hàng chưa đăng ký.\n'
                                '- **Lỗi hồ sơ:** thử lô thiếu chứng từ, hết hạn hoặc đã thu hồi.\n\n'
                                'Danh mục cần có hiệu chỉnh phù hợp để trả lời tự động. Mặt hàng mới đăng ký '
                                'vẫn cần nhân viên xác nhận. Kết quả demo không thay thế đánh giá trên tập test độc lập.')
                    with gr.Accordion('Thông tin hệ thống', open=False):
                        gr.JSON(value=get_service().encoder.describe(), label='Encoder đang dùng')

        gr.HTML(FOOTER)
        common = [co_sku, co_lot, tr_lot, stats, note]
        enrollment_published.then(refresh, co_sku, common)
        invalidated = lambda: (empty_result(), '{}')
        co_image.change(invalidated, None, [co_out, co_json], queue=False)
        co_price.change(invalidated, None, [co_out, co_json], queue=False)
        co_lot.change(invalidated, None, [co_out, co_json], queue=False)
        co_sku.change(lambda sku: gr.update(choices=lot_choices(sku), value=None), co_sku, co_lot,
                      queue=False).then(invalidated, None, [co_out, co_json], queue=False)
        use_claim.change(lambda enabled: (gr.update(interactive=enabled),
                                         gr.update(interactive=enabled, value=None)),
                         use_claim, [co_sku, co_lot], queue=False).then(
                             invalidated, None, [co_out, co_json], queue=False)
        clear_btn.click(lambda: (None, None, None, None, True, empty_result(), '{}'), None,
                        [co_image, co_sku, co_lot, co_price, use_claim, co_out, co_json], queue=False)
        co_btn.click(checkout, [co_image, co_sku, co_lot, co_price, use_claim], [co_out, co_json]).then(
            overview, None, [stats, note])
        refresh_btn.click(refresh, co_sku, common).then(invalidated, None, [co_out, co_json], queue=False)
        btn.click(intake, [sku_id, sku_name, price, category, supplier_name, supplier_addr, origin_country,
                          origin_region, farm, production_date, expiry_date, evidence_types,
                          evidence_verified, images], [intake_out, co_sku, co_lot]).then(refresh, co_sku, common)
        tr_btn.click(trace, [tr_lot, tr_epcis], [tr_out, tr_json])
        tr_lot.change(lambda: ('Chọn lô và bấm **Xem hồ sơ lô** để xem thông tin.', '{}'), None,
                      [tr_out, tr_json], queue=False)
        st_btn.click(set_status, [tr_lot, st_status, st_note], [st_out, tr_lot]).then(
            refresh, co_sku, common).then(invalidated, None, [co_out, co_json], queue=False)
        au_btn.click(audit_view, None, au_out)
        return demo
