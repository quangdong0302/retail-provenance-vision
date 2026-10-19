"""Gradio workflow for guided registration; callbacks use the shared enrollment service."""
from __future__ import annotations

import gradio as gr

from app.presentation import section
from rpv.util import json_dumps


def render_session(d):
    report = d.get('report')
    names = {'needs_samples': 'Cần bổ sung ảnh', 'needs_review': 'Còn nhầm lẫn — cần kiểm tra',
             'validation_passed': 'Đạt tiêu chí kiểm tra nội bộ'}
    lines = [f"**{d['label']}** · Mã mặt hàng: `{d['sku_id']}`",
             '**Chế độ sử dụng: nhân viên xác nhận.**']
    if d['published']:
        lines.append('Đã đăng ký vào danh mục. Danh mục đã thay đổi nên ngưỡng cũ không còn được sử dụng.')
    if not report:
        lines.append('Thêm ảnh rồi bấm **Kiểm tra nhầm lẫn**.')
    else:
        lines += [f"**Kết quả: {names[report['status']]}**", '',
                  '| Vai trò | Số ảnh | Nhóm vật thể/phiên |', '|---|---:|---:|']
        for key, label in [('support', 'Ảnh mẫu'), ('validation', 'Đúng mặt hàng'), ('negative', 'Mặt hàng khác')]:
            lines.append(f"| {label} | {report['counts'][key]} | {report['groups'][key]} |")
        fmt = lambda x: 'Chưa có ảnh' if x is None else f'{x:.1%}'
        lines += ['', f"Nhận đúng ảnh kiểm tra: **{fmt(report['accuracy'])}**. "
                  f"Ảnh hàng khác bị gán vào mặt hàng này: **{fmt(report['negative_assignment_rate'])}**.",
                  '', '**Mặt hàng gần giống trong danh mục:**']
        lines.extend(f"- {r['label']} (`{r['sku_id']}`)" for r in report['nearest_products'])
        if not report['nearest_products']:
            lines.append('- Chưa có mặt hàng đối chứng.')
        lines += ['', '**Hướng dẫn tiếp theo:**'] + [f'- {g}' for g in report['guidance']]
        comparison = report['comparison']
        if comparison['comparable'] and comparison['accuracy_delta'] is not None:
            lines += ['', f"So với lần trước trên cùng ảnh kiểm tra và danh mục: accuracy thay đổi "
                      f"**{comparison['accuracy_delta']*100:+.1f} điểm phần trăm**."]
            if comparison['negative_assignment_delta'] is not None:
                lines += [f"Tỷ lệ gán nhầm hàng khác thay đổi "
                          f"**{comparison['negative_assignment_delta']*100:+.1f} điểm phần trăm** (giảm là tốt hơn)."]
        elif len(d['history']) > 1:
            lines += ['', 'Tập kiểm tra hoặc danh mục đã đổi; không so sánh trực tiếp với lượt trước.']
        if report['revision'] != d['revision'] or d.get('assessment_current') is False:
            lines += ['', '**Kết quả cũ: cần chạy kiểm tra lại sau khi đổi ảnh/danh mục.**']
        lines += ['', 'Đây là kiểm tra phát triển, không phải kết quả test độc lập hoặc chứng nhận độ chính xác.']
    return '\n'.join(lines)


def build_enrollment_tab(get_service):
    def service():
        return get_service().enrollment

    def choices():
        return [(f"{d['label']} — {d['id'][:8]}", d['id']) for d in service().list_sessions()]

    def display(d, prefix=''):
        samples = [[s['id'], s['filename'], s['role'], s['group_id']] for s in d['samples']]
        remove = [(s['filename'] + ' — ' + s['id'][:8], s['id']) for s in d['samples'] if s['role'] == 'support']
        gallery = [(str(service().sample_path(d['id'], s['id'])), s['filename'])
                   for s in d['samples'] if s['role'] == 'support']
        return (prefix + render_session(d), samples, gr.update(choices=remove, value=None), gallery,
                service().reference_images(d['id']), json_dumps(d['history']))

    def create(sku, label):
        try:
            d = service().create(sku, label, 'staff_demo')
            return (d['id'], gr.update(choices=choices(), value=d['id']), *display(d))
        except (ValueError, OSError, RuntimeError) as exc:
            raise gr.Error(str(exc)) from exc

    def load(session_id):
        try:
            return (session_id, *display(service().get(session_id)))
        except (ValueError, OSError, RuntimeError) as exc:
            raise gr.Error(str(exc)) from exc

    def add(session_id, role, group, negative_sku, files):
        try:
            result = service().add_images(session_id, role,
                [f if isinstance(f, str) else f.name for f in (files or [])], group or '', negative_sku or '')
            prefix = f"Đã nhận {len(result['accepted'])} ảnh.\n\n"
            prefix += ''.join(f"- {r['filename']}: {r['reason']}\n" for r in result['rejected']) + '\n'
            return display(result['session'], prefix)
        except (ValueError, OSError, RuntimeError) as exc:
            raise gr.Error(str(exc)) from exc

    def assess(session_id):
        try:
            return display(service().assess(session_id))
        except (ValueError, OSError, RuntimeError) as exc:
            raise gr.Error(str(exc)) from exc

    def remove(session_id, sample_id):
        try:
            return display(service().remove_support(session_id, sample_id))
        except (ValueError, OSError, RuntimeError) as exc:
            raise gr.Error(str(exc)) from exc

    def publish(session_id):
        try:
            return display(get_service().publish_enrollment(session_id))
        except (ValueError, OSError, RuntimeError) as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Tab('Đăng ký mặt hàng', id='enrollment'):
        gr.HTML(section('Đăng ký có kiểm tra nhầm lẫn',
                        'Tạo hồ sơ → thêm ảnh → kiểm tra với ảnh riêng → bổ sung mẫu → đăng ký vào danh mục.'))
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=330, elem_classes='rpv-card'):
                gr.HTML(section('Hồ sơ mặt hàng', 'Dùng một mã SKU nhất quán cho ảnh và hồ sơ lô.', '01'))
                with gr.Row():
                    sku = gr.Textbox(label='Mã mặt hàng', placeholder='tao_moi')
                    label = gr.Textbox(label='Tên hiển thị', placeholder='Táo mới')
                create_btn = gr.Button('Tạo hồ sơ mới')
                with gr.Accordion('Mở hồ sơ đã lưu', open=False):
                    existing = gr.Dropdown(choices=choices(), label='Hồ sơ đăng ký')
                    with gr.Row():
                        refresh = gr.Button('Cập nhật danh sách')
                        load_btn = gr.Button('Mở hồ sơ')
                session = gr.Textbox(label='Mã hồ sơ đang mở', interactive=False)
                gr.HTML(section('Bổ sung ảnh', 'Mỗi vật thể dùng cùng mã nhóm kể cả khi đổi góc hoặc ngày chụp.', '02'))
                role = gr.Dropdown(choices=[('Ảnh mẫu để đăng ký', 'support'),
                                           ('Ảnh kiểm tra đúng mặt hàng', 'validation'),
                                           ('Ảnh kiểm tra mặt hàng khác', 'negative')], value='support', label='Vai trò ảnh')
                group = gr.Textbox(label='Mã vật thể / phiên chụp', placeholder='qua_01_phien_sang')
                negative_sku = gr.Textbox(label='SKU đúng của ảnh hàng khác', visible=False,
                                         placeholder='Mã đã có trong danh mục')
                files = gr.File(label='Ảnh của nhóm này', file_count='multiple', file_types=['image'])
                add_btn = gr.Button('Kiểm tra chất lượng & thêm ảnh')
                with gr.Accordion('Ảnh cần chuẩn bị', open=False):
                    gr.Markdown('Bắt đầu với **3–5 ảnh mẫu từ ít nhất hai nhóm**. Kiểm tra đầy đủ cần '
                                '**10 ảnh đúng mặt hàng và 10 ảnh hàng khác**, mỗi vai trò từ ít nhất hai nhóm riêng. '
                                'Đây là ngưỡng kiểm tra thử nghiệm. Không dùng lại ảnh mẫu làm ảnh kiểm tra; '
                                'ưu tiên hàng khác dễ nhầm trong danh mục.')
            with gr.Column(scale=6, min_width=330, elem_classes='rpv-card'):
                gr.HTML(section('Kiểm tra & cải thiện', 'Xem mẫu đã lưu, kết quả nhầm lẫn và hướng dẫn bổ sung.', '03'))
                gallery = gr.Gallery(label='Ảnh mẫu đã lưu', columns=3, height=210, interactive=False)
                assess_btn = gr.Button('Kiểm tra nhầm lẫn →', variant='primary', elem_classes='rpv-primary')
                output = gr.Markdown('Tạo hoặc mở hồ sơ để bắt đầu. Kết quả kiểm tra sẽ xuất hiện tại đây.')
                references = gr.Gallery(label='Ảnh tham chiếu của mặt hàng dễ nhầm', columns=3, height=200,
                                        interactive=False)
                with gr.Accordion('Quản lý ảnh mẫu', open=False):
                    samples = gr.Dataframe(headers=['Mã ảnh', 'Tên ảnh', 'Vai trò', 'Nhóm'], interactive=False)
                    remove_id = gr.Dropdown(label='Ảnh mẫu cần bỏ')
                    remove_btn = gr.Button('Bỏ ảnh mẫu')
                with gr.Accordion('Lịch sử kiểm tra · JSON', open=False):
                    history = gr.Code(language='json')
                gr.Markdown('Đăng ký sẽ lưu bản sao danh mục cũ và vô hiệu hóa ngưỡng cũ. '
                            'Danh mục cần nhân viên xác nhận tới khi có hiệu chỉnh phù hợp. '
                            'Nhập hồ sơ lô ở tab **Nhập hồ sơ lô** sau khi đăng ký.')
                publish_btn = gr.Button('Đăng ký vào danh mục', elem_classes='rpv-secondary')
        outputs = [output, samples, remove_id, gallery, references, history]
        create_btn.click(create, [sku, label], [session, existing, *outputs])
        refresh.click(lambda: gr.update(choices=choices()), None, existing)
        load_btn.click(load, existing, [session, *outputs])
        add_btn.click(add, [session, role, group, negative_sku, files], outputs)
        role.change(lambda value: gr.update(visible=value == 'negative'), role, negative_sku, queue=False)
        remove_btn.click(remove, [session, remove_id], outputs)
        assess_btn.click(assess, session, outputs)
        return publish_btn.click(publish, session, outputs)
