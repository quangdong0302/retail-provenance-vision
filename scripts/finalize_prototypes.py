"""Install the validation-selected prototype model and evaluate frozen artifacts."""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ['RPV_DEVICE'] = 'cpu'

import numpy as np
from rpv import config
from rpv.data.dataset import build_splits,index_dataset
from rpv.data.extra import extra_partition,risk_extra_partition
from rpv.evaluation.pipeline import fit_catalog,evaluate_system
from rpv.vision.encoders import load_encoder,image_sha256
from rpv.vision.recognizer import Recognizer,STATUS_MATCH
from scripts.improve_prototypes import save,wait_cached


def profile_env(profile,head):
    return dict(RPV_CATALOG_BACKEND='prototype',RPV_PROJECTION_CKPT=str(head),
        RPV_DECISION_RULE=profile['decision_rule'],RPV_ALPHA=str(profile['alpha']),
        RPV_ENERGY_TEMPERATURE=str(profile['temperature']),RPV_OOD_METHOD='energy',
        RPV_TARGET_KNOWN_RECALL=str(profile['target_known_recall']),
        RPV_CONFIDENCE_METHOD=profile['confidence_method'],RPV_TARGET_RISK=str(profile['target_risk']),
        RPV_CONFORMAL_SCORE='relative',RPV_RISK_CALIBRATION='six_extra_cal',
        RPV_CENTROIDS=str(profile.get('centroids_per_sku',1)),
        RPV_PROTOTYPE_COLOR_WEIGHT=str(profile.get('color_weight',0.0)))


def measure(encoder,bank,calibration,known,unknown):
    recognizer = Recognizer(encoder,bank,calibration)
    truth = [c for c in sorted(known) for p in known[c]]
    paths = [p for c in sorted(known) for p in known[c]]
    D,labels = bank.distances_batch(encoder.embed_paths(paths))
    correct = []
    for label,row in zip(truth,D):
        result = recognizer.recognize_distances(dict(zip(labels,map(float,row))))
        if result.status==STATUS_MATCH:
            correct.append(result.prediction_set==[label])
    UD,_ = bank.distances_batch(encoder.embed_paths(unknown))
    uaccepted = sum(recognizer.recognize_distances(dict(zip(labels,map(float,row)))).status==STATUS_MATCH for row in UD)
    coverage = len(correct)/len(truth)
    errors = sum(not x for x in correct)
    far = uaccepted/len(unknown)
    return dict(known_accuracy=float(np.mean([labels[j] == t for j,t in zip(D.argmin(1),truth)])),
        known_coverage=coverage,known_risk=errors/len(correct) if correct else None,
        n_known=len(truth),n_known_accepted=len(correct),n_known_errors=errors,
        n_unknown=len(unknown),n_unknown_accepted=uaccepted,unknown_false_accept_rate=far,
        balanced_mixture_risk=(errors/len(truth)+far)/(coverage+far) if coverage+far else None,
        natural_mixture_risk=(errors+uaccepted)/(len(correct)+uaccepted) if len(correct)+uaccepted else None)


def update_env(path,values):
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    pending = dict(values)
    for i,line in enumerate(lines):
        key = line.split('=',1)[0]
        if key in pending:
            lines[i] = key+'='+pending.pop(key)
    lines.extend(k+'='+v for k,v in pending.items())
    path.write_text('\n'.join(lines)+'\n',encoding='utf8')


def main():
    initial = config.settings
    folder = initial.paths.head_path.parent
    selected = folder/'trials/selection.json'
    while not selected.exists():
        print('[finalize] Waiting for validation selection.',flush=True)
        time.sleep(15)
    selection = json.loads(selected.read_text(encoding='utf8'))
    if selection['state']!='selected' or selection['test_used']:
        raise RuntimeError('A selection independent of final test is required.')
    winner = selection['winner']
    selected_head = Path(selection['representations'][winner['head']]['path'])
    os.environ.update(profile_env(winner,selected_head))
    st = config.reload_settings()
    root = st.paths.data_dir.resolve()
    splits = build_splits(index_dataset(root),st.calibration.k_shot,st.calibration.class_split_seed,st.calibration.calib_test_seed)
    partition = extra_partition(root)
    risk = risk_extra_partition(partition)
    with (root/'metadata.csv').open(encoding='utf-8-sig',newline='') as stream:
        metadata = {str((root/r['path']).resolve()):r for r in csv.DictReader(stream)}
    encoder = load_encoder()
    def refuse_recompute(*args,**kwargs):
        raise RuntimeError('Backbone recomputation blocked: complete cache first.')
    getattr(encoder,'base',encoder)._compute_from_arrays = refuse_recompute
    wait_cached(encoder,list(metadata),metadata,True)
    if hasattr(encoder,'color'):
        wait_cached(encoder.color,list(metadata),metadata,True)
        encoder.color._compute_from_arrays = refuse_recompute
    bank,calibration = fit_catalog(encoder,splits)
    validation = measure(encoder,bank,calibration,splits.validation,
                          [str((root/r['path']).resolve()) for r in risk['selection']])
    save(folder/'trials/runtime_validation.json',validation)
    if (validation['known_risk'] is None or max(validation['known_risk'],
            validation['balanced_mixture_risk'],validation['natural_mixture_risk']) >= .05):
        raise RuntimeError('Actual runtime validation failed <5% constraint; no test evaluated or model installed.')
    # Freeze the choice before the final test is evaluated.
    frozen = dict(head_sha256=image_sha256(selected_head),profile=winner,validation=validation,
                  risk_partition_fingerprint=risk['fingerprint'],test_used_for_selection=False)
    save(folder/'frozen_selection.json',frozen)
    print('[finalize] Selection frozen; evaluating final test once.',flush=True)
    out,bank,calibration = evaluate_system(encoder=encoder,splits=splits,catalog=(bank,calibration))
    report = out.to_json()
    save(folder/'final_evaluation.json',report)
    extra = measure(encoder,bank,calibration,splits.test,
                   [str((root/r['path']).resolve()) for r in partition['holdout']])
    save(folder/'extra_holdout_evaluation.json',extra)
    for suffix in ('.pt','.training.json','.splits.json'):
        source = selected_head.with_suffix(suffix)
        if not source.exists() and suffix=='.splits.json':
            source = initial.paths.head_path.with_suffix(suffix)
        destination = initial.paths.head_path.with_suffix(suffix)
        if source.exists() and source.resolve()!=destination.resolve():
            tmp = destination.with_name(destination.name+'.selected.tmp')
            shutil.copy2(source,tmp)
            tmp.replace(destination)
    training_path=initial.paths.head_path.with_suffix('.training.json')
    training=json.loads(training_path.read_text(encoding='utf8'))
    training['posthoc_validation']=dict(validation,novel_development_2shot_accuracy=
        selection['representations'][winner['head']]['novel_development_2shot_accuracy'])
    training['history_scope']='Parent head episodic training; shared metric/color adaptation is reported in posthoc_validation.'
    save(training_path,training)
    bank.save(initial.paths.bank_path)
    calibration.save(initial.paths.calib_path)
    values = profile_env(winner,initial.paths.head_path)
    for envfile in (ROOT/'.env',ROOT/'.env.example'):
        update_env(envfile,values)
    # The head's SHA and tags are identical after copying; bank/calibration stay valid.
    save(folder/'cache_reuse_verification.json',dict(backbone_forward_calls=0,
        cached_only_enforced=True,cache=encoder._cache().info(),n_metadata_images=len(metadata),
        color_cache=encoder.color._cache().info() if hasattr(encoder,'color') else None))
    baseline = selection['best_baseline']
    representation = selection['representations'][winner['head']]
    text = ['# Kết quả mô hình prototype', '',
        'Nguồn: `D:\\Downloads\\Source`; giữ cấu trúc, notebook và comment, bỏ classifier logistic và clean_caches.py.',
        '', f"Head đã chọn: `{winner['head']}`; quyết định: `{winner['decision_rule']}`.",
        f"{winner.get('centroids_per_sku',1)} prototype/SKU từ 2 ảnh support; DINOv2 đóng băng, 336 px pad, không TTA.", '',
        f"Trọng số đặc trưng màu HS: {winner.get('color_weight',0):g}; màu được ghép trước khi dựng prototype, không có classifier logistic.", '',
        '| Chỉ số validation | Prototype Source | Sau cải thiện |', '|---|---:|---:|',
        f"| Known accuracy | {selection['representations']['source_noaux']['known_val_accuracy']:.2%} | {validation['known_accuracy']:.2%} |",
        f"| Novel-development 2-shot accuracy | {selection['representations']['source_noaux']['novel_development_2shot_accuracy']:.2%} | {representation['novel_development_2shot_accuracy']:.2%} |",
        f"| Known coverage | {baseline['validation']['known_coverage']:.2%} | {validation['known_coverage']:.2%} |" if baseline else '| Known coverage | Không có cấu hình khả thi | '+f"{validation['known_coverage']:.2%} |",
        f"| Known selective risk | {baseline['validation']['known_risk']:.2%} | {validation['known_risk']:.2%} |" if baseline else '',
        f"| Risk hỗn hợp known/unknown cân bằng 50:50 | {baseline['validation']['balanced_mixture_risk']:.2%} | {validation['balanced_mixture_risk']:.2%} |" if baseline else '',
        '', 'Đối chứng là head Source với cổng conformal tốt nhất trong cùng lưới validation, cùng dữ liệu và ràng buộc risk.', '',
        '## Test sau khi chốt', '',
        f"- Known accuracy: {out.closed_set['accuracy_all_classes']:.2%}.",
        f"- Known coverage/risk: {out.selective['operating_point_known']['coverage']:.2%} / {out.selective['operating_point_known']['selective_risk']:.2%}.",
        f"- Risk hỗn hợp main known + unknown: {out.selective['operating_point_known_and_unknown']['selective_risk']:.2%}.",
        f"- Main unknown tự chấp nhận: {out.selective['unknown_auto_accept_rate']:.2%}.",
        f"- Extra holdout tự chấp nhận: {extra['n_unknown_accepted']}/{extra['n_unknown']} ({extra['unknown_false_accept_rate']:.2%}); risk hỗn hợp theo số ảnh: {extra['natural_mixture_risk']:.2%}.",
        '', 'Novel sau enrollment (accuracy nhận dạng; chưa khẳng định risk tự động cho SKU mới):', '',
        '| Shot | Accuracy | Query |', '|---|---:|---:|']
    for row in out.novel_evaluation.get('runs',[]):
        if 'accuracy' in row:
            text.append(f"| {row['k_shot']} | {row['accuracy']:.2%} | {row['n_query']} |")
    text.extend(['', '## Cache và giới hạn', '',
        f"Vector backbone lưu tại `{encoder._cache().path}`. {len(encoder._cache()):,} vector, dim {encoder._cache_dim()}, dùng lại qua các head; trong lượt chốt mô hình không chạy lại backbone.",
        'Ngưỡng risk fit trên known-calibration và sáu lớp reserved; selection dùng known-validation và sáu lớp reserved khác. Không dùng main test hoặc extra holdout để chọn head/ngưỡng.',
        'Risk <5% là tiêu chí thực nghiệm validation; kết quả test được báo nguyên trạng, không bảo đảm mọi phân phối sử dụng. Các test pool đã xuất hiện trong các lượt nghiên cứu trước của dự án; đây không phải bộ test hoàn toàn mới, chưa từng quan sát.',
        'Backbone được pretrained và head học từ nhiều ảnh base; few-shot áp dụng cho support/catalog và đăng ký lớp mới. Ảnh PackEat không thay thế dữ liệu POS Việt Nam; nguồn gốc lấy từ hồ sơ, không suy ra từ ảnh.',
        'Đóng góp có thể đo bằng ablation: episode tách nhóm, hình học mean nhất quán, chọn head trên base/novel, phân tách calibration/selection theo lớp, cổng selective dưới ràng buộc risk và tích hợp provenance/policy/audit.',
        'Tham số và đầy đủ kết quả: `trials/selection.json`, `trials/validation_profiles.json`, `frozen_selection.json`, `final_evaluation.json`, `extra_holdout_evaluation.json`.', ''])
    (folder/'MODEL_RESULTS.md').write_text('\n'.join(text),encoding='utf8')
    save(folder/'finalization_status.json',dict(state='completed',head=winner['head'],validation=validation,
        test_known=out.selective['operating_point_known'],test_mixed=out.selective['operating_point_known_and_unknown'],extra=extra))
    print('[finalize] Complete. '+str(folder/'MODEL_RESULTS.md'),flush=True)


if __name__ == '__main__':
    main()
