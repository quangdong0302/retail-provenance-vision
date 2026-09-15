"""Small deterministic image fixtures, only for unit tests in temporary folders.

These colored shapes exercise embedding/split contracts. They are not product
photos or a dataset for training, demonstrations or reported accuracy.
"""
from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def write_image_fixtures(root: Path) -> None:
    # Preserve test SKU names and split ordering; the images have no retail meaning.
    names = sorted((
        'xoai_cat_hoa_loc', 'xoai_keo', 'thanh_long_ruot_trang', 'thanh_long_ruot_do',
        'cam_sanh', 'cam_vang_uc', 'chuoi_cau', 'chuoi_gia_huong', 'ca_chua_beef',
        'ca_chua_bi', 'khoai_tay_da_lat', 'khoai_lang_mat', 'bap_cai_trang',
        'bap_cai_tim', 'dua_leo', 'bi_xanh', 'ot_sung', 'ot_chi_thien', 'hanh_tay',
        'toi_ly_son', 'nho_ninh_thuan', 'tao_fuji', 'tao_envy', 'le_han_quoc',
        'unknown_dien_thoai', 'unknown_vi_da', 'unknown_chai_nuoc',
        'unknown_do_choi', 'unknown_ban_tay',
    ))
    for index, name in enumerate(names):
        folder = root / name
        folder.mkdir(parents=True)
        color = tuple(round(c * 255) for c in colorsys.hsv_to_rgb(index / len(names), .85, .65))
        for sample in range(6 if name.startswith('unknown_') else 14):
            image = Image.new('RGB', (128, 128), (100, 100, 100))
            draw = ImageDraw.Draw(image)
            offset = sample % 3
            draw.rectangle((20 + offset, 20, 108, 108 - offset), fill=color)
            pixels = np.asarray(image).astype(np.int16)
            noise = np.random.default_rng(index * 100 + sample).integers(-2, 3, pixels.shape)
            Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8)).save(
                folder / f'{sample:03d}.jpg', quality=95)
