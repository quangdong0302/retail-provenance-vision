"""Export reproducible scientific risk-coverage figures and their source data."""
from __future__ import annotations

import csv
from pathlib import Path


def write_risk_coverage(curve, operating, out_dir, stem):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root = Path(out_dir)
    data = root / f'{stem}.risk_coverage.csv'
    with data.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['coverage', 'selective_risk', 'n_answered'])
        writer.writeheader()
        writer.writerows(curve)
    figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    axis.step([p['coverage'] * 100 for p in curve], [p['selective_risk'] * 100 for p in curve],
              where='pre', label='Known-test top-1 diagnostic sweep')
    target_pct = (operating or {}).get('target_risk', .01) * 100
    axis.axhline(target_pct, color='#c64938', linestyle='--', linewidth=1, label=f'Target risk {target_pct:g}%')
    if operating and operating.get('selective_risk') is not None:
        axis.scatter([operating['coverage'] * 100], [operating['selective_risk'] * 100],
                     color='#245b8a', marker='x', zorder=3, label='Deployed gate: threshold fitted on calibration')
    axis.set(xlabel='Coverage (%)', ylabel='Selective risk (%)', xlim=(0, 100))
    axis.set_ylim(bottom=0)
    axis.grid(alpha=.25)
    axis.legend(fontsize=8)
    svg = root / f'{stem}.risk_coverage.svg'
    png = root / f'{stem}.risk_coverage.png'
    figure.savefig(svg)
    figure.savefig(png, dpi=180)
    plt.close(figure)
    return {'risk_coverage_csv': data, 'risk_coverage_svg': svg, 'risk_coverage_png': png}
