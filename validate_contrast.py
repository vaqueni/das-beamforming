"""contrast_speckle 데이터로 무에코 낭종의 CR / CNR / gCNR을 측정한다.

  python validate_contrast.py [--n-waves 75] [--uff PATH]
                              [--xc 0 --zc 30 --r-roi 3 --r-in 5 --r-out 7.5]

- 내 DAS 구현 결과와 PICMUS 파일 동봉 USTB 참조 영상 양쪽에서 측정해 비교한다.
- 결과: results/contrast_metrics.csv, results/contrast_metrics.png
"""
import argparse
import csv
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uffLoader import load_picmus, load_reference_bmode
from dasBeamform import das_beamform
import metrics

DEFAULT_UFF = "./data/PICMUS/PICMUS_simulation_contrast_speckle.uff"
OUT_DIR = "./results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-waves", type=int, default=75)
    ap.add_argument("--uff", default=DEFAULT_UFF)
    ap.add_argument("--xc", type=float, default=0.0)
    ap.add_argument("--zc", type=float, default=30.0)
    ap.add_argument("--r-roi", type=float, default=3.0)
    ap.add_argument("--r-in", type=float, default=5.0)
    ap.add_argument("--r-out", type=float, default=7.5)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    picmus = load_picmus(args.uff)
    x_axis, z_axis = picmus["x_axis"], picmus["z_axis"]
    print(f"loaded {picmus['data'].shape} (waves, ch, samples)  c={picmus['c']} m/s")

    t0 = time.time()
    image = das_beamform(picmus, n_waves=args.n_waves, apodization="dynamic",
                         f_number=1.7, window="tukey50", verbose=False)
    print(f"DAS ({args.n_waves} waves) 완료: {time.time()-t0:.1f} s")

    mine = metrics.to_grid(image, x_axis, z_axis)
    ref = load_reference_bmode(args.uff)

    roi_kw = dict(xc_mm=args.xc, zc_mm=args.zc, r_roi_mm=args.r_roi,
                  r_bg_inner_mm=args.r_in, r_bg_outer_mm=args.r_out)
    m_mine = metrics.contrast_metrics(mine, x_axis, z_axis, **roi_kw)
    m_ref = metrics.contrast_metrics(ref, x_axis, z_axis, **roi_kw)

    print(f"\nROI: 중심 ({args.xc}, {args.zc}) mm  ROI r={args.r_roi} mm  "
          f"bg ring {args.r_in}~{args.r_out} mm   (n_roi={m_mine['n_roi']}, n_bg={m_mine['n_bg']})")
    print(f"{'':>18} | {'CR [dB]':>10} | {'CNR':>8} | {'gCNR':>8}")
    print("-" * 52)
    print(f"{'my DAS':>18} | {m_mine['CR_dB']:10.2f} | {m_mine['CNR']:8.3f} | {m_mine['gCNR']:8.3f}")
    print(f"{'USTB reference':>18} | {m_ref['CR_dB']:10.2f} | {m_ref['CNR']:8.3f} | {m_ref['gCNR']:8.3f}")

    nrmse = metrics.image_nrmse_db(mine, ref)
    print(f"\nUSTB 참조 영상과 dB 영상 NRMSE = {nrmse:.4f}")

    # --- CSV ---
    csv_path = os.path.join(OUT_DIR, "contrast_metrics.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "CR_dB", "CNR", "gCNR", "n_roi", "n_bg", "nrmse_vs_ref"])
        w.writerow(["mine", f"{m_mine['CR_dB']:.3f}", f"{m_mine['CNR']:.4f}",
                    f"{m_mine['gCNR']:.4f}", m_mine["n_roi"], m_mine["n_bg"], f"{nrmse:.4f}"])
        w.writerow(["ustb_ref", f"{m_ref['CR_dB']:.3f}", f"{m_ref['CNR']:.4f}",
                    f"{m_ref['gCNR']:.4f}", m_ref["n_roi"], m_ref["n_bg"], ""])
    print(f"saved {csv_path}")

    # --- figure : B-mode + ROI 원 + gCNR 히스토그램 ---
    ext = [x_axis.min()*1e3, x_axis.max()*1e3, z_axis.max()*1e3, z_axis.min()*1e3]
    bmode = np.clip(metrics.envelope_db(mine), -60, 0)
    th = np.linspace(0, 2*np.pi, 200)

    fig, axs = plt.subplots(1, 2, figsize=(11, 7), gridspec_kw={"width_ratios": [1.3, 1]})
    axs[0].imshow(bmode, cmap="gray", extent=ext, aspect="auto", vmin=-60, vmax=0)
    for rr, col in [(args.r_roi, "r"), (args.r_in, "y"), (args.r_out, "y")]:
        axs[0].plot(args.xc + rr*np.cos(th), args.zc + rr*np.sin(th), col, lw=1.2)
    axs[0].set(title=f"my DAS (n_waves={args.n_waves})\n"
                     f"CR={m_mine['CR_dB']:.1f} dB  CNR={m_mine['CNR']:.2f}  gCNR={m_mine['gCNR']:.3f}",
               xlabel="Lateral [mm]", ylabel="Depth [mm]")

    s = np.abs(mine)
    s_roi = s[m_mine["roi_mask"]]; s_bg = s[m_mine["bg_mask"]]
    lo = 20*np.log10(np.r_[s_roi, s_bg].min() + 1e-30)
    hi = 20*np.log10(s.max())
    edges = np.linspace(lo, hi, 101); ctr = 0.5*(edges[:-1] + edges[1:])
    hi_, _ = np.histogram(20*np.log10(s_roi + 1e-30), bins=edges, density=True)
    ho_, _ = np.histogram(20*np.log10(s_bg + 1e-30), bins=edges, density=True)
    axs[1].plot(ctr, hi_, "r", label="ROI (inside cyst)")
    axs[1].plot(ctr, ho_, "b", label="background ring")
    axs[1].fill_between(ctr, np.minimum(hi_, ho_), color="0.6", label="overlap")
    axs[1].set(title=f"gCNR = 1 - OVL = {m_mine['gCNR']:.3f}",
               xlabel="20·log10|s|  [dB]", ylabel="pdf")
    axs[1].legend()

    fig.tight_layout()
    png = os.path.join(OUT_DIR, "contrast_metrics.png")
    fig.savefig(png, dpi=150)
    print(f"saved {png}")


if __name__ == "__main__":
    main()
