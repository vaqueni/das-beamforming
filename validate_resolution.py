"""resolution_distortion 데이터로 point target FWHM(횡/축 해상도)을 측정한다.

  python validate_resolution.py [--n-waves 75] [--uff PATH]

- 내 DAS 구현 결과와, PICMUS 파일에 동봉된 USTB 참조 영상 양쪽에서 FWHM을 재고
  표로 비교한다.
- 결과: results/resolution_fwhm.csv, results/resolution_fwhm.png
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

DEFAULT_UFF = "./data/PICMUS/PICMUS_simulation_resolution_distortion.uff"
OUT_DIR = "./results"


def _measure(grid, x_axis, z_axis, lam):
    rows = metrics.measure_all_points(grid, x_axis, z_axis,
                                      wavelength_m=lam, win_mm=3.0, drop_db=6.0)
    return rows


def _print_table(title, rows):
    print(f"\n=== {title} ===")
    print(f"{'point (x,z) mm':>16} | {'lat FWHM':>17} | {'axial FWHM':>17}")
    print(f"{'':>16} | {'mm':>8} {'λ':>8} | {'mm':>8} {'λ':>8}")
    print("-" * 58)
    lat, ax = [], []
    for r in rows:
        lm = r["fwhm_lateral_m"] * 1e3
        am = r["fwhm_axial_m"] * 1e3
        print(f"({r['x_nominal_mm']:+5.0f},{r['z_nominal_mm']:4.0f})     | "
              f"{lm:8.3f} {r.get('fwhm_lateral_lambda', float('nan')):8.2f} | "
              f"{am:8.3f} {r.get('fwhm_axial_lambda', float('nan')):8.2f}")
        if np.isfinite(lm):
            lat.append(lm)
        if np.isfinite(am):
            ax.append(am)
    print("-" * 58)
    print(f"{'mean (finite)':>16} | {np.mean(lat):8.3f} {'':>8} | {np.mean(ax):8.3f}")
    return np.mean(lat), np.mean(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-waves", type=int, default=75)
    ap.add_argument("--uff", default=DEFAULT_UFF)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    picmus = load_picmus(args.uff)
    x_axis, z_axis = picmus["x_axis"], picmus["z_axis"]
    lam = picmus["c"] / picmus["f0"]
    print(f"loaded {picmus['data'].shape} (waves, ch, samples)  "
          f"c={picmus['c']} m/s  f0={picmus['f0']/1e6:.3f} MHz  λ={lam*1e3:.4f} mm")

    t0 = time.time()
    image = das_beamform(picmus, n_waves=args.n_waves, apodization="dynamic",
                         f_number=1.7, window="tukey50", verbose=False)
    print(f"DAS ({args.n_waves} waves) 완료: {time.time()-t0:.1f} s")

    mine = metrics.to_grid(image, x_axis, z_axis)
    ref = load_reference_bmode(args.uff)

    rows_mine = _measure(mine, x_axis, z_axis, lam)
    rows_ref = _measure(ref, x_axis, z_axis, lam)

    _print_table(f"my DAS (n_waves={args.n_waves}, dynamic f/1.7 tukey50)", rows_mine)
    _print_table("USTB reference (beamformed_data in .uff)", rows_ref)

    nrmse = metrics.image_nrmse_db(mine, ref)
    print(f"\nUSTB 참조 영상과 dB 영상 NRMSE = {nrmse:.4f}  (0에 가까울수록 일치)")

    # --- CSV ---
    csv_path = os.path.join(OUT_DIR, "resolution_fwhm.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "x_nom_mm", "z_nom_mm", "x_peak_mm", "z_peak_mm",
                    "fwhm_lat_mm", "fwhm_lat_lambda", "fwhm_ax_mm", "fwhm_ax_lambda"])
        for src, rows in [("mine", rows_mine), ("ustb_ref", rows_ref)]:
            for r in rows:
                w.writerow([src, r["x_nominal_mm"], r["z_nominal_mm"],
                            f"{r['x_peak']*1e3:.3f}", f"{r['z_peak']*1e3:.3f}",
                            f"{r['fwhm_lateral_m']*1e3:.4f}", f"{r.get('fwhm_lateral_lambda',float('nan')):.3f}",
                            f"{r['fwhm_axial_m']*1e3:.4f}", f"{r.get('fwhm_axial_lambda',float('nan')):.3f}"])
    print(f"saved {csv_path}")

    # --- figure : B-mode + 측정 point 마커 + 대표 프로파일 ---
    ext = [x_axis.min()*1e3, x_axis.max()*1e3, z_axis.max()*1e3, z_axis.min()*1e3]
    bmode = np.clip(metrics.envelope_db(mine), -60, 0)

    fig, axs = plt.subplots(1, 3, figsize=(13, 7),
                            gridspec_kw={"width_ratios": [2, 1, 1]})
    axs[0].imshow(bmode, cmap="gray", extent=ext, aspect="auto", vmin=-60, vmax=0)
    for r in rows_mine:
        axs[0].plot(r["x_peak"]*1e3, r["z_peak"]*1e3, "r+", ms=9, mew=1.3)
    axs[0].set(title=f"my DAS (n_waves={args.n_waves})", xlabel="Lateral [mm]", ylabel="Depth [mm]")

    # 대표: (0, 20 mm) point의 lateral / axial 프로파일
    r0 = metrics.measure_point_fwhm(mine, x_axis, z_axis, 0.0, 20.0, win_mm=3.0)
    ixw = np.abs(x_axis - r0["x_peak"]) <= 3e-3
    izw = np.abs(z_axis - r0["z_peak"]) <= 3e-3
    iz_pk = np.argmin(np.abs(z_axis - r0["z_peak"]))
    ix_pk = np.argmin(np.abs(x_axis - r0["x_peak"]))

    lp = np.abs(mine[iz_pk, ixw]); lp = 20*np.log10(lp/lp.max()+1e-12)
    axs[1].plot((x_axis[ixw]-r0["x_peak"])*1e3, lp)
    axs[1].axhline(-6, color="r", ls="--", lw=1)
    axs[1].set(title=f"lateral cut @ (0, 20mm)\nFWHM={r0['fwhm_lateral_m']*1e3:.3f} mm",
               xlabel="x - x_peak [mm]", ylabel="dB", ylim=(-40, 1))

    ap_ = np.abs(mine[izw, ix_pk]); ap_ = 20*np.log10(ap_/ap_.max()+1e-12)
    axs[2].plot((z_axis[izw]-r0["z_peak"])*1e3, ap_)
    axs[2].axhline(-6, color="r", ls="--", lw=1)
    axs[2].set(title=f"axial cut @ (0, 20mm)\nFWHM={r0['fwhm_axial_m']*1e3:.3f} mm",
               xlabel="z - z_peak [mm]", ylabel="dB", ylim=(-40, 1))

    fig.tight_layout()
    png = os.path.join(OUT_DIR, "resolution_fwhm.png")
    fig.savefig(png, dpi=150)
    print(f"saved {png}")


if __name__ == "__main__":
    main()
