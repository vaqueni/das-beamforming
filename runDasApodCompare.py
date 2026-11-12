"""dynamic apodization 파라미터(F-number x Tukey roll) 스윕.

  python runDasApodCompare.py [--dataset contrast|resolution]
                              [--backend gpu|cpu] [--n-waves 75] [--no-png]

- 190개 설정(f# 0.1~1.9 x tukey10~100)마다 75-wave compounding 영상을 만들고
  데이터셋에 맞는 화질 지표를 잰다.
    --dataset contrast   : contrast_speckle 낭종에서 CR / CNR / gCNR
    --dataset resolution : resolution_distortion point target 18개의 FWHM(횡/축)
- hilbert(analytic signal)는 f#/window와 무관하므로 루프 밖에서 한 번만 계산한다.
- dataset/backend별로 출력 경로를 나눠 서로의 결과를 덮어쓰지 않는다.
  (contrast는 기존 경로 유지, resolution은 _resolution 접미사)
    timing_log_{suffix}.txt                 설정별 시간 (beamform / metrics / plot / total)
    outputs/tukeyNN_{suffix}/*.png          설정별 B-mode 영상 (--no-png면 생략)
    results/apod_sweep_{suffix}.csv         설정별 지표 + 시간
    results/apod_sweep_{suffix}_heatmap.png f# x tukey 히트맵
    results/apod_sweep_{suffix}_points.csv  (resolution 전용) point별 FWHM 원자료
  (2026-09-04 CPU 스윕 원본은 timing_log.txt, outputs/tukeyNN/에 그대로 둔다.)
- beamform 시간은 GPU 동기화 + 결과를 host로 내리는 시간(D2H)까지 포함한다.
- 두 데이터셋 CSV를 모두 만든 뒤 runParetoFront.py를 돌리면
  CR vs FWHM 파레토 프론트(해상도-대조도 트레이드오프)를 얻는다.
"""
import argparse
import csv
import os
import time
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert

from uffLoader import load_picmus
from dasBeamform import das_beamform, to_bmode
import metrics

DATASETS = {
    "contrast": "./data/PICMUS/PICMUS_simulation_contrast_speckle.uff",
    "resolution": "./data/PICMUS/PICMUS_simulation_resolution_distortion.uff",
}
F_NUMBERS = [f / 10 for f in range(1, 20)]          # 0.1 ~ 1.9
TUKEY_PCTS = [i * 10 for i in range(1, 11)]         # 10 ~ 100
DEFAULT_CFG = (1.7, 50)                              # 다른 검증 스크립트의 기본 설정

# 데이터셋별 지표 컬럼과 히트맵 패널 정의.
#   (csv 컬럼명, 히트맵 제목, colormap, 최적값 선택 함수)
PANELS = {
    "contrast": [
        ("CR_dB", "CR [dB]  (lower = better)", "Blues_r", np.nanargmin),
        ("CNR", "CNR  (higher = better)", "Blues", np.nanargmax),
        ("gCNR", "gCNR  (higher = better)", "Blues", np.nanargmax),
    ],
    "resolution": [
        ("fwhm_lat_mm", "lateral FWHM [mm]  (lower = better)", "Blues_r", np.nanargmin),
        ("fwhm_ax_mm", "axial FWHM [mm]  (lower = better)", "Blues_r", np.nanargmin),
        ("n_points_lat", "detected points (of 18)  (higher = better)", "Blues", np.nanargmax),
    ],
}
METRIC_KEYS = {
    "contrast": ["CR_dB", "CNR", "gCNR"],
    "resolution": ["fwhm_lat_mm", "fwhm_lat_lambda", "fwhm_ax_mm", "fwhm_ax_lambda",
                   "n_points_lat", "n_points_ax"],
}


def make_backend(name, picmus, n_waves):
    """(hilbert 1회 시간 [s], beamform(f_number, window) -> numpy complex 함수) 반환."""
    if name == "gpu":
        import cupy as cp
        from dasBeamformGpu import analytic_signal_gpu, das_beamform_cuda

        t = time.perf_counter()
        analytic = analytic_signal_gpu(picmus["data"])
        cp.cuda.Device().synchronize()
        t_hilbert = time.perf_counter() - t

        def beamform(f_number, window):
            img = das_beamform_cuda(picmus, n_waves=n_waves, apodization="dynamic",
                                    f_number=f_number, window=window, analytic=analytic)
            return cp.asnumpy(img)          # D2H가 끝나야 반환되므로 동기화도 겸한다
        return t_hilbert, beamform

    t = time.perf_counter()
    analytic = hilbert(picmus["data"], axis=-1)
    t_hilbert = time.perf_counter() - t

    def beamform(f_number, window):
        return das_beamform(picmus, n_waves=n_waves, apodization="dynamic",
                            f_number=f_number, window=window, verbose=False,
                            analytic=analytic)
    return t_hilbert, beamform


def measure_contrast(grid, x_axis, z_axis, lam):
    """낭종 ROI/배경 링에서 CR / CNR / gCNR. (row dict, 콘솔 요약 문자열)"""
    m = metrics.contrast_metrics(grid, x_axis, z_axis)
    row = {k: m[k] for k in METRIC_KEYS["contrast"]}
    txt = f"CR={m['CR_dB']:.2f} dB  CNR={m['CNR']:.3f}  gCNR={m['gCNR']:.4f}"
    return row, txt, None


def measure_resolution(grid, x_axis, z_axis, lam):
    """point target 18개의 FWHM 평균. (row dict, 콘솔 요약, point별 원자료 리스트)

    -6 dB 교차점을 ±3 mm 창 안에서 못 찾은 point는 NaN이 되므로 평균에서 제외하고,
    몇 개가 살아남았는지(n_points_*)를 같이 기록한다. f#가 아주 작으면 사이드로브가
    창을 채워 검출이 깨지는데, 그 자체가 설정 품질을 말해 주는 정보다.
    """
    pts = metrics.measure_all_points(grid, x_axis, z_axis,
                                     wavelength_m=lam, win_mm=3.0, drop_db=6.0)
    lat = np.array([p["fwhm_lateral_m"] for p in pts], dtype=float) * 1e3
    ax = np.array([p["fwhm_axial_m"] for p in pts], dtype=float) * 1e3
    n_lat = int(np.isfinite(lat).sum())
    n_ax = int(np.isfinite(ax).sum())
    mean_lat = float(np.nanmean(lat)) if n_lat else float("nan")
    mean_ax = float(np.nanmean(ax)) if n_ax else float("nan")

    row = dict(fwhm_lat_mm=mean_lat,
               fwhm_lat_lambda=mean_lat * 1e-3 / lam if n_lat else float("nan"),
               fwhm_ax_mm=mean_ax,
               fwhm_ax_lambda=mean_ax * 1e-3 / lam if n_ax else float("nan"),
               n_points_lat=n_lat, n_points_ax=n_ax)
    txt = (f"lat FWHM={mean_lat:.3f} mm ({row['fwhm_lat_lambda']:.2f}λ)  "
           f"ax FWHM={mean_ax:.3f} mm  pts={n_lat}/{len(pts)}")
    return row, txt, pts


MEASURE = {"contrast": measure_contrast, "resolution": measure_resolution}


def plot_heatmap(rows, path, dataset, backend, n_waves):
    """f# (가로) x tukey (세로) 격자에 데이터셋별 지표 3개를 색으로 표시."""
    fi = {f: j for j, f in enumerate(F_NUMBERS)}
    ti = {t: i for i, t in enumerate(TUKEY_PCTS)}
    panels = PANELS[dataset]

    fig, axs = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    for ax, (key, title, cmap, pick) in zip(axs, panels):
        grid = np.full((len(TUKEY_PCTS), len(F_NUMBERS)), np.nan)
        for r in rows:
            grid[ti[r["tukey"]], fi[r["f_number"]]] = r[key]
        im = ax.imshow(grid, cmap=cmap, aspect="auto", origin="lower",
                       extent=[F_NUMBERS[0] - 0.05, F_NUMBERS[-1] + 0.05,
                               TUKEY_PCTS[0] - 5, TUKEY_PCTS[-1] + 5])
        fig.colorbar(im, ax=ax, pad=0.01)

        if np.isfinite(grid).any():
            bi, bj = np.unravel_index(pick(grid), grid.shape)
            best_f, best_t, best_v = F_NUMBERS[bj], TUKEY_PCTS[bi], grid[bi, bj]
            ax.plot(best_f, best_t, marker="*", ms=16, mfc="#f2b705", mec="black", mew=1.0,
                    ls="none", label=f"best: f/{best_f:.1f}, tukey{best_t} = {best_v:.3g}")
        d_f, d_t = DEFAULT_CFG
        ax.plot(d_f, d_t, marker="s", ms=12, mfc="none", mec="#d62728", mew=2.0, ls="none",
                label=f"default: f/{d_f}, tukey{d_t} = {grid[ti[d_t], fi[d_f]]:.3g}")
        ax.set_title(title, loc="left")
        ax.set(ylabel="Tukey roll [%]", yticks=TUKEY_PCTS)
        # 범례는 축 위 오른쪽 바깥에 둬서 히트맵 칸(특히 ★)을 가리지 않게 한다
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=8,
                  frameon=False, borderaxespad=0.2)
    axs[-1].set(xlabel="F-number", xticks=F_NUMBERS)
    axs[-1].tick_params(axis="x", labelsize=8)
    fig.suptitle(f"Apodization sweep, {dataset} (n_waves={n_waves}, backend={backend})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["contrast", "resolution"], default="contrast",
                    help="contrast=CR/CNR/gCNR, resolution=point target FWHM")
    ap.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--n-waves", type=int, default=75)
    ap.add_argument("--no-png", action="store_true", help="설정별 B-mode PNG 저장 생략")
    ap.add_argument("--limit", type=int, default=0,
                    help="앞쪽 N개 설정만 돌린다 (0=전체). 코드 검증용 스모크 테스트")
    args = ap.parse_args()

    # contrast는 기존 산출물 경로를 그대로 유지한다(README/그림 링크 보존)
    suffix = args.backend if args.dataset == "contrast" else f"{args.dataset}_{args.backend}"
    timing_log = f"./timing_log_{suffix}.txt"
    png_dir = f"./outputs/tukeyNN_{suffix}"
    csv_path = f"./results/apod_sweep_{suffix}.csv"
    points_csv = f"./results/apod_sweep_{suffix}_points.csv"
    heatmap_path = f"./results/apod_sweep_{suffix}_heatmap.png"
    os.makedirs("./results", exist_ok=True)
    if not args.no_png:
        os.makedirs(png_dir, exist_ok=True)

    picmus = load_picmus(DATASETS[args.dataset])
    x_axis, z_axis = picmus["x_axis"], picmus["z_axis"]
    lam = picmus["c"] / picmus["f0"]
    extent = [x_axis.min() * 1e3, x_axis.max() * 1e3, z_axis.max() * 1e3, z_axis.min() * 1e3]
    measure = MEASURE[args.dataset]

    configs = [(f"dynamic_f{f}_tukey{t}", f, t) for t in TUKEY_PCTS for f in F_NUMBERS]
    if args.limit:
        configs = configs[:args.limit]

    t_all_start = time.perf_counter()
    t_hilbert, beamform = make_backend(args.backend, picmus, args.n_waves)

    # GPU는 첫 호출에 커널 컴파일이 섞이므로 측정 전에 한 번 따로 돌린다
    t_warm = 0.0
    if args.backend == "gpu":
        t = time.perf_counter()
        beamform(DEFAULT_CFG[0], f"tukey{DEFAULT_CFG[1]}")
        t_warm = time.perf_counter() - t

    with open(timing_log, "w", encoding="utf-8") as fh:
        fh.write(f"run start: {datetime.now().isoformat()}\n")
        fh.write(f"dataset: {args.dataset}  ({DATASETS[args.dataset]})\n")
        fh.write(f"backend: {args.backend}\n")
        fh.write(f"n_configs: {len(configs)}  n_waves: {args.n_waves}  png: {not args.no_png}\n")
        fh.write(f"lambda: {lam * 1e3:.4f} mm  (c={picmus['c']} m/s, f0={picmus['f0'] / 1e6:.3f} MHz)\n")
        fh.write(f"hilbert (1회, 전체 설정 공유): {t_hilbert * 1e3:.1f} ms\n")
        if args.backend == "gpu":
            fh.write(f"warm-up (커널 컴파일 포함, 측정 제외): {t_warm * 1e3:.1f} ms\n")
        fh.write("\n[i/N] label: beamform / metrics / plot / total [ms]\n")

    rows = []
    point_rows = []
    for i, (label, f_number, tukey) in enumerate(configs, 1):
        t0 = time.perf_counter()
        image = beamform(f_number, f"tukey{tukey}")
        t1 = time.perf_counter()

        grid = metrics.to_grid(image, x_axis, z_axis)
        m, summary, pts = measure(grid, x_axis, z_axis, lam)
        t2 = time.perf_counter()

        if pts is not None:
            for p in pts:
                point_rows.append(dict(
                    label=label, f_number=f_number, tukey=tukey,
                    x_nom_mm=p["x_nominal_mm"], z_nom_mm=p["z_nominal_mm"],
                    x_peak_mm=p["x_peak"] * 1e3, z_peak_mm=p["z_peak"] * 1e3,
                    fwhm_lat_mm=p["fwhm_lateral_m"] * 1e3,
                    fwhm_ax_mm=p["fwhm_axial_m"] * 1e3))

        if not args.no_png:
            bmode = to_bmode(image, x_axis, z_axis)
            plt.figure(figsize=(5, 8))
            plt.imshow(bmode, cmap="gray", extent=extent, aspect="auto")
            plt.title(f"DAS B-mode ({args.dataset}, n_waves={args.n_waves}, {args.backend})\n{label}")
            plt.xlabel("Lateral [mm]")
            plt.ylabel("Depth [mm]")
            plt.colorbar(label="dB")
            plt.tight_layout()
            plt.savefig(os.path.join(png_dir, f"das_result_apod_{label}.png"), dpi=150)
            plt.close()
        t3 = time.perf_counter()

        ms = [(t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t3 - t0) * 1e3]
        row = dict(label=label, f_number=f_number, tukey=tukey,
                   beamform_ms=ms[0], metrics_ms=ms[1], plot_ms=ms[2], total_ms=ms[3])
        row.update(m)
        rows.append(row)
        print(f"[{i}/{len(configs)}] {label}: beamform {ms[0]:.1f} ms  {summary}")
        with open(timing_log, "a", encoding="utf-8") as fh:
            fh.write(f"[{i}/{len(configs)}] {label}: "
                     f"{ms[0]:.1f} / {ms[1]:.1f} / {ms[2]:.1f} / {ms[3]:.1f}\n")

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if point_rows:
        with open(points_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(point_rows[0].keys()))
            w.writeheader()
            w.writerows(point_rows)
    plot_heatmap(rows, heatmap_path, args.dataset, args.backend, args.n_waves)

    total = time.perf_counter() - t_all_start
    bf = np.array([r["beamform_ms"] for r in rows])
    with open(timing_log, "a", encoding="utf-8") as fh:
        fh.write(f"\nbeamform 합계: {bf.sum() / 1e3:.2f} sec  "
                 f"(설정당 평균 {bf.mean():.1f} ms, 중앙값 {np.median(bf):.1f} ms)\n")
        fh.write(f"metrics 합계: {sum(r['metrics_ms'] for r in rows) / 1e3:.2f} sec\n")
        fh.write(f"plot 합계: {sum(r['plot_ms'] for r in rows) / 1e3:.2f} sec\n")
        fh.write(f"run end: {datetime.now().isoformat()}\n")
        fh.write(f"total_elapsed_sec: {total:.1f}\n")
        fh.write(f"total_elapsed_min: {total / 60:.2f}\n")
    out = [timing_log, csv_path, heatmap_path] + ([points_csv] if point_rows else [])
    print("saved " + ", ".join(out))


if __name__ == "__main__":
    main()
