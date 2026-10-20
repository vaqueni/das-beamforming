"""CPU(NumPy) vs GPU(CuPy vectorized / CUDA RawKernel) DAS 처리시간과 영상 일치도 비교.

  python benchmark_gpu.py [--n-waves 75] [--repeats 10] [--skip-cpu]

- 두 PICMUS 시뮬레이션 데이터(contrast_speckle, resolution_distortion) 각각에 대해
  CPU 1회, GPU 구현별 warm-up 1회 + repeats회 측정 (cuda Event로 GPU 시간만 정확히 잰다).
- 일치도: CPU 결과 대비 dB 영상 NRMSE, 복소 최대 상대오차, USTB 참조 대비 NRMSE,
  그리고 CR/CNR/gCNR(contrast), 평균 FWHM(resolution)이 CPU와 같게 나오는지.
- 결과: results/gpu_benchmark.csv, results/gpu_benchmark.png
"""
import argparse
import csv
import os
import time

import numpy as np
import cupy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uffLoader import load_picmus, load_reference_bmode
from dasBeamform import das_beamform
from dasBeamformGpu import analytic_signal_gpu, das_beamform_cupy, das_beamform_cuda
import metrics

DATASETS = {
    "contrast": "./data/PICMUS/PICMUS_simulation_contrast_speckle.uff",
    "resolution": "./data/PICMUS/PICMUS_simulation_resolution_distortion.uff",
}
OUT_DIR = "./results"
APOD_KW = dict(apodization="dynamic", f_number=1.7, window="tukey50")


def gpu_time_ms(fn, repeats):
    """fn()을 repeats회 실행한 GPU 경과시간 [ms] 리스트 (warm-up 1회 별도)."""
    fn()
    cp.cuda.Device().synchronize()
    start, end = cp.cuda.Event(), cp.cuda.Event()
    times = []
    for _ in range(repeats):
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end))
    return times


def end_to_end_ms(picmus, n_waves, repeats):
    """RF(host) -> H2D -> hilbert -> CUDA kernel -> D2H 전체 시간 [ms]."""
    def run():
        a = analytic_signal_gpu(picmus["data"])
        return cp.asnumpy(das_beamform_cuda(picmus, n_waves=n_waves, analytic=a, **APOD_KW))
    run()
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        run()
        times.append((time.perf_counter() - t) * 1e3)
    return times


def image_metrics(name, grid, picmus):
    x_axis, z_axis = picmus["x_axis"], picmus["z_axis"]
    if name == "contrast":
        m = metrics.contrast_metrics(grid, x_axis, z_axis)
        return dict(CR_dB=m["CR_dB"], CNR=m["CNR"], gCNR=m["gCNR"])
    rows = metrics.measure_all_points(grid, x_axis, z_axis, win_mm=3.0, drop_db=6.0)
    return dict(
        fwhm_lat_mm=np.mean([r["fwhm_lateral_m"] for r in rows]) * 1e3,
        fwhm_ax_mm=np.mean([r["fwhm_axial_m"] for r in rows]) * 1e3,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-waves", type=int, default=75)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--skip-cpu", action="store_true", help="CPU 측정 생략 (GPU만 빠르게 확인)")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    dev = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"GPU: {dev}   n_waves={args.n_waves}   repeats={args.repeats}")

    rows, grids = [], {}
    for name, path in DATASETS.items():
        picmus = load_picmus(path)
        x_axis, z_axis = picmus["x_axis"], picmus["z_axis"]
        to_grid = lambda img: metrics.to_grid(img, x_axis, z_axis)
        ref = load_reference_bmode(path)
        print(f"\n[{name}] {picmus['data'].shape} (waves, ch, samples), {len(picmus['scan_x'])} pixels")

        # --- CPU (float64 NumPy, 기존 구현) ---
        cpu_grid, cpu_ms = None, float("nan")
        if not args.skip_cpu:
            t = time.perf_counter()
            cpu_img = das_beamform(picmus, n_waves=args.n_waves, verbose=False, **APOD_KW)
            cpu_ms = (time.perf_counter() - t) * 1e3
            cpu_grid = to_grid(cpu_img)
            print(f"  CPU numpy          : {cpu_ms/1e3:8.2f} s")

        # --- GPU : hilbert(H2D 포함)는 공통 전처리라 따로 잰다 ---
        analytic = analytic_signal_gpu(picmus["data"])
        hil_ms = np.median(gpu_time_ms(lambda: analytic_signal_gpu(picmus["data"]), args.repeats))
        print(f"  GPU H2D + hilbert  : {hil_ms:8.2f} ms")

        impls = {
            "gpu_cupy": lambda: das_beamform_cupy(picmus, n_waves=args.n_waves, analytic=analytic, **APOD_KW),
            "gpu_cuda": lambda: das_beamform_cuda(picmus, n_waves=args.n_waves, analytic=analytic, **APOD_KW),
        }
        results = {"cpu_numpy": (cpu_ms, cpu_grid)}
        for impl, fn in impls.items():
            t_med = float(np.median(gpu_time_ms(fn, args.repeats)))
            results[impl] = (t_med, to_grid(cp.asnumpy(fn()).astype(np.complex128)))
            print(f"  {impl:<19}: {t_med:8.2f} ms (beamforming만, median)")
        e2e = float(np.median(end_to_end_ms(picmus, args.n_waves, args.repeats)))
        print(f"  gpu_cuda end-to-end: {e2e:8.2f} ms (RF host -> B-mode host)")

        for impl, (t_ms, grid) in results.items():
            if grid is None:
                continue
            r = dict(dataset=name, impl=impl, time_ms=t_ms,
                     speedup_vs_cpu=cpu_ms / t_ms if cpu_grid is not None else float("nan"),
                     nrmse_vs_cpu=metrics.image_nrmse_db(grid, cpu_grid) if cpu_grid is not None else float("nan"),
                     max_rel_err_vs_cpu=(np.abs(grid - cpu_grid).max() / np.abs(cpu_grid).max())
                     if cpu_grid is not None else float("nan"),
                     nrmse_vs_ustb=metrics.image_nrmse_db(grid, ref))
            r.update(image_metrics(name, grid, picmus))
            rows.append(r)
        rows.append(dict(dataset=name, impl="gpu_cuda_end_to_end", time_ms=e2e,
                         speedup_vs_cpu=cpu_ms / e2e if cpu_grid is not None else float("nan")))
        rows.append(dict(dataset=name, impl="gpu_hilbert_h2d", time_ms=hil_ms))
        grids[name] = (results, x_axis, z_axis)

    # --- 표 출력 ---
    print(f"\n{'dataset':<11}{'impl':<21}{'time':>12}{'speedup':>10}{'NRMSE/cpu':>12}{'NRMSE/ustb':>12}")
    for r in rows:
        if "nrmse_vs_ustb" not in r:
            continue
        print(f"{r['dataset']:<11}{r['impl']:<21}{r['time_ms']:>10.1f}ms{r['speedup_vs_cpu']:>9.0f}x"
              f"{r['nrmse_vs_cpu']:>12.2e}{r['nrmse_vs_ustb']:>12.4f}")

    # --- CSV ---
    keys = ["dataset", "impl", "time_ms", "speedup_vs_cpu", "nrmse_vs_cpu", "max_rel_err_vs_cpu",
            "nrmse_vs_ustb", "CR_dB", "CNR", "gCNR", "fwhm_lat_mm", "fwhm_ax_mm"]
    csv_path = os.path.join(OUT_DIR, "gpu_benchmark.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for r in rows:
            w.writerow([("" if r.get(k) is None else
                         (f"{r[k]:.6g}" if isinstance(r[k], float) else r[k])) for k in keys])
    print(f"saved {csv_path}")

    if args.skip_cpu:
        return

    # --- figure : 처리시간(log) + CPU/GPU B-mode + 차이 영상 ---
    res, x_axis, z_axis = grids["contrast"]
    ext = [x_axis.min()*1e3, x_axis.max()*1e3, z_axis.max()*1e3, z_axis.min()*1e3]
    cpu_db = np.clip(metrics.envelope_db(res["cpu_numpy"][1]), -60, 0)
    gpu_db = np.clip(metrics.envelope_db(res["gpu_cuda"][1]), -60, 0)

    fig, axs = plt.subplots(1, 4, figsize=(17, 5.2), gridspec_kw=dict(width_ratios=[1.4, 1, 1, 1]))
    labels = ["CPU\nNumPy", "GPU\nCuPy vectorized", "GPU\nCUDA kernel"]
    times = [res[k][0] for k in ("cpu_numpy", "gpu_cupy", "gpu_cuda")]
    bars = axs[0].bar(labels, times, color=["#888888", "#6a9fd4", "#1f5fa8"])
    axs[0].set_yscale("log")
    axs[0].set_ylabel("time per frame [ms] (log)")
    axs[0].set_title(f"75-wave compounding, {len(x_axis)*len(z_axis)//1000}k px\n{dev}")
    for b, t in zip(bars, times):
        axs[0].text(b.get_x() + b.get_width()/2, t * 1.15,
                    f"{t/1e3:.1f} s" if t > 1e3 else f"{t:.1f} ms\n(x{times[0]/t:.0f})",
                    ha="center", va="bottom", fontsize=9)
    axs[0].set_ylim(top=times[0] * 8)

    for a, img, title in [(axs[1], cpu_db, "CPU (float64)"), (axs[2], gpu_db, "GPU CUDA kernel (float32)")]:
        a.imshow(img, cmap="gray", extent=ext, aspect="auto", vmin=-60, vmax=0)
        a.set_title(title); a.set_xlabel("x [mm]")
    axs[1].set_ylabel("z [mm]")
    diff = gpu_db - cpu_db
    lim = max(np.abs(diff).max(), 1e-6)
    im = axs[3].imshow(diff, cmap="RdBu_r", extent=ext, aspect="auto", vmin=-lim, vmax=lim)
    axs[3].set_title(f"dB difference (GPU - CPU)\nNRMSE = {metrics.image_nrmse_db(res['gpu_cuda'][1], res['cpu_numpy'][1]):.1e}")
    axs[3].set_xlabel("x [mm]")
    fig.colorbar(im, ax=axs[3], label="dB")
    fig.tight_layout()
    png_path = os.path.join(OUT_DIR, "gpu_benchmark.png")
    fig.savefig(png_path, dpi=130)
    print(f"saved {png_path}")


if __name__ == "__main__":
    main()
