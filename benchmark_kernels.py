"""CUDA DAS 커널 최적화 단계별(ablation) 처리시간 / 정확도 비교.

  python benchmark_kernels.py [--repeats 15]

기준 커널(dasBeamformGpu.das_beamform_cuda)에서 출발해 최적화를 하나씩 누적하고,
채택하지 않은 시도와 "데이터가 L2에 다 들어간다면" 하한선도 같이 잰다.
- 정확도: CPU float64 결과 대비 dB NRMSE (CPU 결과는 outputs/cpu_ref_{dataset}.npy에 캐시)
- 결과: results/kernel_opt.csv, results/kernel_opt.png
"""
import argparse
import csv
import os

import numpy as np
import cupy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uffLoader import load_picmus
from dasBeamform import das_beamform
from dasBeamformGpu import analytic_signal_gpu, das_beamform_cuda
from dasKernelVariants import prepare_storage, das_beamform_variant, _kernel
import metrics

DATASETS = {
    "contrast": "./data/PICMUS/PICMUS_simulation_contrast_speckle.uff",
    "resolution": "./data/PICMUS/PICMUS_simulation_resolution_distortion.uff",
}
OUT_DIR = "./results"
CACHE_DIR = "./outputs"

# (이름, 설명, storage, kwargs). 위에서부터 최적화를 하나씩 누적한다.
LADDER = [
    ("v0 baseline", "기준 커널 (1D 블록 128)", None, None),
    ("v1 block 512", "+ 블록 크기 512", 0, dict(block_size=512)),
    ("v2 2D tile", "+ 32 x 8 픽셀 2D 타일 (블록 256)", 0, dict(block_size=256, tile_x=8)),
    ("v3 wave chunk", "+ 평면파 15개씩 launch 분할", 0, dict(block_size=256, tile_x=8, wave_chunk=15)),
    ("v4 half2", "+ analytic을 fp16으로 저장", 1, dict(block_size=256, tile_x=8, wave_chunk=15)),
]
REJECTED = [
    ("x texture", "3D texture 하드웨어 보간 (블록 512)", 2, dict(block_size=512)),
    ("x wave-outer loop", "루프 순서 평면파 바깥 (블록 512)", 0, dict(block_size=512, wave_outer=1)),
]


def gpu_time_ms(fn, repeats):
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
    return float(np.median(times))


def l2_resident_ms(picmus, data, repeats):
    """모든 평면파가 0번 평면파 데이터(1.85 MB)만 읽게 한 v1 커널. 연산량은 같고 데이터는 L2에 상주.
    -> DRAM 병목이 완전히 없어졌을 때의 하한선 (영상은 의미 없음)."""
    N = len(picmus["scan_x"])
    N_z = len(picmus["z_axis"])
    _, N_ch, N_s = picmus["data"].shape
    az = picmus["azimuth"]
    args = (data, cp.asarray(picmus["scan_x"], cp.float32), cp.asarray(picmus["scan_z"], cp.float32),
            cp.asarray(picmus["elem_x"], cp.float32),
            cp.asarray(np.cos(az), cp.float32), cp.asarray(np.sin(az), cp.float32),
            cp.zeros(len(az), cp.int32), cp.empty(N, cp.complex64),
            np.int32(N), np.int32(N_z), np.int32(N_ch), np.int32(N_s),
            np.int32(0), np.int32(N_ch), np.int32(0), np.int32(len(az)), np.int32(0),
            np.float32(picmus["fs"] / picmus["c"]), np.float32(picmus["t0"] * picmus["fs"]),
            np.float32(1.7), np.float32(0.5))
    k = _kernel(0, 0)
    return gpu_time_ms(lambda: k(((N + 511) // 512,), (512,), args), repeats)


def cpu_reference(name, picmus):
    path = os.path.join(CACHE_DIR, f"cpu_ref_{name}.npy")
    if not os.path.exists(path):
        print(f"  CPU 기준 영상 계산 중 (약 2분, {path}에 캐시)")
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(path, das_beamform(picmus, verbose=False))
    return np.load(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=15)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    dev = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"GPU: {dev}")
    rows = []
    for name, path in DATASETS.items():
        picmus = load_picmus(path)
        to_grid = lambda img: metrics.to_grid(img, picmus["x_axis"], picmus["z_axis"])
        cpu = to_grid(cpu_reference(name, picmus))
        analytic = analytic_signal_gpu(picmus["data"])
        stores = {s: prepare_storage(analytic, s) for s in (0, 1, 2)}
        print(f"\n[{name}]")

        base_ms = None
        for group, entries in (("ladder", LADDER), ("rejected", REJECTED)):
            for label, desc, storage, kw in entries:
                if storage is None:
                    fn = lambda: das_beamform_cuda(picmus, analytic=analytic)
                else:
                    fn = lambda: das_beamform_variant(picmus, stores[storage], storage=storage, **kw)
                t = gpu_time_ms(fn, args.repeats)
                base_ms = base_ms or t
                nrmse = metrics.image_nrmse_db(to_grid(cp.asnumpy(fn()).astype(np.complex128)), cpu)
                rows.append(dict(dataset=name, group=group, label=label, desc=desc,
                                 time_ms=t, speedup_vs_v0=base_ms / t, nrmse_vs_cpu=nrmse))
                print(f"  {label:<18} {t:7.2f} ms  x{base_ms / t:4.2f}  NRMSE {nrmse:.2e}   {desc}")

        t = l2_resident_ms(picmus, stores[0], args.repeats)
        rows.append(dict(dataset=name, group="bound", label="L2-resident bound",
                         desc="v1과 같은 연산, 데이터가 전부 L2에 있을 때", time_ms=t,
                         speedup_vs_v0=base_ms / t, nrmse_vs_cpu=float("nan")))
        print(f"  {'L2-resident bound':<18} {t:7.2f} ms  x{base_ms / t:4.2f}  (영상 무의미, 하한선)")
        del stores, analytic
        cp.get_default_memory_pool().free_all_blocks()

    keys = ["dataset", "group", "label", "desc", "time_ms", "speedup_vs_v0", "nrmse_vs_cpu"]
    csv_path = os.path.join(OUT_DIR, "kernel_opt.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for r in rows:
            w.writerow([f"{r[k]:.4g}" if isinstance(r[k], float) else r[k] for k in keys])
    print(f"\nsaved {csv_path}")

    # --- figure : contrast 데이터 기준 처리시간 막대 + NRMSE ---
    sel = [r for r in rows if r["dataset"] == "contrast"]
    colors = {"ladder": "#1f5fa8", "rejected": "#bbbbbb", "bound": "#e8a33d"}
    fig, (ax_t, ax_e) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw=dict(width_ratios=[1.6, 1]))
    y = np.arange(len(sel))[::-1]
    ax_t.barh(y, [r["time_ms"] for r in sel], color=[colors[r["group"]] for r in sel])
    for yi, r in zip(y, sel):
        ax_t.text(r["time_ms"] + 0.8, yi, f"{r['time_ms']:.1f} ms  (x{r['speedup_vs_v0']:.2f})",
                  va="center", fontsize=9)
    ax_t.set_yticks(y, [r["label"] for r in sel])
    ax_t.set_xlabel("time per frame [ms]  (75-wave compounding, 236k px)")
    ax_t.set_xlim(0, max(r["time_ms"] for r in sel) * 1.3)
    ax_t.set_title(f"CUDA kernel optimization ladder — {dev}", loc="left")

    e_sel = [r for r in sel if r["group"] != "bound"]
    ye = np.arange(len(e_sel))[::-1]
    ax_e.barh(ye, [r["nrmse_vs_cpu"] for r in e_sel], color=[colors[r["group"]] for r in e_sel])
    for yi, r in zip(ye, e_sel):
        ax_e.text(r["nrmse_vs_cpu"] * 1.1, yi, f"{r['nrmse_vs_cpu']:.1e}", va="center", fontsize=9)
    ax_e.set_xscale("log")
    ax_e.set_xlim(right=max(r["nrmse_vs_cpu"] for r in e_sel) * 6)
    ax_e.set_yticks(ye, [r["label"] for r in e_sel])
    ax_e.set_xlabel("dB-image NRMSE vs CPU float64 (log)")
    ax_e.set_title("accuracy", loc="left")
    fig.tight_layout()
    png_path = os.path.join(OUT_DIR, "kernel_opt.png")
    fig.savefig(png_path, dpi=130)
    print(f"saved {png_path}")


if __name__ == "__main__":
    main()
