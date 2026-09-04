"""apodization 유무(boxcar 'none' vs dynamic f/1.7 tukey50)에 따른
75-wave compounding 결과를 비교하는 스크립트.
"""
import time
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uffLoader import load_picmus
from dasBeamform import das_beamform, to_bmode

UFF_PATH = r"./data/PICMUS/PICMUS_simulation_contrast_speckle.uff"
TIMING_LOG = r"./timing_log.txt"


def main():
    picmus = load_picmus(UFF_PATH)
    extent = [picmus["x_axis"].min() * 1e3, picmus["x_axis"].max() * 1e3,
              picmus["z_axis"].max() * 1e3, picmus["z_axis"].min() * 1e3]


    configs = []
    for i in range(1, 11):
        for f in range(1, 20):
            f_number = f / 10
            label = f"dynamic_f{f_number}_tukey{i*10}"
            configs.append(
                (
                    label,
                    dict(apodization="dynamic", f_number=f_number, window=f"tukey{i*10}"),
                )
            )

    # configs = [
    #     # ("none", dict(apodization="none")),
    #     ("dynamic_f1.7_tukey40", dict(apodization="dynamic", f_number=1.7, window="tukey40")),
    # ]

    t_all_start = time.time()
    with open(TIMING_LOG, "w") as f:
        f.write(f"run start: {datetime.now().isoformat()}\n")
        f.write(f"n_configs: {len(configs)}\n\n")

    for i, (label, kwargs) in enumerate(configs, 1):
        t0 = time.time()
        print(f"=== [{i}/{len(configs)}] apodization={label} ===")

        image = das_beamform(picmus, n_waves=75, **kwargs)
        bmode = to_bmode(image, picmus["x_axis"], picmus["z_axis"])

        plt.figure(figsize=(5, 8))
        plt.imshow(bmode, cmap="gray", extent=extent, aspect="auto")
        plt.title(f"DAS B-mode (n_waves=75, apodization={label})")
        plt.xlabel("Lateral [mm]")
        plt.ylabel("Depth [mm]")
        plt.colorbar(label="dB")
        plt.tight_layout()
        fname = f"./outputs/tukeyNN/das_result_apod_{label}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"saved {fname}")

        elapsed = time.time() - t0
        with open(TIMING_LOG, "a") as f:
            f.write(f"[{i}/{len(configs)}] {label}: {elapsed:.1f} sec\n")

    total_elapsed = time.time() - t_all_start
    with open(TIMING_LOG, "a") as f:
        f.write(f"\nrun end: {datetime.now().isoformat()}\n")
        f.write(f"total_elapsed_sec: {total_elapsed:.1f}\n")
        f.write(f"total_elapsed_min: {total_elapsed / 60:.2f}\n")
    print(f"timing log saved: {TIMING_LOG}")


if __name__ == "__main__":
    main()
