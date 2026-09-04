"""75개 평면파(1~75번)를 compounding 없이 각각 따로 빔포밍해서 저장하는 스크립트.

outputs/per_wave/ 에 개별 PNG 75장을 만든다 (wave당 1장, 그리드로 합치지 않음).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uffLoader import load_picmus
from dasBeamform import beamform_each_wave, to_bmode

UFF_PATH = r"./data/PICMUS/PICMUS_simulation_contrast_speckle.uff"
OUT_DIR = r"./outputs/per_wave"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    picmus = load_picmus(UFF_PATH)
    azimuth_deg = np.degrees(picmus["azimuth"])
    print(f"loaded: {picmus['data'].shape} (waves, channels, samples)")

    images = beamform_each_wave(picmus)   # (75, N_pixels) complex, wave별로 따로 보관

    extent = [picmus["x_axis"].min() * 1e3, picmus["x_axis"].max() * 1e3,
              picmus["z_axis"].max() * 1e3, picmus["z_axis"].min() * 1e3]

    # wave별 개별 이미지 저장 (75장, 그리드로 합치지 않고 파일 하나씩)
    for i in range(images.shape[0]):
        bmode = to_bmode(images[i], picmus["x_axis"], picmus["z_axis"])
        plt.figure(figsize=(4, 6))
        plt.imshow(bmode, cmap="gray", extent=extent, aspect="auto")
        plt.title(f"wave {i + 1}/75  ({azimuth_deg[i]:+.1f} deg)")
        plt.xlabel("Lateral [mm]")
        plt.ylabel("Depth [mm]")
        plt.tight_layout()
        fname = os.path.join(OUT_DIR, f"wave_{i + 1:02d}_{azimuth_deg[i]:+.1f}deg.png")
        plt.savefig(fname, dpi=100)
        plt.close()
        print(f"saved {fname}")


if __name__ == "__main__":
    main()
