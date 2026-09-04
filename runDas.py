"""PICMUS 데이터로 DAS 빔포밍을 돌리고 B-mode 이미지를 저장/표시하는 진입 스크립트."""
import matplotlib
matplotlib.use("Agg")  # 화면 없이 파일로만 저장할 때. 인터랙티브면 이 줄 지우면 됨.
import matplotlib.pyplot as plt

from uffLoader import load_picmus
from dasBeamform import das_beamform, to_bmode

UFF_PATH = r"./data/PICMUS/PICMUS_simulation_contrast_speckle.uff"


def main(n_waves=1):
    picmus = load_picmus(UFF_PATH)
    print(f"loaded: {picmus['data'].shape} (waves, channels, samples), "
          f"fs={picmus['fs']/1e6:.3f} MHz, c={picmus['c']} m/s")

    image = das_beamform(picmus, n_waves=n_waves)
    bmode = to_bmode(image, picmus["x_axis"], picmus["z_axis"])

    extent = [picmus["x_axis"].min() * 1e3, picmus["x_axis"].max() * 1e3,
              picmus["z_axis"].max() * 1e3, picmus["z_axis"].min() * 1e3]

    plt.figure(figsize=(5, 8))
    plt.imshow(bmode, cmap="gray", extent=extent, aspect="auto")
    plt.xlabel("Lateral [mm]")
    plt.ylabel("Depth [mm]")
    plt.title(f"DAS B-mode (n_waves={n_waves})")
    plt.colorbar(label="dB")
    plt.tight_layout()
    plt.savefig(f"das_result_nwaves{n_waves}f.png", dpi=150)
    print(f"saved: das_result_nwaves{n_waves}f.png")


if __name__ == "__main__":
    main(n_waves=75)   # 우선 0도 단일 평면파로 빠르게 검증. 잘 나오면 n_waves=75로 compounding.
