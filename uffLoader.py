import h5py
import numpy as np

# PICMUS 폴더에서 실행한다고 가정한 상대경로
uff_path = "data/PICMUS/PICMUS_simulation_contrast_speckle.uff"

f = h5py.File(uff_path, "r")

print(list(f.keys()))
print(list(f["channel_data"].keys()))

rf_data = f["channel_data"]["data"][()]
print(rf_data.shape)

c = f["channel_data"]["sound_speed"][()]
print(c)


def load_picmus(path):
    """PICMUS .uff 파일을 읽어 DAS 빔포밍에 필요한 값들을 dict로 반환한다.

    Returns
    -------
    dict:
        data     : (N_waves, N_channels, N_samples) float32, raw RF
        fs       : sampling frequency [Hz]
        t0       : initial_time [s] (샘플 0번이 실제로 몇 초 시점인지)
        c        : sound speed [m/s]
        elem_x   : (N_channels,) 엘리먼트 x좌표 [m] (probe.geometry 1행)
        azimuth  : (N_waves,) 평면파 조향각 [rad]
        scan_x, scan_z : (N_pixels,) 픽셀 좌표 [m] (z가 먼저 변함, MATLAB 순서 그대로)
        x_axis, z_axis : reshape에 쓸 1D 축
    """
    with h5py.File(path, "r") as f:
        cd = f["channel_data"]

        data = cd["data"][()].astype(np.float32)          # (N_waves, N_channels, N_samples)
        fs = float(np.squeeze(cd["sampling_frequency"][()]))
        t0 = float(np.squeeze(cd["initial_time"][()]))
        c = float(np.squeeze(cd["sound_speed"][()]))

        geometry = cd["probe"]["geometry"][()]             # (7, N_channels): x,y,z,theta,phi,w,h
        elem_x = geometry[0, :]

        wave_names = sorted(cd["sequence"].keys())
        azimuth = np.array([
            np.squeeze(cd["sequence"][name]["source"]["azimuth"][()])
            for name in wave_names
        ])

        scan_x = np.squeeze(f["scan"]["x"][()])
        scan_z = np.squeeze(f["scan"]["z"][()])
        x_axis = np.squeeze(f["scan"]["x_axis"][()])
        z_axis = np.squeeze(f["scan"]["z_axis"][()])

    return dict(
        data=data, fs=fs, t0=t0, c=c,
        elem_x=elem_x, azimuth=azimuth,
        scan_x=scan_x, scan_z=scan_z,
        x_axis=x_axis, z_axis=z_axis,
    )

print('함수 실행')
print(load_picmus(uff_path))