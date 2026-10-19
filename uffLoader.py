"""PICMUS .uff(HDF5) 파일에서 DAS 빔포밍에 필요한 채널 데이터를 읽어온다.

USTB(Ultrasound Toolbox)가 만드는 .uff 파일은 HDF5 포맷이라 MATLAB 없이도
h5py로 직접 읽을 수 있다. MATLAB은 column-major라 h5py로 읽으면 배열의
축 순서가 MATLAB에서 본 size()와 반대로 보인다는 점만 주의하면 된다.
    channel_data.data : MATLAB에서 [N_samples, N_channels, N_waves, N_frames]
                          -> h5py로 읽으면 (N_waves, N_channels, N_samples) 형태로 보임

PICMUS 시뮬레이션 파일에는 채널 데이터뿐 아니라 USTB가 직접 빔포밍한
`beamformed_data`(참조 DAS 영상)도 들어 있어서, 내 구현 결과를 정량적으로
대조하는 기준으로 쓸 수 있다 (load_reference_bmode 참고).
"""
import h5py
import numpy as np


def load_picmus(path):
    """PICMUS .uff 파일을 읽어 DAS 빔포밍에 필요한 값들을 dict로 반환한다.

    Returns
    -------
    dict:
        data     : (N_waves, N_channels, N_samples) float32, raw RF
        fs       : sampling frequency [Hz]
        f0       : center(modulation) frequency [Hz]. 파일에 없으면(RF 데이터라
                   modulation_frequency=0) PICMUS L11 관례대로 fs/4로 채운다.
        t0       : initial_time [s] (샘플 0번이 실제로 몇 초 시점인지)
        c        : sound speed [m/s]
        elem_x   : (N_channels,) 엘리먼트 x좌표 [m] (probe.geometry 1행)
        azimuth  : (N_waves,) 평면파 조향각 [rad]
        scan_x, scan_z : (N_pixels,) 픽셀 좌표 [m] (z가 먼저 변함, MATLAB 순서 그대로)
        x_axis, z_axis : reshape에 쓸 1D 축 [m]
    """
    with h5py.File(path, "r") as f:
        cd = f["channel_data"]

        data = cd["data"][()].astype(np.float32)          # (N_waves, N_channels, N_samples)
        fs = float(np.squeeze(cd["sampling_frequency"][()]))
        t0 = float(np.squeeze(cd["initial_time"][()]))
        c = float(np.squeeze(cd["sound_speed"][()]))

        f0 = float(np.squeeze(cd["modulation_frequency"][()])) if "modulation_frequency" in cd else 0.0
        if f0 <= 0.0:
            f0 = fs / 4.0            # PICMUS L11-4v: fs = 4 * f0 (f0 ~ 5.208 MHz)

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
        data=data, fs=fs, f0=f0, t0=t0, c=c,
        elem_x=elem_x, azimuth=azimuth,
        scan_x=scan_x, scan_z=scan_z,
        x_axis=x_axis, z_axis=z_axis,
    )


def load_reference_bmode(path):
    """PICMUS 파일에 들어 있는 USTB 참조 빔포밍 결과(`beamformed_data`)를 읽어
    (N_z, N_x) 복소 배열로 반환한다.

    dasBeamform.to_bmode / metrics.to_grid 와 동일한 픽셀 순서
    (x가 바깥, z가 안쪽 -> reshape(N_x, N_z).T) 로 맞춰 준다.
    """
    with h5py.File(path, "r") as f:
        bd = f["beamformed_data"]["data"]
        flat = np.squeeze(bd["real"][()]).astype(np.float64) + \
            1j * np.squeeze(bd["imag"][()]).astype(np.float64)
        x_axis = np.squeeze(f["scan"]["x_axis"][()])
        z_axis = np.squeeze(f["scan"]["z_axis"][()])
    return flat.reshape(len(x_axis), len(z_axis)).T
