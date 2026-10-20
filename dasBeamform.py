"""Delay-and-Sum(DAS) 빔포밍 구현 (PICMUS 평면파 채널 데이터 기준).

USTB의 +midprocess/das.m, +tools/matlab_beamformer.m 로직을 참고하여 Python/numpy로
포팅한 버전. 정확도(참조 결과와의 일치) 우선으로 작성한 CPU 기준 구현이며,
GPU 가속 버전은 dasBeamformGpu.py에 있다.

알고리즘 순서 (das.m과 동일):
  1. RF 데이터를 fast-time(샘플) 축으로 hilbert 변환 -> analytic signal
     (raw RF를 그대로 지연·보간하면 진동이 심해 오차가 커지므로, USTB도
      먼저 analytic signal로 바꾼 뒤 지연·보간·합산한다.)
  2. 각 픽셀-채널 쌍의 수신 지연(receive delay) = 거리 / 음속
  3. 각 픽셀-평면파 쌍의 송신 지연(transmit delay) = z*cos(az) + x*sin(az)) / 음속
  4. 총 지연을 샘플 인덱스로 변환 후 선형보간으로 값을 뽑아 채널 합산(delay-sum)
  5. 여러 평면파 각도의 결과를 다시 합산(coherent compounding)
"""
import numpy as np
from scipy.signal import hilbert


def compute_apodization(scan_x, scan_z, elem_x, f_number=1.7, window="tukey50"):
    """F-number 기반 dynamic receive apodization (픽셀 깊이에 따라 aperture가 커짐).

    USTB das.m / PICMUS 데모의 기본 설정(F# 1.7, Tukey50)과 동일한 방식.
    각 픽셀 깊이 z에서 receive aperture half-width = z / (2*f_number)로 두고,
    |element_x - pixel_x| 이 그 범위를 벗어난 채널은 0으로 죽인다.

    Parameters
    ----------
    scan_x, scan_z : (N_pixels,) 픽셀 좌표 [m]
    elem_x : (N_channels,) 엘리먼트 x좌표 [m]
    f_number : receive f-number (aperture_width = depth / f_number)
    window : "boxcar" 또는 "tukeyNN" (NN=0~100, 예: "tukey50", "tukey40", "tukey25").
             NN이 클수록 aperture 가장자리를 더 넓게/부드럽게 깎는다. "tukey0"은 boxcar와 동일.

    Returns
    -------
    apod : (N_pixels, N_channels) float32, 픽셀-채널별 수신 가중치
    """
    z_safe = np.maximum(scan_z, 1e-6)                       # z=0 근처 0나눗셈 방지
    aperture_half = z_safe / (2.0 * f_number)                # (N_pixels,)
    dx = np.abs(elem_x[None, :] - scan_x[:, None])            # (N_pixels, N_channels)
    ratio = dx / aperture_half[:, None]

    if window == "boxcar":
        apod = (ratio <= 1.0).astype(np.float32)
    elif window.startswith("tukey"):
        try:
            roll = float(window[len("tukey"):]) / 100.0
        except ValueError:
            raise ValueError(f"알 수 없는 window: {window!r} (boxcar 또는 tukeyNN, 예: tukey50)")
        if not (0.0 <= roll <= 1.0):
            raise ValueError(f"tukey 비율은 0~100 사이여야 함: {window!r}")
        if roll == 0.0:
            apod = (ratio <= 1.0).astype(np.float32)
        else:
            flat_edge = 1.0 - roll
            apod = np.zeros_like(ratio, dtype=np.float32)
            apod[ratio <= flat_edge] = 1.0
            taper = (ratio > flat_edge) & (ratio <= 1.0)
            apod[taper] = 0.5 * (1 + np.cos((np.pi / roll) * (ratio[taper] - flat_edge)))
    else:
        raise ValueError(f"알 수 없는 window: {window!r} (boxcar 또는 tukeyNN, 예: tukey50)")

    return apod


def _resolve_apodization(apodization, scan_x, scan_z, elem_x, N_pixels, N_channels,
                          f_number, window):
    """apodization 인자(str/1D/2D)를 (N_pixels, N_channels) 가중치 맵으로 통일한다."""
    if isinstance(apodization, str):
        if apodization == "dynamic":
            return compute_apodization(scan_x, scan_z, elem_x, f_number=f_number, window=window)
        elif apodization == "none":
            return np.ones((N_pixels, N_channels), dtype=np.float32)
        else:
            raise ValueError(f"알 수 없는 apodization: {apodization!r}")
    apod_map = np.asarray(apodization, dtype=np.float32)
    if apod_map.ndim == 1:
        apod_map = np.broadcast_to(apod_map[None, :], (N_pixels, N_channels))
    return apod_map


def das_beamform(picmus, n_waves=None, apodization="dynamic", f_number=1.7,
                  window="tukey50", verbose=True, analytic=None):
    """PICMUS 데이터를 DAS로 빔포밍한다.

    Parameters
    ----------
    picmus : uffLoader.load_picmus()의 반환값
    n_waves : 사용할 평면파 개수. None이면 75개 전부 coherent compounding.
              정수를 주면 가운데(0도 부근) 각도부터 그만큼만 사용
              (예: n_waves=1 -> 0도 단일 평면파로 빠르게 테스트).
    apodization : 수신 아포다이제이션 지정 방식.
        "dynamic" (기본) : f_number/window로 픽셀 깊이별 dynamic aperture 계산 (compute_apodization).
        "none"           : 전체 1 (boxcar, 고정 aperture 없음. 예전 기본 동작).
        (N_channels,) ndarray      : 모든 픽셀에 동일하게 적용할 고정 채널 가중치.
        (N_pixels, N_channels) ndarray : 이미 계산된 가중치를 직접 전달.
    f_number, window : apodization="dynamic"일 때만 사용 (compute_apodization 참고).
    analytic : 미리 계산한 analytic signal (N_waves, N_channels, N_samples).
               파라미터 스윕처럼 같은 데이터를 여러 번 빔포밍할 때 hilbert를 한 번만 하려고 받는다.
               None이면 내부에서 계산.

    Returns
    -------
    image : (N_pixels,) complex128, envelope 검출 전의 beamformed 신호
    """
    data = picmus["data"]                 # (N_waves, N_channels, N_samples)
    fs, t0, c = picmus["fs"], picmus["t0"], picmus["c"]
    elem_x = picmus["elem_x"]
    azimuth = picmus["azimuth"]
    scan_x, scan_z = picmus["scan_x"], picmus["scan_z"]

    N_waves_total, N_channels, N_samples = data.shape
    N_pixels = scan_x.shape[0]

    if n_waves is None:
        wave_idx = np.arange(N_waves_total)
    else:
        center = N_waves_total // 2
        half = n_waves // 2
        wave_idx = np.arange(center - half, center - half + n_waves)

    # apodization: (N_pixels, N_channels)로 정규화해서 이후 broadcasting 단순화
    apod_map = _resolve_apodization(apodization, scan_x, scan_z, elem_x,
                                     N_pixels, N_channels, f_number, window)

    # 1) RF -> analytic signal (fast-time 축 = 마지막 축)
    if analytic is None:
        analytic = hilbert(data, axis=-1)  # (N_waves, N_channels, N_samples) complex

    image = np.zeros(N_pixels, dtype=np.complex128)

    # 2) 수신 지연: (N_pixels, N_channels)
    receive_delay = np.sqrt((elem_x[None, :] - scan_x[:, None]) ** 2 +
                             scan_z[:, None] ** 2) / c

    for wi in wave_idx:
        az = azimuth[wi]

        # 3) 송신 지연(평면파): (N_pixels,)
        transmit_delay = (scan_z * np.cos(az) + scan_x * np.sin(az)) / c
        total_delay = receive_delay + transmit_delay[:, None]      # (N_pixels, N_channels)

        # 4) 지연 -> fractional 샘플 인덱스 -> 선형보간
        idx = (total_delay - t0) * fs
        idx0 = np.floor(idx).astype(np.int64)
        frac = idx - idx0
        valid = (idx0 >= 0) & (idx0 + 1 < N_samples)
        idx0c = np.clip(idx0, 0, N_samples - 2)

        wave_data = analytic[wi]                                   # (N_channels, N_samples)
        s0 = np.take_along_axis(wave_data, idx0c.T, axis=1).T      # (N_pixels, N_channels)
        s1 = np.take_along_axis(wave_data, idx0c.T + 1, axis=1).T

        sampled = s0 * (1 - frac) + s1 * frac
        sampled[~valid] = 0

        # 채널 합산 (receive), dynamic apodization은 픽셀마다 가중치가 다름
        image += (sampled * apod_map).sum(axis=1)

        if verbose:
            print(f"wave {wi} ({np.degrees(az):+.1f}°) 처리 완료")

    return image


def beamform_each_wave(picmus, apodization="dynamic", f_number=1.7,
                        window="tukey50", verbose=True):
    """75개 평면파를 compounding(합산)하지 않고 각각 따로 빔포밍한다.

    das_beamform()은 여러 wave를 채널까지 합쳐서 하나의 이미지로 만들지만,
    이 함수는 wave별로 이미지를 따로 보관한다 (몇 번째 평면파가 어떤 이미지를
    만드는지 확인하고 싶을 때 사용).

    Returns
    -------
    images : (N_waves, N_pixels) complex128. images[i] = i번째 평면파(순서상
             sequence_%04d % (i+1))만 사용한 단일 beamformed 이미지.
    """
    data = picmus["data"]                 # (N_waves, N_channels, N_samples)
    fs, t0, c = picmus["fs"], picmus["t0"], picmus["c"]
    elem_x = picmus["elem_x"]
    azimuth = picmus["azimuth"]
    scan_x, scan_z = picmus["scan_x"], picmus["scan_z"]

    N_waves_total, N_channels, N_samples = data.shape
    N_pixels = scan_x.shape[0]

    apod_map = _resolve_apodization(apodization, scan_x, scan_z, elem_x,
                                     N_pixels, N_channels, f_number, window)

    # hilbert는 한 번만 계산 (das_beamform을 wave마다 새로 부르면 매번 다시 계산돼 비효율적)
    analytic = hilbert(data, axis=-1)      # (N_waves, N_channels, N_samples) complex

    receive_delay = np.sqrt((elem_x[None, :] - scan_x[:, None]) ** 2 +
                             scan_z[:, None] ** 2) / c

    images = np.zeros((N_waves_total, N_pixels), dtype=np.complex128)

    for wi in range(N_waves_total):
        az = azimuth[wi]
        transmit_delay = (scan_z * np.cos(az) + scan_x * np.sin(az)) / c
        total_delay = receive_delay + transmit_delay[:, None]

        idx = (total_delay - t0) * fs
        idx0 = np.floor(idx).astype(np.int64)
        frac = idx - idx0
        valid = (idx0 >= 0) & (idx0 + 1 < N_samples)
        idx0c = np.clip(idx0, 0, N_samples - 2)

        wave_data = analytic[wi]
        s0 = np.take_along_axis(wave_data, idx0c.T, axis=1).T
        s1 = np.take_along_axis(wave_data, idx0c.T + 1, axis=1).T
        sampled = s0 * (1 - frac) + s1 * frac
        sampled[~valid] = 0

        images[wi] = (sampled * apod_map).sum(axis=1)

        if verbose:
            print(f"wave {wi + 1}/{N_waves_total} ({np.degrees(az):+.1f}°) 완료")

    return images


def to_bmode(image, x_axis, z_axis, dynamic_range=60):
    """빔포밍 결과(complex)를 정규화된 log-compressed B-mode 이미지로 변환한다.

    scan_x/scan_z의 픽셀 순서는 z가 먼저 변하는 MATLAB column-major 순서이므로
    (N_x, N_z)로 reshape한 뒤 전치해서 (depth, lateral) = (N_z, N_x) 형태로 맞춘다.
    """
    img2d = image.reshape(len(x_axis), len(z_axis)).T   # (N_z, N_x)
    envelope = np.abs(img2d)
    envelope_db = 20 * np.log10(envelope / envelope.max() + 1e-12)
    return np.clip(envelope_db, -dynamic_range, 0)
