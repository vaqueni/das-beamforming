"""DAS 빔포밍의 CuPy GPU 버전 (dasBeamform.das_beamform과 동일한 알고리즘).

두 가지 구현을 둔다.

1. das_beamform_cupy : CPU 코드를 numpy -> cupy로 그대로 옮긴 vectorized 버전.
   평면파마다 (N_pixels, N_channels) 지연/보간 배열을 GPU 메모리에 만들고 합산한다.
   CPU와 같은 이유(대형 임시 배열 생성)로 메모리 대역폭에 묶이지만, 포팅 비용이 거의 없다.

2. das_beamform_cuda : CUDA RawKernel 버전. 스레드 1개 = 픽셀 1개.
   각 스레드가 채널 x 평면파 루프를 돌며 지연 계산 -> 선형보간 -> apodization -> 누적을
   레지스터 안에서 끝내므로 (N_pixels, N_channels) 임시 배열이 아예 생기지 않는다.
   dynamic apodization도 커널 안에서 즉석 계산해 120 MB짜리 가중치 맵을 읽지 않는다.

정밀도: GPU는 float32 / complex64로 계산한다 (RTX 20xx는 FP64 처리량이 FP32의 1/32).
지연은 [샘플] 단위로 바꿔 계산하므로 float32로도 보간 위치 오차가 1e-4 샘플 수준이다.
CPU(float64) 결과와의 차이는 benchmark_gpu.py에서 NRMSE로 확인한다.
"""
import numpy as np
import cupy as cp
from cupyx.scipy.signal import hilbert as cp_hilbert

from dasBeamform import _resolve_apodization


def _wave_indices(N_waves_total, n_waves):
    """das_beamform과 동일한 평면파 선택 규칙 (None이면 전부, 정수면 가운데부터)."""
    if n_waves is None:
        return np.arange(N_waves_total)
    center = N_waves_total // 2
    half = n_waves // 2
    return np.arange(center - half, center - half + n_waves)


def _parse_tukey(window):
    """window 문자열 -> Tukey roll 비율 (0이면 boxcar). compute_apodization과 동일 규칙."""
    if window == "boxcar":
        return 0.0
    if window.startswith("tukey"):
        roll = float(window[len("tukey"):]) / 100.0
        if 0.0 <= roll <= 1.0:
            return roll
    raise ValueError(f"알 수 없는 window: {window!r} (boxcar 또는 tukeyNN, 예: tukey50)")


def analytic_signal_gpu(data):
    """RF (N_waves, N_channels, N_samples) -> GPU 위 complex64 analytic signal."""
    d = cp.asarray(data, dtype=cp.float32)
    return cp_hilbert(d, axis=-1).astype(cp.complex64)


# ----------------------------------------------------------------------------
# 1. vectorized CuPy 포팅
# ----------------------------------------------------------------------------
def das_beamform_cupy(picmus, n_waves=None, apodization="dynamic", f_number=1.7,
                      window="tukey50", analytic=None):
    """das_beamform의 cupy 직역판. 반환: (N_pixels,) complex64 cupy 배열.

    analytic : 미리 계산한 GPU analytic signal (반복 측정 시 hilbert 시간을 분리하려고 받음).
    """
    fs, t0, c = picmus["fs"], picmus["t0"], picmus["c"]
    N_waves_total, N_channels, N_samples = picmus["data"].shape
    scan_x = cp.asarray(picmus["scan_x"], dtype=cp.float32)
    scan_z = cp.asarray(picmus["scan_z"], dtype=cp.float32)
    elem_x = cp.asarray(picmus["elem_x"], dtype=cp.float32)
    N_pixels = scan_x.shape[0]

    apod_map = cp.asarray(_resolve_apodization(
        apodization, picmus["scan_x"], picmus["scan_z"], picmus["elem_x"],
        N_pixels, N_channels, f_number, window), dtype=cp.float32)

    if analytic is None:
        analytic = analytic_signal_gpu(picmus["data"])

    # 지연을 [샘플] 단위로: idx = (거리_rx + 거리_tx) * fs/c - t0*fs
    k = np.float32(fs / c)
    rx_samples = cp.sqrt((elem_x[None, :] - scan_x[:, None]) ** 2 + scan_z[:, None] ** 2) * k
    ch_offset = (cp.arange(N_channels, dtype=cp.int64) * N_samples)[None, :]

    image = cp.zeros(N_pixels, dtype=cp.complex64)
    for wi in _wave_indices(N_waves_total, n_waves):
        az = picmus["azimuth"][wi]
        tx_samples = (scan_z * np.float32(np.cos(az)) + scan_x * np.float32(np.sin(az))) * k
        idx = rx_samples + (tx_samples - np.float32(t0 * fs))[:, None]

        idx0 = cp.floor(idx).astype(cp.int64)
        frac = idx - idx0
        valid = (idx0 >= 0) & (idx0 + 1 < N_samples)
        flat = cp.clip(idx0, 0, N_samples - 2) + ch_offset         # 채널별 1D 오프셋

        wave_data = analytic[wi].ravel()
        sampled = wave_data[flat] * (1 - frac) + wave_data[flat + 1] * frac
        sampled = cp.where(valid, sampled, 0)
        image += (sampled * apod_map).sum(axis=1)

    return image


# ----------------------------------------------------------------------------
# 2. CUDA RawKernel (픽셀당 스레드 1개)
# ----------------------------------------------------------------------------
_DAS_KERNEL_SRC = r"""
#include <cupy/complex.cuh>

extern "C" __global__
void das_kernel(const complex<float>* __restrict__ analytic,  // (N_waves, N_ch, N_s)
                const float* __restrict__ scan_x,             // (N_pix,)
                const float* __restrict__ scan_z,
                const float* __restrict__ elem_x,             // (N_ch,)
                const float* __restrict__ cos_az,             // (N_sel,) 선택된 평면파만
                const float* __restrict__ sin_az,
                const int*   __restrict__ wave_idx,           // (N_sel,) analytic의 wave 번호
                const float* __restrict__ apod_map,           // apod_mode==2일 때만 사용
                complex<float>* __restrict__ out,             // (N_pix,)
                int N_pix, int N_ch, int N_s, int N_sel,
                float k_fs_over_c, float t0_fs,
                int apod_mode,          // 0: dynamic(F#, Tukey 즉석 계산), 1: none, 2: map
                float f_number, float roll)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= N_pix) return;

    float px = scan_x[p];
    float pz = scan_z[p];
    float aperture_half = fmaxf(pz, 1e-6f) / (2.0f * f_number);
    float flat_edge = 1.0f - roll;

    complex<float> acc(0.0f, 0.0f);

    for (int ch = 0; ch < N_ch; ++ch) {
        float dx = elem_x[ch] - px;

        // --- receive apodization (compute_apodization과 동일 정의) ---
        float w;
        if (apod_mode == 0) {
            float ratio = fabsf(dx) / aperture_half;
            if (ratio > 1.0f)            w = 0.0f;
            else if (roll == 0.0f)       w = 1.0f;
            else if (ratio <= flat_edge) w = 1.0f;
            else w = 0.5f * (1.0f + cosf((3.14159265358979f / roll) * (ratio - flat_edge)));
        } else if (apod_mode == 1) {
            w = 1.0f;
        } else {
            w = apod_map[(size_t)p * N_ch + ch];
        }
        if (w == 0.0f) continue;        // aperture 밖 채널은 메모리 접근 자체를 생략

        // 수신 지연 [샘플]: 채널마다 한 번만 계산하고 평면파 루프에서 재사용
        float rx = sqrtf(dx * dx + pz * pz) * k_fs_over_c;

        complex<float> ch_acc(0.0f, 0.0f);
        for (int s = 0; s < N_sel; ++s) {
            float tx = (pz * cos_az[s] + px * sin_az[s]) * k_fs_over_c;
            float idx = rx + tx - t0_fs;
            float fl = floorf(idx);
            int i0 = (int)fl;
            if (i0 < 0 || i0 + 1 >= N_s) continue;
            float frac = idx - fl;

            const complex<float>* row =
                analytic + ((size_t)wave_idx[s] * N_ch + ch) * N_s;
            complex<float> s0 = row[i0];
            complex<float> s1 = row[i0 + 1];
            ch_acc += s0 * (1.0f - frac) + s1 * frac;
        }
        acc += ch_acc * w;
    }
    out[p] = acc;
}
"""

_das_kernel = cp.RawKernel(_DAS_KERNEL_SRC, "das_kernel")


def das_beamform_cuda(picmus, n_waves=None, apodization="dynamic", f_number=1.7,
                      window="tukey50", analytic=None, block_size=128):
    """CUDA RawKernel DAS. 반환: (N_pixels,) complex64 cupy 배열.

    apodization="dynamic"이면 가중치를 커널 안에서 계산(apod_mode 0),
    "none"이면 1(apod_mode 1), ndarray면 맵을 GPU로 올려 그대로 읽는다(apod_mode 2).
    """
    fs, t0, c = picmus["fs"], picmus["t0"], picmus["c"]
    N_waves_total, N_channels, N_samples = picmus["data"].shape
    scan_x = cp.asarray(picmus["scan_x"], dtype=cp.float32)
    scan_z = cp.asarray(picmus["scan_z"], dtype=cp.float32)
    elem_x = cp.asarray(picmus["elem_x"], dtype=cp.float32)
    N_pixels = scan_x.shape[0]

    if analytic is None:
        analytic = analytic_signal_gpu(picmus["data"])

    sel = _wave_indices(N_waves_total, n_waves)
    az = picmus["azimuth"][sel]
    cos_az = cp.asarray(np.cos(az), dtype=cp.float32)
    sin_az = cp.asarray(np.sin(az), dtype=cp.float32)
    wave_idx = cp.asarray(sel, dtype=cp.int32)

    roll = 0.0
    apod_map = cp.zeros(1, dtype=cp.float32)            # 사용 안 할 때의 더미 포인터
    if isinstance(apodization, str) and apodization == "dynamic":
        apod_mode, roll = 0, _parse_tukey(window)
    elif isinstance(apodization, str) and apodization == "none":
        apod_mode = 1
    else:
        apod_mode = 2
        apod_map = cp.ascontiguousarray(cp.asarray(_resolve_apodization(
            apodization, picmus["scan_x"], picmus["scan_z"], picmus["elem_x"],
            N_pixels, N_channels, f_number, window), dtype=cp.float32))

    out = cp.empty(N_pixels, dtype=cp.complex64)
    grid = ((N_pixels + block_size - 1) // block_size,)
    _das_kernel(grid, (block_size,), (
        analytic, scan_x, scan_z, elem_x, cos_az, sin_az, wave_idx, apod_map, out,
        np.int32(N_pixels), np.int32(N_channels), np.int32(N_samples), np.int32(len(sel)),
        np.float32(fs / c), np.float32(t0 * fs),
        np.int32(apod_mode), np.float32(f_number), np.float32(roll),
    ))
    return out
