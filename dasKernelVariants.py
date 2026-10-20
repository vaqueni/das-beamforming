"""DAS CUDA 커널 최적화 실험용 변형들.

dasBeamformGpu.das_kernel(기준)과 같은 수식을 쓰되, 한 번에 한 가지씩만 바꿔
무엇이 병목인지 확인한다 (benchmark_kernels.py에서 처리시간 + CPU 대비 NRMSE 비교).

컴파일 옵션(매크로)으로 켜고 끄는 항목:
  WAVE_OUTER : 루프 순서. 0 = 채널 바깥/평면파 안쪽(기준), 1 = 평면파 바깥/채널 안쪽.
               1이면 같은 시점에 모든 warp가 같은 평면파 한 장(1.85 MB)만 읽어 L2(4 MB)에 들어가는 대신
               수신 지연 sqrt와 apodization을 75배 더 계산한다.
  STORAGE    : analytic signal 저장 형식.
               0 = complex64 global 메모리 + 소프트웨어 선형보간 (기준)
               1 = half2 (fp16) global 메모리 -> 읽는 바이트 절반
               2 = 3D texture + 하드웨어 선형보간 (보간 가중치 9-bit 고정소수점)
  TILE_X     : 1 = 블록이 연속 픽셀 1D(기준), N = (block/N) 깊이 x N 가로 줄 2D 타일 (L1 재사용).

launch 분할(wave_chunk / ch_chunk)은 host 쪽 옵션이다 (das_beamform_variant 참고).
결론 설정은 맨 아래 BEST_CONFIG / das_beamform_fast.
"""
import numpy as np
import cupy as cp
from cupy.cuda import texture, runtime

from dasBeamformGpu import _wave_indices, _parse_tukey

_SRC = r"""
#include <cuda_fp16.h>

#ifndef WAVE_OUTER
#define WAVE_OUTER 0
#endif
#ifndef STORAGE
#define STORAGE 0
#endif
#ifndef TILE_X
#define TILE_X 1
#endif

#if STORAGE == 0
typedef float2 sample_t;
#elif STORAGE == 1
typedef __half2 sample_t;
#endif

// (wave s, channel ch, fractional index idx) -> 보간된 복소 샘플. 범위 밖이면 false.
__device__ __forceinline__ bool fetch(
#if STORAGE == 2
        cudaTextureObject_t tex,
#else
        const sample_t* __restrict__ data,
#endif
        int w, int ch, int N_ch, int N_s, float idx, float2& out)
{
    float fl = floorf(idx);
    int i0 = (int)fl;
    if (i0 < 0 || i0 + 1 >= N_s) return false;
#if STORAGE == 2
    // 텍셀 중심이 +0.5에 있으므로 idx+0.5에서 읽으면 i0, i0+1 사이 선형보간이 된다.
    // 채널/평면파 축은 텍셀 중심(+0.5)에 정확히 맞춰 이웃과 섞이지 않게 한다.
    out = tex3D<float2>(tex, idx + 0.5f, ch + 0.5f, w + 0.5f);
#else
    float frac = idx - fl;
    const sample_t* row = data + ((size_t)w * N_ch + ch) * N_s;
#if STORAGE == 0
    float2 s0 = row[i0], s1 = row[i0 + 1];
#else
    float2 s0 = __half22float2(row[i0]), s1 = __half22float2(row[i0 + 1]);
#endif
    out.x = s0.x + (s1.x - s0.x) * frac;
    out.y = s0.y + (s1.y - s0.y) * frac;
#endif
    return true;
}

__device__ __forceinline__ float apod_weight(float dx, float aperture_half, float roll, float flat_edge)
{
    float ratio = fabsf(dx) / aperture_half;
    if (ratio > 1.0f) return 0.0f;
    if (roll == 0.0f || ratio <= flat_edge) return 1.0f;
    return 0.5f * (1.0f + cosf((3.14159265358979f / roll) * (ratio - flat_edge)));
}

extern "C" __global__
void das_variant(
#if STORAGE == 2
                 cudaTextureObject_t data,
#else
                 const sample_t* __restrict__ data,
#endif
                 const float* __restrict__ scan_x, const float* __restrict__ scan_z,
                 const float* __restrict__ elem_x,
                 const float* __restrict__ cos_az, const float* __restrict__ sin_az,
                 const int* __restrict__ wave_idx,
                 float2* __restrict__ out,
                 int N_pix, int N_z, int N_ch, int N_s,
                 int ch_begin, int ch_end, int s_begin, int s_end, int accumulate,
                 float k_fs_over_c, float t0_fs, float f_number, float roll)
{
#if TILE_X == 1
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= N_pix) return;
#else
    // 블록 = (blockDim.x / TILE_X) 깊이 x TILE_X 가로 줄 타일. warp 32개 스레드는 여전히 같은 x 줄의 연속 z.
    // 이웃 x 줄은 같은 채널에서 비슷한 샘플 구간을 읽으므로 같은 SM(L1)에서 재사용된다.
    int tz_n = blockDim.x / TILE_X;
    int nzb = (N_z + tz_n - 1) / tz_n;
    int iz = (blockIdx.x % nzb) * tz_n + threadIdx.x % tz_n;
    int ix = (blockIdx.x / nzb) * TILE_X + threadIdx.x / tz_n;
    int N_x = N_pix / N_z;
    if (iz >= N_z || ix >= N_x) return;
    int p = ix * N_z + iz;          // scan 배열은 z가 먼저 변하는 순서
#endif

    float px = scan_x[p], pz = scan_z[p];
    float aperture_half = fmaxf(pz, 1e-6f) / (2.0f * f_number);
    float flat_edge = 1.0f - roll;
    float2 acc = make_float2(0.0f, 0.0f);
    float2 v;

#if WAVE_OUTER == 0
    for (int ch = ch_begin; ch < ch_end; ++ch) {
        float dx = elem_x[ch] - px;
        float w = apod_weight(dx, aperture_half, roll, flat_edge);
        if (w == 0.0f) continue;
        float rx = sqrtf(dx * dx + pz * pz) * k_fs_over_c;
        float2 ch_acc = make_float2(0.0f, 0.0f);
        for (int s = s_begin; s < s_end; ++s) {
            float idx = rx + (pz * cos_az[s] + px * sin_az[s]) * k_fs_over_c - t0_fs;
            if (fetch(data, wave_idx[s], ch, N_ch, N_s, idx, v)) { ch_acc.x += v.x; ch_acc.y += v.y; }
        }
        acc.x += ch_acc.x * w; acc.y += ch_acc.y * w;
    }
#else
    for (int s = s_begin; s < s_end; ++s) {
        float tx = (pz * cos_az[s] + px * sin_az[s]) * k_fs_over_c - t0_fs;
        int wi = wave_idx[s];
        for (int ch = ch_begin; ch < ch_end; ++ch) {
            float dx = elem_x[ch] - px;
            float w = apod_weight(dx, aperture_half, roll, flat_edge);
            if (w == 0.0f) continue;
            float idx = sqrtf(dx * dx + pz * pz) * k_fs_over_c + tx;
            if (fetch(data, wi, ch, N_ch, N_s, idx, v)) { acc.x += v.x * w; acc.y += v.y * w; }
        }
    }
#endif
    // 채널/평면파를 여러 launch로 나눠 돌릴 때는 이전 launch 결과에 더한다
    // (픽셀마다 담당 스레드가 하나라 launch 사이엔 경쟁이 없다)
    if (accumulate) { float2 o = out[p]; acc.x += o.x; acc.y += o.y; }
    out[p] = acc;
}
"""

_cache = {}


def _kernel(wave_outer, storage, tile_x=1):
    key = (wave_outer, storage, tile_x)
    if key not in _cache:
        _cache[key] = cp.RawKernel(_SRC, "das_variant",
                                   options=(f"-DWAVE_OUTER={wave_outer}", f"-DSTORAGE={storage}",
                                            f"-DTILE_X={tile_x}"))
    return _cache[key]


def prepare_storage(analytic, storage):
    """GPU complex64 analytic (N_waves, N_ch, N_s) -> 커널이 읽을 형식.

    반환값은 커널 첫 인자로 그대로 넘긴다 (texture는 CUDA array가 GC되지 않게 같이 묶어 둔다).
    """
    analytic = cp.ascontiguousarray(analytic, dtype=cp.complex64)
    if storage == 0:
        return analytic
    if storage == 1:
        return analytic.view(cp.float32).astype(cp.float16)          # (re, im) 쌍 = half2
    N_w, N_ch, N_s = analytic.shape
    desc = texture.ChannelFormatDescriptor(32, 32, 0, 0, runtime.cudaChannelFormatKindFloat)
    arr = texture.CUDAarray(desc, N_s, N_ch, N_w)
    arr.copy_from(analytic.view(cp.float32).reshape(N_w, N_ch, N_s * 2))
    res = texture.ResourceDescriptor(runtime.cudaResourceTypeArray, cuArr=arr)
    tdesc = texture.TextureDescriptor((runtime.cudaAddressModeBorder,) * 3,
                                      runtime.cudaFilterModeLinear, runtime.cudaReadModeElementType)
    return texture.TextureObject(res, tdesc)      # ResourceDescriptor가 CUDA array 참조를 들고 있다


def das_beamform_variant(picmus, data, n_waves=None, f_number=1.7, window="tukey50",
                         wave_outer=0, storage=0, block_size=128, wave_chunk=None, ch_chunk=None, tile_x=1):
    """dynamic apodization 전용 변형 커널 실행. data는 prepare_storage(analytic, storage) 결과.

    wave_chunk / ch_chunk : 평면파 / 채널을 이 개수씩 나눠 커널을 여러 번 launch한다 (None이면 한 번에).
        launch 경계에서 GPU 전체가 동기화되므로, 한 launch가 읽는 데이터
        (wave_chunk x ch_chunk x N_s 샘플)를 L2 캐시 안에 묶어 둘 수 있다.
    tile_x : 1이면 블록 = 연속 픽셀 1D, 2 이상이면 (block_size/tile_x) 깊이 x tile_x 가로 줄 2D 타일.
    """
    fs, t0, c = picmus["fs"], picmus["t0"], picmus["c"]
    N_waves_total, N_channels, N_samples = picmus["data"].shape
    scan_x = cp.asarray(picmus["scan_x"], dtype=cp.float32)
    scan_z = cp.asarray(picmus["scan_z"], dtype=cp.float32)
    elem_x = cp.asarray(picmus["elem_x"], dtype=cp.float32)
    N_pixels = scan_x.shape[0]

    sel = _wave_indices(N_waves_total, n_waves)
    az = picmus["azimuth"][sel]
    cos_az = cp.asarray(np.cos(az), dtype=cp.float32)
    sin_az = cp.asarray(np.sin(az), dtype=cp.float32)
    wave_idx = cp.asarray(sel, dtype=cp.int32)

    N_sel = len(sel)
    wave_chunk = wave_chunk or N_sel
    ch_chunk = ch_chunk or N_channels
    kernel = _kernel(wave_outer, storage, tile_x)
    N_z = len(picmus["z_axis"])
    out = cp.empty(N_pixels, dtype=cp.complex64)
    if tile_x == 1:
        grid = ((N_pixels + block_size - 1) // block_size,)
    else:
        tz_n = block_size // tile_x
        grid = (((N_z + tz_n - 1) // tz_n) * ((N_pixels // N_z + tile_x - 1) // tile_x),)
    first = True
    for c0 in range(0, N_channels, ch_chunk):
        for s0 in range(0, N_sel, wave_chunk):
            kernel(grid, (block_size,), (
                data, scan_x, scan_z, elem_x, cos_az, sin_az, wave_idx, out,
                np.int32(N_pixels), np.int32(N_z), np.int32(N_channels), np.int32(N_samples),
                np.int32(c0), np.int32(min(c0 + ch_chunk, N_channels)),
                np.int32(s0), np.int32(min(s0 + wave_chunk, N_sel)), np.int32(0 if first else 1),
                np.float32(fs / c), np.float32(t0 * fs), np.float32(f_number),
                np.float32(_parse_tukey(window)),
            ))
            first = False
    return out


# 실험(benchmark_kernels.py)으로 고른 설정. RTX 2070 SUPER / PICMUS 236k pixels 기준.
BEST_CONFIG = dict(wave_outer=0, block_size=256, tile_x=8, wave_chunk=15)


def das_beamform_fast(picmus, n_waves=None, f_number=1.7, window="tukey50",
                      analytic=None, half=False):
    """최적화 설정(BEST_CONFIG)으로 빔포밍. half=True면 analytic을 fp16으로 저장해 더 빠르지만
    CPU 대비 오차가 약 4배 커진다 (benchmark_kernels.py 참고). 반환: (N_pixels,) complex64 cupy 배열."""
    from dasBeamformGpu import analytic_signal_gpu
    if analytic is None:
        analytic = analytic_signal_gpu(picmus["data"])
    storage = 1 if half else 0
    return das_beamform_variant(picmus, prepare_storage(analytic, storage), n_waves=n_waves,
                                f_number=f_number, window=window, storage=storage, **BEST_CONFIG)
