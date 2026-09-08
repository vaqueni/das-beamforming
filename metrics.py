"""빔포밍 결과의 정량 지표 계산.

두 가지를 잰다.

1. 해상도 (resolution_distortion 데이터)
   - point target의 lateral / axial FWHM(-6 dB full width). 선형보간으로 sub-pixel.

2. 대조도 (contrast_speckle 데이터)
   - CR  : contrast ratio       = 10*log10(mean|s|^2_ROI / mean|s|^2_bg)
   - CNR : contrast-to-noise    = |mean_ROI - mean_bg| / sqrt(var_ROI + var_bg)   (|s|^2 기준)
   - gCNR: generalized CNR      = 1 - OVL(히스토그램 겹침).  Rodriguez-Molares et al. 2020
   위 정의는 USTB `tools.measure_contrast_circles.m` 와 동일하게 맞췄다.

부수적으로, PICMUS 파일에 동봉된 USTB 참조 영상과의 dB 영상 NRMSE도 잰다.
"""
import numpy as np


# ----------------------------------------------------------------------------
# 공통
# ----------------------------------------------------------------------------
def to_grid(image_1d, x_axis, z_axis):
    """(N_pixels,) 1D 빔포밍 결과를 (N_z, N_x) 2D 격자로. dasBeamform.to_bmode와 동일 순서."""
    return np.asarray(image_1d).reshape(len(x_axis), len(z_axis)).T


def envelope_db(grid, ref_max=None):
    """복소/실수 격자 -> 최대값 기준 정규화 dB (log compression 전 단계 표시용)."""
    env = np.abs(grid)
    m = env.max() if ref_max is None else ref_max
    return 20.0 * np.log10(env / m + 1e-12)


# ----------------------------------------------------------------------------
# 1. 해상도 : FWHM
# ----------------------------------------------------------------------------
def _fwhm_from_profile(coord, profile, peak_idx, drop_db=6.0):
    """1D 프로파일에서 peak_idx 좌우로 -drop_db 지점을 선형보간해 full width를 구한다.

    Parameters
    ----------
    coord   : (N,) 좌표 [m] (단조 증가)
    profile : (N,) 선형 envelope 값 (dB 아님)
    peak_idx: 피크 샘플 인덱스
    drop_db : 몇 dB 떨어진 지점의 폭을 잴지 (기본 6 dB = FWHM)

    Returns
    -------
    (width, left_coord, right_coord)  전부 [m]. 한쪽이라도 창 안에서 안 만나면 np.nan.
    """
    thr = profile[peak_idx] * (10.0 ** (-drop_db / 20.0))

    # 왼쪽 교차점
    left = np.nan
    for i in range(peak_idx, 0, -1):
        if profile[i] >= thr >= profile[i - 1]:
            t = (thr - profile[i - 1]) / (profile[i] - profile[i - 1] + 1e-30)
            left = coord[i - 1] + t * (coord[i] - coord[i - 1])
            break

    # 오른쪽 교차점
    right = np.nan
    for i in range(peak_idx, len(profile) - 1):
        if profile[i] >= thr >= profile[i + 1]:
            t = (thr - profile[i]) / (profile[i + 1] - profile[i] + 1e-30)
            right = coord[i] + t * (coord[i + 1] - coord[i])
            break

    return (right - left, left, right) if np.isfinite(left) and np.isfinite(right) else (np.nan, left, right)


def measure_point_fwhm(grid, x_axis, z_axis, x0, z0, win_mm=3.0, drop_db=6.0):
    """(x0, z0) 근처의 point target 하나에 대해 lateral / axial FWHM을 잰다.

    (x0, z0) 를 중심으로 ±win_mm 창 안에서 실제 피크 픽셀을 찾은 뒤,
    그 피크를 지나는 가로선 / 세로선 프로파일에서 -drop_db full width를 구한다.
    win_mm 은 이웃 point(보통 5 mm 간격)를 물지 않을 만큼 작아야 한다.

    Returns
    -------
    dict(x_peak, z_peak [m], fwhm_lateral_m, fwhm_axial_m,
         lat_left, lat_right, ax_left, ax_right [m])
    """
    x_axis = np.asarray(x_axis); z_axis = np.asarray(z_axis)
    env = np.abs(grid)
    w = win_mm * 1e-3

    xm = np.abs(x_axis - x0 * 1e-3) <= w
    zm = np.abs(z_axis - z0 * 1e-3) <= w
    ix = np.where(xm)[0]; iz = np.where(zm)[0]
    sub = env[np.ix_(iz, ix)]
    lz, lx = np.unravel_index(np.argmax(sub), sub.shape)
    z_pk = iz[lz]; x_pk = ix[lx]

    # 피크를 지나는 프로파일 (교차점 탐색은 창 안으로 제한)
    lat_w, lat_l, lat_r = _fwhm_from_profile(x_axis[ix], env[z_pk, ix], lx, drop_db)
    ax_w, ax_l, ax_r = _fwhm_from_profile(z_axis[iz], env[iz, x_pk], lz, drop_db)

    return dict(
        x_peak=float(x_axis[x_pk]), z_peak=float(z_axis[z_pk]),
        fwhm_lateral_m=float(lat_w), fwhm_axial_m=float(ax_w),
        lat_left=lat_l, lat_right=lat_r, ax_left=ax_l, ax_right=ax_r,
    )


# PICMUS resolution_distortion 팬텀의 공칭 point 위치 [mm]
# (동봉된 USTB 참조 영상에서 검출한 값 — 축상 8점 + z=20/40 가로줄)
PICMUS_RESOLUTION_POINTS_MM = [
    (0.0, 10.0), (0.0, 15.0), (0.0, 25.0), (0.0, 30.0), (0.0, 35.0), (0.0, 45.0),
    (-15.0, 20.0), (-10.0, 20.0), (-5.0, 20.0), (0.0, 20.0), (5.0, 20.0), (10.0, 20.0), (15.0, 20.0),
    (-10.0, 40.0), (-5.0, 40.0), (0.0, 40.0), (5.0, 40.0), (10.0, 40.0),
]


def measure_all_points(grid, x_axis, z_axis, points_mm=None, wavelength_m=None, **kw):
    """공칭 point 목록 전체에 대해 measure_point_fwhm을 돌려 리스트로 반환.

    wavelength_m을 주면 각 FWHM을 파장 단위로도 채워 준다.
    """
    if points_mm is None:
        points_mm = PICMUS_RESOLUTION_POINTS_MM
    rows = []
    for (x0, z0) in points_mm:
        r = measure_point_fwhm(grid, x_axis, z_axis, x0, z0, **kw)
        r["x_nominal_mm"] = x0
        r["z_nominal_mm"] = z0
        if wavelength_m:
            r["fwhm_lateral_lambda"] = r["fwhm_lateral_m"] / wavelength_m
            r["fwhm_axial_lambda"] = r["fwhm_axial_m"] / wavelength_m
        rows.append(r)
    return rows


# ----------------------------------------------------------------------------
# 2. 대조도 : CR / CNR / gCNR
# ----------------------------------------------------------------------------
def _disk_mask(x_axis, z_axis, xc_mm, zc_mm, r_mm):
    """(N_z, N_x) 격자에서 중심 (xc,zc), 반지름 r [mm] 안쪽 픽셀 마스크."""
    xx, zz = np.meshgrid(np.asarray(x_axis), np.asarray(z_axis))   # (N_z, N_x) [m]
    d2 = (xx - xc_mm * 1e-3) ** 2 + (zz - zc_mm * 1e-3) ** 2
    return d2 < (r_mm * 1e-3) ** 2


def contrast_metrics(grid, x_axis, z_axis,
                     xc_mm=0.0, zc_mm=30.0,
                     r_roi_mm=3.0, r_bg_inner_mm=5.0, r_bg_outer_mm=7.5):
    """무에코 낭종(ROI)과 그 주변 speckle(background 링)로 CR / CNR / gCNR 계산.

    기본 ROI 좌표는 PICMUS_simulation_contrast_speckle 팬텀 기준
    (USTB UiO IN4015 course, PW_compounding.m): 중심 (0, 30 mm),
    ROI 반지름 3 mm, background 링 5~7.5 mm.

    Returns
    -------
    dict(CR_dB, CNR, gCNR, n_roi, n_bg, roi_mask, bg_mask)
    """
    roi = _disk_mask(x_axis, z_axis, xc_mm, zc_mm, r_roi_mm)
    ring_out = _disk_mask(x_axis, z_axis, xc_mm, zc_mm, r_bg_outer_mm)
    ring_in = _disk_mask(x_axis, z_axis, xc_mm, zc_mm, r_bg_inner_mm)
    bg = ring_out & ~ring_in

    s = np.abs(grid)
    p_roi = s[roi] ** 2          # power
    p_bg = s[bg] ** 2

    cr_db = 10.0 * np.log10(p_roi.mean() / p_bg.mean())
    cnr = abs(p_roi.mean() - p_bg.mean()) / np.sqrt(p_roi.var() + p_bg.var())

    # gCNR : dB 값 히스토그램(100 bin) 겹침. measure_contrast_circles.m 과 동일.
    s_all = s[roi | bg]
    lo = 20.0 * np.log10(s_all[s_all > np.finfo(float).eps].min())
    hi = 20.0 * np.log10(s.max())
    edges = np.linspace(lo, hi, 101)
    hi_, _ = np.histogram(20.0 * np.log10(s[roi] + 1e-30), bins=edges)
    ho_, _ = np.histogram(20.0 * np.log10(s[bg] + 1e-30), bins=edges)
    hi_ = hi_ / hi_.sum(); ho_ = ho_ / ho_.sum()
    ovl = np.minimum(hi_, ho_).sum()

    return dict(
        CR_dB=float(cr_db), CNR=float(cnr), gCNR=float(1.0 - ovl),
        n_roi=int(roi.sum()), n_bg=int(bg.sum()),
        roi_mask=roi, bg_mask=bg,
    )


# ----------------------------------------------------------------------------
# 3. USTB 참조 영상과의 대조
# ----------------------------------------------------------------------------
def image_nrmse_db(test_grid, ref_grid, dynamic_range=60.0):
    """두 빔포밍 결과의 dB 영상(각자 최대값으로 정규화, [-dr, 0] clip) 사이
    normalized RMSE. 0에 가까울수록 참조와 일치.
    """
    a = np.clip(envelope_db(test_grid), -dynamic_range, 0.0)
    b = np.clip(envelope_db(ref_grid), -dynamic_range, 0.0)
    return float(np.sqrt(np.mean((a - b) ** 2)) / dynamic_range)
