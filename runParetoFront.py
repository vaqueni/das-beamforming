"""해상도-대조도 트레이드오프: CR vs FWHM 파레토 프론트.

  python runParetoFront.py [--backend gpu|cpu] [--metric fwhm_lat_mm|fwhm_ax_mm]

전제: 같은 190개 apodization 설정(f# 0.1~1.9 x tukey10~100)에 대해
runDasApodCompare.py를 두 데이터셋으로 돌려 둔 상태.
    results/apod_sweep_{backend}.csv             (contrast : CR / CNR / gCNR)
    results/apod_sweep_resolution_{backend}.csv  (resolution : FWHM)

두 CSV를 label로 조인해서, 한 설정이 해상도(FWHM)와 대조도(CR)를 동시에
얼마나 가져가는지 본다. 둘 다 작을수록 좋은 지표라 최소화 기준으로 비교한다.
    FWHM 작음 = 점이 선명함 (f#를 키우면 유효 구경이 좁아져 FWHM은 커진다)
    CR  작음 = 낭종이 더 검게 비침 (무에코 ROI / 배경 전력비, dB)
어떤 설정도 두 지표를 동시에 지배하지 못하는 집합이 파레토 프론트이고,
그 위에서 어디를 고를지가 곧 "해상도를 얼마 내주고 대조도를 얼마 받는가"다.

결과: results/pareto_cr_fwhm.csv, results/pareto_cr_fwhm.png, results/pareto_summary.md
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "./results"
DEFAULT_CFG = (1.7, 50)


def load_csv(path):
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} 가 없다. 먼저 스윕을 돌려야 한다:\n"
            f"  python runDasApodCompare.py --dataset contrast   --backend gpu --no-png\n"
            f"  python runDasApodCompare.py --dataset resolution --backend gpu --no-png")
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def pareto_front(pts):
    """최소화 기준 2D 비지배 집합의 인덱스. pts: (N, 2) [x, y].

    x 오름차순으로 훑으며 지금까지의 최소 y보다 작아지는 점만 남긴다.
    동일 x에서는 y가 작은 쪽이 먼저 오도록 정렬해 중복 설정도 안전하게 처리한다.
    """
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    keep, best_y = [], np.inf
    for i in order:
        if pts[i, 1] < best_y:
            keep.append(i)
            best_y = pts[i, 1]
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--metric", choices=["fwhm_lat_mm", "fwhm_ax_mm"], default="fwhm_lat_mm",
                    help="해상도 축으로 쓸 FWHM (기본: 횡방향)")
    args = ap.parse_args()

    con = load_csv(f"{OUT_DIR}/apod_sweep_{args.backend}.csv")
    res = load_csv(f"{OUT_DIR}/apod_sweep_resolution_{args.backend}.csv")
    axis_name = "lateral" if args.metric == "fwhm_lat_mm" else "axial"

    res_by_label = {r["label"]: r for r in res}
    merged = []
    for c in con:
        r = res_by_label.get(c["label"])
        if r is None:
            continue
        fwhm = float(r[args.metric])
        cr = float(c["CR_dB"])
        if not (np.isfinite(fwhm) and np.isfinite(cr)):
            continue        # 교차점 검출이 깨진 설정은 비교 대상에서 뺀다
        merged.append(dict(label=c["label"], f_number=float(c["f_number"]),
                           tukey=int(c["tukey"]), fwhm_mm=fwhm, CR_dB=cr,
                           CNR=float(c["CNR"]), gCNR=float(c["gCNR"]),
                           n_points=int(r["n_points_lat"])))
    if not merged:
        raise SystemExit("두 CSV에서 겹치는 유효 설정이 없다. 같은 설정 격자로 돌렸는지 확인.")
    print(f"조인된 설정: {len(merged)} / {len(con)}  "
          f"(FWHM 검출 실패로 제외: {len(con) - len(merged)})")

    pts = np.array([[m["fwhm_mm"], m["CR_dB"]] for m in merged])
    front = pareto_front(pts)
    front_sorted = sorted(front, key=lambda i: pts[i, 0])
    for i in front:
        merged[i]["on_pareto"] = 1
    for m in merged:
        m.setdefault("on_pareto", 0)

    print(f"\n=== 파레토 프론트 ({len(front)}개 설정) ===")
    print(f"{'f#':>5} {'tukey':>6} | {'FWHM mm':>9} {'CR dB':>8} {'CNR':>7} {'gCNR':>7}")
    print("-" * 52)
    for i in front_sorted:
        m = merged[i]
        print(f"{m['f_number']:5.1f} {m['tukey']:6d} | {m['fwhm_mm']:9.3f} "
              f"{m['CR_dB']:8.2f} {m['CNR']:7.3f} {m['gCNR']:7.4f}")

    csv_path = f"{OUT_DIR}/pareto_cr_fwhm.csv"
    cols = ["label", "f_number", "tukey", "fwhm_mm", "CR_dB", "CNR", "gCNR",
            "n_points", "on_pareto"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(sorted(merged, key=lambda m: (-m["on_pareto"], m["fwhm_mm"])))

    # --- figure : 산점도(색=tukey) + 프론트 계단 + 기준점 표시 ---
    fig, ax = plt.subplots(figsize=(9.5, 7))
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=[m["tukey"] for m in merged],
                    cmap="viridis", s=26, alpha=0.75, edgecolors="none")
    fig.colorbar(sc, ax=ax, label="Tukey roll [%]", pad=0.01)

    fx = pts[front_sorted, 0]
    fy = pts[front_sorted, 1]
    ax.step(fx, fy, where="post", color="#d62728", lw=1.6, zorder=3,
            label=f"Pareto front ({len(front)} configs)")
    ax.plot(fx, fy, "o", mfc="none", mec="#d62728", mew=1.6, ms=10, zorder=3)

    # 프론트 양 끝(해상도 우선 / 대조도 우선)과 기존 기본 설정을 라벨링
    for i, tag, off in ((front_sorted[0], "resolution-first", (12, 10)),
                        (front_sorted[-1], "contrast-first", (14, -26))):
        m = merged[i]
        ax.annotate(f"{tag}\nf/{m['f_number']:.1f}, tukey{m['tukey']}",
                    (m["fwhm_mm"], m["CR_dB"]), textcoords="offset points",
                    xytext=off, fontsize=8.5, color="#d62728")
    d_f, d_t = DEFAULT_CFG
    dflt = next((m for m in merged if m["f_number"] == d_f and m["tukey"] == d_t), None)
    if dflt:
        ax.plot(dflt["fwhm_mm"], dflt["CR_dB"], marker="s", ms=13, mfc="none",
                mec="black", mew=2.0, ls="none", zorder=4,
                label=f"default f/{d_f}, tukey{d_t}"
                      f"{'  (on front)' if dflt['on_pareto'] else '  (dominated)'}")

    ax.set(xlabel=f"{axis_name} FWHM [mm]  (lower = sharper)",
           ylabel="CR [dB]  (lower = darker cyst)",
           title="Resolution vs contrast trade-off over 190 apodization configs\n"
                 f"PICMUS plane-wave DAS, n_waves=75, backend={args.backend}")
    ax.grid(alpha=0.3, ls=":")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    # 좌하단이 둘 다 좋은 방향이라는 걸 화살표로 표시
    ax.annotate("better", xy=(0.04, 0.06), xytext=(0.16, 0.2), xycoords="axes fraction",
                textcoords="axes fraction", fontsize=9, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray"))
    fig.tight_layout()
    png = f"{OUT_DIR}/pareto_cr_fwhm.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)

    # --- README에 그대로 붙일 수 있는 마크다운 요약 ---
    md_path = f"{OUT_DIR}/pareto_summary.md"
    dom = len(merged) - len(front)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(f"| f# | Tukey | {axis_name} FWHM [mm] | CR [dB] | CNR | gCNR | 비고 |\n")
        fh.write("|---|---|---|---|---|---|---|\n")
        for i in front_sorted:
            m = merged[i]
            note = []
            if i == front_sorted[0]:
                note.append("해상도 우선")
            if i == front_sorted[-1]:
                note.append("대조도 우선")
            if dflt and m["label"] == dflt["label"]:
                note.append("기존 기본 설정")
            fh.write(f"| {m['f_number']:.1f} | {m['tukey']} | {m['fwhm_mm']:.3f} | "
                     f"{m['CR_dB']:.2f} | {m['CNR']:.3f} | {m['gCNR']:.4f} | "
                     f"{' · '.join(note)} |\n")
        fh.write(f"\n- 비교 대상 {len(merged)}개 설정 중 파레토 프론트 {len(front)}개, "
                 f"지배된 설정 {dom}개.\n")
        if dflt:
            state = "프론트 위에 있음" if dflt["on_pareto"] else "다른 설정에 지배됨"
            fh.write(f"- 기존 기본 설정 f/{d_f} tukey{d_t}: {axis_name} FWHM "
                     f"{dflt['fwhm_mm']:.3f} mm, CR {dflt['CR_dB']:.2f} dB — {state}.\n")
        fh.write(f"- FWHM 범위 {pts[:, 0].min():.3f} ~ {pts[:, 0].max():.3f} mm, "
                 f"CR 범위 {pts[:, 1].min():.2f} ~ {pts[:, 1].max():.2f} dB.\n")

    print(f"\nsaved {csv_path}, {png}, {md_path}")


if __name__ == "__main__":
    main()
