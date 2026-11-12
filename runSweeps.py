"""스윕 2종 + 파레토 프론트를 순서대로 실행하는 엔트리포인트.

  python runSweeps.py [--backend gpu|cpu] [--force] [--png]

단계
  1. contrast 스윕    (190조합) -> results/apod_sweep_{backend}.csv
  2. resolution 스윕  (190조합) -> results/apod_sweep_resolution_{backend}.csv
  3. 파레토 프론트              -> results/pareto_cr_fwhm.{csv,png}, results/pareto_summary.md

- 1·2단계는 같은 설정 격자(f# 0.1~1.9 x Tukey 10~100)를 쓰기 때문에
  3단계에서 label로 1:1 조인이 된다. 한쪽만 새로 돌리면 격자가 어긋나므로
  두 스윕은 항상 이 스크립트로 같이 관리한다.
- 이미 결과 CSV가 있는 단계는 건너뛴다(--force로 강제 재실행).
  스윕 하나가 GPU로 약 1분, CPU로 약 6시간이라 재실행 비용이 크기 때문이다.
- 각 단계는 별도 프로세스로 띄운다. 어느 단계에서 깨졌는지 종료 코드로
  바로 드러나고, GPU 메모리도 단계마다 확실히 회수된다.
"""
import argparse
import os
import subprocess
import sys
import time

RESULTS = "./results"


def run_step(n, total, title, argv, produces, force):
    """한 단계 실행. (건너뛰었으면 None, 돌렸으면 걸린 시간 [s]) 반환."""
    if produces and os.path.exists(produces) and not force:
        print(f"\n[{n}/{total}] {title} — 건너뜀 (이미 있음: {produces}, 다시 돌리려면 --force)")
        return None

    print(f"\n[{n}/{total}] {title}")
    print("      $ " + " ".join(argv))
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable] + argv)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        print(f"\n[{n}/{total}] 실패 (exit {r.returncode}). 위 오류를 확인하세요.")
        if "--backend" in argv and "gpu" in argv:
            print("      GPU 단계라면 cupy / CUDA 환경을 먼저 확인: "
                  'pip install "cupy-cuda12x[ctk]"')
        sys.exit(r.returncode)
    print(f"[{n}/{total}] 완료 — {dt:.1f} s")
    return dt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--n-waves", type=int, default=75)
    ap.add_argument("--force", action="store_true",
                    help="결과 CSV가 있어도 스윕을 다시 돌린다")
    ap.add_argument("--png", action="store_true",
                    help="설정별 B-mode PNG도 저장 (190장 x 2, 느려짐)")
    ap.add_argument("--metric", choices=["fwhm_lat_mm", "fwhm_ax_mm"], default="fwhm_lat_mm",
                    help="파레토의 해상도 축 (기본: 횡방향 FWHM)")
    args = ap.parse_args()

    common = ["--backend", args.backend, "--n-waves", str(args.n_waves)]
    if not args.png:
        common.append("--no-png")

    t0 = time.perf_counter()
    run_step(1, 3, "contrast 스윕 (CR / CNR / gCNR)",
             ["runDasApodCompare.py", "--dataset", "contrast"] + common,
             f"{RESULTS}/apod_sweep_{args.backend}.csv", args.force)
    run_step(2, 3, "resolution 스윕 (point target FWHM)",
             ["runDasApodCompare.py", "--dataset", "resolution"] + common,
             f"{RESULTS}/apod_sweep_resolution_{args.backend}.csv", args.force)
    # 파레토는 입력 CSV가 바뀌었을 수 있으니 항상 다시 계산한다
    run_step(3, 3, "CR vs FWHM 파레토 프론트",
             ["runParetoFront.py", "--backend", args.backend, "--metric", args.metric],
             None, args.force)

    print(f"\n전체 완료 — {time.perf_counter() - t0:.1f} s")
    print(f"결과: {RESULTS}/pareto_cr_fwhm.png, {RESULTS}/pareto_summary.md")


if __name__ == "__main__":
    main()
