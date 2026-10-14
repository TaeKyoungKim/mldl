"""
================================================================================
Two Spirals Noisy Dense 데이터셋 DBSCAN 함수(sklearn.cluster.dbscan) 클러스터링
================================================================================
- Scikit-learn 공식 문서 레퍼런스:
  https://scikit-learn.org/stable/modules/generated/dbscan-function.html#sklearn.cluster.dbscan
  `core_samples, labels = dbscan(X, eps=..., min_samples=..., metric='minkowski', ...)`
- 분석 절차:
  1. 원본 데이터 로드 (x, y 좌표 및 정답 cluster 라벨)
  2. k-최근접 이웃 거리(k-distance graph)를 통한 eps 탐색 가이드라인 제공
  3. 다양한 하이퍼파라미터(eps, min_samples) 그리드 서치 및 성능 지표(ARI, 노이즈 비율 등) 평가
  4. 최적 및 비교 파라미터별 core_samples / border / noise 시각화 및 정답 라벨과의 비교
================================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import dbscan
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

# 콘솔 UTF-8 및 한글 폰트 설정
if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    plt.rc("font", family="Malgun Gothic")
    plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "src" else Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "two_spirals_noisy_dense.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH = OUTPUT_DIR / "two_spirals_dbscan_analysis.png"


def load_dataset():
    """데이터 로드 및 기본 통계 출력"""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"데이터 파일이 존재하지 않습니다: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    X = df[["x", "y"]].values
    y_true = df["cluster"].values

    print(f"[1] 데이터셋 로드 완료: 총 {len(df)}개 샘플")
    print("    - 특성: x (범위: [{:.2f}, {:.2f}]), y (범위: [{:.2f}, {:.2f}])".format(
        X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max()
    ))
    print(f"    - 실제 라벨 분포: Spiral 0 ({np.sum(y_true==0)}개), Spiral 1 ({np.sum(y_true==1)}개), 노이즈 -1 ({np.sum(y_true==-1)}개)")
    return df, X, y_true


def grid_search_dbscan(X, y_true):
    """
    sklearn.cluster.dbscan 함수를 이용한 다양한 파라미터 조합 평가
    """
    print("\n" + "=" * 80)
    print("  [2] sklearn.cluster.dbscan 함수를 활용한 파라미터 그리드 서치")
    print("=" * 80)

    eps_candidates = [0.6, 0.8, 1.0, 1.15, 1.25, 1.5]
    min_samples_candidates = [2, 3, 4, 5]

    search_results = []

    for eps in eps_candidates:
        for min_samples in min_samples_candidates:
            # sklearn.cluster.dbscan 함수 호출 (반환값: core_samples, labels)
            core_samples, labels = dbscan(
                X,
                eps=eps,
                min_samples=min_samples,
                metric="euclidean"
            )

            # 고유 군집 개수 (노이즈 -1 제외)
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
            n_noise = int(np.sum(labels == -1))
            n_core = len(core_samples)

            # 실제 군집 라벨과의 유사도 (Adjusted Rand Index, NMI)
            ari = adjusted_rand_score(y_true, labels)
            nmi = normalized_mutual_info_score(y_true, labels)

            # 유효 군집이 2개 이상일 때 실루엣 계수 계산
            sil = silhouette_score(X, labels) if n_clusters > 1 and len(unique_labels) < len(X) else -1.0

            search_results.append({
                "eps": eps,
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "core_points": n_core,
                "noise_points": n_noise,
                "noise_ratio(%)": (n_noise / len(X)) * 100,
                "ARI": ari,
                "NMI": nmi,
                "Silhouette": sil
            })

    df_results = pd.DataFrame(search_results)
    print(df_results.sort_values(by=["ARI", "NMI"], ascending=False).head(10).to_string(index=False, formatters={
        "eps": lambda x: f"{x:5.2f}",
        "min_samples": lambda x: f"{x:11d}",
        "n_clusters": lambda x: f"{x:10d}",
        "core_points": lambda x: f"{x:11d}",
        "noise_points": lambda x: f"{x:12d}",
        "noise_ratio(%)": lambda x: f"{x:13.1f}%",
        "ARI": lambda x: f"{x:8.4f}",
        "NMI": lambda x: f"{x:8.4f}",
        "Silhouette": lambda x: f"{x:10.4f}" if x > -1 else "       N/A"
    }))
    print("=" * 80)
    return df_results


def run_single_dbscan(X, eps=0.8, min_samples=3):
    """
    특정 파라미터로 dbscan() 함수를 단일 실행하고 결과 분석
    """
    print(f"\n[3] 단일 DBSCAN 실행: eps={eps}, min_samples={min_samples}")
    core_samples, labels = dbscan(X, eps=eps, min_samples=min_samples, metric="euclidean")
    
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    
    print(f"    - 반환된 핵심 샘플 인덱스(core_samples) 수: {len(core_samples)}개")
    print(f"    - 형성된 군집 개수(노이즈 제외): {n_clusters}개")
    print(f"    - 검출된 노이즈(-1) 개수: {n_noise}개 (전체의 {n_noise/len(X)*100:.1f}%)")
    
    return core_samples, labels


def visualize_dbscan_comparison(X, y_true):
    """
    정답 데이터 및 DBSCAN 파라미터별 클러스터링 결과 다중 시각화
    """
    print("\n[4] DBSCAN 파라미터별 클러스터링 비교 시각화 차트 생성 중...")
    
    test_params = [
        (0.8, 3, "ARI 최대화 파라미터 (세밀한 밀도 분할)"),
        (1.0, 4, "중간 대역폭 파라미터"),
        (1.2, 4, "넓은 대역폭 파라미터 (연결성 강조)")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # (1) 실제 정답 라벨 (Ground Truth)
    ax_gt = axes[0, 0]
    unique_gt = sorted(set(y_true))
    colors_gt = {0: "#E76F51", 1: "#2A9D8F", -1: "#264653"}
    for c in unique_gt:
        mask = (y_true == c)
        label_name = f"Spiral {c}" if c >= 0 else "Noise (-1)"
        marker = "o" if c >= 0 else "x"
        ax_gt.scatter(X[mask, 0], X[mask, 1], c=colors_gt.get(c, "gray"),
                      label=label_name, marker=marker, s=50, alpha=0.85, edgecolors="k" if marker == "o" else None)
    ax_gt.set_title("1. 원본 데이터 정답 라벨 (Ground Truth)\n(Spiral 0: 200개, Spiral 1: 200개, Noise: 20개)", fontsize=12, fontweight="bold")
    ax_gt.set_xlabel("X 좌표")
    ax_gt.set_ylabel("Y 좌표")
    ax_gt.grid(True, linestyle=":", alpha=0.5)
    ax_gt.legend(loc="upper right")

    # (2) ~ (4) DBSCAN 함수 실행 결과 비교
    plot_coords = [(0, 1), (1, 0), (1, 1)]
    for (eps, min_samples, desc), (r, c) in zip(test_params, plot_coords):
        ax = axes[r, c]
        core_samples, labels = dbscan(X, eps=eps, min_samples=min_samples)
        
        unique_labels = sorted(set(labels))
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))
        ari = adjusted_rand_score(y_true, labels)

        # 색상 팔레트
        palette = sns.color_palette("tab20", max(n_clusters, 1))

        # 코어 샘플 마스크 생성
        core_mask = np.zeros(len(X), dtype=bool)
        core_mask[core_samples] = True

        for lbl in unique_labels:
            if lbl == -1:
                # 노이즈 점
                mask = (labels == -1)
                ax.scatter(X[mask, 0], X[mask, 1], c="black", marker="x", s=40,
                           label=f"Noise ({n_noise}개)", alpha=0.6)
            else:
                mask = (labels == lbl)
                color = palette[lbl % len(palette)]
                
                # 핵심 포인트 (Core samples)
                ax.scatter(X[mask & core_mask, 0], X[mask & core_mask, 1],
                           color=color, marker="o", s=60, edgecolors="k", linewidths=0.8, alpha=0.9)
                # 경계 포인트 (Border samples)
                ax.scatter(X[mask & ~core_mask, 0], X[mask & ~core_mask, 1],
                           color=color, marker="^", s=45, edgecolors="k", linewidths=0.5, alpha=0.7)

        ax.set_title(f"DBSCAN(eps={eps}, min_samples={min_samples}) | {desc}\n(군집 수: {n_clusters}개, Core: {len(core_samples)}개, Noise: {n_noise}개, ARI: {ari:.4f})",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("X 좌표")
        ax.set_ylabel("Y 좌표")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", fontsize=9)

    plt.suptitle("sklearn.cluster.dbscan 함수를 이용한 Two Spirals 데이터 클러스터링 분석",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(SAVE_PATH, dpi=300)
    plt.close()
    print(f"  - 비교 시각화 차트 저장 완료: {SAVE_PATH}")


def main():
    print("#" * 80)
    print("  TWO SPIRALS NOISY DENSE - SKLEARN.CLUSTER.DBSCAN FUNCTION EVALUATION")
    print("#" * 80)

    # 1. 데이터셋 로드
    df, X, y_true = load_dataset()

    # 2. 그리드 서치 수행
    df_results = grid_search_dbscan(X, y_true)

    # 3. 대표 파라미터 단일 실행
    core_samples, labels = run_single_dbscan(X, eps=0.8, min_samples=3)

    # 4. 시각화 생성 및 저장
    visualize_dbscan_comparison(X, y_true)

    print("\n" + "#" * 80)
    print("  DBSCAN 클러스터링 파이프라인 수행이 완료되었습니다!")
    print(f"  결과 그래프 경로: {SAVE_PATH}")
    print("#" * 80)


if __name__ == "__main__":
    main()
