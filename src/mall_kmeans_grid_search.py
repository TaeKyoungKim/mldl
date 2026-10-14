"""
================================================================================
쇼핑몰 고객(Mall Customers) K-Means 군집화 그리드 서치 및 실루엣 평가 모듈
================================================================================
- 데이터 경로: C:/apps/mldl/data/Mall_Customers.csv
- 분석 내용:
  1. K-Means 군집 수 k = [2, 3, 4, 5, 6, 7, 8] 그리드 서치 (Grid Search)
  2. 각 k별 평가지표 산출:
     - 관성/이너셔 (Inertia, WCSS - Within-Cluster Sum of Squares)
     - 전체 실루엣 계수 (Global Silhouette Score)
     - 데이비스-볼딘 지수 (Davies-Bouldin Index, 낮을수록 우수)
     - 칼린스키-하라바즈 지수 (Calinski-Harabasz Index, 높을수록 우수)
     - 군집별 실루엣 평균 및 음수(오분류 경계) 샘플 수
  3. 그리드 서치 결과 콘솔 테이블 및 상세 분석 출력
  4. 최적 k 선정 및 시각화 저장 (엘보우 곡선 + 실루엣 곡선 + k=5 최적 군집 산점도)
================================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score

# 콘솔 UTF-8 및 matplotlib 한글 폰트 설정
if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    plt.rc("font", family="Malgun Gothic")
    plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "src" else Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "Mall_Customers.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "mall_clustering"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def grid_search_kmeans(X, k_values=range(2, 9), random_state=42):
    """
    k = [2, 3, 4, 5, 6, 7, 8] 범위에서 KMeans를 그리드 서치하고 실루엣 등 복합 평가 지표를 산출
    """
    results = []
    models = {}
    sample_silhouette_dict = {}

    print("\n" + "=" * 80)
    print("  [K-Means 군집 수(k = 2~8) 그리드 서치 및 평가 수행 중...]")
    print("=" * 80)

    for k in k_values:
        # K-Means 모델 생성 및 학습 (n_init=10으로 초기 중심점 안정화)
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=random_state)
        labels = kmeans.fit_predict(X)
        
        inertia = kmeans.inertia_
        sil_score = silhouette_score(X, labels)
        db_score = davies_bouldin_score(X, labels)
        ch_score = calinski_harabasz_score(X, labels)
        
        # 개별 샘플별 실루엣 계수 계산
        sample_sil = silhouette_samples(X, labels)
        neg_count = np.sum(sample_sil < 0)
        neg_ratio = (neg_count / len(X)) * 100
        
        # 군집별 실루엣 평균 리스트
        c_sil_means = [np.mean(sample_sil[labels == c]) for c in range(k)]
        min_c_sil = np.min(c_sil_means)
        
        results.append({
            "k": k,
            "Inertia (WCSS)": inertia,
            "Silhouette Score": sil_score,
            "Davies-Bouldin (↓)": db_score,
            "Calinski-Harabasz (↑)": ch_score,
            "Negative Samples": neg_count,
            "Negative Ratio (%)": neg_ratio,
            "Min Cluster Sil": min_c_sil
        })
        
        models[k] = (kmeans, labels)
        sample_silhouette_dict[k] = sample_sil
        
        print(f"  * k={k:2d} | Silhouette: {sil_score:.4f} | Inertia: {inertia:8.1f} | 음수 샘플: {neg_count}개 ({neg_ratio:.1f}%)")

    df_results = pd.DataFrame(results)
    return df_results, models, sample_silhouette_dict


def display_and_analyze_results(df_results, models, sample_silhouette_dict, X):
    """그리드 서치 결과 요약 표 및 최적 K 분석 출력"""
    print("\n" + "=" * 88)
    print("                      [K-Means 클러스터링 그리드 서치 종합 평가표]")
    print("=" * 88)
    
    # 순위 부여 (실루엣 기준)
    df_sorted = df_results.sort_values("Silhouette Score", ascending=False).reset_index(drop=True)
    best_k = int(df_sorted.iloc[0]["k"])
    best_sil = df_sorted.iloc[0]["Silhouette Score"]
    
    # 포맷팅 출력
    print(df_results.to_string(index=False, formatters={
        "k": lambda x: f"{x:d}",
        "Inertia (WCSS)": lambda x: f"{x:12.2f}",
        "Silhouette Score": lambda x: f"{x:16.4f}",
        "Davies-Bouldin (↓)": lambda x: f"{x:18.4f}",
        "Calinski-Harabasz (↑)": lambda x: f"{x:21.2f}",
        "Negative Samples": lambda x: f"{x:16d}",
        "Negative Ratio (%)": lambda x: f"{x:17.2f}%",
        "Min Cluster Sil": lambda x: f"{x:15.4f}"
    }))
    print("=" * 88)
    
    print(f"\n[그리드 서치 최적 클러스터 수 판정]:")
    print(f"  ★ 최고 실루엣 점수: k = {best_k} (Silhouette Score = {best_sil:.4f})")
    
    # 2위와의 격차
    second_k = int(df_sorted.iloc[1]["k"])
    second_sil = df_sorted.iloc[1]["Silhouette Score"]
    diff = best_sil - second_sil
    print(f"  - 2위 후보: k = {second_k} (Silhouette Score = {second_sil:.4f}, 격차 = +{diff:.4f})")
    
    # 최적 k 군집별 세부 실루엣 분석
    best_model, best_labels = models[best_k]
    best_samples = sample_silhouette_dict[best_k]
    
    print(f"\n[최적 k = {best_k} 군집별 실루엣 세부 평가]:")
    best_details = []
    for c in range(best_k):
        c_vals = best_samples[best_labels == c]
        c_neg = np.sum(c_vals < 0)
        c_center = best_model.cluster_centers_[c]
        best_details.append({
            "Cluster ID": f"C{c}",
            "고객 수": len(c_vals),
            "중심점 (소득, 지출)": f"({c_center[0]:.1f}, {c_center[1]:.1f})",
            "실루엣 평균": np.mean(c_vals),
            "실루엣 최솟값": np.min(c_vals),
            "실루엣 최댓값": np.max(c_vals),
            "음수 샘플 수": c_neg
        })
    df_best_details = pd.DataFrame(best_details)
    print(df_best_details.to_string(index=False, formatters={
        "실루엣 평균": lambda x: f"{x:.4f}",
        "실루엣 최솟값": lambda x: f"{x:.4f}",
        "실루엣 최댓값": lambda x: f"{x:.4f}"
    }))
    print("-" * 88)


def plot_grid_search_visualizations(df_results, models, sample_silhouette_dict, X):
    """그리드 서치 결과 시각화 (엘보우 곡선, 실루엣 평가 곡선, 최적 K 산점도 및 다이어그램)"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    k_vals = df_results["k"].values
    inertias = df_results["Inertia (WCSS)"].values
    sil_scores = df_results["Silhouette Score"].values
    
    best_k = int(df_results.sort_values("Silhouette Score", ascending=False).iloc[0]["k"])
    
    # (1) 엘보우 곡선 (Elbow Method - Inertia)
    axes[0].plot(k_vals, inertias, marker="o", color="#2A9D8F", linewidth=2.5, markersize=7)
    axes[0].axvline(x=best_k, color="#E63946", linestyle="--", linewidth=1.5, label=f"엘보우 분기점 (k={best_k})")
    axes[0].set_title("1. 엘보우 곡선 (Inertia / WCSS)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("클러스터 개수 (k)")
    axes[0].set_ylabel("Inertia (군집 내 오차제곱합)")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()
    for k, v in zip(k_vals, inertias):
        axes[0].annotate(f"{int(v)}", (k, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
        
    # (2) 실루엣 평가 곡선 (Silhouette Score Curve)
    axes[1].plot(k_vals, sil_scores, marker="s", color="#1D3557", linewidth=2.5, markersize=7)
    axes[1].axvline(x=best_k, color="#E63946", linestyle="--", linewidth=1.5, label=f"최고 실루엣 피크 (k={best_k})")
    axes[1].set_title("2. 실루엣 계수 평가 곡선 (Grid Search)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("클러스터 개수 (k)")
    axes[1].set_ylabel("평균 실루엣 계수 (Silhouette Score)")
    axes[1].set_ylim([0.3, 0.65])
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()
    for k, v in zip(k_vals, sil_scores):
        axes[1].annotate(f"{v:.4f}", (k, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, fontweight="bold")

    # (3) 최적 k (k=5) 군집 분할 산점도
    best_model, best_labels = models[best_k]
    centers = best_model.cluster_centers_
    palette = sns.color_palette("tab10", best_k)
    
    for c in range(best_k):
        axes[2].scatter(X[best_labels == c, 0], X[best_labels == c, 1], s=55, color=palette[c],
                        label=f"Cluster {c}", edgecolors="k", alpha=0.8)
    # 중심점 표시
    axes[2].scatter(centers[:, 0], centers[:, 1], s=200, c="red", marker="X", edgecolors="black", linewidths=1.5, label="Centroids")
    axes[2].set_title(f"3. 최적 k={best_k} 군집 산점도 (Silhouette: {sil_scores[best_k-2]:.4f})", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("연간 소득 (Annual Income, k$)")
    axes[2].set_ylabel("소비 점수 (Spending Score, 1-100)")
    axes[2].grid(True, linestyle=":", alpha=0.5)
    axes[2].legend(loc="upper right", fontsize=8)

    plt.suptitle("Mall Customers K-Means 클러스터링 개수(k=2~8) 그리드 서치 및 실루엣 평가", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    out_path = OUTPUT_DIR / "kmeans_grid_search_silhouette.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[시각화 완료] 그리드 서치 종합 차트 저장 완료: {out_path}")
    return out_path


def main():
    print("#" * 80)
    print("  MALL CUSTOMERS K-MEANS CLUSTERING GRID SEARCH & SILHOUETTE EVALUATION")
    print("#" * 80)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"데이터 파일이 존재하지 않습니다: {DATA_PATH}")

    # 데이터 로드 (Annual Income, Spending Score 2변수)
    df = pd.read_csv(DATA_PATH)
    X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values
    print(f"데이터셋 로드: {df.shape[0]}명 고객 샘플")

    # k=2~8 그리드 서치 실행
    df_results, models, sample_silhouette_dict = grid_search_kmeans(X, k_values=range(2, 9), random_state=42)

    # 결과 분석 및 출력
    display_and_analyze_results(df_results, models, sample_silhouette_dict, X)

    # 시각화 저장
    plot_grid_search_visualizations(df_results, models, sample_silhouette_dict, X)

    # 결과 CSV 저장
    csv_out = OUTPUT_DIR / "kmeans_grid_search_results.csv"
    df_results.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f"[CSV 저장] 그리드 서치 수치 평가 결과 저장 완료: {csv_out}")


if __name__ == "__main__":
    main()
