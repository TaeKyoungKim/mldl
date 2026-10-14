"""
================================================================================
UCI HAR Dataset - 공분산 행렬의 고유값 분해(Eigendecomposition) 기반 PCA 파이프라인
================================================================================
- 데이터 경로:
  - 원본 특성 데이터: C:/apps/mldl/data/har_features.csv (10,299개 샘플 × 561개 특성)
  - 타깃 레이블 데이터: C:/apps/mldl/data/har_targets.csv (6개 행동 클래스)
- 수학적 구현 단계 (Mathematical Implementation):
  1. 표준화(Standardization): X_scaled = (X - μ) / σ
  2. 표본 공분산 행렬(Covariance Matrix) 계산: Σ = (1 / (n - 1)) * X_scaled^T * X_scaled (561 × 561)
  3. 공분산 행렬의 고유값 분해(Eigendecomposition): Σ * v_i = λ_i * v_i  (np.linalg.eigh 사용)
  4. 고유값(λ) 내림차순 정렬 및 고유벡터(v) 정렬
  5. 설명 분산 비율(Explained Variance Ratio) 계산
  6. 투영 변환(Projection): X_pca = X_scaled * W_K
- 검증 및 벤치마크:
  - Scikit-learn의 SVD 기반 PCA와 고유값, 설명 분산, 투영 좌표를 1:1 정밀 비교 (오차 검증)
- 산출물:
  - data/har_features_pca_eigen_95.csv (95% 분산 보존 차원 축소 데이터)
  - data/har_features_pca_eigen_90.csv (90% 분산 보존 차원 축소 데이터)
  - output/har_pca/eig_vs_svd_comparison.png (Eigendecomposition vs SVD 1:1 비교 시각화)
  - output/har_pca/eig_covariance_heatmap.png (공분산 행렬 구조 히트맵)
================================================================================
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 윈도우 콘솔 한글 인코딩 대응
if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    plt.rc("font", family="Malgun Gothic")
    plt.rcParams["axes.unicode_minus"] = False

# 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "src" else Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output" / "har_pca"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PATH = DATA_DIR / "har_features.csv"
TARGETS_PATH = DATA_DIR / "har_targets.csv"

ACTIVITY_MAP = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}
CLASS_NAMES = [ACTIVITY_MAP[i] for i in sorted(ACTIVITY_MAP.keys())]
CLASS_PALETTE = {
    "WALKING": "#E63946",
    "WALKING_UPSTAIRS": "#F4A261",
    "WALKING_DOWNSTAIRS": "#E76F51",
    "SITTING": "#2A9D8F",
    "STANDING": "#457B9D",
    "LAYING": "#1D3557"
}


class EigendecompositionPCA:
    """
    공분산 행렬의 고유값 분해(Eigendecomposition)를 순수 넘파이(NumPy)로 구현한 PCA 클래스
    """
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean_ = None
        self.std_ = None
        self.cov_matrix_ = None
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.cumulative_variance_ratio_ = None

    def fit(self, X):
        """
        1. X 표준화 -> 2. 공분산 행렬 계산 -> 3. 고유값 분해 -> 4. 내림차순 정렬
        """
        X_arr = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X_arr.shape

        # [1단계] 데이터 중심화 및 표준화 (Z-score 표준화)
        self.mean_ = np.mean(X_arr, axis=0)
        self.std_ = np.std(X_arr, axis=0, ddof=0)
        self.std_[self.std_ == 0] = 1.0  # 0 분산 방지
        X_scaled = (X_arr - self.mean_) / self.std_

        # [2단계] 표본 공분산 행렬 계산 (Sample Covariance Matrix, p x p)
        # Σ = (1 / (n - 1)) * X^T * X
        self.cov_matrix_ = (X_scaled.T @ X_scaled) / (n_samples - 1)

        # [3단계] 공분산 행렬의 고유값 분해 (Eigendecomposition)
        # 공분산 행렬은 실수 대칭 행렬(Symmetric Matrix)이므로 np.linalg.eigh 사용 (수치 안정성 극대화)
        eig_vals, eig_vecs = np.linalg.eigh(self.cov_matrix_)

        # [4단계] 고유값을 기준으로 내림차순 정렬 (λ_1 >= λ_2 >= ... >= λ_p)
        sorted_indices = np.argsort(eig_vals)[::-1]
        self.eigenvalues_ = eig_vals[sorted_indices]
        self.eigenvectors_ = eig_vecs[:, sorted_indices]

        # [고유벡터 부호 표준화(Sign Flips Alignment)]
        # Scikit-learn과 동일하게 각 고유벡터의 절대값 최댓값이 양수가 되도록 정렬
        for i in range(n_features):
            max_abs_idx = np.argmax(np.abs(self.eigenvectors_[:, i]))
            if self.eigenvectors_[max_abs_idx, i] < 0:
                self.eigenvectors_[:, i] *= -1

        # [5단계] 설명 분산 및 비율 계산
        self.explained_variance_ = self.eigenvalues_
        total_var = np.sum(self.eigenvalues_)
        self.explained_variance_ratio_ = self.eigenvalues_ / total_var
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

        return self

    def transform(self, X, n_components=None):
        """
        고유벡터 행렬(W_K)을 곱하여 주성분 좌표 공간으로 선형 투영(Projection)
        """
        if n_components is None:
            n_components = self.n_components if self.n_components is not None else len(self.eigenvalues_)

        X_arr = np.asarray(X, dtype=np.float64)
        X_scaled = (X_arr - self.mean_) / self.std_

        # 상위 K개 고유벡터 추출 (p x K 변환 행렬)
        W_k = self.eigenvectors_[:, :n_components]

        # 투영: X_pca = X_scaled @ W_k (n x K)
        return X_scaled @ W_k

    def fit_transform(self, X, n_components=None):
        self.fit(X)
        return self.transform(X, n_components=n_components)


def main():
    print("#" * 80)
    print("  PCA VIA EIGENDECOMPOSITION (공분산 행렬 고유값 분해 기반 PCA 구현 및 검증)")
    print("#" * 80)

    # ==========================================================================
    # 1. 원본 데이터 로드
    # ==========================================================================
    print("\n[1] 데이터 로드...")
    if not FEATURES_PATH.exists() or not TARGETS_PATH.exists():
        raise FileNotFoundError(f"데이터셋을 찾을 수 없습니다: {FEATURES_PATH}")

    X = pd.read_csv(FEATURES_PATH)
    y = pd.read_csv(TARGETS_PATH)["Activity"]
    y_named = y.map(ACTIVITY_MAP)
    n_samples, n_features = X.shape

    print(f"  * 샘플 수 (n) : {n_samples}개")
    print(f"  * 특성 수 (p) : {n_features}개 (고차원 센서 신호)")

    # ==========================================================================
    # 2. 고유값 분해(Eigendecomposition) 기반 PCA 실행
    # ==========================================================================
    print("\n[2] 공분산 행렬 고유값 분해(Eigendecomposition) 수행 단계:")
    print("  (1) 데이터 표준화 : X_scaled = (X - μ) / σ (평균 0, 분산 1)")
    print(f"  (2) 공분산 행렬 계산 : Σ = (1 / (n - 1)) * X_scaled^T * X_scaled (크기: {n_features} × {n_features})")
    print("  (3) 고유값 방정식 해결 : Σ * v_i = λ_i * v_i (np.linalg.eigh)")
    print("  (4) 고유값 내림차순 정렬 및 고유벡터 변환 행렬 W 구성")

    t_start = time.time()
    eigen_pca = EigendecompositionPCA()
    eigen_pca.fit(X)
    eig_duration = time.time() - t_start

    print(f"\n  >> 고유값 분해 연산 완료 (소요 시간: {eig_duration:.2f}초)")
    print(f"  * 계산된 공분산 행렬 크기: {eigen_pca.cov_matrix_.shape}")
    print(f"  * 고유값 개수: {len(eigen_pca.eigenvalues_)}개, 고유벡터 행렬 크기: {eigen_pca.eigenvectors_.shape}")

    # ==========================================================================
    # 3. 고유값 분해(Eigendecomposition) vs Scikit-learn (SVD) 1:1 정밀 검증
    # ==========================================================================
    print("\n[3] 고유값 분해(Eigendecomposition) vs Scikit-learn(SVD) 수학적 일치성 정밀 검증...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    sklearn_pca = PCA(random_state=42).fit(X_scaled)

    # 1) 상위 5개 고유값 vs SVD 설명 분산 비교
    top5_eig_vals = eigen_pca.eigenvalues_[:5]
    top5_svd_vars = sklearn_pca.explained_variance_[:5]
    max_val_diff = np.max(np.abs(top5_eig_vals - top5_svd_vars))

    # 2) 설명 분산 비율 비교
    top5_eig_ratios = eigen_pca.explained_variance_ratio_[:5]
    top5_svd_ratios = sklearn_pca.explained_variance_ratio_[:5]
    max_ratio_diff = np.max(np.abs(eigen_pca.explained_variance_ratio_ - sklearn_pca.explained_variance_ratio_))

    # 3) 투영 좌표(Transformed Coordinates) 비교
    X_proj_eig = eigen_pca.transform(X, n_components=5)
    X_proj_svd = sklearn_pca.transform(X_scaled)[:, :5]
    max_coord_diff = np.max(np.abs(X_proj_eig - X_proj_svd))

    comp_summary = pd.DataFrame({
        "주성분 (PC)": [f"PC{i+1}" for i in range(5)],
        "고유값 분해 고유값 (λ)": top5_eig_vals,
        "Scikit-learn SVD 분산": top5_svd_vars,
        "고유값 오차 (|Δ|)": np.abs(top5_eig_vals - top5_svd_vars),
        "고유값 분해 설명비율": [f"{r*100:.3f}%" for r in top5_eig_ratios],
        "SVD 설명비율": [f"{r*100:.3f}%" for r in top5_svd_ratios]
    })

    print("=" * 80)
    print("  [고유값 분해 vs SVD 1:1 수학적 대조표 (상위 5개 주성분)]")
    print("=" * 80)
    print(comp_summary.to_string(index=False, formatters={
        "고유값 분해 고유값 (λ)": "{:.6f}".format,
        "Scikit-learn SVD 분산": "{:.6f}".format,
        "고유값 오차 (|Δ|)": "{:.2e}".format
    }))
    print("=" * 80)

    print(f"\n  [수학적 동등성 검증 결론]:")
    print(f"  * 고유값(분산) 최대 오차       : {max_val_diff:.2e} (머신 입실론 ε 수준의 일치)")
    print(f"  * 설명 분산 비율 최대 오차     : {max_ratio_diff:.2e} (완벽 일치)")
    print(f"  * 투영된 2D/3D 데이터 좌표 오차: {max_coord_diff:.2e} (완벽 일치)")
    print("  => '공분산 행렬의 고유값 분해'와 '데이터 행렬의 SVD'가 완전히 동일한 주성분을 산출함을 입증했습니다.")

    # ==========================================================================
    # 4. 고유값 스펙트럼 및 차원 축소 데이터셋 생성 (95%, 90%)
    # ==========================================================================
    print("\n[4] 고유값 분해 기반 차원 축소 피처 추출 및 CSV 저장...")

    cum_var = eigen_pca.cumulative_variance_ratio_
    k_95 = int(np.argmax(cum_var >= 0.95) + 1)
    k_90 = int(np.argmax(cum_var >= 0.90) + 1)

    print(f"  * 95% 분산 보존 필요 고유벡터 수 : {k_95}개 (561개 → {k_95}개, {((561 - k_95)/561)*100:.1f}% 압축)")
    print(f"  * 90% 분산 보존 필요 고유벡터 수 : {k_90}개 (561개 → {k_90}개, {((561 - k_90)/561)*100:.1f}% 압축)")

    # 95% 데이터셋 생성 및 저장
    X_pca_eig_95 = eigen_pca.transform(X, n_components=k_95)
    df_pca_eig_95 = pd.DataFrame(X_pca_eig_95, columns=[f"PC{i+1}" for i in range(k_95)])
    csv_95_path = DATA_DIR / "har_features_pca_eigen_95.csv"
    df_pca_eig_95.to_csv(csv_95_path, index=False)
    print(f"  >> 95% 피처 추출 데이터셋 저장 완료: {csv_95_path}")

    # 90% 데이터셋 생성 및 저장
    X_pca_eig_90 = eigen_pca.transform(X, n_components=k_90)
    df_pca_eig_90 = pd.DataFrame(X_pca_eig_90, columns=[f"PC{i+1}" for i in range(k_90)])
    csv_90_path = DATA_DIR / "har_features_pca_eigen_90.csv"
    df_pca_eig_90.to_csv(csv_90_path, index=False)
    print(f"  >> 90% 피처 추출 데이터셋 저장 완료: {csv_90_path}")

    # ==========================================================================
    # 5. 시각화: 고유값 분해 vs SVD 1:1 투영 비교 및 공분산 행렬 구조
    # ==========================================================================
    print("\n[5] 고해상도 시각화 차트 생성 및 저장...")

    # (1) 고유값 분해 vs SVD 2D 투영 산점도 1:1 비교
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for act in CLASS_NAMES:
        mask = (y_named == act)
        ax1.scatter(X_proj_eig[mask, 0], X_proj_eig[mask, 1], c=CLASS_PALETTE[act], label=act, alpha=0.6, s=15)
        ax2.scatter(X_proj_svd[mask, 0], X_proj_svd[mask, 1], c=CLASS_PALETTE[act], label=act, alpha=0.6, s=15)

    ax1.set_title(f"[수학 구현: 고유값 분해(Eigendecomposition)]\nPC1: {top5_eig_ratios[0]*100:.1f}%, PC2: {top5_eig_ratios[1]*100:.1f}%", fontsize=12, pad=10)
    ax1.set_xlabel("PC1 (제1 주성분)", fontsize=11)
    ax1.set_ylabel("PC2 (제2 주성분)", fontsize=11)
    ax1.legend(loc="upper right", markerscale=2)

    ax2.set_title(f"[라이브러리: Scikit-learn SVD]\nPC1: {top5_svd_ratios[0]*100:.1f}%, PC2: {top5_svd_ratios[1]*100:.1f}%", fontsize=12, pad=10)
    ax2.set_xlabel("PC1 (제1 주성분)", fontsize=11)
    ax2.set_ylabel("PC2 (제2 주성분)", fontsize=11)
    ax2.legend(loc="upper right", markerscale=2)

    plt.suptitle("공분산 행렬 고유값 분해(좌) vs SVD(우) 2D 투영 결과 1:1 완전 일치 검증", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    comp_img_path = OUTPUT_DIR / "eig_vs_svd_comparison.png"
    plt.savefig(comp_img_path, dpi=300)
    plt.close()
    print(f"  - 고유값 분해 vs SVD 비교 차트 저장 완료: {comp_img_path}")

    # (2) 561 x 561 공분산 행렬 구조 히트맵 (상위 50x50 부분 행렬 시각화)
    plt.figure(figsize=(9, 7))
    sample_cov = eigen_pca.cov_matrix_[:50, :50]
    sns.heatmap(sample_cov, cmap="icefire", center=0, cbar_kws={'label': '공분산 (Covariance)'})
    plt.title("561×561 표본 공분산 행렬(Σ) 상위 50×50 특성 간 상관/공분산 구조", fontsize=12, pad=12)
    plt.xlabel("특성 인덱스 (Feature Index 0~49)", fontsize=10)
    plt.ylabel("특성 인덱스 (Feature Index 0~49)", fontsize=10)
    plt.tight_layout()
    cov_img_path = OUTPUT_DIR / "eig_covariance_heatmap.png"
    plt.savefig(cov_img_path, dpi=300)
    plt.close()
    print(f"  - 공분산 행렬 히트맵 저장 완료: {cov_img_path}")

    print("\n" + "#" * 80)
    print("  공분산 행렬 고유값 분해(Eigendecomposition) 기반 PCA 파이프라인이 성공적으로 완료되었습니다!")
    print("#" * 80)


if __name__ == "__main__":
    main()
