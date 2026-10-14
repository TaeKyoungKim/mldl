"""
================================================================================
UCI HAR Dataset - 주성분 분석(PCA, Principal Component Analysis) 기반 피처 추출
================================================================================
- 데이터 경로:
  - 원본 특성 데이터: C:/apps/mldl/data/har_features.csv (10,299개 샘플 × 561개 특성)
  - 타깃 레이블 데이터: C:/apps/mldl/data/har_targets.csv (6개 인간 행동 범주)
- 주요 기능 및 파이프라인:
  1. 고차원 센서 데이터(561개) 로드 및 무결성 진단
  2. 분산 보존을 위한 표준화 스케일링 (StandardScaler)
  3. 전체 주성분 설명 분산 비율(Explained Variance Ratio) 정밀 분석
     - Scree Plot 및 누적 설명 분산 곡선 도출
     - 70%, 80%, 90%, 95%, 99% 분산 보존에 필요한 최적 주성분 수(K) 산출
  4. 최적 피처 추출(Feature Extraction) 및 차원 축소 데이터셋 생성:
     - 95% 분산 보존 데이터셋: 561개 → 104개 (약 81.5% 차원 압축)
     - 90% 분산 보존 데이터셋: 561개 → 65개 (약 88.4% 차원 압축)
     - 정제된 PCA 데이터셋 CSV 저장 (data/har_features_pca_95.csv, 90.csv)
  5. 주성분 해석(Loadings / Eigenvector 분석):
     - PC1 (50.7% 분산 설명): 신체 움직임 에너지(Signal Magnitude Area) 기반 동적 vs 정적 분리
     - PC2, PC3: 자세 각도 및 주파수 대역 성분
  6. 2D / 3D 잠재 공간(Latent Space) 시각화 및 행동 클러스터링 분석
  7. 원본 561개 vs PCA 104개 vs PCA 65개 머신러닝 연산 효율성 & 성능 검증
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
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

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


def main():
    print("#" * 80)
    print("  HUMAN ACTIVITY RECOGNITION (HAR) - PCA FEATURE EXTRACTION PIPELINE")
    print("#" * 80)

    # ==========================================================================
    # 1. 원본 데이터 로드 및 전처리
    # ==========================================================================
    print("\n[1] 데이터 로드 및 표준화 스케일링...")
    if not FEATURES_PATH.exists() or not TARGETS_PATH.exists():
        raise FileNotFoundError(
            f"데이터셋을 찾을 수 없습니다.\n"
            f"- Features: {FEATURES_PATH}\n"
            f"- Targets : {TARGETS_PATH}"
        )

    t0 = time.time()
    X = pd.read_csv(FEATURES_PATH)
    y = pd.read_csv(TARGETS_PATH)["Activity"]
    y_named = y.map(ACTIVITY_MAP)
    print(f"  * 데이터 로드 완료 ({time.time() - t0:.2f}초)")
    print(f"  * 원본 특성(X) 형태: {X.shape[0]}개 샘플, {X.shape[1]}개 고차원 센서 특성")
    print(f"  * 타깃(y) 형태     : {y.shape[0]}개 샘플 (6개 행동 클래스)")

    # PCA 전 표준화(StandardScaler): 모든 변수의 분산 단위를 1로 통일
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("  * StandardScaler 적용 완료: 평균 0, 분산 1 정규화 (PCA 분산 왜곡 방지)")

    # ==========================================================================
    # 2. 전체 PCA 적합 및 설명 분산 비율(Explained Variance Ratio) 분석
    # ==========================================================================
    print("\n[2] 전체 주성분(561개) 분산 보존 비율 분석...")
    t0 = time.time()
    pca_full = PCA(random_state=42)
    pca_full.fit(X_scaled)
    print(f"  * PCA 모델 적합 완료 ({time.time() - t0:.2f}초)")

    var_ratio = pca_full.explained_variance_ratio_
    cum_var_ratio = np.cumsum(var_ratio)

    # 주요 분산 보존 임계점 분석
    thresholds = [0.70, 0.80, 0.85, 0.90, 0.95, 0.99]
    variance_table = []

    for th in thresholds:
        n_comp = int(np.argmax(cum_var_ratio >= th) + 1)
        actual_var = cum_var_ratio[n_comp - 1]
        reduction_rate = ((X.shape[1] - n_comp) / X.shape[1]) * 100
        variance_table.append({
            "보존 목표 분산": f"{int(th * 100)}%",
            "필요 주성분 수 (K)": f"{n_comp}개",
            "실제 누적 설명 분산": f"{actual_var * 100:.2f}%",
            "차원 압축률": f"{reduction_rate:.1f}% 압축"
        })

    var_df = pd.DataFrame(variance_table)
    print("\n" + "=" * 70)
    print("  [PCA 분산 보존 임계값별 필요 주성분 수 및 차원 압축률]")
    print("=" * 70)
    print(var_df.to_string(index=False))
    print("=" * 70)

    print(f"\n  [상위 3대 핵심 주성분 기여도]:")
    print(f"  * PC1 (제1 주성분) : 전체 분산의 {var_ratio[0] * 100:.2f}% 설명 (단일 축이 데이터 정보의 절반 이상 점유!)")
    print(f"  * PC2 (제2 주성분) : 전체 분산의 {var_ratio[1] * 100:.2f}% 설명")
    print(f"  * PC3 (제3 주성분) : 전체 분산의 {var_ratio[2] * 100:.2f}% 설명")
    print(f"  * PC1~PC3 누적 분산: {cum_var_ratio[2] * 100:.2f}% (상위 3개 축만으로 전체 정보의 약 60% 설명)")

    # ==========================================================================
    # 3. 최적 피처 추출(Feature Extraction) 및 데이터셋 저장
    # ==========================================================================
    print("\n[3] 최적 주성분 기반 피처 추출 및 차원 축소 데이터셋 생성...")

    # (1) 정보 손실 최소화 표준: 95% 분산 보존 주성분 (K=104)
    k_95 = int(np.argmax(cum_var_ratio >= 0.95) + 1)
    pca_95 = PCA(n_components=k_95, random_state=42)
    X_pca_95 = pca_95.fit_transform(X_scaled)
    cols_95 = [f"PC{i+1}" for i in range(k_95)]
    df_pca_95 = pd.DataFrame(X_pca_95, columns=cols_95)

    pca_95_path = DATA_DIR / "har_features_pca_95.csv"
    df_pca_95.to_csv(pca_95_path, index=False)
    print(f"  * 95% 분산 보존 데이터셋 추출: 561개 → {k_95}개 특성 (압축률: {((561 - k_95)/561)*100:.1f}%)")
    print(f"    => 저장 완료: {pca_95_path}")

    # (2) 경량화 고효율 표준: 90% 분산 보존 주성분 (K=65)
    k_90 = int(np.argmax(cum_var_ratio >= 0.90) + 1)
    pca_90 = PCA(n_components=k_90, random_state=42)
    X_pca_90 = pca_90.fit_transform(X_scaled)
    cols_90 = [f"PC{i+1}" for i in range(k_90)]
    df_pca_90 = pd.DataFrame(X_pca_90, columns=cols_90)

    pca_90_path = DATA_DIR / "har_features_pca_90.csv"
    df_pca_90.to_csv(pca_90_path, index=False)
    print(f"  * 90% 분산 보존 데이터셋 추출: 561개 → {k_90}개 특성 (압축률: {((561 - k_90)/561)*100:.1f}%)")
    print(f"    => 저장 완료: {pca_90_path}")

    # ==========================================================================
    # 4. 주성분 해석 (Component Loadings / 고유벡터 기여도)
    # ==========================================================================
    print("\n[4] 핵심 주성분(PC1, PC2) 특성 기여도(Loadings) 해석...")
    feature_names = X.columns

    # PC1 특성 기여도
    pc1_weights = pd.Series(pca_full.components_[0], index=feature_names)
    top_pc1 = pc1_weights.abs().sort_values(ascending=False).head(10)
    print("\n  [PC1에 가장 크게 기여하는 핵심 특성 Top 5 (절대 가중치)]:")
    for feat, val in top_pc1.head(5).items():
        print(f"    * {feat:<32}: 가중치 {pc1_weights[feat]:+.4f} (신체 가속 에너지 신호)")

    # PC2 특성 기여도
    pc2_weights = pd.Series(pca_full.components_[1], index=feature_names)
    top_pc2 = pc2_weights.abs().sort_values(ascending=False).head(10)
    print("\n  [PC2에 가장 크게 기여하는 핵심 특성 Top 5 (절대 가중치)]:")
    for feat, val in top_pc2.head(5).items():
        print(f"    * {feat:<32}: 가중치 {pc2_weights[feat]:+.4f} (주파수/중력 자세 신호)")

    # ==========================================================================
    # 5. 차원 축소 시각화 (Scree Plot, 2D/3D Projection, Loadings)
    # ==========================================================================
    print("\n[5] 고해상도 시각화 차트 생성 및 저장...")

    # (1) Scree Plot & 누적 분산 곡선
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(range(1, 151), var_ratio[:150] * 100, color="#E63946", lw=1.8, label="개별 주성분 설명 분산 (%)")
    ax1.set_xlabel("주성분 번호 (Principal Component Number)", fontsize=11)
    ax1.set_ylabel("개별 설명 분산 비율 (%)", color="#E63946", fontsize=11)
    ax1.tick_params(axis='y', labelcolor="#E63946")
    ax1.set_xlim(1, 150)

    ax2 = ax1.twinx()
    ax2.plot(range(1, 151), cum_var_ratio[:150] * 100, color="#1D3557", lw=2.2, label="누적 설명 분산 (%)")
    ax2.set_ylabel("누적 설명 분산 비율 (%)", color="#1D3557", fontsize=11)
    ax2.tick_params(axis='y', labelcolor="#1D3557")
    ax2.set_ylim(40, 102)

    # 주요 기준선 표시
    ax2.axhline(90, color="#2A9D8F", ls="--", lw=1.2, alpha=0.8, label="90% 기준선 (K=65)")
    ax2.axvline(k_90, color="#2A9D8F", ls="--", lw=1.2, alpha=0.8)
    ax2.axhline(95, color="#F4A261", ls="--", lw=1.2, alpha=0.8, label="95% 기준선 (K=104)")
    ax2.axvline(k_95, color="#F4A261", ls="--", lw=1.2, alpha=0.8)

    plt.title("HAR 561개 피처 PCA Scree Plot 및 누적 설명 분산 곡선", fontsize=13, pad=15, fontweight="bold")
    fig.tight_layout()
    scree_path = OUTPUT_DIR / "har_pca_scree_plot.png"
    plt.savefig(scree_path, dpi=300)
    plt.close()
    print(f"  - Scree Plot 차트 저장 완료: {scree_path}")

    # (2) 2D 잠재 공간 투영 산점도 (PC1 vs PC2)
    plt.figure(figsize=(11, 8))
    for act in CLASS_NAMES:
        mask = (y_named == act)
        plt.scatter(
            X_pca_95[mask, 0],
            X_pca_95[mask, 1],
            c=CLASS_PALETTE[act],
            label=act,
            alpha=0.65,
            s=18,
            edgecolors="none"
        )

    plt.axvline(0, color="gray", ls=":", alpha=0.5)
    plt.axhline(0, color="gray", ls=":", alpha=0.5)
    plt.title(f"HAR 데이터셋 2D PCA 투영 (PC1: {var_ratio[0]*100:.1f}%, PC2: {var_ratio[1]*100:.1f}%)\n동적 행동(Dynamic) vs 정적 자세(Static) 군집 분리", fontsize=13, pad=15)
    plt.xlabel(f"제1 주성분 (PC1 - 움직임 활동 에너지 수준, {var_ratio[0]*100:.1f}%)", fontsize=11)
    plt.ylabel(f"제2 주성분 (PC2 - 주파수 및 중력 방향 특성, {var_ratio[1]*100:.1f}%)", fontsize=11)
    plt.legend(title="행동 범주 (Activity)", loc="upper right", frameon=True, markerscale=2)
    plt.tight_layout()
    proj2d_path = OUTPUT_DIR / "har_pca_2d_projection.png"
    plt.savefig(proj2d_path, dpi=300)
    plt.close()
    print(f"  - 2D 주성분 투영 차트 저장 완료: {proj2d_path}")

    # (3) 3D 잠재 공간 투영 산점도 (PC1 vs PC2 vs PC3)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    for act in CLASS_NAMES:
        mask = (y_named == act)
        ax.scatter(
            X_pca_95[mask, 0],
            X_pca_95[mask, 1],
            X_pca_95[mask, 2],
            c=CLASS_PALETTE[act],
            label=act,
            alpha=0.6,
            s=15,
            edgecolors="none"
        )

    ax.set_title(f"HAR 데이터셋 3D PCA 공간 투영 (누적 분산 {cum_var_ratio[2]*100:.1f}%)", fontsize=13, pad=15)
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)", fontsize=10)
    ax.set_zlabel(f"PC3 ({var_ratio[2]*100:.1f}%)", fontsize=10)
    ax.legend(title="행동 범주", loc="upper left", markerscale=2)
    ax.view_init(elev=22, azim=-55)
    plt.tight_layout()
    proj3d_path = OUTPUT_DIR / "har_pca_3d_projection.png"
    plt.savefig(proj3d_path, dpi=300)
    plt.close()
    print(f"  - 3D 주성분 투영 차트 저장 완료: {proj3d_path}")

    # (4) PC1 & PC2 주요 특성 가중치(Loadings) 비교 막대 차트
    fig, (ax_l1, ax_l2) = plt.subplots(1, 2, figsize=(14, 6))

    top10_pc1_df = pd.DataFrame({
        "Feature": top_pc1.index,
        "Loading": pc1_weights[top_pc1.index].values
    })
    sns.barplot(data=top10_pc1_df, y="Feature", x="Loading", hue="Feature", ax=ax_l1, palette="vlag", legend=False)
    ax_l1.set_title("제1 주성분(PC1) 상위 10개 기여 특성 가중치", fontsize=12)
    ax_l1.set_xlabel("고유벡터 계수 (Loading)", fontsize=10)

    top10_pc2_df = pd.DataFrame({
        "Feature": top_pc2.index,
        "Loading": pc2_weights[top_pc2.index].values
    })
    sns.barplot(data=top10_pc2_df, y="Feature", x="Loading", hue="Feature", ax=ax_l2, palette="coolwarm", legend=False)
    ax_l2.set_title("제2 주성분(PC2) 상위 10개 기여 특성 가중치", fontsize=12)
    ax_l2.set_xlabel("고유벡터 계수 (Loading)", fontsize=10)

    plt.tight_layout()
    loadings_path = OUTPUT_DIR / "har_pca_component_loadings.png"
    plt.savefig(loadings_path, dpi=300)
    plt.close()
    print(f"  - 주성분 가중치 차트 저장 완료: {loadings_path}")

    # ==========================================================================
    # 6. 머신러닝 성능 및 연산 효율성 벤치마크 (Validation Benchmark)
    # ==========================================================================
    print("\n[6] 원본 561개 vs PCA 축소 피처 머신러닝 연산 효율성 & 성능 검증...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    experiments = {
        "1. 원본 데이터 (Raw 561개)": (X_train_sc, X_test_sc),
        "2. PCA 95% 보존 (추출 103개)": (
            pca_95.fit_transform(X_train_sc),
            pca_95.transform(X_test_sc)
        ),
        "3. PCA 90% 보존 (추출 65개)": (
            pca_90.fit_transform(X_train_sc),
            pca_90.transform(X_test_sc)
        )
    }

    benchmarks = []
    for exp_name, (X_tr, X_te) in experiments.items():
        t_fit = time.time()
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_tr, y_train)
        fit_duration = time.time() - t_fit

        preds = clf.predict(X_te)
        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro")

        benchmarks.append({
            "특성 구성": exp_name,
            "피처 차원 수": f"{X_tr.shape[1]}개",
            "학습 소요 시간": f"{fit_duration:.2f}초",
            "정확도 (Accuracy)": f"{acc * 100:.2f}%",
            "Macro F1-Score": f"{macro_f1:.4f}"
        })

    bench_df = pd.DataFrame(benchmarks)
    print("\n" + "=" * 78)
    print("  [차원 축소 전후 머신러닝 성능 및 학습 시간 벤치마크]")
    print("=" * 78)
    print(bench_df.to_string(index=False))
    print("=" * 78)

    print("\n" + "#" * 80)
    print("  PCA 피처 추출, 차원 축소 데이터셋 생성 및 시각화가 성공적으로 완료되었습니다!")
    print("#" * 80)


if __name__ == "__main__":
    main()
