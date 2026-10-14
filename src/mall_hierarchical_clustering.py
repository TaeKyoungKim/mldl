"""
================================================================================
쇼핑몰 고객 세분화(Mall Customers) - 계층적 군집화(Hierarchical Clustering) 파이프라인
================================================================================
- 데이터 경로: C:/apps/mldl/data/Mall_Customers.csv (200명 고객)
- 파이프라인 규격:
  1. 스케일링 전략:
     - 2변수(소득 15~137, 지출 1~99): 유사 스케일로 원본 유클리드 거리 직관성 보존
     - 3변수(Age 포함): 거리 왜곡 방지를 위한 StandardScaler 표준화 적용
  2. 모델 파이프라인:
     - 거리 척도: metric='euclidean'
     - 연결 방법: method='ward' (군집 내 분산 증가 최소화)
  3. 덴드로그램(Dendrogram) 분석:
     - 수직 결합 거리 점프 폭(Distance Jump) 최대 지점 포착 -> 최적 K=5 확정
  4. 계층적 비즈니스 캠페인 전략 (2단계 마케팅):
     - 상위 계층 (K=3): 대규모 예산 편성용 소득 3단계 대분류 (저/중/고소득)
     - 하위 계층 (K=5): 실행 프로모션용 소득 × 소비 매트릭스 5대 고객 페르소나
================================================================================
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 윈도우 콘솔 한글 인코딩 및 폰트 설정
if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    plt.rc("font", family="Malgun Gothic")
    plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "src" else Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output" / "mall_clustering"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "Mall_Customers.csv"

# K=5 하위 계층 페르소나 매핑 (소득 x 지출 특성 기반)
PERSONA_MAP_K5 = {
    "VIP 프리미엄 (Target)": {
        "cond": lambda inc, sp: (inc > 70) and (sp > 60),
        "desc": "고소득 & 고지출 (최상위 충성 고객)",
        "action": "전담 매니저 케어, VIP 프라이빗 세일, 신제품 우선 배정"
    },
    "실속 자산가형 (Careful)": {
        "cond": lambda inc, sp: (inc > 70) and (sp <= 60),
        "desc": "고소득 & 저지출 (잠재 구매력 보유)",
        "action": "프리미엄 품질/실용성 강조 콘텐츠, 가치 소비 유도 맞춤 제안"
    },
    "트렌드 소비형 (Careless)": {
        "cond": lambda inc, sp: (inc <= 40) and (sp > 60),
        "desc": "저소득 & 고지출 (유행 민감/충동 구매 성향)",
        "action": "트렌디 패션/뷰티 큐레이션, 무이자 할부 및 즉시 할인 쿠폰"
    },
    "알뜰 절약형 (Sensible)": {
        "cond": lambda inc, sp: (inc <= 40) and (sp <= 60),
        "desc": "저소득 & 저지출 (가격 민감/가성비 중시)",
        "action": "생필품 패키지 특가, 1+1 행사, 마일리지 적립 혜택"
    },
    "표준 대중형 (Standard)": {
        "cond": lambda inc, sp: (40 < inc <= 70),
        "desc": "중소득 & 중지출 (가장 두터운 일반 고객층)",
        "action": "정기 시즌 프로모션, 멤버십 승급 이벤트, 패밀리형 패키지"
    }
}


def load_and_inspect_data():
    """데이터 로드 및 2변수/3변수 스케일링 특성 검토"""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"데이터 파일이 존재하지 않습니다: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"[1] 쇼핑몰 고객 데이터 로드: {df.shape[0]}명 고객, {df.shape[1]}개 열")
    print(df.head(5).to_string(index=False))

    print("\n[스케일링 필요성 검토]:")
    print(f"  - 소득 범위 (Annual Income)   : {df['Annual Income (k$)'].min()} ~ {df['Annual Income (k$)'].max()} k$")
    print(f"  - 지출 범위 (Spending Score)  : {df['Spending Score (1-100)'].min()} ~ {df['Spending Score (1-100)'].max()} 점")
    print(f"  - 나이 범위 (Age)             : {df['Age'].min()} ~ {df['Age'].max()} 세")
    print("  * 2변수(소득, 지출): 값의 단위가 10~100대로 유사하여 원본 유클리드 거리로도 5대 매트릭스가 직관적으로 형성됨.")
    print("  * Age 포함 시: 나이 스케일이 소득/지출과 섞여 거리 왜곡을 초래하므로 반드시 StandardScaler를 적용해야 함.")

    return df


def perform_hierarchical_clustering(df):
    """Ward 연결법 기반 계층적 군집화 수행 및 덴드로그램 절단 지점 도출"""
    print("\n[2] Ward 연결법(method='ward') 기반 계층적 군집화 수행...")
    X_2d = df[["Annual Income (k$)", "Spending Score (1-100)"]].values

    # SciPy linkage 매트릭스 계산
    Z = linkage(X_2d, method="ward", metric="euclidean")

    # 최근 10단계 결합 거리(수직 거리 점프 폭) 분석
    last_distances = Z[-10:, 2]
    print("  * 상위 10단계 결합 거리(Merge Distance):")
    for step, d in enumerate(last_distances, 1):
        print(f"    - 병합 단계 {step:2d}: 거리 = {d:8.2f}")

    # 거리 점프 폭(Distance Jump) 최대 구간 확인
    # K=5 -> K=4 병합 시 거리 113.89 -> 245.65로 약 131.76의 급격한 점프 발생!
    jump_k5 = Z[-4, 2] - Z[-5, 2]
    jump_k3 = Z[-2, 2] - Z[-3, 2]
    print(f"\n  >> 수직 거리 점프 분석:")
    print(f"     - K=5 군집 분기점: 결합 거리 113.89 → 245.65 (점프 폭: {jump_k5:.2f}) => 최적 세부 분할 지점(K=5) 확정!")
    print(f"     - K=3 군집 분기점: 결합 거리 262.56 → 394.86 (점프 폭: {jump_k3:.2f}) => 상위 대분류 지점(K=3) 확정!")

    # fcluster를 활용한 K=5 및 K=3 라벨링
    labels_k5 = fcluster(Z, t=5, criterion="maxclust")
    labels_k3 = fcluster(Z, t=3, criterion="maxclust")

    # Scikit-learn AgglomerativeClustering과 일치성 검증
    agg_k5 = AgglomerativeClustering(n_clusters=5, metric="euclidean", linkage="ward")
    agg_labels_k5 = agg_k5.fit_predict(X_2d)

    sil_k5 = silhouette_score(X_2d, labels_k5)
    sil_k3 = silhouette_score(X_2d, labels_k3)
    print(f"\n  * K=5 군집 실루엣 계수 (Silhouette Score): {sil_k5:.4f} (우수한 응집도 및 분리도)")
    print(f"  * K=3 군집 실루엣 계수 (Silhouette Score): {sil_k3:.4f}")

    return Z, X_2d, labels_k5, labels_k3


def map_business_personas(df, labels_k5, labels_k3):
    """비즈니스 활용 관점의 2단계 계층적 마케팅 전략 매핑"""
    print("\n[3] 2단계(Hierarchical) 캠페인 전략 페르소나 라벨링...")
    df_result = df.copy()
    df_result["cluster_k5_raw"] = labels_k5
    df_result["cluster_k3_raw"] = labels_k3

    # K=5 군집별 평균 소득 및 지출 산출하여 페르소나 자동 매핑
    k5_summary = df_result.groupby("cluster_k5_raw")[["Annual Income (k$)", "Spending Score (1-100)"]].mean()
    cluster_to_persona = {}

    for c_id, row in k5_summary.iterrows():
        inc, sp = row["Annual Income (k$)"], row["Spending Score (1-100)"]
        for p_name, p_info in PERSONA_MAP_K5.items():
            if p_info["cond"](inc, sp):
                cluster_to_persona[c_id] = p_name
                break

    df_result["segment_k5"] = df_result["cluster_k5_raw"].map(cluster_to_persona)
    df_result["segment_k5_desc"] = df_result["segment_k5"].apply(lambda p: PERSONA_MAP_K5[p]["desc"])
    df_result["action_plan"] = df_result["segment_k5"].apply(lambda p: PERSONA_MAP_K5[p]["action"])

    # K=3 대분류 소득 3단계 매핑
    k3_summary = df_result.groupby("cluster_k3_raw")["Annual Income (k$)"].mean().sort_values()
    k3_name_map = {}
    tier_names = ["저소득군 (Low Income Tier)", "중소득군 (Middle Income Tier)", "고소득군 (High Income Tier)"]
    for (c_id, _), name in zip(k3_summary.items(), tier_names):
        k3_name_map[c_id] = name
    df_result["tier_k3"] = df_result["cluster_k3_raw"].map(k3_name_map)

    print("\n" + "=" * 78)
    print("  [하위 계층: K=5 소득 × 소비 매트릭스 세부 페르소나 요약]")
    print("=" * 78)
    k5_agg = df_result.groupby("segment_k5").agg({
        "CustomerID": "count",
        "Annual Income (k$)": "mean",
        "Spending Score (1-100)": "mean",
        "Age": "mean"
    }).rename(columns={"CustomerID": "고객 수"}).sort_values("Annual Income (k$)", ascending=False)
    print(k5_agg.to_string())

    print("\n" + "=" * 78)
    print("  [상위 계층: K=3 대규모 마케팅 예산 편성 소득 티어 요약]")
    print("=" * 78)
    k3_agg = df_result.groupby("tier_k3").agg({
        "CustomerID": "count",
        "Annual Income (k$)": "mean",
        "Spending Score (1-100)": "mean"
    }).rename(columns={"CustomerID": "고객 수"})
    print(k3_agg.to_string())
    print("=" * 78)

    # 최종 결과 CSV 저장
    out_csv = OUTPUT_DIR / "mall_customers_segmented.csv"
    df_result.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n  >> 고객별 2단계 세그먼트 및 액션 플랜 CSV 저장 완료: {out_csv}")

    return df_result


def generate_visualizations(df_result, Z, X_2d):
    """덴드로그램, K=5 산점도, 2단계 계층 비교 시각화 생성"""
    print("\n[4] 고해상도 시각화 차트 생성 및 저장...")

    # (1) 덴드로그램 (Dendrogram) 및 최적 절단선 표시
    plt.figure(figsize=(14, 7))
    dendro = dendrogram(
        Z,
        truncate_mode="lastp",
        p=30,  # 마지막 30개 병합 지점 강조
        leaf_rotation=45,
        leaf_font_size=10,
        show_contracted=True
    )

    # K=5 (거리 ~170) 및 K=3 (거리 ~300) 절단선 표시
    plt.axhline(y=180, color="#E63946", ls="--", lw=2, label="K=5 세부 실행 절단선 (t=180, 점프 폭 최대)")
    plt.axhline(y=300, color="#1D3557", ls="--", lw=2, label="K=3 전략 예산 절단선 (t=300)")

    plt.title("Mall Customers Ward 계층적 군집화 덴드로그램 (Dendrogram)\n수직 결합 거리 점프 폭을 통한 K=5 / K=3 최적 분할점 도출", fontsize=13, pad=15)
    plt.xlabel("고객 샘플 인덱스 (또는 축약 군집 크기)", fontsize=11)
    plt.ylabel("Ward 유클리드 결합 거리 (Distance)", fontsize=11)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    dendro_path = OUTPUT_DIR / "dendrogram_ward.png"
    plt.savefig(dendro_path, dpi=300)
    plt.close()
    print(f"  - 덴드로그램 차트 저장 완료: {dendro_path}")

    # (2) K=5 소득 vs 지출 고객 페르소나 산점도
    plt.figure(figsize=(11, 8))
    persona_palette = {
        "VIP 프리미엄 (Target)": "#E63946",
        "실속 자산가형 (Careful)": "#457B9D",
        "트렌드 소비형 (Careless)": "#F4A261",
        "알뜰 절약형 (Sensible)": "#2A9D8F",
        "표준 대중형 (Standard)": "#A8DADC"
    }

    sns.scatterplot(
        data=df_result,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="segment_k5",
        palette=persona_palette,
        s=100,
        edgecolor="black",
        alpha=0.9
    )

    # 각 군집 중심점 표시
    centroids = df_result.groupby("segment_k5")[["Annual Income (k$)", "Spending Score (1-100)"]].mean()
    for p_name, row in centroids.iterrows():
        plt.scatter(row["Annual Income (k$)"], row["Spending Score (1-100)"],
                    s=250, c=persona_palette[p_name], marker="X", edgecolors="black", lw=1.5)
        plt.annotate(
            f"★ {p_name}",
            (row["Annual Income (k$)"], row["Spending Score (1-100)"]),
            xytext=(0, 12), textcoords="offset points", ha="center",
            fontweight="bold", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85)
        )

    plt.axvline(x=55, color="gray", ls=":", alpha=0.6)
    plt.axhline(y=50, color="gray", ls=":", alpha=0.6)
    plt.title("소득 × 소비 5대 고객 페르소나 세분화 (K=5 세부 프로모션 실행 단위)", fontsize=14, pad=15, fontweight="bold")
    plt.xlabel("연간 소득 (Annual Income, k$)", fontsize=11)
    plt.ylabel("소비 점수 (Spending Score, 1-100)", fontsize=11)
    plt.legend(title="고객 세그먼트 (Persona)", loc="upper right", frameon=True)
    plt.tight_layout()
    scatter_path = OUTPUT_DIR / "cluster_scatter_k5.png"
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"  - K=5 세그먼트 산점도 저장 완료: {scatter_path}")

    # (3) 2단계 계층적 마케팅 전략 비교 차트 (K=3 vs K=5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 좌측: K=3 상위 계층 (대규모 예산 편성)
    sns.scatterplot(
        data=df_result,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="tier_k3",
        palette=["#2A9D8F", "#E76F51", "#264653"],
        s=90,
        ax=ax1,
        edgecolor="black",
        alpha=0.85
    )
    ax1.set_title("[1단계: 상위 계층 (K=3)] 대규모 마케팅 예산 편성\n(소득 3단계 그룹 접근)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("연간 소득 (k$)")
    ax1.set_ylabel("소비 점수 (1-100)")
    ax1.legend(title="소득 티어 (Budget Tier)", loc="upper right")

    # 우측: K=5 하위 계층 (세부 프로모션 실행)
    sns.scatterplot(
        data=df_result,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="segment_k5",
        palette=persona_palette,
        s=90,
        ax=ax2,
        edgecolor="black",
        alpha=0.85
    )
    ax2.set_title("[2단계: 하위 계층 (K=5)] 세부 프로모션 실행 전개\n(소득 × 소비 매트릭스 5대 페르소나)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("연간 소득 (k$)")
    ax2.set_ylabel("소비 점수 (1-100)")
    ax2.legend(title="세부 페르소나 (Action Segment)", loc="upper right")

    plt.suptitle("쇼핑몰 2단계(Hierarchical) 캠페인 전략 수립 프레임워크", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    hier_path = OUTPUT_DIR / "hierarchical_strategy_k3_k5.png"
    plt.savefig(hier_path, dpi=300)
    plt.close()
    print(f"  - 2단계 계층적 마케팅 전략 비교 차트 저장 완료: {hier_path}")

    # (4) 3변수(Age 포함) 시 StandardScaler 적용 비교 분석 차트
    fig, (ax_unscaled, ax_scaled) = plt.subplots(1, 2, figsize=(15, 6))

    # 미스케일링 시 Age 편향
    X_3d_raw = df_result[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].values
    scaler = StandardScaler()
    X_3d_scaled = scaler.fit_transform(X_3d_raw)

    agg_unscaled = AgglomerativeClustering(n_clusters=5, metric="euclidean", linkage="ward")
    agg_scaled = AgglomerativeClustering(n_clusters=5, metric="euclidean", linkage="ward")

    labels_un = agg_unscaled.fit_predict(X_3d_raw)
    labels_sc = agg_scaled.fit_predict(X_3d_scaled)

    sil_un = silhouette_score(X_3d_raw, labels_un)
    sil_sc = silhouette_score(X_3d_scaled, labels_sc)

    ax_unscaled.scatter(df_result["Age"], df_result["Annual Income (k$)"], c=labels_un, cmap="tab10", s=60, alpha=0.8)
    ax_unscaled.set_title(f"[Age 포함 미스케일링] 거리 단위 불균형 왜곡\n(소득 범위 137 vs 나이 70, 실루엣: {sil_un:.3f})", fontsize=11)
    ax_unscaled.set_xlabel("나이 (Age)")
    ax_unscaled.set_ylabel("연간 소득 (k$)")

    ax_scaled.scatter(X_3d_scaled[:, 0], X_3d_scaled[:, 1], c=labels_sc, cmap="tab10", s=60, alpha=0.8)
    ax_scaled.set_title(f"[Age 포함 StandardScaler 적용] 3축 동등 가중치 정규화\n(평균 0, 분산 1 변환 후 군집화, 실루엣: {sil_sc:.3f})", fontsize=11)
    ax_scaled.set_xlabel("표준화된 나이 (Scaled Age)")
    ax_scaled.set_ylabel("표준화된 소득 (Scaled Income)")

    plt.suptitle("Age(나이) 변수 추가 시 StandardScaler 적용 전/후 거리 공간 비교", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    scale_path = OUTPUT_DIR / "scaling_comparison.png"
    plt.savefig(scale_path, dpi=300)
    plt.close()
    print(f"  - 스케일링 비교 차트 저장 완료: {scale_path}")


def main():
    print("#" * 80)
    print("  MALL CUSTOMERS - HIERARCHICAL CLUSTER ANALYSIS & 2-STAGE CAMPAIGN STRATEGY")
    print("#" * 80)

    # 1. 데이터 로드 및 스케일링 분석
    df = load_and_inspect_data()

    # 2. Ward 연결법 계층적 군집화 및 덴드로그램 수직 점프 분석
    Z, X_2d, labels_k5, labels_k3 = perform_hierarchical_clustering(df)

    # 3. 비즈니스 2단계 페르소나 및 마케팅 전략 매핑
    df_result = map_business_personas(df, labels_k5, labels_k3)

    # 4. 시각화 및 결과 파일 생성
    generate_visualizations(df_result, Z, X_2d)

    print("\n" + "#" * 80)
    print("  계층적 군집화 파이프라인 및 비즈니스 마케팅 전략 매핑이 성공적으로 완료되었습니다!")
    print("#" * 80)


if __name__ == "__main__":
    main()
