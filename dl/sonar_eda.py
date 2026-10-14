import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 및 마이너스 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def run_sonar_eda(file_path):
    print("=" * 80)
    print(" 1. 데이터 로드 및 기본 정보 확인")
    print("=" * 80)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    # 데이터 로드
    df = pd.read_csv(file_path)
    
    print(f"데이터 형태 (Rows, Columns): {df.shape}")
    print("\n[상위 5개 샘플 미리보기]")
    print(df.head())
    
    print("\n[데이터 컬럼 목록 및 데이터 타입(info)]")
    df.info()

    # ============================================================
    # 2. 결측치 및 중복값 점검
    # ============================================================
    print("\n" + "=" * 80)
    print(" 2. 결측치 및 중복값 점검")
    print("=" * 80)
    null_counts = df.isnull().sum()
    total_null = null_counts.sum()
    print(f"총 결측치(NaN) 개수: {total_null}개")
    if total_null > 0:
        print(null_counts[null_counts > 0])
        
    duplicate_rows = df.duplicated().sum()
    print(f"중복된 행 개수: {duplicate_rows}개")

    # ============================================================
    # 3. 타깃 레이블(Label) 분포 확인
    # ============================================================
    print("\n" + "=" * 80)
    print(" 3. 타깃 변수(Label: R vs M) 분포 확인")
    print("=" * 80)
    label_counts = df['Label'].value_counts()
    label_ratio = df['Label'].value_counts(normalize=True) * 100
    
    label_summary = pd.DataFrame({'Count': label_counts, 'Ratio(%)': label_ratio})
    print(label_summary)
    
    # ============================================================
    # 4. 수치형 피처 요약 통계량 (기술통계)
    # ============================================================
    print("\n" + "=" * 80)
    print(" 4. 피처 요약 통계량 (기술통계량 기초)")
    print("=" * 80)
    feature_cols = [col for col in df.columns if col != 'Label']
    print(df[feature_cols].describe().T[['mean', 'std', 'min', '50%', 'max']].head(10))
    print(f"... (총 {len(feature_cols)}개 주파수 대역 피처)")

    # ============================================================
    # 5. 필수 시각화 분석
    # ============================================================
    fig = plt.figure(figsize=(16, 12))
    
    # 5-1. 타깃 분포 막대 그래프 & 파이 차트
    plt.subplot(2, 2, 1)
    sns.countplot(x='Label', data=df, palette='Set2')
    plt.title("타깃 클래스 분포 (R: 암석 vs M: 기뢰)", fontsize=13)
    plt.xlabel("Class (R=Rock, M=Mine)")
    plt.ylabel("Count")
    for idx, count in enumerate(label_counts):
        plt.text(idx, count / 2, f"{count} ({count/len(df)*100:.1f}%)", 
                 ha='center', va='center', color='white', fontweight='bold', fontsize=12)

    # 5-2. 클래스별 평균 주파수 시그니처 (R vs M Spectral Profile)
    plt.subplot(2, 2, 2)
    mean_by_label = df.groupby('Label')[feature_cols].mean()
    freq_indices = range(1, len(feature_cols) + 1)
    plt.plot(freq_indices, mean_by_label.loc['R'], label='Rock (R)', color='royalblue', lw=2)
    plt.plot(freq_indices, mean_by_label.loc['M'], label='Mine (M)', color='crimson', lw=2)
    plt.title("클래스별 평균 주파수 반사 강도 (Spectral Signature)", fontsize=13)
    plt.xlabel("Frequency Band Index (1~60)")
    plt.ylabel("Mean Energy")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # 5-3. 주파수 피처 간 상관관계 히트맵 (일부 대표 대역 샘플링: 5개 간격)
    plt.subplot(2, 2, 3)
    sampled_cols = feature_cols[::5] # 1, 6, 11, ... 56 대역
    corr_matrix = df[sampled_cols].corr()
    sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, annot=False, cbar=True)
    plt.title("주파수 대역 간 상관관계 히트맵 (대표 대역 샘플)", fontsize=13)

    # 5-4. 주요 차이를 보이는 대역의 박스플롯 (이상치 및 분포 비교)
    plt.subplot(2, 2, 4)
    # 클래스 간 평균 차이가 큰 주요 피처 선정
    mean_diff = (mean_by_label.loc['M'] - mean_by_label.loc['R']).abs().sort_values(ascending=False)
    top_diff_features = mean_diff.head(4).index.tolist()
    
    df_melted = df.melt(id_vars='Label', value_vars=top_diff_features, var_name='Feature', value_name='Energy')
    sns.boxplot(x='Feature', y='Energy', hue='Label', data=df_melted, palette='Set2')
    plt.title(f"클래스 간 차이가 가장 큰 상위 4개 대역 박스플롯", fontsize=13)
    plt.xlabel("Top Differentiating Bands")
    plt.ylabel("Energy")

    plt.tight_layout()
    plt.show()
    print("\nEDA 시각화 완료!")

if __name__ == "__main__":
    file_path = "C:/apps/mldl/dl/data/sonar.all-data-uci.csv"
    run_sonar_eda(file_path)
