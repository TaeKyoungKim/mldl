"""
Bank Marketing 데이터셋 (C:/apps/mldl/data/bank/bank-full.csv)
전문 탐색적 데이터 분석(EDA) 스크립트

[분석 내용]
1. 기본 메타데이터 확인 (행/열 크기, 데이터 타입, 결측치 및 'unknown' 값 파악)
2. 타깃 변수(y: 정기예금 가입 여부 'yes' vs 'no') 클래스 불균형 분석
3. 수치형 변수(age, balance, duration, campaign 등)의 기술통계량 및 분포 확인
4. 주요 범주형 변수(job, education, housing, loan 등)와 가입률(Target y) 간의 관계 분석
5. 시각화 대시보드 (타깃 분포, 수치형 변수 박스플롯, 직업별 가입률, 상관관계 히트맵)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 및 마이너스 기호 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def run_bank_eda(file_path: str):
    print("=" * 85)
    print(" 1. Bank Marketing 데이터 로드 및 기본 구조 확인")
    print("=" * 85)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    # 구분자가 세미콜론(;)으로 되어 있는 CSV 파일 로드
    df = pd.read_csv(file_path, sep=';')

    print(f"▶ 데이터 행 x 열 크기: {df.shape[0]:,}행 x {df.shape[1]}개 열")
    print("\n[상위 5개 샘플 미리보기]")
    print(df.head())

    print("\n[컬럼 목록 및 데이터 타입 (info)]")
    df.info()

    # ============================================================
    # 2. 결측치 및 'unknown' 값 점검
    # ============================================================
    print("\n" + "=" * 85)
    print(" 2. 결측치(Null) 및 'unknown' 범주값 점검")
    print("=" * 85)
    
    null_counts = df.isnull().sum()
    print(f"▶ 공식적인 결측치(NaN/Null) 개수: {null_counts.sum()}개")

    # Bank 데이터셋 특유의 'unknown' 값 비율 확인
    print("\n[주요 범주형 컬럼 내 'unknown' 값 비율]")
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    unknown_summary = []
    for col in cat_cols:
        if (df[col] == 'unknown').any():
            count = (df[col] == 'unknown').sum()
            ratio = (count / len(df)) * 100
            unknown_summary.append({'컬럼': col, 'unknown 건수': count, '비율(%)': f"{ratio:.2f}%"})
    
    if unknown_summary:
        print(pd.DataFrame(unknown_summary).to_string(index=False))

    # ============================================================
    # 3. 타깃 변수(y: 정기예금 가입 여부) 분포 분석
    # ============================================================
    print("\n" + "=" * 85)
    print(" 3. 타깃 변수(y: 정기예금 가입 여부) 불균형 점검")
    print("=" * 85)

    y_counts = df['y'].value_counts()
    y_ratio = df['y'].value_counts(normalize=True) * 100
    y_summary = pd.DataFrame({'건수': y_counts, '비율(%)': y_ratio})
    print(y_summary)
    print(f"▶ 불균형 비율 (No : Yes) = {y_counts['no'] / y_counts['yes']:.2f} : 1")

    # ============================================================
    # 4. 수치형 피처 요약 통계량
    # ============================================================
    print("\n" + "=" * 85)
    print(" 4. 수치형 피처 주요 기술통계량")
    print("=" * 85)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(df[num_cols].describe().T[['mean', 'std', 'min', '50%', 'max']].to_string())

    # ============================================================
    # 5. 종합 EDA 시각화 대시보드 (4-Panel)
    # ============================================================
    fig = plt.figure(figsize=(18, 12))

    # 5-1. 타깃 변수(y) 분포 막대 차트
    ax1 = plt.subplot(2, 2, 1)
    sns.countplot(x='y', data=df, palette=['#E57373', '#81C784'], ax=ax1)
    ax1.set_title("1. 타깃 클래스(y: 가입 여부) 분포", fontsize=12, fontweight='bold')
    ax1.set_xlabel("정기예금 가입 여부 (y)")
    ax1.set_ylabel("고객 수")
    for i, count in enumerate(y_counts[['no', 'yes']]):
        ax1.text(i, count / 2, f"{count:,}명\n({count/len(df)*100:.1f}%)", 
                 ha='center', va='center', color='white', fontweight='bold', fontsize=11)

    # 5-2. 통화 시간(duration)과 가입 여부(y)의 관계 (핵심 피처)
    ax2 = plt.subplot(2, 2, 2)
    sns.boxplot(x='y', y='duration', data=df, palette=['#E57373', '#81C784'], ax=ax2, showfliers=False)
    ax2.set_title("2. 마지막 통화 시간(duration, 초)과 가입 여부의 관계", fontsize=12, fontweight='bold')
    ax2.set_xlabel("정기예금 가입 여부 (y)")
    ax2.set_ylabel("통화 시간 (초, 이상치 제외)")

    # 5-3. 직업군(job)별 가입 성공률(%) 분석
    ax3 = plt.subplot(2, 2, 3)
    job_target = df.groupby('job')['y'].apply(lambda s: (s == 'yes').mean() * 100).sort_values(ascending=False)
    sns.barplot(x=job_target.values, y=job_target.index, palette='Blues_r', ax=ax3)
    ax3.set_title("3. 직업군(job)별 정기예금 가입률 (%)", fontsize=12, fontweight='bold')
    ax3.set_xlabel("가입률 (%)")
    ax3.set_ylabel("직업군")
    for idx, val in enumerate(job_target.values):
        ax3.text(val + 0.3, idx, f"{val:.1f}%", va='center', fontsize=9)

    # 5-4. 수치형 변수 간 상관관계 히트맵
    ax4 = plt.subplot(2, 2, 4)
    # y 변수를 0과 1로 변환하여 수치형 상관계수 계산에 포함
    df_corr = df[num_cols].copy()
    df_corr['y_numeric'] = (df['y'] == 'yes').astype(int)
    corr_matrix = df_corr.corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, ax=ax4, cbar_kws={'label': '상관계수'})
    ax4.set_title("4. 수치형 피처 및 가입 여부(y) 간 상관관계 히트맵", fontsize=12, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)

    plt.suptitle("Bank Marketing 데이터셋 종합 EDA 대시보드", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    # 결과 이미지 저장 디렉터리
    output_dir = "C:/apps/mldl/output/bank_eda"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "bank_eda_summary.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n▶ EDA 대시보드 시각화 차트 저장 완료: {save_path}")
    plt.close()

if __name__ == "__main__":
    file_path = "C:/apps/mldl/data/bank/bank-full.csv"
    run_bank_eda(file_path)
