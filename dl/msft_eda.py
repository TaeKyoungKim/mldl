"""
MSFT 주가 데이터(C:/apps/mldl/dl/data/MSFT.csv) 전문 시계열 EDA 스크립트

[분석 전문가 관점의 EDA 목표]
1. 시계열 연속성, 결측치 및 이상치 점검
2. 목표 변수(Close(t), Close_forcast)의 추세(Trend) 및 변동성(Volatility) 확인
3. 이동평균선(MA5, 20, 50, 200) 및 볼린저 밴드 분석
4. 보조지표(RSI, MACD) 및 거래량(Volume) 상관관계 분석
5. CNN-LSTM 모델링을 위한 입력 특성(Feature) 선정 및 시퀀스 윈도우 적합성 검토
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 및 마이너스 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def run_msft_eda(file_path: str):
    print("=" * 85)
    print(" [1] 데이터 로드 및 기본 메타데이터 점검")
    print("=" * 85)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    df = pd.read_csv(file_path)
    
    # Date 컬럼 Datetime 파싱 및 정렬
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    print(f"▶ 데이터 행(일수) x 컬럼 수 : {df.shape[0]:,}행 x {df.shape[1]}개 컬럼")
    print(f"▶ 관측 기간 (Start ~ End)  : {df['Date'].min().strftime('%Y-%m-%d')} ~ {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"▶ 총 거래 일수 (Trading Days): {len(df):,} 영업일")

    # ============================================================
    # 2. 결측치 및 데이터 정합성 점검
    # ============================================================
    print("\n" + "=" * 85)
    print(" [2] 결측치(Null) 및 무한값(Inf) 점검")
    print("=" * 85)
    null_counts = df.isnull().sum()
    total_null = null_counts.sum()
    print(f"총 결측치 수: {total_null}개")
    if total_null > 0:
        print("[결측치가 존재하는 컬럼]")
        print(null_counts[null_counts > 0])

    # ============================================================
    # 3. 주요 가격 및 기술적 지표 기술통계량
    # ============================================================
    print("\n" + "=" * 85)
    print(" [3] 핵심 변수(가격, 거래량, 보조지표) 기술통계량")
    print("=" * 85)
    key_cols = ['Open', 'High', 'Low', 'Close(t)', 'Close_forcast', 'Volume', 'RSI', 'MACD', 'ATR']
    print(df[key_cols].describe().T[['mean', 'std', 'min', '50%', 'max']].to_string())

    # ============================================================
    # 4. 시계열 상관관계 분석 (Target: Close_forcast)
    # ============================================================
    print("\n" + "=" * 85)
    print(" [4] 다음 날 예측 대상(Close_forcast)과 상관관계 상위 10개 피처")
    print("=" * 85)
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()['Close_forcast'].drop(labels=['Close_forcast']).sort_values(ascending=False)
    print(correlations.head(10).to_frame(name='상관계수(r)'))

    # ============================================================
    # 5. 심층 시계열 EDA 종합 시각화 (4-Panel 대시보드)
    # ============================================================
    fig = plt.figure(figsize=(18, 12))

    # 5-1. 주가 추세 및 이동평균선/볼린저 밴드
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(df['Date'], df['Close(t)'], label='종가 (Close)', color='black', lw=1.2)
    ax1.plot(df['Date'], df['MA20'], label='MA20', color='blue', alpha=0.7, lw=1)
    ax1.plot(df['Date'], df['MA50'], label='MA50', color='orange', alpha=0.7, lw=1)
    ax1.plot(df['Date'], df['MA200'], label='MA200 (장기)', color='red', alpha=0.7, lw=1)
    ax1.fill_between(df['Date'], df['Lower_Band'], df['Upper_Band'], color='gray', alpha=0.15, label='볼린저 밴드')
    ax1.set_title("1. 마이크로소프트(MSFT) 장기 주가 추세 및 이동평균선", fontsize=12, fontweight='bold')
    ax1.set_ylabel("주가 (USD)")
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 5-2. 일별 거래량(Volume) 및 변동성
    ax2 = plt.subplot(2, 2, 2)
    ax2.bar(df['Date'], df['Volume'] / 1e6, color='#5C6BC0', width=1.5, alpha=0.7)
    ax2.set_title("2. 일별 거래량 (Volume, 단위: 백만 주)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("거래량 (Millions)")
    ax2.grid(True, linestyle='--', alpha=0.5)

    # 5-3. 보조지표: RSI (상대강도지수) 과매수/과매도 분석
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(df['Date'], df['RSI'], color='purple', lw=1)
    ax3.axhline(70, color='red', linestyle='--', alpha=0.8, label='과매수 기준선 (70)')
    ax3.axhline(30, color='blue', linestyle='--', alpha=0.8, label='과매도 기준선 (30)')
    ax3.fill_between(df['Date'], 70, df['RSI'], where=(df['RSI'] >= 70), color='red', alpha=0.3)
    ax3.fill_between(df['Date'], 30, df['RSI'], where=(df['RSI'] <= 30), color='blue', alpha=0.3)
    ax3.set_title("3. 기술적 보조지표 RSI (과매수/과매도 구간)", fontsize=12, fontweight='bold')
    ax3.set_ylabel("RSI 값")
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, linestyle='--', alpha=0.5)

    # 5-4. 주요 가격 & 지수 피처 간 히트맵 (다중공선성 확인)
    ax4 = plt.subplot(2, 2, 4)
    selected_corr_cols = ['Close(t)', 'Open', 'High', 'Low', 'Volume', 'QQQ_Close', 'SnP_Close', 'DJIA_Close', 'RSI', 'Close_forcast']
    sns.heatmap(df[selected_corr_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, ax=ax4, cbar_kws={'label': '상관계수'})
    ax4.set_title("4. 주요 시장 지수 및 가격 피처 간 상관관계 히트맵", fontsize=12, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)

    plt.suptitle("마이크로소프트(MSFT) 주가 시계열 심층 EDA (CNN-LSTM 모델링 준비)", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

    # ============================================================
    # 6. CNN-LSTM 모델링을 위한 데이터 엔지니어링 브리핑
    # ============================================================
    print("\n" + "=" * 85)
    print(" [5] 데이터 엔지니어링 & CNN-LSTM 모델링 관점의 전략 요약")
    print("=" * 85)
    print("1. 시계열 윈도우(Sliding Window) 제안:")
    print("   - 과거 30일~60일(Window Size: 30~60)의 멀티채널 피처를 묶어 3차원 텐서 (Batch, TimeSteps, Features) 생성 필요")
    print("2. CNN-LSTM 역할 분담:")
    print("   - Conv1D 계층: 여러 피처(가격, QQQ/S&P 지수, 거래량, RSI) 간의 국소적 조합 및 단기 패턴 추출")
    print("   - LSTM 계층  : 윈도우 내 시간의 흐름(장단기 시계열 종속성)을 학습하여 내일의 종가(Close_forcast) 예측")
    print("3. 스케일링(Scaling):")
    print("   - 주가가 17달러에서 200달러 이상으로 장기 상승 추세이므로, Train/Val/Test 분할 후 'MinMaxScaler' 또는 'StandardScaler' 적용 필수")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    file_path = "C:/apps/mldl/dl/data/MSFT.csv"
    run_msft_eda(file_path)
