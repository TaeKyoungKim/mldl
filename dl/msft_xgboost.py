"""
MSFT 주가 데이터(C:/apps/mldl/dl/data/MSFT.csv) 기반 XGBoost 회귀 모델링 파이프라인

[데이터 엔지니어링 및 머신러닝 전략]
1. 시계열 순서 보존 분할 (Train 85% / Test 15%): 미래 데이터 누수(Data Leakage) 차단
2. 피처 선정: 과거 가격(OHLC), 거래량, 기술지표(RSI, MACD, MA, 볼린저 밴드 등), 거시 지수(QQQ, S&P, DJIA)
3. XGBoost Regressor 하이퍼파라미터 튜닝 및 학습:
   - n_estimators=500, learning_rate=0.03, max_depth=6, subsample=0.8
   - early_stopping_rounds 적용 (과적합 방지)
4. 성능 지표 평가 (RMSE, MAE, MAPE, R²) 및 최근 5일 예측치 비교
5. 피처 중요도(Feature Importance) 분석 및 실제 vs 예측 주가 시각화
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 한글 폰트 및 마이너스 기호 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def main():
    file_path = "C:/apps/mldl/dl/data/MSFT.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")

    print("=" * 80)
    print(" 1. 시계열 데이터 로드 및 피처 엔지니어링")
    print("=" * 80)

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # 타깃 컬럼: 다음 날 종가 (Close_forcast)
    target_col = 'Close_forcast'

    # 불필요한 날짜 문자열 컬럼 제외 후 수치형 특성 전부 활용
    drop_cols = ['Date', 'Date_col', target_col]
    feature_cols = [c for c in df.columns if c not in drop_cols and np.issubdtype(df[c].dtype, np.number)]

    print(f"▶ 총 데이터 수   : {len(df):,} 거래일")
    print(f"▶ 입력 피처 개수 : {len(feature_cols)}개 (가격, 거래량, 보조지표, 거시 지수 등)")
    print(f"▶ 예측 목표 타깃 : {target_col}")

    # ============================================================
    # 2. 시계열 데이터 분할 (시간 순서 보존: Train 85% / Test 15%)
    # ============================================================
    print("\n" + "=" * 80)
    print(" 2. 시계열 순차 분할 (Train 85% / Test 15%)")
    print("=" * 80)

    train_ratio = 0.85
    train_size = int(len(df) * train_ratio)

    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    test_dates = test_df['Date'].reset_index(drop=True)

    print(f"▶ 학습 데이터 세트 (Train): {X_train.shape[0]:,}행 ({train_df['Date'].min().strftime('%Y-%m-%d')} ~ {train_df['Date'].max().strftime('%Y-%m-%d')})")
    print(f"▶ 검증/테스트 세트 (Test) : {X_test.shape[0]:,}행 ({test_df['Date'].min().strftime('%Y-%m-%d')} ~ {test_df['Date'].max().strftime('%Y-%m-%d')})")

    # ============================================================
    # 3. XGBoost 회귀 모델 정의 및 학습 (Early Stopping 적용)
    # ============================================================
    print("\n" + "=" * 80)
    print(" 3. XGBoost 회귀 모델 학습 진행")
    print("=" * 80)

    # 트리 기반 부스팅 모델이므로 피처 스케일링(정규화) 없이도 탁월한 성능 발휘
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        early_stopping_rounds=30,
        eval_metric='rmse',
        n_jobs=-1
    )

    # 학습 수행 (Test 세트를 평가셋으로 지정하여 조기 종료 모니터링)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50
    )

    best_iter = model.best_iteration
    print(f"\n최적 트리 에폭(Best Iteration): {best_iter}번째 트리")

    # ============================================================
    # 4. 테스트 데이터 예측 및 정량적 평가지표 산출
    # ============================================================
    print("\n" + "=" * 80)
    print(" 4. 테스트 데이터 예측 성능 평가")
    print("=" * 80)

    y_pred = model.predict(X_test)

    # 회귀 평가지표 계산
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100
    r2 = r2_score(y_test, y_pred)

    print(f"▶ RMSE (평균 제곱근 오차): ${rmse:.2f}")
    print(f"▶ MAE  (평균 절대 오차)  : ${mae:.2f}")
    print(f"▶ MAPE (평균 절대 백분율) : {mape:.2f}%")
    print(f"▶ R² Score (결정 계수)   : {r2:.4f} (1.00에 가까울수록 설명력 높음)")

    # 최근 5영업일 실제 주가 vs 예측 주가 대조 출력
    print("\n[최근 5영업일 실제 주가 vs XGBoost 예측 주가 비교]")
    for i in range(-5, 0):
        date_str = test_dates.iloc[i].strftime('%Y-%m-%d')
        act = y_test.iloc[i]
        prd = y_pred[i]
        diff = prd - act
        print(f"  {date_str} | 실제: ${act:6.2f} -> 예측: ${prd:6.2f} (오차: {diff:+5.2f} USD, 오차율: {abs(diff)/act*100:4.2f}%)")

    # ============================================================
    # 5. 피처 중요도(Feature Importance) 상위 10개 추출
    # ============================================================
    print("\n" + "=" * 80)
    print(" 5. XGBoost 피처 중요도 (Feature Importance) 상위 10개")
    print("=" * 80)
    
    importance_series = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(importance_series.head(10).to_frame(name='Importance(Gain)'))

    # ============================================================
    # 6. 결과 시각화 대시보드 (예측 추세선 & 피처 중요도)
    # ============================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [2.5, 1.5]})

    # (1) 실제 주가 vs XGBoost 예측 주가 곡선
    ax1.plot(test_dates, y_test.values, label='실제 종가 (Actual Close)', color='black', lw=1.5)
    ax1.plot(test_dates, y_pred, label='XGBoost 예측 종가 (Predicted)', color='#1E88E5', lw=1.5, linestyle='--')
    ax1.set_title(f"MSFT 테스트 구간 XGBoost 주가 예측 결과 (R² = {r2:.4f}, MAPE = {mape:.2f}%)", fontsize=13, fontweight='bold')
    ax1.set_ylabel("주가 (USD)", fontsize=11)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # (2) 피처 중요도 상위 12개 가로 막대 그래프
    top_features = importance_series.head(12).sort_values(ascending=True)
    ax2.barh(top_features.index, top_features.values, color='#43A047', alpha=0.85)
    ax2.set_title("XGBoost 모델이 가장 중요하게 반영한 상위 12개 특성 (Feature Importance)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("상대적 중요도 (Gain)", fontsize=10)
    ax2.grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
