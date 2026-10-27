"""
MSFT 주가 데이터 기반 CNN-LSTM 주가 예측 파이프라인

[아키텍처 및 데이터 엔지니어링 전략]
1. 시계열 슬라이딩 윈도우 (Sliding Window):
   - 과거 30일(WINDOW_SIZE=30)의 멀티 피처를 묶어 3D 텐서 (Samples, 30, Features) 생성
2. 피처 엔지니어링 및 데이터 누수 방지 스케일링:
   - 핵심 피처: 가격(OHLC), 거래량, 기술적 지표(RSI, MACD, MA20), 시장 지수(QQQ, SnP)
   - 미래 데이터 누수(Data Leakage)를 방지하기 위해 시간 순 분할 후 Train 데이터로만 MinMaxScaler fit
3. CNN-LSTM 하이브리드 모델:
   - Conv1D + MaxPool1D : 피처 간의 국소적 상호작용 및 단기 패턴 추출
   - LSTM               : 윈도우 시퀀스에 걸친 장단기 시계열 종속성 학습
   - Dense              : 최종 다음 날 종가(Close_forcast) 회귀 예측
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# 한글 폰트 및 마이너스 기호 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 재현성을 위한 시드 고정
np.random.seed(42)
tf.random.set_seed(42)

def create_sliding_windows(X_data, y_data, window_size=30):
    """
    2차원 배열 데이터를 입력받아 (Samples, Window_Size, Features) 형태의 3D 시계열 텐서로 변환
    """
    X_windows, y_targets = [], []
    for i in range(len(X_data) - window_size):
        X_windows.append(X_data[i : i + window_size])
        y_targets.append(y_data[i + window_size])
    return np.array(X_windows), np.array(y_targets)

def main():
    file_path = "C:/apps/mldl/dl/data/MSFT.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"데이터 파일이 없습니다: {file_path}")

    print("=" * 80)
    print(" 1. 시계열 데이터 로드 및 피처 엔지니어링")
    print("=" * 80)

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # 모델링에 사용할 핵심 멀티채널 피처 선정 (가격 + 기술지표 + 거시지수)
    feature_cols = [
        'Open', 'High', 'Low', 'Close(t)', 'Volume',
        'MA5', 'MA20', 'MA50', 'RSI', 'MACD',
        'QQQ_Close', 'SnP_Close'
    ]
    target_col = 'Close_forcast'  # 예측 목표: 다음 날 종가

    print(f"총 샘플 수   : {len(df):,} 거래일")
    print(f"사용 피처 수 : {len(feature_cols)}개 {feature_cols}")
    print(f"타깃 변수   : {target_col}")

    # ============================================================
    # 2. 시계열 Train / Test 분할 및 스케일링 (Data Leakage 방지)
    # ============================================================
    print("\n" + "=" * 80)
    print(" 2. 시계열 순차 분할 (Train 85% / Test 15%) 및 MinMax 정규화")
    print("=" * 80)

    # 시계열 데이터는 랜덤 셔플을 하지 않고 시간 순으로 분할
    train_ratio = 0.85
    train_size = int(len(df) * train_ratio)

    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    # 입력 특성(X)과 타깃(y)을 분리하여 스케일러 적용 (역변환을 위해 y 전용 스케일러 유지)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(train_df[feature_cols])
    y_train_scaled = scaler_y.fit_transform(train_df[[target_col]])

    X_test_scaled = scaler_X.transform(test_df[feature_cols])
    y_test_scaled = scaler_y.transform(test_df[[target_col]])

    # 30일 시계열 윈도우 생성
    WINDOW_SIZE = 30
    X_train, y_train = create_sliding_windows(X_train_scaled, y_train_scaled, WINDOW_SIZE)
    X_test, y_test = create_sliding_windows(X_test_scaled, y_test_scaled, WINDOW_SIZE)

    # 테스트 구간 날짜 (시각화용)
    test_dates = test_df['Date'].iloc[WINDOW_SIZE:].reset_index(drop=True)

    print(f"학습용 3D 텐서 Shape : X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"테스트 3D 텐서 Shape : X_test={X_test.shape}, y_test={y_test.shape}")

    # ============================================================
    # 3. CNN-LSTM 하이브리드 딥러닝 모델 구축
    # ============================================================
    print("\n" + "=" * 80)
    print(" 3. CNN-LSTM 하이브리드 신경망 모델 구조")
    print("=" * 80)

    n_timesteps = X_train.shape[1]   # 30
    n_features = X_train.shape[2]    # 12

    model = models.Sequential([
        layers.Input(shape=(n_timesteps, n_features), name='input_sequence'),
        
        # [CNN 계층] 피처 간 국소적 조합 및 단기 변동 패턴 추출
        layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', name='conv1d_1'),
        layers.BatchNormalization(name='bn_1'),
        layers.MaxPooling1D(pool_size=2, name='pool1d_1'),  # 30 시퀀스 -> 15로 압축
        
        # [LSTM 계층] 압축된 시퀀스 피처맵에서 장단기 시계열 추세 및 모멘텀 학습
        layers.LSTM(64, return_sequences=True, name='lstm_1'),
        layers.Dropout(0.2, name='dropout_1'),
        layers.LSTM(32, return_sequences=False, name='lstm_2'),
        layers.Dropout(0.2, name='dropout_2'),

        # [회귀 출력층]
        layers.Dense(32, activation='relu', name='dense_1'),
        layers.Dense(1, name='output_price')
    ])

    model.summary()

    # 모델 컴파일
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )

    # ============================================================
    # 4. 모델 학습 진행 (EarlyStopping 적용)
    # ============================================================
    print("\n" + "=" * 80)
    print(" 4. CNN-LSTM 모델 학습 진행")
    print("=" * 80)

    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=40,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    # ============================================================
    # 5. 테스트 데이터 예측 및 스케일 역변환 (실제 달러 가격 복원)
    # ============================================================
    print("\n" + "=" * 80)
    print(" 5. 테스트 데이터 예측 및 평가 지표 산출")
    print("=" * 80)

    # 모델 예측 (스케일된 값: 0~1)
    y_pred_scaled = model.predict(X_test, verbose=0)

    # 원래 주가 단위(USD)로 역변환 (Inverse Transform)
    y_pred_actual = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_test_actual = scaler_y.inverse_transform(y_test).flatten()

    # 평가 지표 계산
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
    r2 = r2_score(y_test_actual, y_pred_actual)

    print(f"▶ RMSE (평균 제곱근 오차): ${rmse:.2f}")
    print(f"▶ MAE  (평균 절대 오차)  : ${mae:.2f}")
    print(f"▶ MAPE (평균 절대 백분율) : {mape:.2f}%")
    print(f"▶ R² Score (결정 계수)   : {r2:.4f} (1.00에 가까울수록 설명력 높음)")

    # 최근 5일 예측값 vs 실제값 대조 출력
    print("\n[최근 5영업일 실제 주가 vs 예측 주가 비교]")
    for i in range(-5, 0):
        date_str = test_dates.iloc[i].strftime('%Y-%m-%d')
        act = y_test_actual[i]
        prd = y_pred_actual[i]
        diff = prd - act
        print(f"  {date_str} | 실제: ${act:6.2f} -> 예측: ${prd:6.2f} (오차: {diff:+5.2f} USD)")

    # ============================================================
    # 6. 예측 결과 및 오차 시각화
    # ============================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={'height_ratios': [3, 1]})

    # (1) 실제 주가 vs CNN-LSTM 예측 주가 곡선
    ax1.plot(test_dates, y_test_actual, label='실제 종가 (Actual Close)', color='black', lw=1.5)
    ax1.plot(test_dates, y_pred_actual, label='CNN-LSTM 예측 주가 (Predicted)', color='#E53935', lw=1.5, linestyle='--')
    ax1.set_title(f"MSFT 테스트 구간 주가 예측 결과 (R² = {r2:.4f}, MAPE = {mape:.2f}%)", fontsize=13, fontweight='bold')
    ax1.set_ylabel("주가 (USD)", fontsize=11)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # (2) 잔차(Residuals: 예측값 - 실제값) 분포
    residuals = y_pred_actual - y_test_actual
    ax2.plot(test_dates, residuals, color='gray', lw=1, alpha=0.8, label='오차 (Residual = Pred - Actual)')
    ax2.axhline(0, color='red', linestyle='--', lw=1.2)
    ax2.set_title("일별 예측 오차 추이 (USD)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("오차 ($)", fontsize=10)
    ax2.set_xlabel("날짜 (Date)", fontsize=11)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
