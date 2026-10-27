"""
MSFT 주가 데이터(C:/apps/mldl/dl/data/MSFT.csv) 기반 TCN(Temporal Convolutional Network) 모델링 파이프라인

[가장 효과적인 TCN 아키텍처: Dilated Causal Residual TCN]
1. Causal Padding: 미래 정보가 과거로 누출(Leakage)되지 않도록 과거 방향으로만 패딩
2. Dilated Convolution (팽창률 d = 1, 2, 4, 8): 지수적으로 수용 영역(Receptive Field)을 넓혀
   파라미터 폭증 없이 과거 장기 시계열 패턴(수십 일간의 추세)을 포착
3. Residual Connection (잔차 연결, Skip Connection): 깊은 층에서도 기울기 소실(Vanishing Gradient) 방지
4. SpatialDropout1D: 시계열 피처 간의 과적합을 방지
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

def build_tcn_residual_block(x, filters, kernel_size, dilation_rate, dropout_rate=0.2):
    """
    TCN의 핵심 단위: Dilated Causal Convolution + Residual Block
    """
    shortcut = x

    # 1. First Dilated Conv Layer (Causal padding: 미래 정보 차단)
    y = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding='causal',
        activation='relu'
    )(x)
    y = layers.BatchNormalization()(y)
    y = layers.SpatialDropout1D(dropout_rate)(y)

    # 2. Second Dilated Conv Layer
    y = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding='causal',
        activation='relu'
    )(y)
    y = layers.BatchNormalization()(y)
    y = layers.SpatialDropout1D(dropout_rate)(y)

    # 3. Residual Connection (채널 수 일치 작업)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters=filters, kernel_size=1, padding='same')(shortcut)

    res = layers.Add()([shortcut, y])
    return layers.Activation('relu')(res)

def create_sliding_windows(X_data, y_data, window_size=30):
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

    feature_cols = [
        'Open', 'High', 'Low', 'Close(t)', 'Volume',
        'MA5', 'MA20', 'MA50', 'RSI', 'MACD',
        'QQQ_Close', 'SnP_Close'
    ]
    target_col = 'Close_forcast'

    # 시계열 순서 보존 분할 (Train 85% / Test 15%)
    train_ratio = 0.85
    train_size = int(len(df) * train_ratio)

    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(train_df[feature_cols])
    y_train_scaled = scaler_y.fit_transform(train_df[[target_col]])

    X_test_scaled = scaler_X.transform(test_df[feature_cols])
    y_test_scaled = scaler_y.transform(test_df[[target_col]])

    WINDOW_SIZE = 30
    X_train, y_train = create_sliding_windows(X_train_scaled, y_train_scaled, WINDOW_SIZE)
    X_test, y_test = create_sliding_windows(X_test_scaled, y_test_scaled, WINDOW_SIZE)
    test_dates = test_df['Date'].iloc[WINDOW_SIZE:].reset_index(drop=True)

    print(f"학습용 3D 텐서 Shape : X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"테스트 3D 텐서 Shape : X_test={X_test.shape}, y_test={y_test.shape}")

    # ============================================================
    # 2. 가장 효과적인 Dilated Residual TCN 모델 구축
    # ============================================================
    print("\n" + "=" * 80)
    print(" 2. Dilated Residual TCN 신경망 모델 구조")
    print("=" * 80)

    inputs = layers.Input(shape=(WINDOW_SIZE, len(feature_cols)), name='input_sequence')

    # Dilation Rate 지수적 증가 (1 -> 2 -> 4 -> 8)를 적용한 4개의 TCN 잔차 블록
    x = inputs
    dilation_rates = [1, 2, 4, 8]
    filters = 64
    kernel_size = 3

    for idx, d_rate in enumerate(dilation_rates):
        x = build_tcn_residual_block(
            x,
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=d_rate,
            dropout_rate=0.2
        )

    # 시계열의 마지막 시점 특징 추출 (Causal 구조이므로 마지막 위치가 전체 과거를 반영함)
    x = layers.Lambda(lambda t: t[:, -1, :], name='last_timestep')(x)
    x = layers.Dense(32, activation='relu', name='dense_buffer')(x)
    outputs = layers.Dense(1, name='output_price')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='MSFT_TCN_Model')
    model.summary()

    # 모델 컴파일
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )

    # ============================================================
    # 3. 모델 학습 진행 (EarlyStopping 적용)
    # ============================================================
    print("\n" + "=" * 80)
    print(" 3. TCN 모델 학습 진행")
    print("=" * 80)

    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=12,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=45,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    # ============================================================
    # 4. 테스트 데이터 예측 및 정량적 평가지표 산출
    # ============================================================
    print("\n" + "=" * 80)
    print(" 4. TCN 테스트 데이터 예측 성능 평가")
    print("=" * 80)

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred_actual = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_test_actual = scaler_y.inverse_transform(y_test).flatten()

    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
    r2 = r2_score(y_test_actual, y_pred_actual)

    print(f"▶ RMSE (평균 제곱근 오차): ${rmse:.2f}")
    print(f"▶ MAE  (평균 절대 오차)  : ${mae:.2f}")
    print(f"▶ MAPE (평균 절대 백분율) : {mape:.2f}%")
    print(f"▶ R² Score (결정 계수)   : {r2:.4f} (1.00에 가까울수록 설명력 높음)")

    # 최근 5영업일 실제 주가 vs 예측 주가 대조 출력
    print("\n[최근 5영업일 실제 주가 vs TCN 예측 주가 비교]")
    for i in range(-5, 0):
        date_str = test_dates.iloc[i].strftime('%Y-%m-%d')
        act = y_test_actual[i]
        prd = y_pred_actual[i]
        diff = prd - act
        print(f"  {date_str} | 실제: ${act:6.2f} -> 예측: ${prd:6.2f} (오차: {diff:+5.2f} USD, 오차율: {abs(diff)/act*100:4.2f}%)")

    # ============================================================
    # 5. 예측 결과 및 오차 시각화
    # ============================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={'height_ratios': [3, 1]})

    # (1) 실제 주가 vs TCN 예측 주가 곡선
    ax1.plot(test_dates, y_test_actual, label='실제 종가 (Actual Close)', color='black', lw=1.5)
    ax1.plot(test_dates, y_pred_actual, label='TCN 예측 종가 (Predicted)', color='#7B1FA2', lw=1.5, linestyle='--')
    ax1.set_title(f"MSFT 테스트 구간 TCN(Temporal Convolutional Network) 주가 예측 결과 (R² = {r2:.4f}, MAPE = {mape:.2f}%)",
                  fontsize=13, fontweight='bold')
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
