"""
MSFT 주가 데이터 기반 WaveNet 표준 Residual TCN (Temporal Convolutional Network) 아키텍처

[WaveNet TCN 핵심 설계]
1. Dilated Causal Convolution: 
   - 팽창률(d = 1, 2, 4, 8, 16)을 지수적으로 증가시켜 넓은 과거 수용 영역(Receptive Field) 확보
   - 미래 데이터 누수 차단을 위한 'causal' 패딩 적용
2. Gated Activation Unit (게이트 활성화):
   - W_f(Filter: tanh) 와 W_g(Gate: sigmoid)의 원소별 곱 z = tanh(W_f * x) ⊙ σ(W_g * x) 적용
   - WaveNet 원작 논문의 핵심 비선형 게이팅 기법
3. Dual Path Residual & Skip Connections:
   - Residual Path: 다음 블록으로 전달되는 잔차 연결 (x + conv_1x1(z))
   - Skip Path    : 모든 블록의 출력을 집계하여 최종 분류/회귀 헤드로 전달하는 스킵 연결 (Σ skip)
4. SpatialDropout1D: 시계열 피처맵 전반에 걸친 과적합 방지
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# 한글 폰트 및 마이너스 기호 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 재현성을 위한 시드 고정
np.random.seed(42)
tf.random.set_seed(42)

def wavenet_residual_block(x, filters, kernel_size, dilation_rate, dropout_rate=0.2):
    """
    WaveNet 원작 논문의 표준 Gated Activation + Residual + Skip 블록
    """
    # 1. 팽창 인과 합성곱 (Filter: tanh 활성화)
    conv_filter = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding='causal',
        activation='tanh',
        name=f'filter_d{dilation_rate}'
    )(x)

    # 2. 팽창 인과 합성곱 (Gate: sigmoid 활성화)
    conv_gate = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding='causal',
        activation='sigmoid',
        name=f'gate_d{dilation_rate}'
    )(x)

    # 3. Gated Activation: tanh(Filter) ⊙ sigmoid(Gate)
    gated = layers.Multiply()([conv_filter, conv_gate])
    gated = layers.SpatialDropout1D(dropout_rate)(gated)

    # 4. 1x1 Conv를 이용한 차원 투영 (Residual & Skip 분기)
    res_out = layers.Conv1D(filters=filters, kernel_size=1, padding='same')(gated)
    skip_out = layers.Conv1D(filters=filters, kernel_size=1, padding='same')(gated)

    # 5. Residual 연결: 입력 x와 더하여 다음 블록으로 보냄
    # 입력과 채널 수가 다르면 맞춰줌
    if x.shape[-1] != filters:
        x = layers.Conv1D(filters=filters, kernel_size=1, padding='same')(x)

    residual_output = layers.Add()([x, res_out])
    return residual_output, skip_out

def create_sliding_windows(X_data, y_data, window_size=60):
    X_windows, y_targets = [], []
    for i in range(len(X_data) - window_size):
        X_windows.append(X_data[i : i + window_size])
        y_targets.append(y_data[i + window_size])
    return np.array(X_windows), np.array(y_targets)

def main():
    file_path = "C:/apps/mldl/dl/data/MSFT.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"데이터 파일이 없습니다: {file_path}")

    print("=" * 85)
    print(" 1. 시계열 데이터 로드 및 피처 엔지니어링")
    print("=" * 85)

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # 멀티채널 피처 선정 (가격, 거래량, 보조지표, 거시 지수)
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

    # 과거 60일의 윈도우 생성 (WaveNet의 넓은 수용 영역에 맞춤)
    WINDOW_SIZE = 60
    X_train, y_train = create_sliding_windows(X_train_scaled, y_train_scaled, WINDOW_SIZE)
    X_test, y_test = create_sliding_windows(X_test_scaled, y_test_scaled, WINDOW_SIZE)
    test_dates = test_df['Date'].iloc[WINDOW_SIZE:].reset_index(drop=True)

    print(f"▶ 학습용 3D 텐서 Shape : X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"▶ 테스트 3D 텐서 Shape : X_test={X_test.shape}, y_test={y_test.shape}")

    # ============================================================
    # 2. WaveNet 표준 Residual TCN 아키텍처 구축
    # ============================================================
    print("\n" + "=" * 85)
    print(" 2. WaveNet 표준 Residual TCN 아키텍처 구축 (Gated + Skip-Accumulation)")
    print("=" * 85)

    inputs = layers.Input(shape=(WINDOW_SIZE, len(feature_cols)), name='input_sequence')

    # 초기 채널 맞춤용 1x1 Conv
    filters = 64
    x = layers.Conv1D(filters=filters, kernel_size=1, padding='same', name='initial_conv')(inputs)

    # 지수적 팽창률 d = [1, 2, 4, 8, 16] (5단계 확장)
    dilation_rates = [1, 2, 4, 8, 16]
    kernel_size = 3
    skip_connections = []

    for d_rate in dilation_rates:
        x, skip = wavenet_residual_block(
            x,
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=d_rate,
            dropout_rate=0.2
        )
        skip_connections.append(skip)

    # 3. WaveNet의 핵심: 모든 블록의 Skip Connections 합산 (Feature Aggregation)
    total_skip = layers.Add(name='accumulate_skips')(skip_connections)
    total_skip = layers.Activation('relu')(total_skip)

    # 4. 마지막 시점(t=-1) 추출 (Causal Convolution으로 과거 전체 정보가 집약됨)
    last_step = layers.Lambda(lambda t: t[:, -1, :], name='extract_last_step')(total_skip)

    # 5. 최종 예측 헤드 (Dense 계층들)
    dense_1 = layers.Dense(64, activation='relu', name='head_dense_1')(last_step)
    dense_1 = layers.Dropout(0.2)(dense_1)
    dense_2 = layers.Dense(32, activation='relu', name='head_dense_2')(dense_1)
    outputs = layers.Dense(1, name='output_price')(dense_2)

    model = models.Model(inputs=inputs, outputs=outputs, name='WaveNet_Residual_TCN')
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
    print("\n" + "=" * 85)
    print(" 3. WaveNet TCN 모델 학습 진행")
    print("=" * 85)

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
    print("\n" + "=" * 85)
    print(" 4. WaveNet TCN 테스트 데이터 예측 성능 평가")
    print("=" * 85)

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
    print("\n[최근 5영업일 실제 주가 vs WaveNet TCN 예측 주가 비교]")
    for i in range(-5, 0):
        date_str = test_dates.iloc[i].strftime('%Y-%m-%d')
        act = y_test_actual[i]
        prd = y_pred_actual[i]
        diff = prd - act
        print(f"  {date_str} | 실제: ${act:6.2f} -> 예측: ${prd:6.2f} (오차: {diff:+5.2f} USD, 오차율: {abs(diff)/act*100:4.2f}%)")

    # ============================================================
    # 5. 예측 결과 및 오차 시각화 (대시보드)
    # ============================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={'height_ratios': [3, 1]})

    # (1) 실제 주가 vs WaveNet TCN 예측 주가 곡선
    ax1.plot(test_dates, y_test_actual, label='실제 종가 (Actual Close)', color='black', lw=1.5)
    ax1.plot(test_dates, y_pred_actual, label='WaveNet TCN 예측 종가 (Predicted)', color='#D81B60', lw=1.5, linestyle='--')
    ax1.set_title(f"MSFT 테스트 구간 WaveNet 표준 Residual TCN 주가 예측 결과 (R² = {r2:.4f}, MAPE = {mape:.2f}%)",
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
