import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# 한글 폰트 및 마이너스 기호 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 재현성을 위한 시드 고정
np.random.seed(42)
tf.random.set_seed(42)

def main():
    data_path = "C:/apps/mldl/data/telco_churn_preprocessed.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"전처리된 데이터를 찾을 수 없습니다: {data_path}")

    # ============================================================
    # 1. 데이터 로드 및 1D-CNN 입력 형태로 변환
    # ============================================================
    df = pd.read_csv(data_path)
    
    # 특성(X)과 타깃(y: Churn) 분리
    X = df.drop(columns=['Churn']).values
    y = df['Churn'].values
    feature_names = df.drop(columns=['Churn']).columns.tolist()

    # Train / Test 분리 (8:2 비율, 계층적 분할)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1D-CNN 입력 형태: (샘플 수, 시퀀스 길이/특성 수, 채널 수)
    # (N, 19) -> (N, 19, 1)
    X_train_cnn = np.expand_dims(X_train, axis=-1)
    X_test_cnn = np.expand_dims(X_test, axis=-1)

    print(f"훈련 데이터 shape : {X_train_cnn.shape}")
    print(f"테스트 데이터 shape: {X_test_cnn.shape}")

    # ============================================================
    # 2. 1D-CNN 모델 구성 (정형 데이터 이탈 예측용)
    # ============================================================
    # Conv1D -> BatchNorm -> MaxPooling1D -> Conv1D -> GlobalMaxPooling1D -> Dense -> Sigmoid
    model = models.Sequential([
        layers.Input(shape=(X_train_cnn.shape[1], 1), name='input_layer'),
        
        # 첫 번째 Conv1D 블록 (인접한 특성 간 국소 패턴 추출)
        layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', name='conv1d_1'),
        layers.BatchNormalization(name='bn_1'),
        layers.MaxPooling1D(pool_size=2, name='pool1d_1'),

        # 두 번째 Conv1D 블록 (고차원 특징 결합)
        layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', name='conv1d_2'),
        layers.BatchNormalization(name='bn_2'),
        layers.GlobalMaxPooling1D(name='global_pool'),

        # 완전 연결 계층 (분류기)
        layers.Dense(32, activation='relu', name='dense_1'),
        layers.Dropout(0.3, name='dropout'),
        layers.Dense(1, activation='sigmoid', name='output_layer')
    ])

    print("\n[1D-CNN 모델 구조 요약]")
    model.summary()

    # ============================================================
    # 3. 모델 컴파일 및 학습
    # ============================================================
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # 조기 종료 설정
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    print("\n1D-CNN 모델 학습 시작...")
    history = model.fit(
        X_train_cnn, y_train,
        validation_split=0.2,
        epochs=40,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    # 전체 테스트 세트 평가
    test_loss, test_acc = model.evaluate(X_test_cnn, y_test, verbose=0)
    print(f"\n[전체 테스트 데이터셋 성능] Loss: {test_loss:.4f}, Accuracy: {test_acc * 100:.2f}%")

    # ============================================================
    # 4. 테스트 데이터 중 단일 샘플(고객) 예측 및 비교
    # ============================================================
    # 예시: 테스트 세트의 50번째 고객 샘플 선택
    sample_idx = 49
    sample_features = X_test_cnn[sample_idx:sample_idx + 1]  # shape: (1, 19, 1)
    actual_churn = y_test[sample_idx]

    # 예측 수행 (이탈 확률 0.0 ~ 1.0)
    churn_prob = model.predict(sample_features, verbose=0)[0][0]
    predicted_churn = 1 if churn_prob >= 0.5 else 0

    actual_text = "이탈 (Yes)" if actual_churn == 1 else "유지 (No)"
    pred_text = "이탈 (Yes)" if predicted_churn == 1 else "유지 (No)"
    confidence = churn_prob * 100 if predicted_churn == 1 else (1 - churn_prob) * 100

    print("\n" + "=" * 65)
    print(f"       테스트 데이터 [{sample_idx + 1}번째 고객] 이탈 예측 결과")
    print("=" * 65)
    print(f" 실제 정답 (Actual)    : [{actual_churn}] {actual_text}")
    print(f" 예측 결과 (Predicted) : [{predicted_churn}] {pred_text} (신뢰도: {confidence:.2f}%)")
    print(f" 예측된 이탈 확률      : {churn_prob * 100:.2f}%")
    print(f" 일치 여부             : {'정답 (O)' if actual_churn == predicted_churn else '오답 (X)'}")
    print("=" * 65 + "\n")

    # ============================================================
    # 5. 결과 시각화 (예측 확률 게이지 & 고객 주요 특성 값)
    # ============================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # (1) 이탈 확률 게이지 바
    colors = ['#4CAF50', '#F44336']  # 녹색(유지), 빨간색(이탈)
    bar = ax1.barh(['이탈 확률'], [churn_prob * 100], color='#E53935' if churn_prob >= 0.5 else '#43A047', height=0.4)
    ax1.axvline(x=50, color='gray', linestyle='--', linewidth=1.5, label='기준선 (50%)')
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("확률 (%)", fontsize=11)
    ax1.set_title(
        f"고객 이탈 확률 예측: {churn_prob * 100:.1f}%\n"
        f"[실제: {actual_text} | 예측: {pred_text}]",
        fontsize=12, fontweight='bold', pad=12
    )
    ax1.text(churn_prob * 100 + 2, 0, f"{churn_prob * 100:.1f}%", va='center', fontweight='bold', fontsize=11)
    ax1.legend(loc='lower right')
    ax1.grid(axis='x', linestyle='--', alpha=0.5)

    # (2) 해당 고객의 19개 특성값 프로파일 (입력 텐서 시각화)
    raw_sample = X_test[sample_idx]
    ax2.bar(range(len(raw_sample)), raw_sample, color='#1E88E5', alpha=0.8)
    ax2.set_xticks(range(len(feature_names)))
    ax2.set_xticklabels(feature_names, rotation=60, ha='right', fontsize=8.5)
    ax2.set_ylabel("정규화된 특성값 (Z-score)", fontsize=10)
    ax2.set_title("해당 고객의 19개 피처 프로파일", fontsize=12, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
