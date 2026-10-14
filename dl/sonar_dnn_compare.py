import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# 한글 및 마이너스 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 재현성을 위한 시드 고정
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data(file_path):
    # 1. 데이터 로드
    df = pd.read_csv(file_path)
    X = df.drop(columns=['Label']).values
    y = df['Label'].values

    # 2. 레이블 인코딩 (M: 1, R: 0)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # 3. 학습용 / 테스트용 데이터 분리 (8:2 비율, 계층적 샘플링)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 4. 피처 스케일링 (표준화)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, encoder.classes_

def build_model(hidden_layer_count, input_dim=60):
    """
    히든층 수(2, 6, 12)에 맞춰 점진적으로 특징을 압축하는 피라미드 구조의 노드 설계.
    층이 깊어질 때(6, 12층) 기울기 소실 및 과적합을 방지하기 위해
    BatchNormalization과 Dropout(0.2)을 함께 적용.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    # 각 히든층 수에 따른 레이어별 노드 구성 정의
    layer_architectures = {
        # 2개 히든층: 기본적인 2단계 특징 추출
        2: [64, 32],

        # 6개 히든층: 6단계에 걸쳐 점진적 압축
        6: [128, 96, 64, 48, 32, 16],

        # 12개 히든층: 고차원 깊은 신경망 구조
        12: [128, 128, 96, 96, 64, 64, 48, 48, 32, 32, 16, 16]
    }

    nodes_list = layer_architectures.get(hidden_layer_count)
    if nodes_list is None:
        raise ValueError(f"지원하지 않는 히든층 수입니다: {hidden_layer_count}")

    for idx, nodes in enumerate(nodes_list):
        model.add(layers.Dense(nodes, activation='relu', name=f'hidden_{idx+1}_{nodes}'))
        # 깊은 모델(6, 12층) 학습 안정화를 위한 정규화 및 드롭아웃
        if hidden_layer_count >= 6:
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(0.2))

    # 최종 출력층 (이진 분류: Sigmoid)
    model.add(layers.Dense(1, activation='sigmoid', name='output'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    file_path = "C:/apps/mldl/dl/data/sonar.all-data-uci.csv"
    X_train, X_test, y_train, y_test, classes = load_and_preprocess_data(file_path)

    hidden_counts = [2, 6, 12]
    histories = {}
    evaluation_results = []

    print("\n" + "=" * 70)
    print(" Sonar 데이터셋 DNN 히든층별(2층, 6층, 12층) 모델 학습 및 성능 평가")
    print("=" * 70)

    for h_count in hidden_counts:
        print(f"\n>>> [모델: 히든층 {h_count}개] 구성 및 학습 시작...")
        model = build_model(h_count, input_dim=X_train.shape[1])
        model.summary()

        # 조기 종료(EarlyStopping) 설정
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=0
        )

        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=16,
            callbacks=[early_stop],
            verbose=0
        )
        histories[f"{h_count} Layers"] = history

        # 테스트 데이터셋을 이용한 최종 평가
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        evaluation_results.append({
            'Hidden Layers': f'{h_count}층',
            'Test Loss': loss,
            'Test Accuracy (%)': accuracy * 100,
            'Trained Epochs': len(history.history['loss'])
        })
        print(f">>> [결과] {h_count}층 모델 - Test Loss: {loss:.4f}, Test Accuracy: {accuracy * 100:.2f}% (학습 종료 에폭: {len(history.history['loss'])})")

    # ============================================================
    # 비교 결과 테이블 출력
    # ============================================================
    results_df = pd.DataFrame(evaluation_results)
    print("\n" + "=" * 70)
    print(" [최종 성능 비교 결과 요약]")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("=" * 70 + "\n")

    # ============================================================
    # 학습 곡선 및 테스트 정확도 비교 시각화
    # ============================================================
    plt.figure(figsize=(16, 5))

    # 1. Validation Accuracy 추이
    plt.subplot(1, 3, 1)
    for name, hist in histories.items():
        plt.plot(hist.history['val_accuracy'], label=f'{name} (Val Acc)')
    plt.title("검증 정확도 (Val Accuracy) 비교", fontsize=12)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # 2. Validation Loss 추이
    plt.subplot(1, 3, 2)
    for name, hist in histories.items():
        plt.plot(hist.history['val_loss'], label=f'{name} (Val Loss)')
    plt.title("검증 손실 (Val Loss) 비교", fontsize=12)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # 3. 테스트 데이터셋 정확도 막대 그래프 비교
    plt.subplot(1, 3, 3)
    bars = plt.bar(
        results_df['Hidden Layers'],
        results_df['Test Accuracy (%)'],
        color=['#4A90E2', '#50E3C2', '#B8E986'],
        edgecolor='gray'
    )
    plt.title("테스트 데이터 최종 정확도 비교", fontsize=12)
    plt.ylabel("Test Accuracy (%)")
    plt.ylim([0, 105])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1.5, f"{yval:.2f}%", ha='center', fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
