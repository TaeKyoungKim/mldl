import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# 한글 및 마이너스 기호 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 재현성을 위한 난수 시드 고정
np.random.seed(42)
tf.random.set_seed(42)

def main():
    print("=" * 70)
    print(" 1. Wine 데이터셋 로드 및 전처리")
    print("=" * 70)

    # 1. Scikit-learn Wine 데이터셋 로드 (178개 샘플, 13개 특성, 3개 클래스)
    wine = load_wine()
    X = wine.data
    y = wine.target
    target_names = wine.target_names
    feature_names = wine.feature_names

    print(f"데이터 크기: {X.shape[0]}개 샘플, {X.shape[1]}개 특성")
    print(f"클래스 종류: {target_names} (총 {len(target_names)}개 클래스)")

    # 2. Train / Test 데이터셋 분리 (8:2 비율, 계층적 분할)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. 데이터 표준화 (Feature Scaling)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ============================================================
    # 2. 다중 분류(Multi-class) DNN 모델 구조 정의
    # ============================================================
    print("\n" + "=" * 70)
    print(" 2. 다중 분류 DNN 모델 구성")
    print("=" * 70)

    input_dim = X_train.shape[1]   # 13개 특성
    num_classes = len(target_names) # 3개 클래스

    model = models.Sequential([
        layers.Input(shape=(input_dim,), name='input_layer'),
        layers.Dense(64, activation='relu', name='hidden_1'),
        layers.BatchNormalization(name='bn_1'),
        layers.Dropout(0.2, name='dropout_1'),
        layers.Dense(32, activation='relu', name='hidden_2'),
        layers.BatchNormalization(name='bn_2'),
        layers.Dropout(0.2, name='dropout_2'),
        layers.Dense(num_classes, activation='softmax', name='output_layer')
    ])

    model.summary()

    # 3. 모델 컴파일 (다중 분류: sparse_categorical_crossentropy)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # ============================================================
    # 3. 조기종료(EarlyStopping) & 모델체크포인트(ModelCheckpoint) 설정
    # ============================================================
    print("\n" + "=" * 70)
    print(" 3. 콜백 함수 설정 (EarlyStopping & ModelCheckpoint)")
    print("=" * 70)

    save_dir = "C:/apps/mldl/dl/saved_models"
    os.makedirs(save_dir, exist_ok=True)

    # 1. 콜백에 넘길 저장용 템플릿 경로 (에폭 및 검증 손실값 포함)
    checkpoint_template = os.path.join(save_dir, "wine_model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.keras")

    # (1) 조기종료: val_loss가 15에폭 동안 개선되지 않으면 조기 학습 중단
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    # (2) 모델체크포인트: 검증 손실(val_loss) 기준 가장 우수한 가중치와 구조를 파일로 저장
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=checkpoint_template,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )

    # ============================================================
    # 4. 모델 학습 실행
    # ============================================================
    print("\n" + "=" * 70)
    print(" 4. 모델 학습 진행")
    print("=" * 70)

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=150,
        batch_size=16,
        callbacks=[early_stopping, checkpoint_cb],
        verbose=1
    )

    # ============================================================
    # 5. 저장된 최적 모델 로드 및 테스트 평가
    # ============================================================
    print("\n" + "=" * 70)
    print(" 5. 저장된 최적 모델 파일 로드 및 테스트 데이터 평가")
    print("=" * 70)

    # 3. 저장된 실제 파일 목록에서 로드할 파일 찾기
    saved_files = glob.glob(os.path.join(save_dir, "wine_model_epoch_*.keras"))

    if not saved_files:
        raise FileNotFoundError("저장된 체크포인트 파일이 없습니다.")

    # 방법 A: 가장 최근에 저장(갱신)된 파일 가져오기 (save_best_only면 마지막 저장된 게 최고 성능)
    best_model_path = max(saved_files, key=os.path.getmtime)

    # 4. 저장된 모델 불러오기
    loaded_best_model = tf.keras.models.load_model(best_model_path)
    print(f"로드 성공: {best_model_path}")

    # 테스트 데이터셋 평가
    test_loss, test_accuracy = loaded_best_model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[로드된 모델의 테스트 평가 결과]")
    print(f" - Test Loss    : {test_loss:.4f}")
    print(f" - Test Accuracy: {test_accuracy * 100:.2f}%")

    # 샘플 5개에 대한 실제값 vs 예측값 비교
    sample_preds = loaded_best_model.predict(X_test[:5])
    sample_pred_labels = np.argmax(sample_preds, axis=1)

    print("\n[테스트 샘플 5개 예측 비교]")
    for i in range(5):
        actual = target_names[y_test[i]]
        pred = target_names[sample_pred_labels[i]]
        prob = sample_preds[i][sample_pred_labels[i]] * 100
        print(f"샘플 {i+1} | 실제: {actual:10s} -> 예측: {pred:10s} (확률: {prob:5.1f}%) | {'O' if actual == pred else 'X'}")

    # ============================================================
    # 6. 학습 과정 시각화 (Loss & Accuracy)
    # ============================================================
    plt.figure(figsize=(12, 5))

    # Loss 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss', color='royalblue')
    plt.plot(history.history['val_loss'], label='Val Loss', color='crimson')
    plt.axvline(x=len(history.history['loss']) - 15 - 1, color='gray', linestyle='--', label='Best Val Loss')
    plt.title("학습 및 검증 손실 (Loss)", fontsize=13)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Accuracy 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='royalblue')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='crimson')
    plt.title("학습 및 검증 정확도 (Accuracy)", fontsize=13)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
