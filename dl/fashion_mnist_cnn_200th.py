import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# 한글 폰트 및 마이너스 기호 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def main():
    # ============================================================
    # 1. Fashion-MNIST 데이터셋 로드 및 전처리
    # ============================================================
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # 10개 클래스 이름 정의
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    # 픽셀 정규화 (0~255 -> 0.0~1.0)
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # CNN 입력 채널 차원 추가: (28, 28) -> (28, 28, 1) 단일 채널(흑백)
    x_train_cnn = np.expand_dims(x_train, axis=-1)
    x_test_cnn = np.expand_dims(x_test, axis=-1)

    print(f"훈련 데이터 shape : {x_train_cnn.shape}")
    print(f"테스트 데이터 shape: {x_test_cnn.shape}")

    # ============================================================
    # 2. CNN 모델 구조 정의
    # ============================================================
    # Conv2D(32) -> MaxPooling -> Conv2D(64) -> MaxPooling -> Flatten -> Dense(64) -> Dropout -> Output(10)
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1), name='input_layer'),
        
        # 첫 번째 Conv Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1'),
        layers.BatchNormalization(name='bn1'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        
        # 두 번째 Conv Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        layers.BatchNormalization(name='bn2'),
        layers.MaxPooling2D((2, 2), name='pool2'),

        # 분류를 위한 Dense Block
        layers.Flatten(name='flatten'),
        layers.Dense(64, activation='relu', name='dense1'),
        layers.Dropout(0.3, name='dropout'),
        layers.Dense(10, activation='softmax', name='output')
    ])

    print("\n[CNN 모델 구조 요약]")
    model.summary()

    # ============================================================
    # 3. 모델 컴파일 및 학습
    # ============================================================
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nCNN 모델 학습 시작 (5 Epochs)...")
    history = model.fit(
        x_train_cnn, y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    # 테스트 데이터셋 전체 성능 평가
    test_loss, test_acc = model.evaluate(x_test_cnn, y_test, verbose=0)
    print(f"\n[전체 테스트 데이터셋 성능] Loss: {test_loss:.4f}, Accuracy: {test_acc * 100:.2f}%")

    # ============================================================
    # 4. 테스트 데이터 200번째 이미지 (인덱스: 199) 예측
    # ============================================================
    target_idx = 199  # 200번째 샘플 (0-based indexing: 200 - 1)
    
    # 단일 샘플 배치 형태 변환: (28, 28, 1) -> (1, 28, 28, 1)
    sample_image = np.expand_dims(x_test_cnn[target_idx], axis=0)
    actual_label_idx = y_test[target_idx]
    actual_class_name = class_names[actual_label_idx]

    # 모델 예측 수행 (10개 클래스 확률 반환)
    predictions = model.predict(sample_image)
    predicted_label_idx = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_label_idx]
    confidence = predictions[0][predicted_label_idx] * 100

    # 콘솔 결과 출력
    print("\n" + "=" * 65)
    print(f"       테스트 데이터 [200번째] 샘플 (인덱스: {target_idx}) 예측 결과")
    print("=" * 65)
    print(f" 실제 정답 (Actual)    : [{actual_label_idx}] {actual_class_name}")
    print(f" 예측 결과 (Predicted) : [{predicted_label_idx}] {predicted_class_name} (신뢰도: {confidence:.2f}%)")
    print(f" 일치 여부             : {'정답 (O)' if actual_label_idx == predicted_label_idx else '오답 (X)'}")
    print("=" * 65 + "\n")

    # ============================================================
    # 5. 결과 시각화 (200번째 이미지 + 클래스별 확률 분포)
    # ============================================================
    plt.figure(figsize=(10, 4.5))

    # (1) 200번째 패션 아이템 이미지 시각화
    plt.subplot(1, 2, 1)
    # 2D 형태로 표시하기 위해 마지막 채널 squeeze: (28, 28, 1) -> (28, 28)
    plt.imshow(x_test[target_idx], cmap='gray')
    is_correct = (actual_label_idx == predicted_label_idx)
    color = 'blue' if is_correct else 'red'
    plt.title(
        f"200번째 테스트 이미지\n정답: {actual_class_name} | 예측: {predicted_class_name}\n(신뢰도: {confidence:.1f}%)",
        color=color, fontsize=12
    )
    plt.axis('off')

    # (2) 10개 클래스 예측 확률 막대 그래프
    plt.subplot(1, 2, 2)
    bars = plt.bar(range(10), predictions[0], color='#A0A0A0')
    bars[actual_label_idx].set_color('blue')        # 실제 정답: 파란색
    if not is_correct:
        bars[predicted_label_idx].set_color('red')  # 오답 시 예측값: 빨간색

    plt.xticks(range(10), class_names, rotation=45, ha='right')
    plt.ylim([0, 1.05])
    plt.ylabel("예측 확률 (Probability)")
    plt.title("클래스별 소프트맥스 확률 분포", fontsize=12)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
