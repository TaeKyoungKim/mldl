import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet50 import ResNet50

# 한글 폰트 및 마이너스 기호 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("=" * 70)
    print(" 1. Fashion-MNIST 데이터셋 로드 및 전처리")
    print("=" * 70)

    # 1. Fashion-MNIST 데이터셋 로드
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # 10개 클래스 이름 정의
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    # 정규화 (0~255 -> 0.0~1.0)
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # 1채널 흑백 -> 3채널(RGB) 변환: ResNet50은 3개의 컬러 채널을 요구합니다.
    # (N, 28, 28) -> (N, 28, 28, 3)
    x_train_rgb = np.repeat(np.expand_dims(x_train, axis=-1), 3, axis=-1)
    x_test_rgb = np.repeat(np.expand_dims(x_test, axis=-1), 3, axis=-1)

    print(f"변환된 학습 데이터 shape: {x_train_rgb.shape}")
    print(f"변환된 테스트 데이터 shape: {x_test_rgb.shape}")

    # ============================================================
    # 2. ResNet50 기반 전이학습 모델 구축
    # ============================================================
    print("\n" + "=" * 70)
    print(" 2. ResNet50 모델 구성 (Resizing & Transfer Learning)")
    print("=" * 70)

    # ResNet50은 최소 32x32 이상의 입력을 필요로 하므로, 
    # 모델 내부 첫 번째 레이어에서 32x32로 업샘플링합니다.
    inputs = layers.Input(shape=(28, 28, 3), name='input_image')
    x = layers.Resizing(32, 32, interpolation='bilinear', name='resize_to_32')(inputs)

    # ImageNet 가중치 기반 ResNet50 백본 로드 (최상단 분류기 제외)
    base_resnet = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(32, 32, 3)
    )

    # 사전 학습된 가중치 동결 (특징 추출기로 사용)
    base_resnet.trainable = False

    x = base_resnet(x, training=False)
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dropout(0.3, name='dropout')(x)
    outputs = layers.Dense(10, activation='softmax', name='fashion_classifier')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='ResNet50_FashionMNIST')
    model.summary()

    # ============================================================
    # 3. 모델 컴파일 및 미니 학습 (특징 분류기 레이어 튜닝)
    # ============================================================
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nResNet50 모델 학습 진행 (3 에폭)...")
    # 빠른 실습을 위해 3 에폭 학습 진행
    model.fit(
        x_train_rgb, y_train,
        epochs=3,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    # ============================================================
    # 4. 테스트 데이터 200번째 이미지 (인덱스: 199) 예측
    # ============================================================
    target_idx = 199  # 200번째 샘플 (0-based indexing: 200 - 1)

    # 단일 샘플 배치 차원 추가: (28, 28, 3) -> (1, 28, 28, 3)
    sample_image = np.expand_dims(x_test_rgb[target_idx], axis=0)
    actual_label_idx = y_test[target_idx]
    actual_class_name = class_names[actual_label_idx]

    # ResNet50 모델 예측 수행
    predictions = model.predict(sample_image, verbose=0)
    predicted_label_idx = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_label_idx]
    confidence = predictions[0][predicted_label_idx] * 100

    # 콘솔 결과 출력
    print("\n" + "=" * 65)
    print(f"       ResNet50 테스트 데이터 [200번째] 샘플 (인덱스: {target_idx}) 예측 결과")
    print("=" * 65)
    print(f" 실제 정답 (Actual)    : [{actual_label_idx}] {actual_class_name}")
    print(f" 예측 결과 (Predicted) : [{predicted_label_idx}] {predicted_class_name} (신뢰도: {confidence:.2f}%)")
    print(f" 일치 여부             : {'정답 (O)' if actual_label_idx == predicted_label_idx else '오답 (X)'}")
    print("=" * 65 + "\n")

    # ============================================================
    # 5. 결과 시각화 (200번째 원본 이미지 + 10개 클래스 확률 분포)
    # ============================================================
    plt.figure(figsize=(10, 4.5))

    # (1) 200번째 이미지 시각화
    plt.subplot(1, 2, 1)
    plt.imshow(x_test[target_idx], cmap='gray')
    is_correct = (actual_label_idx == predicted_label_idx)
    color = 'blue' if is_correct else 'red'
    plt.title(
        f"200번째 테스트 데이터 (ResNet50)\n정답: {actual_class_name} | 예측: {predicted_class_name}\n(신뢰도: {confidence:.1f}%)",
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
    plt.title("ResNet50 클래스별 소프트맥스 확률 분포", fontsize=12)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
