import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

def main():
    # 1. 데이터셋 로드 및 전처리 (Fashion-MNIST 28x28 이미지)
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Fashion-MNIST 10개 클래스 이름 정의
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    # 픽셀 정규화 (0~255 -> 0.0~1.0)
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # 2. 다이어그램 구조대로 모델 구성
    # Input(28x28) -> Flatten(784) -> Hidden 1(128, relu) -> Hidden 2(64, relu) -> Output(10, softmax)
    model = models.Sequential([
        layers.Input(shape=(28, 28)),                  # Input Data: 28 x 28 이미지 입력
        layers.Flatten(),                              # Preprocessing: Flatten (784 Nodes)
        layers.Dense(128, activation='relu'),          # Hidden Layer 1: Dense (128, relu)
        layers.Dense(64, activation='relu'),           # Hidden Layer 2: Dense (64, relu)
        layers.Dense(10, activation='softmax')         # Output Layer: Dense (10, softmax)
    ])

    # 모델 구조 요약
    model.summary()

    # 3. 모델 컴파일 및 학습
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n모델 학습 시작...")
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

    # 4. 테스트 데이터의 500번째 값 (Index: 499) 예측 및 비교
    target_index = 499  # 500번째 샘플 (0-based indexing)

    # 단일 샘플 배치 차원 추가: (28, 28) -> (1, 28, 28)
    sample_image = np.expand_dims(x_test[target_index], axis=0)
    actual_label_idx = y_test[target_index]
    actual_class_name = class_names[actual_label_idx]

    # 예측 수행 (10개 클래스별 소프트맥스 확률값 반환)
    predictions = model.predict(sample_image)
    predicted_label_idx = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_label_idx]
    confidence = predictions[0][predicted_label_idx] * 100

    # 5. 콘솔 결과 출력
    print("\n" + "=" * 60)
    print(f"테스트 데이터 [{target_index + 1}번째] 샘플 예측 결과")
    print(f"실제 정답 (Actual)    : [{actual_label_idx}] {actual_class_name}")
    print(f"예측 결과 (Predicted) : [{predicted_label_idx}] {predicted_class_name} (신뢰도: {confidence:.2f}%)")
    print(f"일치 여부             : {'정답 (O)' if actual_label_idx == predicted_label_idx else '오답 (X)'}")
    print("=" * 60 + "\n")

    # 6. 결과 시각화 (이미지 + 예측 클래스별 확률 분포 막대 그래프)
    plt.figure(figsize=(9, 4))

    # (1) 500번째 패션 아이템 이미지
    plt.subplot(1, 2, 1)
    plt.imshow(x_test[target_index], cmap='gray')
    is_correct = (actual_label_idx == predicted_label_idx)
    title_color = 'blue' if is_correct else 'red'
    plt.title(
        f"Actual: {actual_class_name}\nPred: {predicted_class_name} ({confidence:.1f}%)",
        color=title_color
    )
    plt.axis('off')

    # (2) 10개 클래스에 대한 소프트맥스 확률 분포 막대 그래프
    plt.subplot(1, 2, 2)
    bar_plot = plt.bar(range(10), predictions[0], color='#777777')
    bar_plot[actual_label_idx].set_color('blue')           # 실제 정답: 파란색
    if not is_correct:
        bar_plot[predicted_label_idx].set_color('red')     # 오답 예측 시: 빨간색
    plt.xticks(range(10), class_names, rotation=45, ha='right')
    plt.ylim([0, 1])
    plt.ylabel("Probability")
    plt.title("Class Probability Distribution")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
