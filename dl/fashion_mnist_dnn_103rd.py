import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# 한글 폰트 및 마이너스 기호 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def visualize_dnn_architecture():
    """
    외부 graphviz 설치 없이 순수 matplotlib으로
    DNN 레이어 파이프라인 및 노드 연결 구조를 직관적으로 시각화합니다.
    """
    layers_info = [
        {"name": "Input Data\n(28 × 28 이미지)", "nodes_text": "28 × 28 Matrix", "color": "#1E88E5"},
        {"name": "Flatten\n(1차원 평탄화)", "nodes_text": "784 Nodes", "color": "#8E24AA"},
        {"name": "Hidden Layer 1\nDense (ReLU)", "nodes_text": "128 Nodes", "color": "#039BE5"},
        {"name": "Hidden Layer 2\nDense (ReLU)", "nodes_text": "64 Nodes", "color": "#00897B"},
        {"name": "Output Layer\nDense (Softmax)", "nodes_text": "10 Nodes (0~9)", "color": "#43A047"}
    ]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('#F8F9FA')

    n_layers = len(layers_info)
    box_width = 3.6
    box_height = 0.7

    for idx, info in enumerate(layers_info):
        y_pos = (n_layers - 1 - idx) * 1.25

        # 레이어 박스
        rect = plt.Rectangle((1.0, y_pos - box_height / 2), box_width, box_height,
                             facecolor=info['color'], edgecolor='#2c3e50',
                             linewidth=1.5, alpha=0.9, zorder=3)
        ax.add_patch(rect)

        # 텍스트 라벨 (좌측 설명)
        ax.text(1.2, y_pos, info['name'],
                ha='left', va='center', color='white',
                fontsize=10.5, fontweight='bold', zorder=4)

        # 노드 수 정보 (우측 배지)
        badge = plt.Rectangle((3.4, y_pos - 0.22), 1.1, 0.44,
                              facecolor='white', edgecolor='#333333',
                              linewidth=1, zorder=4)
        ax.add_patch(badge)
        ax.text(3.95, y_pos, info['nodes_text'],
                ha='center', va='center', color='#222222',
                fontsize=9, fontweight='bold', zorder=5)

        # 아래 레이어로 연결 화살표
        if idx < n_layers - 1:
            next_y = (n_layers - 1 - (idx + 1)) * 1.25
            ax.annotate('', xy=(1.0 + box_width / 2, next_y + box_height / 2),
                        xytext=(1.0 + box_width / 2, y_pos - box_height / 2),
                        arrowprops=dict(facecolor='#424242', edgecolor='#424242',
                                        width=2.5, headwidth=9, headlength=7),
                        zorder=2)

    ax.set_xlim(0.2, 5.5)
    ax.set_ylim(-0.8, (n_layers - 1) * 1.25 + 0.9)
    ax.axis('off')
    plt.title("Fashion-MNIST DNN 네트워크 구조 다이어그램\nInput(28×28) → Flatten(784) → Dense(128) → Dense(64) → Output(10)",
              fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.show()

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

    # 픽셀값 정규화 (0~255 -> 0.0~1.0)
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # ============================================================
    # 2. DNN 모델 구성 (은닉층 2개: 128, 64 노드)
    # ============================================================
    # 구조: Flatten(784) -> Dense(128, ReLU) -> Dense(64, ReLU) -> Dense(10, Softmax)
    model = models.Sequential([
        layers.Input(shape=(28, 28), name='input_layer'),
        layers.Flatten(name='flatten'),
        layers.Dense(128, activation='relu', name='hidden_1_128'),
        layers.Dense(64, activation='relu', name='hidden_2_64'),
        layers.Dense(10, activation='softmax', name='output_layer')
    ])

    print("\n[모델 구조 요약]")
    model.summary()

    # DNN 아키텍처 다이어그램 시각화 (Matplotlib)
    visualize_dnn_architecture()

    # ============================================================
    # 3. 모델 컴파일 및 학습
    # ============================================================
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n모델 학습 시작...")
    model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    # 전체 테스트 데이터셋 평가
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n[전체 테스트 데이터셋 성능] Loss: {test_loss:.4f}, Accuracy: {test_acc * 100:.2f}%")

    # ============================================================
    # 4. 테스트 데이터 103번째 이미지 (인덱스: 102) 예측 및 결과 출력
    # ============================================================
    target_idx = 102  # 103번째 샘플 (0-based indexing: 103 - 1)
    
    # 단일 샘플 배치 차원 추가: (28, 28) -> (1, 28, 28)
    sample_image = np.expand_dims(x_test[target_idx], axis=0)
    actual_label_idx = y_test[target_idx]
    actual_class_name = class_names[actual_label_idx]

    # 모델 예측 수행 (10개 클래스 확률 반환)
    predictions = model.predict(sample_image)
    predicted_label_idx = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_label_idx]
    confidence = predictions[0][predicted_label_idx] * 100

    # 콘솔 출력
    print("\n" + "=" * 65)
    print(f"       테스트 데이터 [103번째] 샘플 (인덱스: {target_idx}) 예측 결과")
    print("=" * 65)
    print(f" 실제 정답 (Actual)    : [{actual_label_idx}] {actual_class_name}")
    print(f" 예측 결과 (Predicted) : [{predicted_label_idx}] {predicted_class_name} (신뢰도: {confidence:.2f}%)")
    print(f" 일치 여부             : {'정답 (O)' if actual_label_idx == predicted_label_idx else '오답 (X)'}")
    print("=" * 65 + "\n")

    # ============================================================
    # 5. 결과 시각화 (103번째 이미지 + 클래스별 확률 분포)
    # ============================================================
    plt.figure(figsize=(10, 4.5))

    # (1) 103번째 이미지 출력
    plt.subplot(1, 2, 1)
    plt.imshow(x_test[target_idx], cmap='gray')
    is_correct = (actual_label_idx == predicted_label_idx)
    color = 'blue' if is_correct else 'red'
    plt.title(
        f"103번째 테스트 데이터\n정답: {actual_class_name} | 예측: {predicted_class_name}\n(신뢰도: {confidence:.1f}%)",
        color=color, fontsize=12
    )
    plt.axis('off')

    # (2) 10개 클래스별 예측 확률 분포 막대 그래프
    plt.subplot(1, 2, 2)
    bar_plot = plt.bar(range(10), predictions[0], color='#A0A0A0')
    bar_plot[actual_label_idx].set_color('blue')       # 실제 정답: 파란색
    if not is_correct:
        bar_plot[predicted_label_idx].set_color('red') # 오답 시 예측값: 빨간색

    plt.xticks(range(10), class_names, rotation=45, ha='right')
    plt.ylim([0, 1.05])
    plt.ylabel("예측 확률 (Probability)")
    plt.title("클래스별 소프트맥스 확률 분포", fontsize=12)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
