import matplotlib.pyplot as plt
import tensorflow as tf

# 한글 폰트 및 마이너스 기호 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def main():
    # 1. Fashion-MNIST 데이터셋 로드
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (_, _) = fashion_mnist.load_data()

    # 클래스 레이블 이름 정의
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    # 2. 10번째 데이터 선택 (0-based 인덱스: 9)
    target_idx = 9
    img = train_images[target_idx]
    label = train_labels[target_idx]
    label_name = class_names[label]

    # 3. 픽셀 단위 시각화
    fig, ax = plt.subplots(figsize=(14, 14))
    cax = ax.imshow(img, cmap='gray')

    # 그리드 격자선 표시
    ax.set_xticks(range(28))
    ax.set_yticks(range(28))
    ax.set_xticklabels(range(28), fontsize=8)
    ax.set_yticklabels(range(28), fontsize=8)

    # 4. 각 픽셀 셀 위에 실제 픽셀값(0~255) 수치 텍스트 표시
    # (픽셀값이 밝으면 검은색 글씨, 어두우면 흰색 글씨로 가독성 확보)
    threshold = img.max() / 2.0
    for i in range(28):
        for j in range(28):
            pixel_val = img[i, j]
            text_color = "black" if pixel_val > threshold else "white"
            ax.text(j, i, f"{pixel_val}",
                    ha="center", va="center",
                    color=text_color, fontsize=6.5)

    plt.title(f"Fashion-MNIST 10번째 데이터 픽셀 상세 시각화\n[클래스: {label} ({label_name})]", fontsize=14, pad=15)
    plt.colorbar(cax, fraction=0.046, pad=0.04, label="Pixel Value (0~255)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
