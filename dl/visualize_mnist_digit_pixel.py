import matplotlib.pyplot as plt
import tensorflow as tf

# 한글 폰트 및 마이너스 기호 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def main():
    # 1. MNIST 데이터셋 로드 (이미지의 샘플은 MNIST 훈련 데이터의 첫 번째 샘플 '5'입니다)
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (_, _) = mnist.load_data()

    # 첫 번째 샘플 (숫자 5)
    img = x_train[0]
    label = y_train[0]

    # 2. 픽셀 숫자 시각화 창 설정
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # 이미지 배경 렌더링 (그레이스케일, 배경 흰색/글자 검은색인 'gray_r' 또는 기본 'gray')
    cax = ax.imshow(img, cmap='gray_r')

    # 3. 28x28 격자 및 축 눈금 설정
    ax.set_xticks(range(28))
    ax.set_yticks(range(28))
    ax.set_xticklabels(range(28), fontsize=8)
    ax.set_yticklabels(range(28), fontsize=8)

    # 4. 각 픽셀 셀마다 실제 수치값(0~255) 텍스트 오버레이
    # 배경 밝기에 따라 가독성을 위해 텍스트 색상 전환
    threshold = img.max() / 2.0
    for i in range(28):
        for j in range(28):
            pixel_val = img[i, j]
            # gray_r 기준: 픽셀값이 클수록(진할수록) 어두워지므로 글자색을 흰색으로 표시
            text_color = "white" if pixel_val > threshold else "black"
            ax.text(j, i, f"{pixel_val}",
                    ha="center", va="center",
                    color=text_color, fontsize=6.5)

    plt.title(f"MNIST 손글씨 데이터 [라벨: {label}] 픽셀 숫자 상세 시각화 (28×28)", fontsize=14, pad=15)
    plt.colorbar(cax, fraction=0.046, pad=0.04, label="Pixel Intensity (0~255)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
