import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models

# 한글 폰트 및 마이너스 기호 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def main():
    img_path = "C:/apps/mldl/dl/data/hd.png"
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {img_path}")

    # ============================================================
    # 1. 이미지 로드 및 전처리
    # ============================================================
    raw_img = Image.open(img_path).convert('RGB')

    # hd.png 중앙의 픽셀 아트 영역(사람 & 강아지) 크롭 (554x554 영역)
    # x: 1099 ~ 1653, y: 497 ~ 1051
    cropped_img = raw_img.crop((1099, 497, 1653, 1051))

    # 컨볼루션 입력용 텐서 변환 (정규화 0.0 ~ 1.0)
    # Keras Conv2D 입력 차원: (batch_size, height, width, channels) -> (1, H, W, 3)
    img_array = np.array(cropped_img, dtype=np.float32) / 255.0
    input_tensor = np.expand_dims(img_array, axis=0)

    # ============================================================
    # 2. 10개의 필터를 가진 Conv2D 모델 구성
    # ============================================================
    # 필터 크기 3x3, 필터 개수 10개, padding='same' (출력 크기 유지), 활성화 함수 'relu'
    num_filters = 10
    conv_model = models.Sequential([
        layers.Input(shape=img_array.shape),
        layers.Conv2D(
            filters=num_filters,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            name='conv_10_filters'
        )
    ])

    print("[모델 정보]")
    conv_model.summary()

    # ============================================================
    # 3. Convolution 연산 수행 (피처맵 추출)
    # ============================================================
    # 출력 형태: (1, Height, Width, 10)
    feature_maps = conv_model.predict(input_tensor)
    feature_maps = feature_maps[0]  # 배치 차원 제거 -> (Height, Width, 10)

    # ============================================================
    # 4. 원본 이미지 및 10개 필터로 처리된 컨볼루션 피처맵 시각화
    # ============================================================
    # 상단 1행: 원본 이미지 (가운데 배치)
    # 하단 2행: 10개의 필터 결과 피처맵 (2행 5열)
    fig = plt.figure(figsize=(15, 8))

    # [1] 원본 이미지 출력 (좌측 상단)
    ax_orig = plt.subplot2grid((3, 5), (0, 2))
    ax_orig.imshow(cropped_img)
    ax_orig.set_title("원본 이미지 (Cropped Art)", fontsize=12, fontweight='bold', pad=10)
    ax_orig.axis('off')

    # [2] 10개 필터가 추출한 특징 맵(Feature Map) 각각 출력
    for i in range(num_filters):
        # 3행 5열 그리드 중 1~2행(index 1, 2)에 5개씩 배치
        row = 1 + (i // 5)
        col = i % 5
        ax = plt.subplot2grid((3, 5), (row, col))
        
        # 각 필터의 활성화 맵 (viridis / gray 등의 컬러맵 활용)
        f_map = feature_maps[:, :, i]
        cax = ax.imshow(f_map, cmap='viridis')
        ax.set_title(f"Filter #{i+1} Feature Map", fontsize=10, pad=6)
        ax.axis('off')

    plt.suptitle("hd.png 이미지 10개 필터(Conv2D) Convolution 연산 결과", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
