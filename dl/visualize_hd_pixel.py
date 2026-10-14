import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 한글 폰트 및 마이너스 기호 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def visualize_hd_pixels(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {image_path}")

    # 1. 원본 이미지 로드
    raw_img = Image.open(image_path).convert('RGB')
    
    # 2. 중앙의 픽셀 아트 영역 크롭 (554 x 554 크기)
    # x: 1099 ~ 1653, y: 497 ~ 1051
    cropped_art = raw_img.crop((1099, 497, 1653, 1051))
    
    # 3. 픽셀 아트의 원본 해상도(24 x 24 그리드)로 정확하게 샘플링
    grid_size = 24
    pixel_art_24x24 = cropped_art.resize((grid_size, grid_size), Image.Resampling.NEAREST)
    pixel_array = np.array(pixel_art_24x24)

    # 4. 시각화 창 구성 (좌측: 픽셀 아트 이미지 / 우측: 24x24 픽셀 격자 및 RGB 값)
    fig = plt.figure(figsize=(18, 9))

    # [좌측] 크롭된 픽셀 아트 원본
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(cropped_art)
    ax1.set_title("1. 중앙 픽셀 아트 영역 (Cropped)", fontsize=14, pad=12)
    ax1.axis('off')

    # [우측] 24 x 24 픽셀 그리드 시각화
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(pixel_array)

    # 눈금 및 격자 설정 (0~23)
    ax2.set_xticks(range(grid_size))
    ax2.set_yticks(range(grid_size))
    ax2.set_xticklabels(range(grid_size), fontsize=8)
    ax2.set_yticklabels(range(grid_size), fontsize=8)
    ax2.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax2.grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax2.tick_params(which='minor', size=0)

    # 각 픽셀 셀 위에 RGB HEX 코드 또는 밝기 기준 텍스트 표시
    for y in range(grid_size):
        for x in range(grid_size):
            r, g, b = pixel_array[y, x]
            # 배경 밝기(Luminance)에 따라 글자색 반전
            brightness = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "black" if brightness > 128 else "white"
            hex_color = f"{r:02X}{g:02X}{b:02X}"
            
            # 셀 안에 RGB 16진수 색상 코드 표기
            ax2.text(x, y, hex_color,
                     ha="center", va="center",
                     color=text_color, fontsize=5.0, fontweight='bold')

    ax2.set_title(f"2. 24 x 24 픽셀 격자 & 색상값(Hex RGB) 시각화", fontsize=14, pad=12)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    img_path = "C:/apps/mldl/dl/data/hd.png"
    visualize_hd_pixels(img_path)
