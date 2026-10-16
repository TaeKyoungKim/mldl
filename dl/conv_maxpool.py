import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# 한글 폰트 및 마이너스 기호 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 이미지 불러오기 및 전처리
# ============================================================
image_path = "C:/apps/mldl/dl/data/hd.png"
if not os.path.exists(image_path):
    raise FileNotFoundError(f"파일을 찾을 수 없습니다: {image_path}")

raw_img = Image.open(image_path).convert("RGB")

# hd.png 중앙의 실제 픽셀 아트 영역(사람 & 강아지 캐릭터) 크롭 (554x554 영역)
# (불필요한 회색 배경을 제거하여 특징 추출 효과를 극대화)
cropped_img = raw_img.crop((1099, 497, 1653, 1051))

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),  # (C, H, W), 0.0 ~ 1.0 정규화
])
img_tensor = transform(cropped_img).unsqueeze(0)  # (1, 3, 128, 128) 배치 차원 추가

# ============================================================
# 2. 필터 10개짜리 Convolution 레이어 정의
# ============================================================
# in_channels=3 (RGB), out_channels=10 (필터 10개), kernel_size=3, padding=1 (크기 유지)
conv = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1)

# ============================================================
# 3. 맥스풀링(MaxPool) 레이어 정의 (2x2 영역에서 최댓값 추출 -> 가로/세로 절반 축소)
# ============================================================
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

# ============================================================
# 4. Convolution -> ReLU -> Maxpooling 순서로 연산 수행
# ============================================================
relu = nn.ReLU()

with torch.no_grad():
    conv_output = conv(img_tensor)           # (1, 10, 128, 128)
    relu_output = relu(conv_output)          # 활성화 함수 적용
    pool_output = maxpool(relu_output)       # (1, 10, 64, 64)

print("입력 이미지 shape    :", img_tensor.shape)
print("Convolution 결과 shape:", conv_output.shape)
print("MaxPooling 결과 shape :", pool_output.shape)  # 가로세로가 128 -> 64로 절반 축소

# ============================================================
# 5. 원본 이미지 & Conv(위) vs Maxpooling(아래) 결과 비교 시각화
# ============================================================
fig = plt.figure(figsize=(22, 6))

# 상단: 원본 크롭 이미지 (좌측 배치)
ax_orig = plt.subplot2grid((2, 11), (0, 0), rowspan=2)
ax_orig.imshow(cropped_img)
ax_orig.set_title("원본 이미지\n(Cropped)", fontsize=11, fontweight='bold')
ax_orig.axis("off")

# 10개 필터에 대해 Conv 결과(위) vs MaxPool 결과(아래) 나란히 시각화
for i in range(10):
    # 상단 행: Conv 피처맵 (128x128)
    ax_conv = plt.subplot2grid((2, 11), (0, i + 1))
    ax_conv.imshow(relu_output[0, i].numpy(), cmap="viridis")
    ax_conv.set_title(f"Conv #{i + 1}\n(128×128)", fontsize=9)
    ax_conv.axis("off")

    # 하단 행: MaxPool 피처맵 (64x64)
    ax_pool = plt.subplot2grid((2, 11), (1, i + 1))
    ax_pool.imshow(pool_output[0, i].numpy(), cmap="viridis")
    ax_pool.set_title(f"MaxPool #{i + 1}\n(64×64)", fontsize=9)
    ax_pool.axis("off")

plt.suptitle("hd.png 이미지 Convolution(10개 필터) 및 MaxPooling(2×2) 비교", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()

# 결과 저장 폴더 생성 및 저장
output_dir = "C:/apps/mldl/dl/output"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, "conv_maxpool_result.png")
plt.savefig(save_path, dpi=150)
print(f"\n결과 이미지 저장 완료: {save_path}")

# 화면에도 표시
plt.show()
