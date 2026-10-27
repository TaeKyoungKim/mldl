"""
DCGAN (Deep Convolutional Generative Adversarial Network) 
64x64 컬러 강아지 이미지(dog_image_preprocessing.png) 생성 및 시각화 스크립트

[구조 및 특징]
1. 입력 데이터: 64x64x3 RGB 강아지 이미지 전처리 (데이터 증강 배치 구성)
2. 64x64 컬러 전용 DCGAN 아키텍처:
   - Generator: 100차원 잠재 벡터 z -> 4x4x512 -> 8x8x256 -> 16x16x128 -> 32x32x64 -> 64x64x3 (tanh)
   - Discriminator: 64x64x3 -> 32x32x64 -> 16x16x128 -> 8x8x256 -> 4x4x512 -> Flatten -> Dense(1)
3. 학습 및 생성 결과 비교:
   - 좌측: 원본 64x64 강아지 이미지
   - 우측: DCGAN 생성자가 새롭게 합성해낸 64x64 강아지 이미지 그리드 (3x3)
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models

# 한글 및 마이너스 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 시드 고정
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# 1. 하이퍼파라미터 및 경로 설정
# ============================================================
NOISE_DIM = 100
BATCH_SIZE = 16
EPOCHS = 150          # 단일 이미지를 기반으로 특징을 학습 및 재구성하기 위한 에폭
LEARNING_RATE = 2e-4
BETA_1 = 0.5

IMAGE_PATH = "C:/apps/mldl/dl/data/dog_image_preprocessing.png"
OUTPUT_DIR = "C:/apps/mldl/dl/output/dcgan_dog"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. 64x64 컬러 전용 Generator (생성자) 모델 구축
# ============================================================
def build_generator():
    """
    100차원 잠재 벡터 z -> 64x64x3 RGB 이미지 생성
    4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 (4단계 업샘플링)
    """
    model = models.Sequential(name="Generator_64x64")

    # Foundation for 4x4 image
    model.add(layers.Dense(4 * 4 * 512, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.2))
    model.add(layers.Reshape((4, 4, 512)))

    # Upsample 1: 4x4 -> 8x8
    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.2))

    # Upsample 2: 8x8 -> 16x16
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.2))

    # Upsample 3: 16x16 -> 32x32
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.2))

    # Upsample 4: 32x32 -> 64x64x3 (RGB 컬러 이미지, tanh: -1 ~ 1)
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model

# ============================================================
# 3. 64x64 컬러 전용 Discriminator (판별자) 모델 구축
# ============================================================
def build_discriminator():
    """
    64x64x3 RGB 이미지 -> 진짜/가짜 판별 로짓
    64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4 (다운샘플링)
    """
    model = models.Sequential(name="Discriminator_64x64")

    # Downsample 1: 64x64 -> 32x32
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[64, 64, 3]))
    model.add(layers.LeakyReLU(negative_slope=0.2))
    model.add(layers.Dropout(0.3))

    # Downsample 2: 32x32 -> 16x16
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.2))
    model.add(layers.Dropout(0.3))

    # Downsample 3: 16x16 -> 8x8
    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.2))
    model.add(layers.Dropout(0.3))

    # Downsample 4: 8x8 -> 4x4
    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.2))
    model.add(layers.Dropout(0.3))

    # Output Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

# ============================================================
# 4. 손실 함수 및 옵티마이저
# ============================================================
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator = build_generator()
discriminator = build_discriminator()

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)

# ============================================================
# 5. tf.function 학습 스텝
# ============================================================
@tf.function
def train_step(images):
    noise = tf.random.normal([tf.shape(images)[0], NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

# ============================================================
# 6. 메인 실행 함수
# ============================================================
def main():
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"강아지 이미지를 찾을 수 없습니다: {IMAGE_PATH}")

    print("=" * 80)
    print(" 1. 원본 64x64 강아지 이미지 로드 및 데이터 증강 배치 구성")
    print("=" * 80)

    raw_img = Image.open(IMAGE_PATH).convert('RGB')
    base_array = np.array(raw_img, dtype=np.float32)

    # [-1, 1] tanh 범위로 정규화
    normalized_img = (base_array - 127.5) / 127.5

    # 단일 이미지를 기반으로 회전/좌우반전/밝기 변화 증강을 주어 128개 학습 배치 생성
    augmented_list = []
    for _ in range(128):
        # 좌우 반전
        img_variant = np.fliplr(normalized_img) if np.random.rand() > 0.5 else normalized_img.copy()
        # 미세 노이즈 추가
        img_variant += np.random.normal(0, 0.02, img_variant.shape)
        img_variant = np.clip(img_variant, -1.0, 1.0)
        augmented_list.append(img_variant)

    train_dataset = tf.data.Dataset.from_tensor_slices(augmented_list).shuffle(128).batch(BATCH_SIZE)

    print(f"▶ 원본 이미지 크기: {raw_img.size} (64x64, RGB)")
    print(f"▶ 학습 데이터셋 크기: {len(augmented_list)}개 증강 샘플, 배치 크기: {BATCH_SIZE}")

    print("\n[DCGAN Generator 64x64 구조 요약]")
    generator.summary()

    # ============================================================
    # 7. DCGAN 모델 학습 루프
    # ============================================================
    print("\n" + "=" * 80)
    print(f" 2. DCGAN 강아지 이미지 학습 시작 (총 {EPOCHS} Epochs)")
    print("=" * 80)

    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        g_losses, d_losses = [], []
        for batch in train_dataset:
            g_loss, d_loss = train_step(batch)
            g_losses.append(g_loss.numpy())
            d_losses.append(d_loss.numpy())

        if epoch % 25 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"Epoch [{epoch:03d}/{EPOCHS:03d}] Gen Loss: {np.mean(g_losses):.4f} | Disc Loss: {np.mean(d_losses):.4f}")

    print(f"\n학습 완료! 총 소요시간: {time.time() - start_time:.1f}초")

    # ============================================================
    # 8. DCGAN이 새롭게 만들어낸 이미지 생성 및 원본과 비교 출력
    # ============================================================
    print("\n" + "=" * 80)
    print(" 3. DCGAN으로 새로운 강아지 이미지 9개 생성 및 비교 시각화")
    print("=" * 80)

    # 9개의 무작위 잠재 벡터 z로 새로운 강아지 이미지 생성
    num_samples = 9
    random_noise = tf.random.normal([num_samples, NOISE_DIM])
    generated_images = generator(random_noise, training=False).numpy()

    # [-1, 1] -> [0, 1] 복원
    generated_images = (generated_images * 127.5 + 127.5) / 255.0
    generated_images = np.clip(generated_images, 0.0, 1.0)

    fig = plt.figure(figsize=(14, 7))

    # [좌측] 원본 학습 대상 강아지 이미지 (64x64)
    ax_orig = plt.subplot2grid((3, 5), (0, 0), rowspan=3, colspan=2)
    ax_orig.imshow(raw_img)
    ax_orig.set_title("원본 강아지 이미지\n(Original 64×64)", fontsize=13, fontweight='bold', pad=12)
    ax_orig.axis('off')

    # [우측 3x3 그리드] DCGAN이 새롭게 합성/생성한 강아지 이미지들
    for i in range(num_samples):
        row = i // 3
        col = 2 + (i % 3)
        ax = plt.subplot2grid((3, 5), (row, col))
        ax.imshow(generated_images[i])
        ax.set_title(f"생성 #{i+1}", fontsize=10, pad=6)
        ax.axis('off')

    plt.suptitle("DCGAN(Deep Convolutional GAN) 64×64 강아지 이미지 학습 및 생성 결과", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    # 이미지 저장
    result_path = os.path.join(OUTPUT_DIR, "dcgan_dog_generated_result.png")
    plt.savefig(result_path, dpi=150)
    print(f"▶ 생성 결과 이미지 저장 완료: {result_path}")
    plt.show()

if __name__ == "__main__":
    main()
