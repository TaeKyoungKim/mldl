"""
DCGAN (Deep Convolutional Generative Adversarial Network) MNIST 생성 파이프라인

[주요 기능]
1. 생성자(Generator) & 판별자(Discriminator) 표준 DCGAN 아키텍처 구축
   - Generator: Conv2DTranspose + BatchNorm + LeakyReLU (tanh 출력: 28x28x1)
   - Discriminator: Conv2D(stride=2) + LeakyReLU + Dropout (이진 판별 출력: 1)
2. 고정된 잠재 벡터(Fixed Noise)를 사용하여 에폭별 이미지 생성 발전 과정을 동일 노이즈로 추적
3. 단계별(에폭별) 생성 이미지 자동 저장 (output/dcgan_mnist/epoch_XX.png)
4. 최종 학습 완료 후 에폭별 발전 과정을 한 화면에 비교하는 진화(Evolution) 그리드 시각화
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# 한글 및 마이너스 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 재현성을 위한 시드 고정
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# 1. 하이퍼파라미터 및 경로 설정
# ============================================================
NOISE_DIM = 100       # 잠재 벡터(Latent Vector z) 차원
NUM_EXAMPLES_TO_GENERATE = 16  # 에폭별 모니터링할 샘플 수 (4x4 그리드)
EPOCHS = 10           # 단계별 생성 과정 확인을 위한 기본 에폭 (필요에 따라 조절)
BATCH_SIZE = 128
LEARNING_RATE = 2e-4
BETA_1 = 0.5          # DCGAN 원작 논문 권장 Adam beta_1 값

OUTPUT_DIR = "C:/apps/mldl/dl/output/dcgan_mnist"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 단계별 진행 과정을 고정된 잠재 벡터로 추적하기 위해 생성
fixed_seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])

# ============================================================
# 2. Generator (생성자) 모델 구축
# ============================================================
def make_generator_model():
    """
    100차원 잠재 벡터 z를 입력받아 28x28x1 크기의 가짜(Fake) 이미지를 생성
    7x7 -> 14x14 -> 28x28 순서로 업샘플링(Conv2DTranspose)
    """
    model = models.Sequential(name="Generator")

    # Foundation for 7x7 image: 100 -> 7 * 7 * 256 = 12,544
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.2))
    model.add(layers.Reshape((7, 7, 256)))

    # Upsample 1: 7x7 -> 14x14
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.2))

    # Upsample 2: 14x14 -> 28x28
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(negative_slope=0.2))

    # Output Layer: 28x28x1, tanh 활성화 함수 (-1.0 ~ 1.0)
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model

# ============================================================
# 3. Discriminator (판별자) 모델 구축
# ============================================================
def make_discriminator_model():
    """
    28x28x1 이미지를 입력받아 진짜(1)인지 가짜(0)인지 판별하는 이진 분류기
    MaxPooling 대신 Strided Convolution을 사용
    """
    model = models.Sequential(name="Discriminator")

    # Downsample 1: 28x28 -> 14x14
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU(negative_slope=0.2))
    model.add(layers.Dropout(0.3))

    # Downsample 2: 14x14 -> 7x7
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(negative_slope=0.2))
    model.add(layers.Dropout(0.3))

    # Output Layer: 이진 분류 로짓 (Logit)
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

# ============================================================
# 4. 손실 함수 (Loss) 및 옵티마이저 정의
# ============================================================
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    # 진짜 이미지는 1에 가깝게, 가짜 이미지는 0에 가깝게 판별해야 함
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    # 생성자는 판별자가 가짜 이미지를 진짜(1)로 속도록 만들어야 함
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)

# ============================================================
# 5. 에폭별 생성 이미지 시각화 및 파일 저장 함수
# ============================================================
def generate_and_save_images(model, epoch, test_input, save_path=None):
    # training=False로 설정하여 BatchNorm이 추론 모드로 동작하게 함
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        # [-1, 1] 범위를 [0, 1] 범위로 되돌려 렌더링
        img = (predictions[i, :, :, 0] * 127.5 + 127.5) / 255.0
        plt.imshow(img, cmap='gray')
        plt.axis('off')

    plt.suptitle(f"Epoch {epoch:02d} 생성 결과", fontsize=11, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.close()

# ============================================================
# 6. 학습 스텝 (Train Step) 정의 (tf.function)
# ============================================================
generator = make_generator_model()
discriminator = make_discriminator_model()

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

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
# 7. 메인 학습 루프 및 단계별 발전 과정 시각화
# ============================================================
def main():
    print("=" * 80)
    print(" 1. MNIST 데이터셋 로드 및 [-1, 1] 정규화 전처리")
    print("=" * 80)

    (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
    # 4차원 변환 (60000, 28, 28, 1) 및 [-1, 1] tanh 범위로 정규화
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5

    BUFFER_SIZE = 60000
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    print("\n[생성자(Generator) 구조 요약]")
    generator.summary()

    print("\n[판별자(Discriminator) 구조 요약]")
    discriminator.summary()

    # 학습 시작 전(Epoch 0) 완전한 노이즈 상태 이미지 저장
    initial_save_path = os.path.join(OUTPUT_DIR, "epoch_00_initial_noise.png")
    generate_and_save_images(generator, 0, fixed_seed, initial_save_path)
    print(f"\n[초기화] 에폭 0 (순수 노이즈 상태) 이미지 저장 완료: {initial_save_path}")

    # ============================================================
    # 에폭별 학습 및 단계별 이미지 저장
    # ============================================================
    print("\n" + "=" * 80)
    print(f" 2. DCGAN 모델 학습 시작 (총 {EPOCHS} Epochs)")
    print("=" * 80)

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        gen_loss_list, disc_loss_list = [], []

        for image_batch in train_dataset:
            g_loss, d_loss = train_step(image_batch)
            gen_loss_list.append(g_loss.numpy())
            disc_loss_list.append(d_loss.numpy())

        elapsed = time.time() - start
        avg_g_loss = np.mean(gen_loss_list)
        avg_d_loss = np.mean(disc_loss_list)

        # 고정된 동일 노이즈(fixed_seed)로 단계별 생성 이미지 저장
        save_path = os.path.join(OUTPUT_DIR, f"epoch_{epoch:02d}.png")
        generate_and_save_images(generator, epoch, fixed_seed, save_path)

        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] 소요시간: {elapsed:5.1f}초 | Gen Loss: {avg_g_loss:.4f} | Disc Loss: {avg_d_loss:.4f} -> 저장: {save_path}")

    # ============================================================
    # 3. 단계별 발전 과정(Evolution) 종합 비교 시각화
    # ============================================================
    print("\n" + "=" * 80)
    print(" 3. 에폭별 이미지 생성 발전 과정(Evolution) 종합 비교 출력")
    print("=" * 80)

    # 주요 마일스톤 에폭 선택 (초기 노이즈 -> 1 -> 3 -> 5 -> 최종)
    milestones = [0, 1, 3, 5, EPOCHS] if EPOCHS >= 5 else [0, 1, EPOCHS]
    milestones = sorted(list(set(milestones)))

    plt.figure(figsize=(15, 4))
    for idx, ep in enumerate(milestones):
        img_file = os.path.join(OUTPUT_DIR, f"epoch_{ep:02d}.png" if ep > 0 else "epoch_00_initial_noise.png")
        if os.path.exists(img_file):
            loaded_img = plt.imread(img_file)
            plt.subplot(1, len(milestones), idx + 1)
            plt.imshow(loaded_img)
            plt.title(f"단계 {idx+1}: Epoch {ep}", fontsize=11, fontweight='bold')
            plt.axis('off')

    plt.suptitle("DCGAN MNIST 손글씨 생성 발전 과정 (순수 노이즈 -> 형태 형성 -> 선명한 숫자)", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    evolution_path = os.path.join(OUTPUT_DIR, "dcgan_mnist_evolution_summary.png")
    plt.savefig(evolution_path, dpi=150)
    print(f"▶ 단계별 발전 과정 종합 요약 차트 저장 완료: {evolution_path}")
    plt.show()

if __name__ == "__main__":
    main()
