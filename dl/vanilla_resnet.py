"""
[바닐라(Vanilla) ResNet 핵심 구조 이해 코드]
- ResNet(Residual Network)의 핵심 원리인 '잔차 연결(Residual Connection / Skip Connection)'을
  가장 단순하고 명확하게 구현한 최소 단위(Vanilla) 코드입니다.
- H(x) = F(x) + x 수식을 코드로 직접 확인하고, 기울기 소실(Vanishing Gradient) 문제를
  해결하는 원리를 시각화합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# 한글 및 마이너스 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def build_residual_block(input_tensor, filters, stride=1):
    """
    [ResNet의 핵심: Residual Block (잔차 블록)]
    
    기존 일반 합성곱 블록:
        x  ──> [ Conv -> BN -> ReLU -> Conv -> BN ] ──> 출력: F(x)
        
    ResNet의 Residual 블록:
        x  ──┬──> [ Conv -> BN -> ReLU -> Conv -> BN ] ──> F(x)
             │                                              │
             └───────────── (Shortcut / Skip) ─────────────(+) ──> ReLU(F(x) + x)
    """
    shortcut = input_tensor

    # 1. 주 경로 (Main Path: 합성곱 계층들 F(x))
    x = layers.Conv2D(filters, (3, 3), strides=stride, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # 2. 지름길 경로 (Shortcut Path)
    # 만약 stride > 1 이거나 채널 수가 바뀌어 shape가 다르면 1x1 Conv로 맞춰줌
    if stride != 1 or input_tensor.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # 3. [핵심] 잔차 더하기 (F(x) + x)
    # 입력을 그대로 건너뛰게 만들어 더해줌으로써 기울기가 1만큼 항상 보존됨!
    x = layers.Add()([x, shortcut])
    
    # 4. 마지막에 활성화 함수 적용
    output = layers.ReLU()(x)
    return output

def build_vanilla_resnet():
    """
    바닐라(Vanilla) ResNet 미니 모델
    Input(28x28x1) -> Conv -> ResBlock 1 -> ResBlock 2 -> GlobalAvgPool -> Dense(10)
    """
    inputs = layers.Input(shape=(28, 28, 1), name='input_image')

    # 초기 특징 추출 (Stem Conv)
    x = layers.Conv2D(32, (3, 3), padding='same', use_bias=False, name='stem_conv')(inputs)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.ReLU(name='stem_relu')(x)

    # 잔차 블록 1 (채널 32 유지)
    x = build_residual_block(x, filters=32, stride=1)

    # 잔차 블록 2 (채널 64로 증가, 가로세로 크기 1/2 다운샘플링: stride=2)
    x = build_residual_block(x, filters=64, stride=2)

    # 최종 분류기 (FC Layer 대신 GlobalAveragePooling 사용)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    outputs = layers.Dense(10, activation='softmax', name='classifier')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='Vanilla_ResNet')
    return model

def visualize_resnet_concept():
    """
    ResNet의 지름길(Shortcut Connection) 개념 다이어그램 시각화
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor('#FAFAFA')

    # 1. 주 경로 박스들
    boxes = [
        {"x": 2.5, "y": 2.5, "w": 2.2, "h": 0.8, "text": "Weight (Conv2D)\n+ BatchNorm", "color": "#4A90E2"},
        {"x": 2.5, "y": 1.2, "w": 2.2, "h": 0.8, "text": "Weight (Conv2D)\n+ BatchNorm", "color": "#4A90E2"},
    ]

    for b in boxes:
        rect = plt.Rectangle((b['x'], b['y']), b['w'], b['h'], facecolor=b['color'], edgecolor='#1A365D', alpha=0.9, zorder=3)
        ax.add_patch(rect)
        ax.text(b['x'] + b['w']/2, b['y'] + b['h']/2, b['text'], ha='center', va='center', color='white', fontweight='bold', fontsize=10, zorder=4)

    # 화살표 (Main Path)
    ax.annotate('', xy=(3.6, 3.3), xytext=(3.6, 4.0), arrowprops=dict(facecolor='black', width=2, headwidth=8), zorder=2)
    ax.text(3.6, 4.2, "입력: x", ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.annotate('', xy=(3.6, 2.0), xytext=(3.6, 2.5), arrowprops=dict(facecolor='black', width=2, headwidth=8), zorder=2)
    ax.text(3.6, 2.25, "ReLU", ha='center', va='center', fontsize=9, color='#C53030', fontweight='bold')
    
    ax.annotate('', xy=(3.6, 0.4), xytext=(3.6, 1.2), arrowprops=dict(facecolor='black', width=2, headwidth=8), zorder=2)
    ax.text(3.9, 0.8, "F(x)", fontsize=11, fontweight='bold', color='#1A365D')

    # 2. Skip Connection (지름길 곡선 화살표)
    # x 지점에서 시작해 덧셈 노드(+)로 바로 직행
    ax.annotate('', xy=(3.6, 0.3), xytext=(3.6, 3.8),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.7", color="#E53E3E", lw=3),
                zorder=2)
    ax.text(6.0, 2.0, "Skip / Shortcut Connection\n(변형 없이 x 그대로 전달)", ha='center', va='center', fontsize=11, color='#E53E3E', fontweight='bold')

    # 3. 덧셈 노드 (+)
    circle = plt.Circle((3.6, 0.1), 0.3, facecolor='#ECC94B', edgecolor='#744210', lw=2, zorder=5)
    ax.add_patch(circle)
    ax.text(3.6, 0.1, "+", ha='center', va='center', fontsize=16, fontweight='bold', zorder=6)

    # 4. 출력
    ax.annotate('', xy=(3.6, -0.7), xytext=(3.6, -0.2), arrowprops=dict(facecolor='black', width=2, headwidth=8), zorder=2)
    ax.text(3.6, -0.9, "출력: ReLU( F(x) + x )", ha='center', va='center', fontsize=12, fontweight='bold', color='#2F855A')

    ax.set_xlim(1.5, 7.8)
    ax.set_ylim(-1.2, 4.6)
    ax.axis('off')
    plt.title("[ResNet 핵심 원리: Residual Learning Block]", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.show()

def main():
    print("=" * 70)
    print(" 바닐라(Vanilla) ResNet 모델 빌드 및 구조 확인")
    print("=" * 70)

    model = build_vanilla_resnet()
    model.summary()

    # 가상 데이터(Fashion-MNIST 규격: 4개 샘플, 28x28x1)로 순전파 테스트
    dummy_input = np.random.randn(4, 28, 28, 1).astype(np.float32)
    output = model(dummy_input)
    print(f"\n[순전파 테스트 성공]")
    print(f" - 입력 Shape: {dummy_input.shape}")
    print(f" - 출력 Shape: {output.shape} (10개 클래스 확률값)")

    # ResNet 원리 시각화 팝업
    print("\nResNet 구조 개념 다이어그램을 출력합니다...")
    visualize_resnet_concept()

if __name__ == "__main__":
    main()
