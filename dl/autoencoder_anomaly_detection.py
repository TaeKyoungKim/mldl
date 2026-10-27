"""
오토인코더(Autoencoder) 기반 산업 센서 시계열 이상치 탐지(Anomaly Detection) 실습 파이프라인

[시나리오]
- 산업 장비의 진동/온도 센서 데이터 모사:
  1) 정상 데이터: 정상 주파수와 진폭을 가지는 사인/코사인 파형 + 정규분포 노이즈
  2) 이상 데이터: 스파이크(충격), 급격한 진폭 변동, 고주파 노이즈 혼입
- 파이프라인:
  1. 정상 데이터(Normal Only)로 오토인코더(Encoder -> Bottleneck -> Decoder) 학습
  2. 정상 검증 데이터의 재구성 오차(Reconstruction Error, MSE) 계산
  3. 통계적 임계값(Threshold tau) 설정: (평균 + 3 * 표준편차) 방식 적용
  4. 테스트 데이터(정상 + 이상)에 대해 이상치 판정 및 평가(정밀도, 재현율, F1-Score)
  5. 이상 탐지 원리 시각화 (신호 복원 비교 & 오차 분포 및 임계값 차트)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import tensorflow as tf
from tensorflow.keras import layers, models

# 한글 폰트 및 마이너스 기호 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 재현성을 위한 시드 고정
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# 1. 센서 시계열 시뮬레이션 데이터 생성 함수
# ============================================================
def generate_sensor_data(n_samples=3000, seq_len=50):
    """
    seq_len=50개의 연속된 센서 측정값을 1개의 샘플로 정의
    - 정상 데이터: 주기적 파동 (sin + cos) + 백색 가우시안 노이즈
    - 이상 데이터: 비주기적 스파이크, 신호 왜곡, 비정상 진폭
    """
    t = np.linspace(0, 50 * np.pi, n_samples * seq_len)
    
    # 기본 정상 신호
    normal_signal = np.sin(t) + 0.5 * np.cos(2 * t) + np.random.normal(0, 0.1, len(t))
    
    # (n_samples, seq_len) 형태로 분할
    X_normal = normal_signal.reshape(n_samples, seq_len)
    
    # 이상 데이터 생성 (500개 샘플)
    n_anomalies = 500
    X_anomaly = []
    for _ in range(n_anomalies):
        # 무작위 파형 생성 후 이상 패턴 삽입
        t_sub = np.linspace(0, 4 * np.pi, seq_len)
        sig = np.sin(t_sub) + np.random.normal(0, 0.1, seq_len)
        
        anomaly_type = np.random.choice(['spike', 'amplitude', 'noise'])
        if anomaly_type == 'spike': # 급격한 충격 스파이크
            idx = np.random.randint(10, seq_len - 10)
            sig[idx:idx+3] += np.random.choice([3.5, -3.5])
        elif anomaly_type == 'amplitude': # 진폭 3배 급증
            sig = sig * 3.0
        else: # 강한 고주파 노이즈
            sig += np.random.normal(0, 1.2, seq_len)
            
        X_anomaly.append(sig)
        
    X_anomaly = np.array(X_anomaly)
    return X_normal, X_anomaly

# ============================================================
# 2. 오토인코더(Autoencoder) 신경망 모델 정의
# ============================================================
def build_autoencoder(input_dim=50):
    """
    입력(50) -> 인코더(32 -> 16 -> 잠재공간: 8) -> 디코더(16 -> 32 -> 복원: 50)
    과적합을 방지하고 정상 특징만 학습하도록 병목(Bottleneck: 8) 설계
    """
    inputs = layers.Input(shape=(input_dim,), name='input_signal')
    
    # [Encoder] 차원을 50 -> 8로 압축
    x = layers.Dense(32, activation='relu', name='enc_1')(inputs)
    x = layers.Dense(16, activation='relu', name='enc_2')(x)
    latent_z = layers.Dense(8, activation='relu', name='bottleneck_latent')(x)
    
    # [Decoder] 잠재 벡터 8 -> 원래 신호 50으로 복원
    x = layers.Dense(16, activation='relu', name='dec_1')(latent_z)
    x = layers.Dense(32, activation='relu', name='dec_2')(x)
    reconstructed_x = layers.Dense(input_dim, activation='linear', name='reconstruction')(x)
    
    autoencoder = models.Model(inputs=inputs, outputs=reconstructed_x, name='Anomaly_Autoencoder')
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def main():
    print("=" * 80)
    print(" 1. 센서 시계열 데이터셋 생성 및 전처리")
    print("=" * 80)
    
    # 정상 데이터 3,000개, 이상 데이터 500개 생성 (각 샘플당 50개 시점 특성)
    X_normal_raw, X_anomaly_raw = generate_sensor_data(n_samples=3000, seq_len=50)
    
    # 데이터 분할:
    # - Train: 오직 정상 데이터 2,000개만 사용 (원리에 입각하여 정상만 학습!)
    # - Val  : 정상 데이터 500개 (임계값 결정을 위한 검증용)
    # - Test : 정상 데이터 500개 + 이상 데이터 500개 = 총 1,000개
    X_train_normal = X_normal_raw[:2000]
    X_val_normal   = X_normal_raw[2000:2500]
    X_test_normal  = X_normal_raw[2500:]
    
    # 표준화 (StandardScaler) - Train 정상 데이터 기준으로만 fit
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_normal)
    X_val   = scaler.transform(X_val_normal)
    
    # 테스트 세트 구성 (정상 500개 + 이상 500개)
    X_test_combined = np.vstack([scaler.transform(X_test_normal), scaler.transform(X_anomaly_raw)])
    y_test_true = np.array([0] * len(X_test_normal) + [1] * len(X_anomaly_raw)) # 0: 정상, 1: 이상
    
    print(f"▶ 학습 데이터 (정상만 학습)  : {X_train.shape} (라벨 없음 / 비지도학습)")
    print(f"▶ 검증 데이터 (임계값 산출용): {X_val.shape} (정상 500개)")
    print(f"▶ 테스트 데이터 (최종 검증) : {X_test_combined.shape} (정상 500개 + 이상 500개)")

    # ============================================================
    # 2. 오토인코더 모델 학습 (오직 정상 데이터로만 진행)
    # ============================================================
    print("\n" + "=" * 80)
    print(" 2. 정상 데이터 전용 오토인코더 모델 학습")
    print("=" * 80)
    
    autoencoder = build_autoencoder(input_dim=50)
    autoencoder.summary()
    
    history = autoencoder.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=30,
        batch_size=64,
        verbose=1
    )

    # ============================================================
    # 3. 임계값(Threshold, tau) 결정 및 산출 근거
    # ============================================================
    print("\n" + "=" * 80)
    print(" 3. 통계적 이상치 판단 임계값(Threshold tau) 결정")
    print("=" * 80)
    
    # 정상 검증 데이터(X_val)에 대한 재구성 오차(MSE) 계산
    val_recon = autoencoder.predict(X_val, verbose=0)
    val_mse = np.mean(np.square(X_val - val_recon), axis=1) # 각 샘플별 MSE
    
    val_mean = np.mean(val_mse)
    val_std = np.std(val_mse)
    val_95th = np.percentile(val_mse, 95)
    
    # [핵심] 임계값 설정 공식: 평균 + 3 * 표준편차 (정규분포 99.7% 신뢰구간)
    threshold = val_mean + 3 * val_std
    
    print(f"▶ 정상 검증 데이터 평균 재구성 오차 (Mean) : {val_mean:.5f}")
    print(f"▶ 정상 검증 데이터 오차 표준편차 (Std)     : {val_std:.5f}")
    print(f"▶ 95 백분위수 (95th Percentile)             : {val_95th:.5f}")
    print(f"▶ 최종 채택 임계값 (tau = Mean + 3*Std)   : {threshold:.5f}")
    print("-" * 80)
    print("[임계값 설정 이유]")
    print(f" 1) 정상 데이터의 재구성 오차는 평균 {val_mean:.4f} 부근에 조밀하게 밀집되어 있습니다.")
    print(f" 2) 3-시그마(3*sigma) 규칙에 따라, '평균 + 3*표준편차'({threshold:.4f})를 초과하는 오차는")
    print("    정상 패턴에서 발생할 확률이 0.3% 미만(통계적 극단치)이므로 '이상치'로 판정합니다.")

    # ============================================================
    # 4. 테스트 데이터셋 이상 탐지 평가
    # ============================================================
    print("\n" + "=" * 80)
    print(" 4. 테스트 데이터 이상 탐지 평가 (정상 500 + 이상 500)")
    print("=" * 80)
    
    test_recon = autoencoder.predict(X_test_combined, verbose=0)
    test_mse = np.mean(np.square(X_test_combined - test_recon), axis=1)
    
    # 임계값 초과 시 이상(1), 이하 시 정상(0) 판정
    y_pred = (test_mse > threshold).astype(int)
    
    print("\n[분류 성능 평가 지표 (Classification Report)]")
    print(classification_report(y_test_true, y_pred, target_names=['정상(Normal)', '이상(Anomaly)'], digits=4))
    
    cm = confusion_matrix(y_test_true, y_pred)
    print(f"[혼동 행렬 (Confusion Matrix)]")
    print(f" - 정상 예측 성공(TN): {cm[0,0]}개 | 정상을 이상으로 오탐(FP): {cm[0,1]}개")
    print(f" - 이상을 정상으로 미탐(FN): {cm[1,0]}개 | 이상 탐지 성공(TP): {cm[1,1]}개")
    print(f" - 최종 F1-Score: {f1_score(y_test_true, y_pred):.4f}")

    # ============================================================
    # 5. 이상 탐지 원리 시각화 (신호 복원 차이 & 오차 분포 히스토그램)
    # ============================================================
    fig = plt.figure(figsize=(16, 9))
    
    # 5-1. 정상 샘플의 복원 비교 (잘 복원됨)
    sample_normal = X_test_combined[10] # 정상 샘플
    recon_normal  = test_recon[10]
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(sample_normal, label='원본 신호 (x)', color='#1E88E5', lw=2)
    ax1.plot(recon_normal, label='복원 신호 (x̂)', color='#43A047', linestyle='--', lw=2)
    ax1.set_title(f"1. 정상 샘플: 복원 성공 (오차 MSE: {test_mse[10]:.4f} <= tau)", fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 5-2. 이상 샘플의 복원 비교 (복원 실패하여 큰 차이 발생)
    sample_anomaly = X_test_combined[510] # 이상 샘플 (스파이크/왜곡)
    recon_anomaly  = test_recon[510]
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(sample_anomaly, label='이상 신호 (x)', color='#E53935', lw=2)
    ax2.plot(recon_anomaly, label='복원 시도 (x̂)', color='gray', linestyle='--', lw=2)
    ax2.set_title(f"2. 이상 샘플: 복원 실패 (오차 MSE: {test_mse[510]:.4f} > tau)", fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    # 5-3. 테스트 데이터 샘플별 재구성 오차 산점도
    ax3 = plt.subplot(2, 2, 3)
    ax3.scatter(range(500), test_mse[:500], color='#1E88E5', alpha=0.6, label='실제 정상 샘플', s=15)
    ax3.scatter(range(500, 1000), test_mse[500:], color='#E53935', alpha=0.6, label='실제 이상 샘플', s=15)
    ax3.axhline(threshold, color='black', linestyle='--', lw=2, label=f'임계값 tau ({threshold:.3f})')
    ax3.set_title("3. 샘플별 재구성 오차(MSE) 및 판정선", fontsize=11, fontweight='bold')
    ax3.set_xlabel("샘플 번호")
    ax3.set_ylabel("재구성 오차 (MSE)")
    ax3.set_yscale('log') # 오차 편차가 크므로 log scale 적용
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.5)

    # 5-4. 재구성 오차 분포 히스토그램 (정상 vs 이상)
    ax4 = plt.subplot(2, 2, 4)
    ax4.hist(test_mse[:500], bins=30, color='#1E88E5', alpha=0.6, label='정상 데이터 오차')
    ax4.hist(test_mse[500:], bins=30, color='#E53935', alpha=0.6, label='이상 데이터 오차')
    ax4.axvline(threshold, color='black', linestyle='--', lw=2, label=f'임계값 tau = {threshold:.3f}')
    ax4.set_title("4. 재구성 오차 분포 비교 및 임계값 기준선", fontsize=11, fontweight='bold')
    ax4.set_xlabel("재구성 오차 (MSE)")
    ax4.set_ylabel("샘플 수 (빈도)")
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("오토인코더 기반 이상치 탐지(Anomaly Detection) 전체 프로세스 실습", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    out_dir = "C:/apps/mldl/dl/output/autoencoder_anomaly"
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "autoencoder_anomaly_result.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n▶ 결과 시각화 차트 저장 완료: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()
