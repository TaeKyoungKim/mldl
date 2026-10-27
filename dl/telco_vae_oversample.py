"""
Telco Churn - 정통 VAE (Variational Autoencoder) 기반 오버샘플링 파이프라인

[구현 특징]
1. V (Variational)의 핵심 수학적 메커니즘 완전 구현:
   - 인코더: 특징 추출 후 평균(mu)과 로그 분산(logvar)을 각각 예측
   - Reparameterization Trick: z = mu + sigma * epsilon (역전파 가능하게 표준 가우시안 샘플링)
   - VAE Loss: Reconstruction Loss (MSE) + KL Divergence (잠재공간을 N(0, 1)로 정규화)
2. 디코더를 통한 정통 확률적 샘플링 생성:
   - z ~ N(0, 1) 또는 소수 클래스 잠재 분포로부터 직접 고품질 합성 데이터(Synthetic Data) 생성
3. 오버샘플링 전후 Random Forest 분류 성능(ROC-AUC, Precision, Recall, F1) 비교 및 시각화
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 한글 및 마이너스 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 시드 고정
np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# 1. 데이터 로드 및 전처리
# ============================================================
data_path = "C:/apps/mldl/data/telco_churn_preprocessed.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"데이터 파일이 없습니다: {data_path}")

df = pd.read_csv(data_path)
X = df.drop("Churn", axis=1).values.astype(np.float32)
y = df["Churn"].values.astype(int)

# 데이터 누수 방지를 위한 Train/Test 선행 분할 (8:2)
X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=" * 80)
print(" 1. 원본 데이터 클래스 분포 (Train 세트 기준)")
print("=" * 80)
unique, counts = np.unique(y_train_raw, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Churn={u}: {c}개 ({c/len(y_train_raw)*100:.1f}%)")

# 소수 클래스 (Churn=1, 이탈 고객)만 추출
X_minority = X_train_raw[y_train_raw == 1]
X_majority = X_train_raw[y_train_raw == 0]

n_to_generate = len(X_majority) - len(X_minority)
print(f"▶ 균형을 맞추기 위한 VAE 합성 데이터 생성 목표: {n_to_generate}개")

INPUT_DIM  = X.shape[1]   # 19개 피처
HIDDEN_DIM = 32
LATENT_DIM = 8           # 잠재 공간 차원
BATCH_SIZE = 64
EPOCHS     = 250
LR         = 1e-3

# ============================================================
# 2. 정통 VAE (Variational Autoencoder) 모델 정의
# ============================================================
class VAE(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        # 공통 특징 추출층
        self.encoder_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # [V의 핵심 1] 잠재 변수의 확률 분포 파라미터 (평균 mu, 로그분산 logvar)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder (복원 생성기)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    # [V의 핵심 2] Reparameterization Trick: z = mu + sigma * epsilon
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder_fc(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar

# [V의 핵심 3] VAE 전용 복합 손실 함수: Reconstruction Loss + KL Divergence
def vae_loss_function(recon_x, x, mu, logvar, beta=0.1):
    # 1) 재구성 오차 (MSE)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    
    # 2) KL Divergence: 잠재분포 q(z|x)가 표준정규분포 N(0, I)와 얼마나 가까운지 측정
    # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    
    return recon_loss + beta * kl_div, recon_loss, kl_div

# ============================================================
# 3. 소수 클래스(Churn=1) 전용 VAE 학습
# ============================================================
print("\n" + "=" * 80)
print(" 2. 소수 클래스(Churn=1) 데이터를 활용한 VAE 모델 학습")
print("=" * 80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"학습 실행 디바이스: {device}")

tensor_min = torch.tensor(X_minority, dtype=torch.float32)
loader = DataLoader(TensorDataset(tensor_min), batch_size=BATCH_SIZE, shuffle=True)

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

model.train()
loss_history = []

for epoch in range(EPOCHS):
    total_loss, total_recon, total_kl = 0.0, 0.0, 0.0
    for (batch_x,) in loader:
        batch_x = batch_x.to(device)
        optimizer.zero_grad()
        
        recon_x, mu, logvar = model(batch_x)
        loss, r_loss, k_loss = vae_loss_function(recon_x, batch_x, mu, logvar)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += r_loss.item()
        total_kl += k_loss.item()
        
    avg_loss = total_loss / len(loader)
    loss_history.append(avg_loss)
    
    if (epoch + 1) % 50 == 0 or epoch == 0:
        print(f"  Epoch [{epoch+1:03d}/{EPOCHS}] Total Loss: {avg_loss:.6f} (Recon: {total_recon/len(loader):.6f}, KL: {total_kl/len(loader):.6f})")

# ============================================================
# 4. 정통 VAE 잠재 공간 샘플링을 통한 새 소수 데이터 생성
# ============================================================
print("\n" + "=" * 80)
print(" 3. VAE 정규 잠재 공간(Latent Space)에서 오버샘플링 데이터 생성")
print("=" * 80)

model.eval()
with torch.no_grad():
    # 학습된 소수 클래스의 mu, logvar를 구하여 해당 영역의 잠재 분포에서 샘플링
    tensor_min_dev = tensor_min.to(device)
    _, mu_all, logvar_all = model(tensor_min_dev)
    
    # 소수 클래스가 형성한 잠재 공간의 평균과 분산 추정
    mean_mu = mu_all.mean(dim=0)
    mean_std = torch.exp(0.5 * logvar_all).mean(dim=0)
    
    # [VAE 정통 샘플링] N(mean_mu, mean_std^2) 분포로부터 z 생성
    z_sampled = mean_mu + torch.randn(n_to_generate, LATENT_DIM, device=device) * mean_std
    
    # Decoder를 통과시켜 새 이탈 고객 피처 복원
    X_synthetic = model.decoder(z_sampled).cpu().numpy()

print(f"▶ VAE로 생성된 합성 이탈 고객 데이터 수: {X_synthetic.shape[0]}개 (Shape: {X_synthetic.shape})")

# ============================================================
# 5. 오버샘플링된 균형 학습 세트 구축
# ============================================================
X_train_resampled = np.vstack([X_train_raw, X_synthetic])
y_train_resampled = np.concatenate([y_train_raw, np.ones(n_to_generate, dtype=int)])

print("\n[오버샘플링 후 Train 세트 클래스 분포]")
u_res, c_res = np.unique(y_train_resampled, return_counts=True)
for u, c in zip(u_res, c_res):
    print(f"  Churn={u}: {c}개 ({c/len(y_train_resampled)*100:.1f}%) -> 완벽한 50:50 균형 달성!")

# ============================================================
# 6. 머신러닝 분류기(Random Forest)를 통한 성능 평가 대조
# ============================================================
print("\n" + "=" * 80)
print(" 4. 원본 불균형 데이터 vs VAE 오버샘플링 데이터 분류 성능 비교")
print("=" * 80)

# 1) 원본 데이터로 학습한 모델
rf_orig = RandomForestClassifier(n_estimators=100, random_state=42)
rf_orig.fit(X_train_raw, y_train_raw)
y_pred_orig = rf_orig.predict(X_test)
y_prob_orig = rf_orig.predict_proba(X_test)[:, 1]
auc_orig = roc_auc_score(y_test, y_prob_orig)

# 2) VAE 오버샘플링 데이터로 학습한 모델
rf_vae = RandomForestClassifier(n_estimators=100, random_state=42)
rf_vae.fit(X_train_resampled, y_train_resampled)
y_pred_vae = rf_vae.predict(X_test)
y_prob_vae = rf_vae.predict_proba(X_test)[:, 1]
auc_vae = roc_auc_score(y_test, y_prob_vae)

print("[A. 원본 불균형 데이터로 학습 시 (Baseline)]")
print(classification_report(y_test, y_pred_orig, target_names=["유지 (No Churn)", "이탈 (Churn)"], digits=4))
print(f"ROC-AUC: {auc_orig:.4f}")

print("\n" + "-" * 80)
print("[B. VAE 오버샘플링 데이터로 학습 시 (Proposed)]")
print(classification_report(y_test, y_pred_vae, target_names=["유지 (No Churn)", "이탈 (Churn)"], digits=4))
print(f"ROC-AUC: {auc_vae:.4f}")
print("-" * 80)

# ============================================================
# 7. 결과 시각화 (학습 손실 곡선 및 피처 분포 비교)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 7-1. VAE 학습 Loss 추이
ax1.plot(loss_history, color='#1E88E5', lw=2)
ax1.set_title("1. VAE 학습 손실 곡선 (Reconstruction + KL Loss)", fontsize=11, fontweight='bold')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True, linestyle='--', alpha=0.5)

# 7-2. 실제 이탈 고객 vs VAE 합성 고객의 주요 피처 분포 비교 (예: 첫 번째 피처)
ax2.hist(X_minority[:, 0], bins=25, alpha=0.5, color='#43A047', label='실제 이탈 고객 (Real Churn)', density=True)
ax2.hist(X_synthetic[:, 0], bins=25, alpha=0.5, color='#E53935', label='VAE 합성 고객 (Synthetic)', density=True)
ax2.set_title("2. 실제 이탈 고객 vs VAE 합성 고객 피처 분포 일치성", fontsize=11, fontweight='bold')
ax2.set_xlabel("정규화된 특성값 (Feature #1)")
ax2.set_ylabel("확률 밀도 (Density)")
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.5)

plt.suptitle("Telco Churn - 정통 VAE(Variational Autoencoder) 오버샘플링 및 성능 검증", fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()

out_dir = "C:/apps/mldl/dl/output/vae_oversample"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "vae_oversampling_result.png")
plt.savefig(out_path, dpi=150)
print(f"\n▶ 시각화 결과 저장 완료: {out_path}")
plt.show()

if __name__ == "__main__":
    pass
