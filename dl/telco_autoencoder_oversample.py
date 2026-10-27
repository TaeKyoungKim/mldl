"""
Telco Churn - Autoencoder 기반 오버샘플링
불균형: Churn=0 (73.5%) vs Churn=1 (26.5%)
소수 클래스(Churn=1)를 오토인코더로 생성하여 균형을 맞춤
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ── 1. 데이터 로드 ────────────────────────────────────────────────
df = pd.read_csv("C:/apps/mldl/data/telco_churn_preprocessed.csv")

X = df.drop("Churn", axis=1).values.astype(np.float32)
y = df["Churn"].values

print("=== 원본 클래스 분포 ===")
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Churn={u}: {c} ({c/len(y)*100:.1f}%)")

# ── 2. 소수 클래스 추출 ───────────────────────────────────────────
X_minority = X[y == 1]          # Churn=1 (1869개)
X_majority  = X[y == 0]

n_to_generate = len(X_majority) - len(X_minority)   # 부족한 수만큼 생성
print(f"\n생성 목표: {n_to_generate}개")

INPUT_DIM  = X.shape[1]    # 19
LATENT_DIM = 8
HIDDEN_DIM = 32
EPOCHS     = 200
BATCH_SIZE = 64
LR         = 1e-3

# ── 3. 오토인코더 정의 ────────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, LATENT_DIM),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, INPUT_DIM),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

# ── 4. 소수 클래스만으로 오토인코더 학습 ─────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n학습 디바이스: {device}")

tensor_min = torch.tensor(X_minority, dtype=torch.float32)
loader = DataLoader(TensorDataset(tensor_min), batch_size=BATCH_SIZE, shuffle=True)

model = Autoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

losses = []
model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for (batch,) in loader:
        batch = batch.to(device)
        recon, _ = model(batch)
        loss = criterion(recon, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(loader)
    losses.append(avg_loss)
    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS}  Loss: {avg_loss:.6f}")

# ── 5. 잠재 공간에서 노이즈 추가 → 새 샘플 생성 ─────────────────
model.eval()
with torch.no_grad():
    tensor_min_dev = tensor_min.to(device)
    _, z_real = model(tensor_min_dev)           # 실제 잠재 벡터
    z_real = z_real.cpu().numpy()

# 잠재 벡터의 평균·분산을 보존하며 샘플링
np.random.seed(42)
n_real   = len(z_real)
indices  = np.random.choice(n_real, size=n_to_generate, replace=True)
z_base   = z_real[indices]

# 잠재 공간 노이즈 스케일 = 실제 잠재 벡터 표준편차의 비율
noise_scale = 0.15
noise = np.random.normal(0, noise_scale * z_real.std(axis=0), size=z_base.shape)
z_synthetic = z_base + noise

with torch.no_grad():
    z_tensor  = torch.tensor(z_synthetic, dtype=torch.float32).to(device)
    X_synth   = model.decoder(z_tensor).cpu().numpy()

print(f"\n생성 완료: {X_synth.shape[0]}개")

# ── 6. 오버샘플링 데이터셋 구성 ───────────────────────────────────
X_resampled = np.vstack([X, X_synth])
y_resampled = np.concatenate([y, np.ones(n_to_generate, dtype=int)])

print("\n=== 오버샘플링 후 클래스 분포 ===")
unique, counts = np.unique(y_resampled, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Churn={u}: {c} ({c/len(y_resampled)*100:.1f}%)")

# ── 7. 분류 성능 비교 (원본 vs 오버샘플링) ────────────────────────
def evaluate(X_train, y_train, X_test, y_test, label):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"\n── {label} ──")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print(f"ROC-AUC: {auc:.4f}")
    return auc

X_tr_orig, X_te, y_tr_orig, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 오버샘플링 학습셋: 원본 train + 합성 샘플
X_tr_over = np.vstack([X_tr_orig, X_synth])
y_tr_over = np.concatenate([y_tr_orig, np.ones(n_to_generate, dtype=int)])

auc_orig = evaluate(X_tr_orig, y_tr_orig, X_te, y_te, "원본 데이터")
auc_over = evaluate(X_tr_over, y_tr_over, X_te, y_te, "오토인코더 오버샘플링")

# ── 8. 시각화 ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# (a) 학습 손실
axes[0].plot(losses, color="#4C72B0")
axes[0].set_title("Autoencoder Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].grid(True, alpha=0.3)

# (b) 클래스 분포 비교
labels = ["No Churn (0)", "Churn (1)"]
orig_counts = [5174, 1869]
over_counts = [5174, 1869 + n_to_generate]
x = np.arange(2)
w = 0.35
axes[1].bar(x - w/2, orig_counts, w, label="Original", color="#4C72B0")
axes[1].bar(x + w/2, over_counts, w, label="Oversampled", color="#DD8452")
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels)
axes[1].set_title("Class Distribution")
axes[1].set_ylabel("Count")
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis="y")

# (c) 잠재 공간: 실제 vs 합성 (첫 두 차원)
axes[2].scatter(z_real[:, 0], z_real[:, 1], alpha=0.4, s=10, label="Real minority", color="#4C72B0")
axes[2].scatter(z_synthetic[:, 0], z_synthetic[:, 1], alpha=0.3, s=6, label="Synthetic", color="#DD8452")
axes[2].set_title("Latent Space (dim 0 vs 1)")
axes[2].set_xlabel("Latent dim 0")
axes[2].set_ylabel("Latent dim 1")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle("Autoencoder Oversampling - Telco Churn", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("output/telco_autoencoder_oversample.png", dpi=150, bbox_inches="tight")
print("\n시각화 저장: output/telco_autoencoder_oversample.png")

print(f"\n=== 최종 요약 ===")
print(f"  원본 ROC-AUC     : {auc_orig:.4f}")
print(f"  오버샘플링 ROC-AUC: {auc_over:.4f}")
print(f"  개선폭           : {auc_over - auc_orig:+.4f}")
