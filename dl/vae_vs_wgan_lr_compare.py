"""
Telco Churn 정형 데이터 오버샘플링 비교 파이프라인
[비교 대상]
1. Baseline: 원본 불균형 데이터 Logistic Regression
2. VAE (Variational Autoencoder): 잠재공간 확률적 샘플링 증강
3. WGAN-GP (Wasserstein GAN with Gradient Penalty):
   - 정형(Tabular) 수치 데이터에 가장 적합한 GAN 아키텍처
   - 모드 붕괴(Mode Collapse) 및 기울기 소실을 방지하여 안정적인 소수 클래스 데이터 합성
4. Class-Weighted LR: 가중치 조정 기준 모델

실행:
    python vae_vs_wgan_lr_compare.py
"""

import os
import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. VAE 모델 정의
# ============================================================
class VAE(nn.Module):
    def __init__(self, n_features, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        self.mu = nn.Linear(16, latent_dim)
        self.logvar = nn.Linear(16, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, n_features)
        )

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.mu(h), self.logvar(h).clamp(-10, 10)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decoder(z), mu, logvar

def train_vae(x_minority, seed=42, epochs=150, latent_dim=4):
    x_fit, x_stop = train_test_split(x_minority, test_size=0.2, random_state=seed)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_fit, dtype=torch.float32)),
        batch_size=32, shuffle=True,
        generator=torch.Generator().manual_seed(seed)
    )
    model = VAE(x_minority.shape[1], latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        beta_now = 0.1 * min(1.0, (epoch + 1) / 30)
        for (batch,) in loader:
            recon, mu, logvar = model(batch)
            recon_loss = (recon - batch).square().sum(1).mean()
            kl_loss = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean()
            loss = recon_loss + beta_now * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

# ============================================================
# 2. WGAN-GP (정형 데이터에 최적화된 GAN) 모델 정의
# ============================================================
class TabularGenerator(nn.Module):
    def __init__(self, latent_dim, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, n_features)
        )

    def forward(self, z):
        return self.net(z)

class TabularCritic(nn.Module):
    """WGAN의 비평자(Critic / Discriminator)"""
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1)  # WGAN은 시그모이드 없이 실수 스코어 출력
        )

    def forward(self, x):
        return self.net(x)

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """WGAN-GP의 립시츠 연속성 제약을 위한 그래디언트 패널티 계산"""
    alpha = torch.rand((real_samples.size(0), 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates)
    fake = torch.ones((real_samples.size(0), 1), device=device)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_wgan_gp(x_minority, seed=42, epochs=150, latent_dim=8, lambda_gp=10.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = x_minority.shape[1]
    
    generator = TabularGenerator(latent_dim, n_features).to(device)
    critic = TabularCritic(n_features).to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=1e-3, betas=(0.5, 0.9))
    opt_c = torch.optim.Adam(critic.parameters(), lr=1e-3, betas=(0.5, 0.9))

    dataset = TensorDataset(torch.tensor(x_minority, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(seed))

    for epoch in range(epochs):
        for (real_batch,) in loader:
            real_batch = real_batch.to(device)
            batch_size = real_batch.size(0)

            # 1) Train Critic (WGAN 손실 + GP)
            opt_c.zero_grad()
            z = torch.randn((batch_size, latent_dim), device=device)
            fake_batch = generator(z)

            critic_real = critic(real_batch)
            critic_fake = critic(fake_batch.detach())

            gp = compute_gradient_penalty(critic, real_batch, fake_batch.detach(), device)
            loss_c = -(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_gp * gp

            loss_c.backward()
            opt_c.step()

            # 2) Train Generator
            opt_g.zero_grad()
            gen_fake = critic(fake_batch)
            loss_g = -torch.mean(gen_fake)

            loss_g.backward()
            opt_g.step()

    return generator

# ============================================================
# 3. 모델 평가 헬퍼 함수
# ============================================================
def evaluate(name, y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        'model': name,
        'threshold': threshold,
        'accuracy': accuracy_score(y_true, pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, pred),
        'precision': precision_score(y_true, pred, zero_division=0),
        'recall': recall_score(y_true, pred, zero_division=0),
        'f1': f1_score(y_true, pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, prob),
        'average_precision': average_precision_score(y_true, prob),
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)
    }

# ============================================================
# 4. 메인 파이프라인
# ============================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_csv = Path("C:/apps/mldl/data/telco_churn_preprocessed.csv")
    parser.add_argument('--csv', type=Path, default=default_csv)
    parser.add_argument('--target', default='Churn')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--out', type=Path, default=Path('C:/apps/mldl/dl/output/vae_vs_wgan'))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" 1. Telco Churn 데이터 로드 및 Train / Validation / Test 분할")
    print("=" * 80)
    
    frame = pd.read_csv(args.csv)
    y = frame.pop(args.target).to_numpy(dtype=int)
    X = frame.to_numpy(dtype=float)
    feature_names = frame.columns.tolist()

    # 60:20:20 분할
    indices = np.arange(len(y))
    train_idx, rest_idx = train_test_split(indices, test_size=0.4, stratify=y, random_state=args.seed)
    val_idx, test_idx = train_test_split(rest_idx, test_size=0.5, stratify=y[rest_idx], random_state=args.seed)

    scaler = StandardScaler().fit(X[train_idx])
    x_train, x_val, x_test = [scaler.transform(X[ix]).astype(np.float32) for ix in [train_idx, val_idx, test_idx]]
    y_train, y_val, y_test = [y[ix] for ix in [train_idx, val_idx, test_idx]]

    x_minority = x_train[y_train == 1]
    n_majority = (y_train == 0).sum()
    n_minority = (y_train == 1).sum()
    n_to_generate = n_majority - n_minority

    print(f"▶ 학습 세트 크기 : 총 {len(y_train)}개 (정상: {n_majority}개 vs 이탈: {n_minority}개)")
    print(f"▶ 1:1 균형 생성을 위한 목표 합성 샘플 수: {n_to_generate}개")

    # ============================================================
    # 2. VAE vs WGAN-GP 학습 및 데이터 생성
    # ============================================================
    print("\n" + "=" * 80)
    print(" 2. VAE 및 WGAN-GP 모델 학습 및 소수 클래스 데이터 합성")
    print("=" * 80)

    # (1) VAE 학습 및 샘플링
    print("▶ [1/2] VAE 모델 학습 중...")
    vae_model = train_vae(x_minority, seed=args.seed, epochs=args.epochs, latent_dim=4)
    with torch.no_grad():
        z_vae = torch.randn(n_to_generate, 4)
        x_synthetic_vae = vae_model.decoder(z_vae).cpu().numpy()

    # (2) WGAN-GP 학습 및 샘플링
    print("▶ [2/2] WGAN-GP 모델 학습 중 (정형 데이터 안정적 적대적 생성)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wgan_generator = train_wgan_gp(x_minority, seed=args.seed, epochs=args.epochs, latent_dim=8)
    with torch.no_grad():
        z_wgan = torch.randn(n_to_generate, 8, device=device)
        x_synthetic_wgan = wgan_generator(z_wgan).cpu().numpy()

    print("▶ 두 생성 모델의 오버샘플링 데이터 생성 완료!")

    # 데이터셋 구성
    x_aug_vae = np.vstack([x_train, x_synthetic_vae])
    y_aug_vae = np.concatenate([y_train, np.ones(n_to_generate, dtype=int)])

    x_aug_wgan = np.vstack([x_train, x_synthetic_wgan])
    y_aug_wgan = np.concatenate([y_train, np.ones(n_to_generate, dtype=int)])

    # ============================================================
    # 3. 로지스틱 회귀 모델 학습 및 테스트 세트 평가 대조
    # ============================================================
    print("\n" + "=" * 80)
    print(" 3. 분류기(Logistic Regression) 성능 비교 평가 (동일한 Unseen Test Set)")
    print("=" * 80)

    models_to_test = {
        'Baseline_LR': LogisticRegression(max_iter=1000, random_state=args.seed).fit(x_train, y_train),
        'ClassWeighted_LR': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=args.seed).fit(x_train, y_train),
        'VAE_Oversampled_LR': LogisticRegression(max_iter=1000, random_state=args.seed).fit(x_aug_vae, y_aug_vae),
        'WGAN_Oversampled_LR': LogisticRegression(max_iter=1000, random_state=args.seed).fit(x_aug_wgan, y_aug_wgan),
    }

    results = []
    predictions = {}
    for name, clf in models_to_test.items():
        prob = clf.predict_proba(x_test)[:, 1]
        predictions[name] = prob
        results.append(evaluate(name, y_test, prob, threshold=0.5))

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(out / 'metrics_summary.csv', index=False)

    print("\n" + metrics_df[['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision', 'TP', 'FN']].to_string(index=False))

    # ============================================================
    # 4. 시각화 (PR Curve & PCA 잠재 공간 분포 대조)
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Precision-Recall Curve
    for name, prob in predictions.items():
        prec, rec, _ = precision_recall_curve(y_test, prob)
        ap = average_precision_score(y_test, prob)
        axes[0].plot(rec, prec, label=f'{name} (AP={ap:.3f})', lw=1.5)
    axes[0].axhline(y_test.mean(), ls='--', color='gray', label='Random baseline')
    axes[0].set_title("1. Precision-Recall Curve (PR 곡선)", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Recall (재현율)")
    axes[0].set_ylabel("Precision (정밀도)")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # 2) PCA 2D 투영: VAE 합성 데이터 분포
    pca = PCA(n_components=2).fit(x_train)
    real_2d = pca.transform(x_train)
    vae_2d = pca.transform(x_synthetic_vae)
    wgan_2d = pca.transform(x_synthetic_wgan)

    axes[1].scatter(*real_2d[y_train == 0].T, s=8, alpha=0.15, color='gray', label='Original No-Churn (0)')
    axes[1].scatter(*real_2d[y_train == 1].T, s=15, alpha=0.7, color='red', label='Original Churn (1)')
    axes[1].scatter(*vae_2d.T, s=10, alpha=0.3, color='blue', label='VAE Synthetic Churn')
    axes[1].set_title("2. VAE 생성 데이터 공간 분포 (PCA)", fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # 3) PCA 2D 투영: WGAN-GP 합성 데이터 분포
    axes[2].scatter(*real_2d[y_train == 0].T, s=8, alpha=0.15, color='gray', label='Original No-Churn (0)')
    axes[2].scatter(*real_2d[y_train == 1].T, s=15, alpha=0.7, color='red', label='Original Churn (1)')
    axes[2].scatter(*wgan_2d.T, s=10, alpha=0.3, color='green', label='WGAN-GP Synthetic Churn')
    axes[2].set_title("3. WGAN-GP 생성 데이터 공간 분포 (PCA)", fontsize=11, fontweight='bold')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("Telco Churn: VAE vs WGAN-GP 오버샘플링 성능 및 생성 분포 비교", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    fig_path = out / 'vae_vs_wgan_comparison.png'
    plt.savefig(fig_path, dpi=160)
    plt.close()

    print(f"\n▶ 비교 결과 차트 저장 완료: {fig_path.resolve()}")
    print(f"▶ 지표 요약 CSV 저장 완료: {(out / 'metrics_summary.csv').resolve()}\n")

if __name__ == '__main__':
    main()
