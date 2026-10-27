"""VAE 소수 클래스 증강 전후 Logistic Regression 비교.
실행: python vae_lr_compare.py
실제 수치형 데이터: python vae_lr_compare.py --csv my_data.csv --target target
설치: pip install numpy pandas scikit-learn matplotlib torch
"""
import argparse
import copy
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
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class VAE(nn.Module):
    """연속형 수치 특성용 VAE: 선형 decoder 출력 + MSE 재구성 손실."""
    def __init__(self, n_features, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_features, 32), nn.ReLU(),
                                     nn.Linear(32, 16), nn.ReLU())
        self.mu = nn.Linear(16, latent_dim)
        self.logvar = nn.Linear(16, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 16), nn.ReLU(),
                                    nn.Linear(16, 32), nn.ReLU(),
                                    nn.Linear(32, n_features))

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.mu(h), self.logvar(h).clamp(-10, 10)
        # Reparameterization trick: 이 경로를 통해 encoder에도 역전파된다.
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decoder(z), mu, logvar


def train_vae(x_minority, seed=42, epochs=250, beta=0.1, latent_dim=4):
    """VAE 내부 검증도 소수 클래스 학습 표본 안에서만 분리한다."""
    x_fit, x_stop = train_test_split(x_minority, test_size=0.2, random_state=seed)
    loader = DataLoader(TensorDataset(torch.tensor(x_fit, dtype=torch.float32)),
                        batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    stop_tensor = torch.tensor(x_stop, dtype=torch.float32)
    model = VAE(x_minority.shape[1], latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # 고정 epsilon으로 epoch 간 내부 검증 손실 비교의 잡음을 줄인다.
    stop_eps = torch.randn((len(x_stop), latent_dim),
                           generator=torch.Generator().manual_seed(seed + 1))
    best_loss, best_state = float('inf'), None
    history = []
    for epoch in range(epochs):
        model.train()
        beta_now = beta * min(1.0, (epoch + 1) / 50)
        total_recon, total_kl = 0.0, 0.0
        for (batch,) in loader:
            reconstructed, mu, logvar = model(batch)
            recon = (reconstructed - batch).square().sum(1).mean()
            kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean()
            loss = recon + beta_now * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_recon += recon.item() * len(batch)
            total_kl += kl.item() * len(batch)
        model.eval()
        with torch.no_grad():
            h = model.encoder(stop_tensor)
            mu, logvar = model.mu(h), model.logvar(h).clamp(-10, 10)
            pred = model.decoder(mu + (0.5 * logvar).exp() * stop_eps)
            v_recon = (pred - stop_tensor).square().sum(1).mean()
            v_kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean()
            val_loss = (v_recon + beta * v_kl).item()
        history.append({'epoch': epoch + 1,
                        'train_reconstruction': total_recon / len(x_fit),
                        'train_kl': total_kl / len(x_fit), 'vae_val_loss': val_loss})
        if val_loss < best_loss:
            best_loss, best_state = val_loss, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history)


def choose_threshold(y_val, p_val):
    """평가용 test를 보지 않고 validation F1이 최대인 임계값을 선택."""
    precision, recall, thresholds = precision_recall_curve(y_val, p_val)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.argmax(f1)])


def evaluate(name, y, prob, threshold, policy):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'model': name, 'threshold_policy': policy, 'threshold': threshold,
            'accuracy': accuracy_score(y, pred),
            'balanced_accuracy': balanced_accuracy_score(y, pred),
            'precision': precision_score(y, pred, zero_division=0),
            'recall': recall_score(y, pred, zero_division=0),
            'f1': f1_score(y, pred, zero_division=0),
            'roc_auc': roc_auc_score(y, prob),
            'average_precision': average_precision_score(y, prob),
            'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}


def main(argv=None):
    default_csv = Path("C:/apps/mldl/data/telco_churn_preprocessed.csv")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=default_csv if default_csv.exists() else None, help="기본값: C:/apps/mldl/data/telco_churn_preprocessed.csv")
    parser.add_argument('--target', default='Churn', help="타깃 컬럼명 (기본값: Churn)")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--out', type=Path, default=Path('results'))
    args = parser.parse_args(argv)
    if args.epochs < 1:
        raise ValueError('epochs는 1 이상이어야 합니다.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # 1. 95:5에 가까운 이진분류 예제. 실제 데이터는 수치형 CSV로 교체 가능.
    if args.csv is None:
        X, y = make_classification(n_samples=6000, n_features=12,
            n_informative=8, n_redundant=2, n_clusters_per_class=2,
            weights=[0.95, 0.05], class_sep=1.0, flip_y=0.005,
            random_state=args.seed)
        columns = [f'x{i + 1}' for i in range(X.shape[1])]
    else:
        frame = pd.read_csv(args.csv)
        y = frame.pop(args.target).to_numpy()
        X = frame.to_numpy(dtype=float)
        columns = frame.columns.tolist()
    if set(np.unique(y)) != {0, 1} or not np.isfinite(X).all():
        raise ValueError('target은 0/1이며, 특성은 결측치 없는 유한한 수치여야 합니다.')
    if min(np.bincount(y.astype(int))) < 30:
        raise ValueError('예제 분할에는 각 클래스가 최소 30개 필요합니다. VAE에는 더 많은 표본을 권장합니다.')
    if (y == 1).sum() >= (y == 0).sum():
        raise ValueError('이 코드는 증강할 소수 클래스가 target=1이라고 가정합니다.')

    # 2. 먼저 train/validation/test = 60/20/20 분할. test는 마지막 평가에만 사용.
    indices = np.arange(len(y))
    train_idx, rest_idx = train_test_split(indices, test_size=0.4,
                                          stratify=y, random_state=args.seed)
    val_idx, test_idx = train_test_split(rest_idx, test_size=0.5,
                                        stratify=y[rest_idx], random_state=args.seed)
    scaler = StandardScaler().fit(X[train_idx])
    x_train, x_val, x_test = [scaler.transform(X[ix]).astype(np.float32)
                            for ix in [train_idx, val_idx, test_idx]]
    y_train, y_val, y_test = [y[ix].astype(int) for ix in [train_idx, val_idx, test_idx]]

    # 3. train의 소수 클래스만으로 VAE 학습. 추가 생성으로 train만 1:1로 조정.
    vae, history = train_vae(x_train[y_train == 1], args.seed, args.epochs)
    n_new = int((y_train == 0).sum() - (y_train == 1).sum())
    with torch.no_grad():
        z = torch.randn(n_new, 4)
        x_synthetic = vae.decoder(z).numpy()
    x_aug = np.vstack([x_train, x_synthetic])
    y_aug = np.concatenate([y_train, np.ones(n_new, dtype=int)])
    assert set(train_idx).isdisjoint(val_idx) and set(train_idx).isdisjoint(test_idx)
    assert set(val_idx).isdisjoint(test_idx)
    assert np.bincount(y_aug)[0] == np.bincount(y_aug)[1]

    # 4. 같은 LR 설정. class_weight 방법은 저비용 비교 기준으로 추가.
    specs = [('Original_LR', x_train, y_train, None),
             ('VAE_LR', x_aug, y_aug, None),
             ('Weighted_LR', x_train, y_train, 'balanced')]
    rows, predictions = [], {}
    for name, x_fit, y_fit, weight in specs:
        clf = LogisticRegression(max_iter=2000, class_weight=weight,
                                 random_state=args.seed)
        clf.fit(x_fit, y_fit)
        p_val = clf.predict_proba(x_val)[:, 1]
        p_test = clf.predict_proba(x_test)[:, 1]
        predictions[name] = p_test
        for policy, threshold in [('fixed_0.5', 0.5),
                                  ('validation_f1', choose_threshold(y_val, p_val))]:
            rows.append(evaluate(name, y_test, p_test, threshold, policy))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / 'metrics.csv', index=False)
    pd.DataFrame({'y_true': y_test, **predictions}).to_csv(out / 'test_predictions.csv', index=False)
    history.to_csv(out / 'vae_training_history.csv', index=False)

    # 5. 실제 단위로 되돌린 원본/증강 데이터 + 분할표도 저장.
    original = pd.DataFrame(X, columns=columns)
    original['target'] = y
    original['split'] = ''
    for label, ix in [('train', train_idx), ('validation', val_idx), ('test', test_idx)]:
        original.loc[ix, 'split'] = label
    original.to_csv(out / 'original_data.csv', index=False)
    augmented = pd.DataFrame(scaler.inverse_transform(x_aug), columns=columns)
    augmented['target'] = y_aug
    augmented['is_synthetic'] = np.arange(len(x_aug)) >= len(x_train)
    augmented.to_csv(out / 'augmented_train_data.csv', index=False)
    counts = pd.DataFrame([
        {'split': label, 'class_0': int((labels == 0).sum()),
         'class_1': int((labels == 1).sum())}
        for label, labels in [('train_before', y_train), ('train_after_vae', y_aug),
                              ('validation', y_val), ('test', y_test)]])
    counts.to_csv(out / 'class_counts.csv', index=False)
    metadata = {'seed': args.seed, 'epochs': args.epochs,
        'generated_samples': n_new, 'test_positive_rate': float(y_test.mean()),
        'torch_version': torch.__version__,
        'note': 'AP is average precision, not trapezoidal PR AUC. Single split demo.'}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    # 6. PR 곡선: 불균형 test의 양성 비율이 무작위 기준선.
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, prob in predictions.items():
        prec, rec, _ = precision_recall_curve(y_test, prob)
        ap = average_precision_score(y_test, prob)
        ax.plot(rec, prec, label=f'{name} (AP={ap:.3f})')
    ax.axhline(y_test.mean(), ls='--', color='gray', label='Random baseline')
    ax.set(xlabel='Recall', ylabel='Precision', title='Same untouched test set',
           xlim=(0, 1), ylim=(0, 1.02))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'precision_recall.png', dpi=160)
    plt.close(fig)

    # PCA도 원본 train에만 fit. 분포 진단용이며 생성 품질의 증명은 아니다.
    pca = PCA(n_components=2).fit(x_train)
    real_2d, synthetic_2d = pca.transform(x_train), pca.transform(x_synthetic)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    for ax in axes:
        ax.scatter(*real_2d[y_train == 0].T, s=8, alpha=.15, label='Original class 0')
        ax.scatter(*real_2d[y_train == 1].T, s=15, alpha=.7, label='Original class 1')
        ax.set(xlabel='PC1', ylabel='PC2')
    axes[1].scatter(*synthetic_2d.T, s=8, alpha=.2, label='VAE synthetic class 1')
    axes[0].set_title('Before augmentation')
    axes[1].set_title('After augmentation (train only)')
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'data_distribution.png', dpi=160)
    plt.close(fig)
    print('\nCLASS COUNTS\n', counts.to_string(index=False))
    print('\nTEST METRICS\n', metrics.round(4).to_string(index=False))
    print('\nSaved:', out.resolve())
    return metrics


if __name__ == '__main__':
    main()
