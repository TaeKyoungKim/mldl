"""
Bank Marketing 데이터셋: TabDDPM(조건부 확산 모델) vs SMOTE vs ADASYN 오버샘플링 비교

파이프라인 요약:
1. Baseline: 불균형 원본 데이터 (No Oversampling)
2. SMOTENC / SMOTE: 전통적 최근접 이웃 선형 보간 오버샘플링
3. ADASYN: 분류하기 어려운 경계면 소수 샘플에 가중치를 두어 적응형 합성
4. Conditional TabDDPM (Tabular Denoising Diffusion Probabilistic Model):
   - 클래스 라벨(y=0, y=1)을 조건(Conditioning Embedding)으로 부여
   - 순방향 확산(Forward Noising) 및 역방향 디노이징(Reverse Denoising MLP) 학습
   - 라벨 y=1 조건부 샘플링을 통해 고품질의 소수 클래스 합성 데이터 생성
5. 공통 분류기(Random Forest)로 독립 Test Set(원본 불균형 20%)에서 평가
   - 핵심 비교 지표: PR-AUC (Average Precision) 및 Recall (재현율)
"""

import os
import sys
import argparse
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, balanced_accuracy_score,
    confusion_matrix, precision_recall_curve
)
from imblearn.over_sampling import SMOTE, ADASYN

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# 1. Conditional TabDDPM (조건부 테이블 확산 모델) 구현
# -----------------------------------------------------------------------------

class ConditionalDiffusionMLP(nn.Module):
    """
    라벨 조건부 노이즈 예측 네트워크 (Epsilon Predictor)
    입력: x_t (노이즈 낀 테이블 데이터), t (시간 임베딩), y (라벨 조건 임베딩)
    출력: 예측 노이즈 epsilon_theta(x_t, t, y)
    """
    def __init__(self, data_dim, time_dim=32, num_classes=2, class_emb_dim=16, hidden_dim=256):
        super().__init__()
        self.time_dim = time_dim
        
        # 라벨 임베딩
        self.class_emb = nn.Embedding(num_classes, class_emb_dim)
        
        # 시간 스텝 MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 메인 백본 MLP
        in_features = data_dim + hidden_dim + class_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, data_dim)
        )

    def sinusoidal_embedding(self, timesteps):
        """시간 스텝 Sinusoidal Positional Embedding"""
        half_dim = self.time_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

    def forward(self, x, t, y):
        t_emb = self.sinusoidal_embedding(t)
        t_feat = self.time_mlp(t_emb)
        y_feat = self.class_emb(y)
        feat = torch.cat([x, t_feat, y_feat], dim=1)
        return self.net(feat)


class ConditionalTabDDPM:
    """
    조건부 Tabular DDPM 스케줄러 및 샘플러
    """
    def __init__(self, data_dim, num_classes=2, timesteps=100, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.device = torch.device(device)
        self.timesteps = timesteps
        self.data_dim = data_dim
        
        # 선형 베타 스케줄링
        self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 모델 빌드
        self.model = ConditionalDiffusionMLP(data_dim=data_dim, num_classes=num_classes).to(self.device)

    def q_sample(self, x_0, t, noise=None):
        """순방향 노이즈 주입: q(x_t | x_0) = sqrt(alpha_bar_t)*x_0 + sqrt(1 - alpha_bar_t)*noise"""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    def train_ddpm(self, dataloader, epochs=20, lr=1e-3):
        """DDPM 훈련"""
        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        mse_loss = nn.MSELoss()

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                b_size = batch_x.size(0)

                # t ~ Uniform(0, T-1)
                t = torch.randint(0, self.timesteps, (b_size,), device=self.device).long()
                noise = torch.randn_like(batch_x)
                x_t = self.q_sample(batch_x, t, noise)

                optimizer.zero_grad()
                pred_noise = self.model(x_t, t, batch_y)
                loss = mse_loss(pred_noise, noise)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * b_size

            avg_loss = total_loss / len(dataloader.dataset)
            if epoch % 5 == 0 or epoch == 1:
                print(f"    [TabDDPM Epoch {epoch:02d}/{epochs}] Loss: {avg_loss:.5f}")

    @torch.no_grad()
    def sample_conditioned(self, num_samples, class_label=1):
        """조건부 역방향 디노이징 샘플링 (y=class_label로 유도)"""
        self.model.eval()
        x = torch.randn(num_samples, self.data_dim, device=self.device)
        y = torch.full((num_samples,), class_label, dtype=torch.long, device=self.device)

        for step in reversed(range(self.timesteps)):
            t = torch.full((num_samples,), step, dtype=torch.long, device=self.device)
            pred_noise = self.model(x, t, y)
            
            beta_t = self.betas[step]
            alpha_t = self.alphas[step]
            alpha_bar_t = self.alphas_cumprod[step]

            # DDPM 역방향 평균
            mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise)

            if step > 0:
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt(beta_t)
                x = mean + sigma_t * noise
            else:
                x = mean

        return x.cpu().numpy()


# -----------------------------------------------------------------------------
# 2. 성능 평가 및 유틸리티 함수
# -----------------------------------------------------------------------------

def evaluate_model(y_true, y_prob, threshold=0.5):
    """불균형 분류 지표 산출"""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "PR-AUC (AP)": average_precision_score(y_true, y_prob),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "Balanced Acc": balanced_accuracy_score(y_true, y_pred),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    }

def run_tabddpm_comparison(args):
    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print(" 1. Bank Marketing 데이터셋 전처리 및 독립 Test Set(20%) 분리")
    print("=" * 85)

    df = pd.read_csv(data_path, sep=";")
    print(f"▶ 데이터 파일: {data_path.resolve()}")
    print(f"▶ 전체 데이터: {len(df):,}행 x {df.shape[1]}열")

    y = df["y"].eq("yes").astype(int).to_numpy()
    X = df.drop(columns=["y", "duration"], errors="ignore").copy()

    if "pdays" in X:
        X["previously_contacted"] = X["pdays"].ne(-1).astype(int)
        X["pdays"] = X["pdays"].replace(-1, np.nan)

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.columns.difference(numeric_cols).tolist()

    # 독립 80:20 분할
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())

    print(f"▶ Train Set (학습 80%): {len(X_train_df):,}건 (No: {n_neg:,}건, Yes: {n_pos:,}건 / 가입률: {y_train.mean():.2%})")
    print(f"▶ Test Set  (독립평가 20%): {len(X_test_df):,}건 (No: {(y_test == 0).sum():,}건, Yes: {(y_test == 1).sum():,}건 / 가입률: {y_test.mean():.2%})")

    # 전처리 파이프라인
    preprocessor = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric_cols),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", sparse_output=False)), categorical_cols)
    ])

    X_train_proc = preprocessor.fit_transform(X_train_df)
    X_test_proc = preprocessor.transform(X_test_df)
    data_dim = X_train_proc.shape[1]
    print(f"▶ 원핫 인코딩 & 스케일링 후 특성 차원수: {data_dim}차원")

    # -------------------------------------------------------------------------
    # 2단계: 오버샘플링 기법 적용
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print(" 2. 오버샘플링 기법별 데이터 합성 및 학습 데이터 구축")
    print("=" * 85)

    # 생성할 목표 소수 샘플 수: 소수 클래스 비율을 다수의 약 33% (sampling_strategy=0.33) 수준으로 증강
    target_ratio = 0.33
    n_target_pos = int(n_neg * target_ratio)
    n_synthetic_needed = n_target_pos - n_pos

    print(f"▶ 증강 목표: 다수 클래스(0) 대비 약 {target_ratio*100:.0f}% 수준으로 증강 (합성 필요 소수 샘플 수: {n_synthetic_needed:,}건)")

    datasets = {}

    # 1. Baseline (미적용)
    datasets["1. Baseline (원본)"] = (X_train_proc, y_train, 0.0)

    # 2. SMOTE
    print("\n▶ [SMOTE] k-최근접 이웃 선형 보간 합성 중...")
    t0 = perf_counter()
    smote = SMOTE(sampling_strategy=target_ratio, random_state=args.seed)
    X_smote, y_smote = smote.fit_resample(X_train_proc, y_train)
    t_smote = perf_counter() - t0
    datasets["2. SMOTE (33%)"] = (X_smote, y_smote, t_smote)
    print(f"   완료 ({t_smote:.2f}초): 소수 샘플 {int((y_smote==1).sum()):,}건")

    # 3. ADASYN
    print("\n▶ [ADASYN] 경계면 적응형 가중 합성 중...")
    t0 = perf_counter()
    adasyn = ADASYN(sampling_strategy=target_ratio, random_state=args.seed)
    X_ada, y_ada = adasyn.fit_resample(X_train_proc, y_train)
    t_ada = perf_counter() - t0
    datasets["3. ADASYN (33%)"] = (X_ada, y_ada, t_ada)
    print(f"   완료 ({t_ada:.2f}초): 소수 샘플 {int((y_ada==1).sum()):,}건")

    # 4. Conditional TabDDPM
    print("\n▶ [Conditional TabDDPM] 라벨 조건부 확산 모델 학습 및 합성 중...")
    t0 = perf_counter()
    
    # PyTorch DataLoader 구성
    tensor_x = torch.tensor(X_train_proc, dtype=torch.float32)
    tensor_y = torch.tensor(y_train, dtype=torch.long)
    dataset = TensorDataset(tensor_x, tensor_y)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    tab_ddpm = ConditionalTabDDPM(data_dim=data_dim, timesteps=args.timesteps, device=args.device)
    tab_ddpm.train_ddpm(loader, epochs=args.epochs, lr=1e-3)

    print(f"   → 라벨 조건 y=1 (소수 클래스) {n_synthetic_needed:,}건 역방향 디노이징 샘플링 중...")
    synthetic_minority = tab_ddpm.sample_conditioned(num_samples=n_synthetic_needed, class_label=1)
    synthetic_minority_labels = np.ones(n_synthetic_needed, dtype=int)

    # 원본 Train 데이터와 결합
    X_ddpm = np.vstack([X_train_proc, synthetic_minority])
    y_ddpm = np.concatenate([y_train, synthetic_minority_labels])
    t_ddpm = perf_counter() - t0
    datasets["4. TabDDPM (조건부 생성 33%)"] = (X_ddpm, y_ddpm, t_ddpm)
    print(f"   완료 ({t_ddpm:.2f}초): 소수 샘플 {int((y_ddpm==1).sum()):,}건")

    # -------------------------------------------------------------------------
    # 3단계: RandomForest 모델링 및 독립 Test Set 비교평가
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print(" 3. RandomForest 모델링 및 독립 Test Set 비교평가 (PR-AUC & Recall 중점)")
    print("=" * 85)

    results = []
    test_probs = {}

    for name, (X_tr, y_tr, prep_time) in datasets.items():
        print(f"\n▶ [{name}] RandomForest 학습 진행 중...")
        t_fit = perf_counter()
        rf = RandomForestClassifier(
            n_estimators=args.trees,
            max_features="sqrt",
            min_samples_leaf=1,
            random_state=args.seed,
            n_jobs=args.jobs
        )
        rf.fit(X_tr, y_tr)
        fit_time = perf_counter() - t_fit

        prob_test = rf.predict_proba(X_test_proc)[:, 1]
        test_probs[name] = prob_test
        m = evaluate_model(y_test, prob_test, threshold=0.5)

        row = {
            "기법": name,
            "학습 데이터수": len(y_tr),
            "소수(Yes)수": int((y_tr == 1).sum()),
            "PR-AUC (AP)": m["PR-AUC (AP)"],
            "Recall": m["Recall"],
            "Precision": m["Precision"],
            "F1-Score": m["F1-Score"],
            "ROC-AUC": m["ROC-AUC"],
            "합성시간(초)": round(prep_time, 2),
            "학습시간(초)": round(fit_time, 2)
        }
        results.append(row)
        print(f"  → 완료: PR-AUC(AP): {m['PR-AUC (AP)']:.4f} | Recall: {m['Recall']:.4f} | Precision: {m['Precision']:.4f} | F1: {m['F1-Score']:.4f}")

    df_results = pd.DataFrame(results)

    # -------------------------------------------------------------------------
    # 4단계: 결과 요약 및 시각화 저장
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print(" 4. TabDDPM vs SMOTE vs ADASYN 최종 비교 결과표")
    print("=" * 85)
    display_cols = ["기법", "학습 데이터수", "소수(Yes)수", "PR-AUC (AP)", "Recall", "Precision", "F1-Score", "ROC-AUC"]
    print(df_results[display_cols].to_string(index=False))

    df_results.to_csv(output_dir / "tabddpm_vs_smote_adasyn_rf_comparison.csv", index=False, encoding="utf-8-sig")

    # 시각화 대시보드
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    short_names = ["1. Baseline", "2. SMOTE", "3. ADASYN", "4. TabDDPM"]
    x_pos = np.arange(len(short_names))
    bw = 0.35

    # 차트 1: 핵심 지표 Recall & Precision 비교
    ax1 = axes[0]
    ax1.bar(x_pos - bw/2, df_results["Recall"], width=bw, label="Recall (재현율)", color="#EF5350", alpha=0.9)
    ax1.bar(x_pos + bw/2, df_results["Precision"], width=bw, label="Precision (정밀도)", color="#26A69A", alpha=0.9)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(short_names, fontsize=9, fontweight='bold', rotation=15)
    ax1.set_ylim(0, 0.75)
    ax1.set_title("1. Recall (재현율) 및 Precision 비교", fontsize=11, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.legend()
    for i in range(len(short_names)):
        ax1.text(i - bw/2, df_results["Recall"].iloc[i] + 0.015, f"{df_results['Recall'].iloc[i]:.3f}", ha='center', fontsize=8, fontweight='bold')
        ax1.text(i + bw/2, df_results["Precision"].iloc[i] + 0.015, f"{df_results['Precision'].iloc[i]:.3f}", ha='center', fontsize=8)

    # 차트 2: 핵심 지표 PR-AUC (Average Precision) 및 F1 비교
    ax2 = axes[1]
    ax2.bar(x_pos - bw/2, df_results["PR-AUC (AP)"], width=bw, label="PR-AUC (AP)", color="#42A5F5", alpha=0.9)
    ax2.bar(x_pos + bw/2, df_results["F1-Score"], width=bw, label="F1-Score", color="#AB47BC", alpha=0.9)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(short_names, fontsize=9, fontweight='bold', rotation=15)
    ax2.set_ylim(0, 0.60)
    ax2.set_title("2. PR-AUC (AP) 및 F1-Score 비교", fontsize=11, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.legend()
    for i in range(len(short_names)):
        ax2.text(i - bw/2, df_results["PR-AUC (AP)"].iloc[i] + 0.012, f"{df_results['PR-AUC (AP)'].iloc[i]:.3f}", ha='center', fontsize=8, fontweight='bold')
        ax2.text(i + bw/2, df_results["F1-Score"].iloc[i] + 0.012, f"{df_results['F1-Score'].iloc[i]:.3f}", ha='center', fontsize=8)

    # 차트 3: 독립 Test Set PR 곡선 대조
    ax3 = axes[2]
    colors = ["#78909C", "#FFA726", "#26A69A", "#AB47BC"]
    for i, (name, _) in enumerate(datasets.items()):
        prob = test_probs[name]
        p, r, _ = precision_recall_curve(y_test, prob)
        ap = df_results.loc[df_results["기법"] == name, "PR-AUC (AP)"].values[0]
        ax3.plot(r, p, label=f"{short_names[i]} (AP={ap:.3f})", color=colors[i], linewidth=2.0)

    ax3.axhline(y_test.mean(), linestyle="--", color="gray", label=f"기준선 ({y_test.mean():.1%})")
    ax3.set_xlabel("Recall (재현율)", fontsize=10)
    ax3.set_ylabel("Precision (정밀도)", fontsize=10)
    ax3.set_title("3. 독립 Test Set PR (정밀도-재현율) 곡선", fontsize=11, fontweight='bold')
    ax3.grid(linestyle='--', alpha=0.4)
    ax3.legend(fontsize=8, loc='upper right')

    plt.suptitle("Bank Marketing: TabDDPM vs SMOTE vs ADASYN RandomForest 성능 비교", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    chart_file = output_dir / "tabddpm_oversampling_dashboard.png"
    plt.savefig(chart_file, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"\n▶ 비교 시각화 대시보드 저장 완료: {chart_file}")
    print(f"▶ 결과 CSV 파일 저장 완료: {output_dir / 'tabddpm_vs_smote_adasyn_rf_comparison.csv'}")

def main():
    parser = argparse.ArgumentParser(description="TabDDPM vs SMOTE vs ADASYN Comparison on Bank Marketing")
    default_data = ROOT / "data/bank/bank-full.csv"
    if not default_data.exists():
        default_data = ROOT / "data/bank+marketing/bank-full.csv"
    default_output = ROOT / "output/bank_tabddpm"

    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=-1)
    run_tabddpm_comparison(parser.parse_args())

if __name__ == "__main__":
    main()
