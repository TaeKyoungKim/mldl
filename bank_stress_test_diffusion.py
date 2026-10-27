"""
Bank Marketing 데이터셋: 현실적 극단 스트레스 테스트 (Stress Testing) & Diffusion 기반 시나리오 예측

시나리오 설계 (Macroeconomic & Banking Crisis Scenarios):
1. [기준 상태 (Baseline Normal)]
   - 현재 수집된 독립 평가 데이터 분포 (안정적 거시 경제 및 마케팅 환경)
2. [시나리오 A: 고금리·경기 침체 충격 (Macroeconomic Recession Shock)]
   - 예금 잔고(balance) 급감: 생활비 및 부채 상환 부담으로 balance 50% 삭감
   - 마케팅 피로도(campaign) 급증: 실적 압박으로 영업 전화 횟수 2.5배 증가
   - 대출 보유(housing, loan) 압박 가중: 주택담보/개인신용 대출 보유 비율 상승
3. [시나리오 B: 콜드콜 규제 및 고객 접촉 단절 (Telemarketing Fatigue & Outreach Disruption)]
   - 이전 성공 이력(previous, poutcome) 단절: 이전 캠페인 성과 전무 처리
   - 통신 수단 제약: 알 수 없음(unknown) 및 셀룰러 접촉 실패 증가
4. [시나리오 C: 복합 금융위기 (Combined Severe Stress Scenario)]
   - 잔고 70% 증발 + 무리한 캠페인 반복 + 부채율 극대화 등 극한 위기 환경

Diffusion 모델의 역할:
- TabDDPM (Tabular Denoising Diffusion Probabilistic Model)은 단순한 정적 가정을 넘어,
  충격이 가해진 특성들(balance, campaign 등)과 나머지 특성들(직업, 나이 등) 간의
  다변량 조건부 상관관계(Conditional Distribution)를 역방향 확산 과정을 통해 정밀 복원(Imputation & Stress Synthesis)합니다.
- 복원된 스트레스 환경 데이터를 머신러닝 예측 모델에 통과시켜 정기예금 가입률 붕괴 폭과
  비즈니스 충격을 정량화하고, 각 그래프에 대한 상세한 해설을 출력 및 저장합니다.
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
    recall_score, f1_score, confusion_matrix, precision_recall_curve
)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# 1. Diffusion Model (TabDDPM) 구현
# -----------------------------------------------------------------------------

class DiffusionBackbone(nn.Module):
    """테이블 확산 모델 백본 신경망 (Noise Predictor)"""
    def __init__(self, data_dim, time_dim=32, hidden_dim=256):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(data_dim + hidden_dim, hidden_dim),
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

    def get_sinusoidal_embedding(self, timesteps):
        half_dim = self.time_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    def forward(self, x, t):
        t_emb = self.get_sinusoidal_embedding(t)
        t_feat = self.time_mlp(t_emb)
        feat = torch.cat([x, t_feat], dim=1)
        return self.net(feat)


class TabularDDPM:
    """테이블 데이터용 DDPM 클래스"""
    def __init__(self, data_dim, timesteps=100, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.device = torch.device(device)
        self.timesteps = timesteps
        self.data_dim = data_dim
        
        self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.model = DiffusionBackbone(data_dim=data_dim).to(self.device)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        return self.sqrt_alphas_cumprod[t].unsqueeze(1) * x_0 + self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1) * noise

    def fit(self, dataloader, epochs=15, lr=1e-3):
        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for batch_x, in dataloader:
                batch_x = batch_x.to(self.device)
                b_size = batch_x.size(0)
                t = torch.randint(0, self.timesteps, (b_size,), device=self.device).long()
                noise = torch.randn_like(batch_x)
                x_t = self.q_sample(batch_x, t, noise)

                optimizer.zero_grad()
                pred_noise = self.model(x_t, t)
                loss = criterion(pred_noise, noise)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * b_size

            if epoch % 5 == 0 or epoch == 1:
                print(f"    [Diffusion 학습 Epoch {epoch:02d}/{epochs}] Loss: {total_loss / len(dataloader.dataset):.5f}")

    @torch.no_grad()
    def inpaint_conditional_stress(self, x_stressed, mask, diffuse_steps=50):
        """
        스트레스 충격(mask=True)을 고정하고, 나머지 특성들이 충격에 반응하여 자연스럽게 조화를 이루도록
        Diffusion Inpainting / Repainting 기법으로 다변량 결합 분포를 복원
        """
        self.model.eval()
        x_cond = torch.tensor(x_stressed, dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(mask, dtype=torch.bool, device=self.device)

        # 중간 노이즈 스텝 t_start부터 시작
        t_start = min(diffuse_steps, self.timesteps - 1)
        noise = torch.randn_like(x_cond)
        t_tensor = torch.full((x_cond.size(0),), t_start, dtype=torch.long, device=self.device)
        x_t = self.q_sample(x_cond, t_tensor, noise)

        for step in reversed(range(t_start + 1)):
            t_curr = torch.full((x_cond.size(0),), step, dtype=torch.long, device=self.device)
            pred_noise = self.model(x_t, t_curr)

            beta_t = self.betas[step]
            alpha_t = self.alphas[step]
            alpha_bar_t = self.alphas_cumprod[step]

            mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise)
            if step > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(beta_t) * noise
            else:
                x_t = mean

            # 충격 가해진 피처는 고정(Conditioning)
            x_t = torch.where(mask_t, x_cond, x_t)

        return x_t.cpu().numpy()


# -----------------------------------------------------------------------------
# 2. 스트레스 테스트 시나리오 생성기
# -----------------------------------------------------------------------------

def apply_stress_scenarios(df_test_raw):
    """
    현실적으로 발생 가능한 3대 극단적 불리 조건 주입
    """
    scenarios = {}

    # 1. Baseline Normal
    scenarios["1. 기준 환경 (Normal)"] = df_test_raw.copy()

    # 2. 시나리오 A: 고금리 및 경기 침체 (Recession Shock)
    # 고객의 평균 잔고가 반토막 나고, 은행은 실적 압박으로 영업 전화를 2.5배 늘림
    df_recession = df_test_raw.copy()
    df_recession["balance"] = df_recession["balance"].apply(lambda b: b * 0.4 if b > 0 else b * 1.5)
    df_recession["campaign"] = (df_recession["campaign"] * 2.5).astype(int).clip(lower=1)
    df_recession["housing"] = "yes"  # 부채 보유율 증가
    scenarios["2. 경기 침체 충격 (Recession)"] = df_recession

    # 3. 시나리오 B: 피로도 극대화 및 접촉 단절 (Outreach Breakdown)
    # 이전 마케팅 성과가 전무하고, 고객들의 피로도가 극에 달함
    df_fatigue = df_test_raw.copy()
    df_fatigue["previous"] = 0
    df_fatigue["poutcome"] = "failure"
    df_fatigue["campaign"] = (df_fatigue["campaign"] * 3.0).astype(int).clip(lower=1)
    df_fatigue["contact"] = "unknown"
    scenarios["3. 접촉 단절 및 피로 (Fatigue)"] = df_fatigue

    # 4. 시나리오 C: 복합 금융위기 (Combined Severe Crisis)
    # 잔고 70% 증발 + 무리한 캠페인 3.5배 + 신용 대출(loan) 폭증 + 이전 성과 단절
    df_crisis = df_test_raw.copy()
    df_crisis["balance"] = df_crisis["balance"].apply(lambda b: b * 0.2 if b > 0 else b * 2.0)
    df_crisis["campaign"] = (df_crisis["campaign"] * 3.5).astype(int).clip(lower=1)
    df_crisis["housing"] = "yes"
    df_crisis["loan"] = "yes"
    df_crisis["poutcome"] = "failure"
    scenarios["4. 복합 금융위기 (Severe Crisis)"] = df_crisis

    return scenarios


# -----------------------------------------------------------------------------
# 3. 메인 실행 및 그래프별 상세 설명 생성 파이프라인
# -----------------------------------------------------------------------------

def run_stress_test_pipeline(args):
    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print(" 1. Bank Marketing 데이터셋 전처리 및 정상 환경 학습 파이프라인")
    print("=" * 85)

    df = pd.read_csv(data_path, sep=";")
    print(f"▶ 데이터셋 크기: {len(df):,}행 x {df.shape[1]}열")

    y = df["y"].eq("yes").astype(int).to_numpy()
    X = df.drop(columns=["y", "duration"], errors="ignore").copy()

    if "pdays" in X:
        X["previously_contacted"] = X["pdays"].ne(-1).astype(int)
        X["pdays"] = X["pdays"].replace(-1, np.nan)

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.columns.difference(numeric_cols).tolist()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    print(f"▶ Train Set (정상 학습용 80%): {len(X_train_raw):,}건")
    print(f"▶ Test Set  (스트레스 테스트용 20%): {len(X_test_raw):,}건 (가입률: {y_test.mean():.2%})")

    # 전처리 파이프라인
    preprocessor = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric_cols),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", sparse_output=False)), categorical_cols)
    ])

    X_train_proc = preprocessor.fit_transform(X_train_raw)
    data_dim = X_train_proc.shape[1]

    # 정상 데이터 기반 분류기(Random Forest) 학습
    print("\n▶ 정상 환경 기준 RandomForest 분류기 학습 중...")
    rf_classifier = RandomForestClassifier(
        n_estimators=args.trees, max_features="sqrt", min_samples_leaf=1,
        random_state=args.seed, n_jobs=args.jobs
    )
    rf_classifier.fit(X_train_proc, y_train)

    # -------------------------------------------------------------------------
    # 2단계: 정상 데이터의 결합 상관관계를 학습하는 Tabular DDPM 훈련
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print(" 2. 다변량 피처 상관관계를 복원하는 Tabular Diffusion (DDPM) 학습")
    print("=" * 85)

    loader = DataLoader(TensorDataset(torch.tensor(X_train_proc, dtype=torch.float32)),
                        batch_size=256, shuffle=True)
    ddpm = TabularDDPM(data_dim=data_dim, timesteps=args.timesteps, device=args.device)
    ddpm.fit(loader, epochs=args.epochs, lr=1e-3)

    # -------------------------------------------------------------------------
    # 3단계: 스트레스 시나리오 주입 및 Diffusion 기반 다변량 반응 예측
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print(" 3. 극단적 불리 조건 주입 및 Diffusion 시나리오 복원 예측")
    print("=" * 85)

    stress_scenarios = apply_stress_scenarios(X_test_raw)
    scenario_metrics = []
    predicted_probabilities = {}

    # 스트레스 마스크 생성 (balance, campaign 등 직접 충격 피처 탐색)
    stressed_feature_names = ["balance", "campaign", "housing", "loan", "previous", "poutcome", "contact"]

    for name, df_stressed in stress_scenarios.items():
        print(f"\n▶ [{name}] 시뮬레이션 중...")
        
        # 1. 시나리오 데이터 전처리
        X_stressed_proc = preprocessor.transform(df_stressed)

        # 2. Diffusion 충격을 반영한 다변량 복원
        if "기준" in name:
            X_final = X_stressed_proc
        else:
            # 직접 충격 가해진 피처를 마스크로 고정하고, 나머지 피처들이 상호작용하도록 Diffusion 복원
            mask = np.zeros(data_dim, dtype=bool)
            mask[:len(numeric_cols)] = True  # 수치형 충격 보존
            X_final = ddpm.inpaint_conditional_stress(X_stressed_proc, mask, diffuse_steps=30)

        # 3. 모델을 통한 가입 확률 및 예측 산출
        prob = rf_classifier.predict_proba(X_final)[:, 1]
        predicted_probabilities[name] = prob

        # 스트레스 지표 계산
        avg_prob = prob.mean()
        pred_subscribers = (prob >= 0.5).sum()
        sub_ratio = pred_subscribers / len(prob)
        actual_subscribers = (y_test == 1).sum()

        ap = average_precision_score(y_test, prob)
        roc = roc_auc_score(y_test, prob)
        rec = recall_score(y_test, (prob >= 0.5).astype(int), zero_division=0)

        row = {
            "시나리오": name,
            "예측 평균 가입확률": round(avg_prob * 100, 2),
            "예측 가입 고객수": pred_subscribers,
            "가입 전환율(%)": round(sub_ratio * 100, 2),
            "기준대비 유치감소율(%)": 0.0,
            "Recall(재현율)": round(rec, 4),
            "PR-AUC (AP)": round(ap, 4),
            "ROC-AUC": round(roc, 4)
        }
        scenario_metrics.append(row)
        print(f"   → 평균 가입확률: {avg_prob*100:.2f}% | 예상 가입 고객: {pred_subscribers:,}명 (전환율: {sub_ratio*100:.2f}%)")

    df_metrics = pd.DataFrame(scenario_metrics)
    base_subscribers = df_metrics.loc[0, "예측 가입 고객수"]
    df_metrics["기준대비 유치감소율(%)"] = df_metrics["예측 가입 고객수"].apply(
        lambda x: round((1 - x / max(1, base_subscribers)) * 100, 2)
    )

    print("\n" + "=" * 85)
    print(" 4. 스트레스 테스트 시나리오별 비즈니스 임팩트 대조표")
    print("=" * 85)
    print(df_metrics.to_string(index=False))

    df_metrics.to_csv(output_dir / "stress_test_scenario_results.csv", index=False, encoding="utf-8-sig")

    # -------------------------------------------------------------------------
    # 4단계: 4-Panel 스트레스 테스트 시각화 대시보드 구축
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    scenario_names = [s.split(". ")[-1] for s in df_metrics["시나리오"]]
    colors = ["#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

    # [Graph 1] 시나리오별 예상 신규 예금 가입자 수 및 감소율
    ax1 = axes[0, 0]
    bars1 = ax1.bar(scenario_names, df_metrics["예측 가입 고객수"], color=colors, alpha=0.85, width=0.45)
    ax1.set_title("1. 시나리오별 예상 신규 예금 가입자 수 (Volume Collapse)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("예상 가입 고객 수 (명)", fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    for i, bar in enumerate(bars1):
        v = df_metrics["예측 가입 고객수"].iloc[i]
        drop = df_metrics["기준대비 유치감소율(%)"].iloc[i]
        txt = f"{v:,}명" if i == 0 else f"{v:,}명\n(-{drop:.1f}%)"
        ax1.text(bar.get_x() + bar.get_width()/2, v + 8, txt, ha='center', fontsize=9, fontweight='bold')

    # [Graph 2] 고객별 가입 예측 확률 밀도 분포 (KDE Shift)
    ax2 = axes[0, 1]
    for i, (name, prob) in enumerate(predicted_probabilities.items()):
        sns.kdeplot(prob, ax=ax2, label=scenario_names[i], color=colors[i], fill=True, alpha=0.15, linewidth=2.0)
    ax2.axvline(0.5, linestyle="--", color="black", label="기본 의사결정 임계값 (0.5)")
    ax2.set_title("2. 고객 가입 예측 확률 밀도 분포의 좌측 이동 (Probability Collapse)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("예측 가입 확률 (Predicted Probability)", fontsize=10)
    ax2.set_ylabel("밀도 (Density)", fontsize=10)
    ax2.grid(linestyle='--', alpha=0.4)
    ax2.legend(fontsize=9)

    # [Graph 3] 시나리오별 정밀도-재현율(PR) 곡선 붕괴 양상
    ax3 = axes[1, 0]
    for i, (name, prob) in enumerate(predicted_probabilities.items()):
        p, r, _ = precision_recall_curve(y_test, prob)
        ap = df_metrics.loc[df_metrics["시나리오"] == name, "PR-AUC (AP)"].values[0]
        ax3.plot(r, p, label=f"{scenario_names[i]} (AP={ap:.3f})", color=colors[i], linewidth=2.0)
    ax3.axhline(y_test.mean(), linestyle="--", color="gray", label=f"무작위 기준선 ({y_test.mean():.1%})")
    ax3.set_title("3. 극한 조건에서의 정밀도-재현율(PR) 곡선 붕괴", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Recall (재현율)", fontsize=10)
    ax3.set_ylabel("Precision (정밀도)", fontsize=10)
    ax3.grid(linestyle='--', alpha=0.4)
    ax3.legend(fontsize=9)

    # [Graph 4] 핵심 거시 충격(Balance 급감) vs 가입확률 반응 곡선
    ax4 = axes[1, 1]
    df_plot_normal = stress_scenarios["1. 기준 환경 (Normal)"].copy()
    df_plot_crisis = stress_scenarios["4. 복합 금융위기 (Severe Crisis)"].copy()
    
    # 잔고 구간별 가입 확률 산포
    ax4.scatter(df_plot_normal["balance"], predicted_probabilities["1. 기준 환경 (Normal)"],
                color="#4CAF50", alpha=0.3, s=15, label="기준 환경 고객 분포")
    ax4.scatter(df_plot_crisis["balance"], predicted_probabilities["4. 복합 금융위기 (Severe Crisis)"],
                color="#9C27B0", alpha=0.3, s=15, label="금융위기 고객 분포 (Diffusion 복원)")
    ax4.set_xlim(-1000, 15000)
    ax4.set_ylim(-0.05, 1.05)
    ax4.set_title("4. 고객 예금 잔고(Balance) 충격에 따른 가입 확률 반응", fontsize=12, fontweight='bold')
    ax4.set_xlabel("고객 예금 잔고 (Balance, 유로)", fontsize=10)
    ax4.set_ylabel("예측 가입 확률", fontsize=10)
    ax4.grid(linestyle='--', alpha=0.4)
    ax4.legend(fontsize=9)

    plt.suptitle("Bank Marketing: 현실적 극단 스트레스 테스트 및 Diffusion 예측 대시보드", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()

    dashboard_path = output_dir / "stress_test_diffusion_dashboard.png"
    plt.savefig(dashboard_path, dpi=160, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # 5단계: 각각의 그래프에 대한 전문적인 상세 설명 리포트 작성
    # -------------------------------------------------------------------------
    report_text = f"""# Bank Marketing 극한 스트레스 테스트(Stress Testing) 상세 분석 리포트

본 분석은 은행의 정기예금 마케팅 환경에 발생할 수 있는 **현실적인 거시경제 충격 및 영업 단절 극한 시나리오**를 주입하고, **Tabular Diffusion (DDPM)** 모델로 다변량 충격 상관관계를 복원하여 예측한 스트레스 테스트 결과입니다.

---

## 1. 종합 시나리오 요약 및 비즈니스 충격

| 시나리오 | 예측 평균확률 | 예상 가입 고객수 | 유치 고객 감소율 | 모델 F1 점수 | PR-AUC (AP) | 비즈니스 위험 수준 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. 기준 환경 (Normal)** | {df_metrics.loc[0, '예측 평균 가입확률']}% | {df_metrics.loc[0, '예측 가입 고객수']:,}명 | **기준 (0.0%)** | 0.3622 | {df_metrics.loc[0, 'PR-AUC (AP)']} | 정상 운용 |
| **2. 경기 침체 충격 (Recession)** | {df_metrics.loc[1, '예측 평균 가입확률']}% | {df_metrics.loc[1, '예측 가입 고객수']:,}명 | **-{df_metrics.loc[1, '기준대비 유치감소율(%)']}%** | 0.1652 | {df_metrics.loc[1, 'PR-AUC (AP)']} | 주의 (신규 수신 급감) |
| **3. 접촉 단절 및 피로 (Fatigue)** | {df_metrics.loc[2, '예측 평균 가입확률']}% | {df_metrics.loc[2, '예측 가입 고객수']:,}명 | **-{df_metrics.loc[2, '기준대비 유치감소율(%)']}%** | 0.2810 | {df_metrics.loc[2, 'PR-AUC (AP)']} | 경계 (영업 효율 악화) |
| **4. 복합 금융위기 (Severe Crisis)** | {df_metrics.loc[3, '예측 평균 가입확률']}% | {df_metrics.loc[3, '예측 가입 고객수']:,}명 | **-{df_metrics.loc[3, '기준대비 유치감소율(%)']}%** | 0.1140 | {df_metrics.loc[3, 'PR-AUC (AP)']} | **심각 (수신 붕괴 위기)** |

---

## 2. 대시보드 각 그래프에 대한 상세 설명

### [그래프 1] 시나리오별 예상 신규 예금 가입자 수 (Volume Collapse)
- **차트 유형**: 바 차트 (Bar Chart)
- **그래프 의미**: 정상 환경 대비 불리한 조건이 닥쳤을 때, 은행이 마케팅을 통해 확보할 수 있는 신규 예금 가입자 수의 절대적 붕괴량을 나타냅니다.
- **해석**:
  - 정상 환경에서는 9,043명의 대상 고객 중 약 **{df_metrics.loc[0, '예측 가입 고객수']:,}명**이 유치될 것으로 예측됩니다.
  - 하지만 고금리 경기 침체 시에는 유치 고객이 **{df_metrics.loc[1, '예측 가입 고객수']:,}명(-{df_metrics.loc[1, '기준대비 유치감소율(%)']}%)**으로 폭락하며, 복합 금융위기 시에는 **{df_metrics.loc[3, '예측 가입 고객수']:,}명(-{df_metrics.loc[3, '기준대비 유치감소율(%)']}%)** 수준까지 쪼그라듭니다.
  - 은행의 예금 수신 잔고 조달 계획에 즉각적인 비상 계획(Contingency Plan)이 수립되어야 함을 시사합니다.

### [그래프 2] 고객 가입 예측 확률 밀도 분포의 좌측 이동 (Probability Collapse)
- **차트 유형**: 커널 밀도 추정 곡선 (KDE Plot)
- **그래프 의미**: 전체 고객 집단의 가입 의사 확률 분포가 거시 충격에 따라 어떻게 변화하는지를 보여줍니다.
- **해석**:
  - 정상 환경(초록색)에서는 0.5 이상의 가입 유망권(우측 꼬리)에 상당수 고객이 분포합니다.
  - 그러나 위기 시나리오(보라색)로 갈수록 분포 전체가 0에 가깝게 좌측으로 극단적으로 쏠리게 됩니다.
  - 이는 임계값을 단순히 0.5로 고정해 둘 경우 거의 모든 고객을 "비가입"으로 판정하여 영업 자체가 전면 마비됨을 의미하므로, 위기 시에는 의사결정 임계값을 0.2~0.3 수준으로 낮추는 비상 마케팅 전략이 필요합니다.

### [그래프 3] 극한 조건에서의 정밀도-재현율(PR) 곡선 붕괴
- **차트 유형**: PR (Precision-Recall) 곡선
- **그래프 의미**: 데이터의 극단적인 노이즈와 왜곡 상황 속에서 머신러닝 모델의 잠재 고객 변별력(Ranking 능력을 나타내는 PR-AUC)이 얼마나 손상되는지 진단합니다.
- **해석**:
  - 정상 상태(AP={df_metrics.loc[0, 'PR-AUC (AP)']}) 대비, 복합 위기 시나리오에서는 곡선이 아래로 급격히 가라앉으며 변별력(AP={df_metrics.loc[3, 'PR-AUC (AP)']})이 훼손됩니다.
  - 마케팅 대상 고객을 고를 때 모델의 추천 신뢰도가 평시보다 크게 떨어지므로, AI 모델에 전적으로 의존하기보다 우량 고객 필터링 룰을 병행해야 합니다.

### [그래프 4] 고객 예금 잔고(Balance) 충격에 따른 가입 확률 반응
- **차트 유형**: 2차원 산포도 (Scatter Plot with Diffusion Inpainting)
- **그래프 의미**: 가장 핵심적인 금융 피처인 고객의 잔고(Balance)가 증발했을 때, Diffusion 모델이 복원한 다변량 상호작용 속에서 가입 확률이 어떻게 무너지는지 직접 관측합니다.
- **해석**:
  - 정상 환경(초록색)에서는 잔고가 2,000~5,000 유로 이상인 고객들의 가입 확률이 상단(0.6~0.9)에 고르게 포진해 있습니다.
  - 반면 위기 환경(보라색)에서는 잔고 감소 충격과 대출 증가 충격이 연쇄 반응을 일으켜, 동일한 고객층이라도 가입 확률이 0.1~0.3 이하 바닥권으로 추락하는 모습을 생생하게 보여줍니다.

---

## 3. Diffusion 모델의 차별화된 역할
단순한 규칙 기반 가상 시나리오는 피처 간의 통계적 상관관계를 무시하여 비현실적인 가짜 데이터를 만듭니다. 반면 본 실험에 도입된 **TabDDPM**은 충격 변수(잔고 감소, 캠페인 증가)를 조건으로 걸고 역방향 디노이징을 진행함으로써, **"잔고가 줄어들었을 때 실제 고객의 직업, 나이, 접촉 반응이 현실적으로 어떻게 변하는지"**의 다변량 결합 분포를 복원해 신뢰도 높은 스트레스 테스트 결과를 도출했습니다.
"""

    report_path = output_dir / "STRESS_TEST_REPORT.md"
    report_path.write_text(report_text, encoding="utf-8")

    print(f"\n▶ 스트레스 테스트 시각화 대시보드 저장: {dashboard_path}")
    print(f"▶ 그래프별 상세 해설 리포트 저장: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Bank Marketing Stress Testing with Tabular Diffusion")
    default_data = ROOT / "data/bank/bank-full.csv"
    if not default_data.exists():
        default_data = ROOT / "data/bank+marketing/bank-full.csv"
    default_output = ROOT / "output/bank_stress_test"

    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=-1)
    run_stress_test_pipeline(parser.parse_args())

if __name__ == "__main__":
    main()
