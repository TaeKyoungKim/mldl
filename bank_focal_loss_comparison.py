"""
Bank Marketing 데이터셋: Focal Loss vs Baseline(가중치 미부여) 비교 실험

추천 알고리즘:
- 정형 불균형 데이터에서 2차 테일러 전개(Hessian)를 정밀하게 지원하고 안정적인 수렴을 보이는 XGBoost/LightGBM 기반 Focal Loss
- Focal Loss 핵심 수식:
  FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
  1) 모듈레이팅 팩터 (1 - p_t)^gamma: 쉬운 샘플(p_t가 큰 샘플)의 손실 비중을 급격히 다운웨이팅(Down-weighting)하여
     분류하기 어려운 경계면 샘플(Hard Examples)에 모델 학습력을 집중시킴 (gamma=2.0)
  2) 알파 팩터 alpha_t: 소수 클래스와 다수 클래스 간의 기본 빈도 균형 조절 (alpha=0.75 또는 0.25)

비교 실험군:
1. Baseline: 표준 Cross-Entropy (가중치 미부여, scale_pos_weight=1.0)
2. Class Weighting: 전통적 단순 빈도 역수 가중치 (scale_pos_weight = 7.55)
3. Focal Loss (gamma=2.0, alpha=0.25): 표준 Focal Loss 설정
4. Focal Loss (gamma=2.0, alpha=0.75): 소수 클래스 집중형 Focal Loss (어려운 샘플 + 소수 가중)

평가 설계 원칙:
- 전체 데이터를 Train(80%)과 독립 Test(20%)로 분할 (Stratified).
- 독립 Test 셋은 원본 불균형 상태(11.7%) 그대로 유지하여 일반화 성능 측정.
- 불균형 평가지표: Precision, Recall, F1-Score, ROC-AUC, PR-AUC(Average Precision).
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

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, balanced_accuracy_score,
    confusion_matrix, precision_recall_curve
)
from xgboost import XGBClassifier

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parent

class FocalLossObjective:
    """
    XGBoost 맞춤형 2계 도함수(Hessian) 지원 Focal Loss 목적함수
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, y_true, y_pred):
        # y_pred는 시그모이드 통과 전 raw margin
        p = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -15, 15)))
        pt = np.where(y_true == 1, p, 1.0 - p)
        at = np.where(y_true == 1, self.alpha, 1.0 - self.alpha)

        # 1차 도함수 (Gradient): d(FL)/dy_pred
        # g = at * (1 - pt)^gamma * (p - y)
        grad = at * ((1.0 - pt) ** self.gamma) * (p - y_true)

        # 2차 도함수 (Hessian): d^2(FL)/dy_pred^2
        hess_term1 = at * ((1.0 - pt) ** self.gamma) * (p * (1.0 - p))
        hess_term2 = at * self.gamma * ((1.0 - pt) ** max(self.gamma - 1.0, 0)) * ((p - y_true) ** 2)
        hess = hess_term1 + hess_term2
        hess = np.maximum(hess, 1e-6)

        return grad, hess

def evaluate_predictions(y_true, y_prob, threshold=0.5):
    """
    불균형 분류 성능 지표 산출
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "PR-AUC (AP)": average_precision_score(y_true, y_prob),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "Balanced Acc": balanced_accuracy_score(y_true, y_pred),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    }

def build_preprocessor(numeric_cols, categorical_cols):
    """
    전처리 파이프라인
    """
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric_cols),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", sparse_output=False)), categorical_cols)
    ])

def run_focal_loss_comparison(args):
    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print(" 1. Bank Marketing 데이터셋 로드 및 Train/Test 분할")
    print("=" * 85)

    df = pd.read_csv(data_path, sep=";")
    print(f"▶ 데이터 파일: {data_path.resolve()}")
    print(f"▶ 데이터셋 크기: {len(df):,}행 x {df.shape[1]}열")

    y = df["y"].eq("yes").astype(int).to_numpy()
    X = df.drop(columns=["y", "duration"], errors="ignore").copy()

    if "pdays" in X:
        X["previously_contacted"] = X["pdays"].ne(-1).astype(int)
        X["pdays"] = X["pdays"].replace(-1, np.nan)

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.columns.difference(numeric_cols).tolist()

    # Train (80%) vs Test (20%) 계층화 분할
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_weight = round(n_neg / n_pos, 2)

    print(f"▶ Train Set (학습 80%): {len(X_train_df):,}건 (No: {n_neg:,}건, Yes: {n_pos:,}건 / 가입률: {y_train.mean():.2%})")
    print(f"▶ Test Set  (평가 20%): {len(X_test_df):,}건 (No: {(y_test == 0).sum():,}건, Yes: {(y_test == 1).sum():,}건 / 가입률: {y_test.mean():.2%}) - 격리 보관")

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_train_proc = preprocessor.fit_transform(X_train_df)
    X_test_proc = preprocessor.transform(X_test_df)

    # -------------------------------------------------------------------------
    # 2단계: 실험 대상 정의
    # -------------------------------------------------------------------------
    experiments = [
        {
            "name": "1. Baseline (가중치 미부여)",
            "desc": "표준 Cross Entropy (scale_pos_weight=1.0)",
            "type": "baseline",
            "obj": None,
            "scale_pos_weight": 1.0
        },
        {
            "name": "2. 단순 클래스 가중치",
            "desc": f"전통적 빈도 역수 가중치 (scale_pos_weight={scale_weight})",
            "type": "class_weight",
            "obj": None,
            "scale_pos_weight": scale_weight
        },
        {
            "name": "3. Focal Loss (γ=2.0, α=0.25)",
            "desc": "표준 Focal Loss (어려운 샘플 가중 + 감쇠 알파)",
            "type": "focal",
            "obj": FocalLossObjective(alpha=0.25, gamma=2.0),
            "scale_pos_weight": 1.0
        },
        {
            "name": "4. Focal Loss (γ=2.0, α=0.75)",
            "desc": "소수 클래스 강화 Focal Loss (어려운 샘플 + 소수 집중)",
            "type": "focal",
            "obj": FocalLossObjective(alpha=0.75, gamma=2.0),
            "scale_pos_weight": 1.0
        }
    ]

    print("\n" + "=" * 85)
    print(" 2. 실험군 모델 학습 및 독립 Test Set 평가")
    print("=" * 85)

    test_results = []
    test_probs = {}

    for exp in experiments:
        name = exp["name"]
        obj = exp["obj"]
        spw = exp["scale_pos_weight"]

        print(f"\n▶ [{name}] 학습 진행 중... ({exp['desc']})")
        t_start = perf_counter()

        if obj is not None:
            xgb = XGBClassifier(
                n_estimators=args.trees,
                learning_rate=args.lr,
                max_depth=args.depth,
                objective=obj,
                random_state=args.seed,
                eval_metric="logloss",
                n_jobs=args.jobs
            )
        else:
            xgb = XGBClassifier(
                n_estimators=args.trees,
                learning_rate=args.lr,
                max_depth=args.depth,
                scale_pos_weight=spw,
                random_state=args.seed,
                eval_metric="logloss",
                n_jobs=args.jobs
            )

        xgb.fit(X_train_proc, y_train)
        fit_time = perf_counter() - t_start

        prob_test = xgb.predict_proba(X_test_proc)[:, 1]
        test_probs[name] = prob_test

        metrics = evaluate_predictions(y_test, prob_test, threshold=0.5)

        row = {
            "실험": name,
            "설명": exp["desc"],
            "정밀도(Precision)": metrics["Precision"],
            "재현율(Recall)": metrics["Recall"],
            "F1-Score": metrics["F1-Score"],
            "PR-AUC (AP)": metrics["PR-AUC (AP)"],
            "ROC-AUC": metrics["ROC-AUC"],
            "Balanced Acc": metrics["Balanced Acc"],
            "FP(오탐)": metrics["FP"],
            "FN(미탐)": metrics["FN"],
            "학습시간(초)": round(fit_time, 2)
        }
        test_results.append(row)
        print(f"  → 완료: Precision: {metrics['Precision']:.4f} | Recall: {metrics['Recall']:.4f} | F1: {metrics['F1-Score']:.4f} | AP: {metrics['PR-AUC (AP)']:.4f}")

    df_results = pd.DataFrame(test_results)

    # -------------------------------------------------------------------------
    # 3단계: 결과 대조표 출력 및 저장
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print(" 3. Focal Loss vs 가중치 미부여 vs 클래스 가중치 종합 비교표")
    print("=" * 85)
    display_cols = ["실험", "정밀도(Precision)", "재현율(Recall)", "F1-Score", "PR-AUC (AP)", "ROC-AUC", "FP(오탐)", "FN(미탐)"]
    print(df_results[display_cols].to_string(index=False))

    df_results.to_csv(output_dir / "focal_loss_comparison.csv", index=False, encoding="utf-8-sig")

    # -------------------------------------------------------------------------
    # 4단계: 시각화 차트 생성
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    short_labels = ["1. Baseline", "2. Class Weight", "3. Focal (α=0.25)", "4. Focal (α=0.75)"]
    x_indices = np.arange(len(short_labels))

    # 차트 1: Precision vs Recall 트레이드오프
    ax1 = axes[0]
    bw = 0.35
    ax1.bar(x_indices - bw/2, df_results["정밀도(Precision)"], width=bw, label="Precision (정밀도)", color="#26A69A", alpha=0.9)
    ax1.bar(x_indices + bw/2, df_results["재현율(Recall)"], width=bw, label="Recall (재현율)", color="#FFA726", alpha=0.9)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(short_labels, fontsize=9, fontweight='bold', rotation=15)
    ax1.set_ylim(0, 0.85)
    ax1.set_title("1. Precision vs Recall 트레이드오프", fontsize=11, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.legend()
    for i in range(len(short_labels)):
        ax1.text(i - bw/2, df_results["정밀도(Precision)"].iloc[i] + 0.015, f"{df_results['정밀도(Precision)'].iloc[i]:.2f}", ha='center', fontsize=8)
        ax1.text(i + bw/2, df_results["재현율(Recall)"].iloc[i] + 0.015, f"{df_results['재현율(Recall)'].iloc[i]:.2f}", ha='center', fontsize=8, fontweight='bold')

    # 차트 2: F1-Score 및 PR-AUC (Average Precision)
    ax2 = axes[1]
    ax2.bar(x_indices - bw/2, df_results["F1-Score"], width=bw, label="F1-Score", color="#AB47BC", alpha=0.9)
    ax2.bar(x_indices + bw/2, df_results["PR-AUC (AP)"], width=bw, label="PR-AUC (AP)", color="#42A5F5", alpha=0.9)
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(short_labels, fontsize=9, fontweight='bold', rotation=15)
    ax2.set_ylim(0, 0.60)
    ax2.set_title("2. F1-Score 및 PR-AUC (랭킹 능력)", fontsize=11, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.legend()
    for i in range(len(short_labels)):
        ax2.text(i - bw/2, df_results["F1-Score"].iloc[i] + 0.015, f"{df_results['F1-Score'].iloc[i]:.3f}", ha='center', fontsize=8, fontweight='bold')
        ax2.text(i + bw/2, df_results["PR-AUC (AP)"].iloc[i] + 0.015, f"{df_results['PR-AUC (AP)'].iloc[i]:.3f}", ha='center', fontsize=8)

    # 차트 3: Test Set PR 곡선
    ax3 = axes[2]
    colors = ["#78909C", "#EF5350", "#26A69A", "#3F51B5"]
    for i, exp in enumerate(experiments):
        name = exp["name"]
        prob = test_probs[name]
        p, r, _ = precision_recall_curve(y_test, prob)
        ap = df_results.loc[df_results["실험"] == name, "PR-AUC (AP)"].values[0]
        ax3.plot(r, p, label=f"{short_labels[i]} (AP={ap:.3f})", color=colors[i], linewidth=2.0)

    ax3.axhline(y_test.mean(), linestyle="--", color="gray", label=f"기준선 ({y_test.mean():.1%})")
    ax3.set_xlabel("Recall (재현율)", fontsize=10)
    ax3.set_ylabel("Precision (정밀도)", fontsize=10)
    ax3.set_title("3. 독립 Test Set PR (정밀도-재현율) 곡선", fontsize=11, fontweight='bold')
    ax3.grid(linestyle='--', alpha=0.4)
    ax3.legend(fontsize=8, loc='upper right')

    plt.suptitle("Bank Marketing: Focal Loss vs Baseline 성능 비교 대시보드", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    chart_file = output_dir / "focal_loss_dashboard.png"
    plt.savefig(chart_file, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"\n▶ 비교 시각화 대시보드 저장 완료: {chart_file}")
    print(f"▶ 결과 CSV 파일 저장 완료: {output_dir / 'focal_loss_comparison.csv'}")

def main():
    parser = argparse.ArgumentParser(description="Bank Marketing Focal Loss Comparison")
    default_data = ROOT / "data/bank/bank-full.csv"
    if not default_data.exists():
        default_data = ROOT / "data/bank+marketing/bank-full.csv"
    default_output = ROOT / "output/bank_focal_loss"

    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=-1)
    run_focal_loss_comparison(parser.parse_args())

if __name__ == "__main__":
    main()
