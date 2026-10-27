"""
Bank Marketing 데이터셋: Cost-sensitive Learning vs Class Weighting (소수 클래스 패널티) 비교

비교 대상:
1. Baseline: 가중치/비용 미부여 (Standard Logloss, scale_pos_weight=1.0)
2. Class Weighting (scale_pos_weight = 7.55): 소수 클래스 역수 비율 패널티
3. Cost-sensitive Learning (비용 행렬 기반 비대칭 목적함수 최적화):
   - 비용 시나리오: FN Cost = 7.55 (가입 고객을 놓쳤을 때의 기회비용), FP Cost = 1.0 (비가입자 마케팅 헛수고 비용)
   - Custom Objective: L = Cost(y) * LogLoss(y, p) 의 1차(Gradient), 2차(Hessian) 도함수를 트리 분기 손실에 직접 반영
4. Cost-sensitive Learning (Threshold Tuning): 비용 민감성 사후 임계값(Threshold = FP_cost / (FP_cost + FN_cost)) 적용

평가 설계 원칙:
- 전체 데이터를 Train(80%)과 독립 Test(20%)로 분할 (Stratified).
- 독립 Test 셋은 원본 불균형 상태 그대로 유지.
- 비즈니스 총 비용(Total Expected Cost = FP * C_fp + FN * C_fn) 지표를 도입하여 비용 절감 효과 정량화.
- 불균형 성능 지표(Precision, Recall, F1-Score, ROC-AUC, PR-AUC) 동시 비교.
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

# 한글 폰트 및 마이너스 기호 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parent

def make_cost_sensitive_objective(cost_fn, cost_fp):
    """
    XGBoost 맞춤형 비용 민감 손실 함수 (Cost-sensitive Loss) 생성기
    - cost_fn: False Negative 비용 (소수 클래스를 놓쳤을 때의 비용)
    - cost_fp: False Positive 비용 (다수 클래스 오탐 비용)
    """
    def cost_sensitive_obj(y_true, y_pred):
        # y_pred는 시그모이드 통과 전 마진(logits) 값
        p = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -15, 15)))
        # 샘플별 가중치 할당
        weight = np.where(y_true == 1, cost_fn, cost_fp)
        # 1차 도함수 (Gradient) 및 2차 도함수 (Hessian)
        grad = weight * (p - y_true)
        hess = weight * np.maximum(p * (1.0 - p), 1e-6)
        return grad, hess
    return cost_sensitive_obj

def evaluate_with_cost(y_true, y_prob, threshold=0.5, cost_fn=7.55, cost_fp=1.0):
    """
    분류 성능 지표 및 비즈니스 총 손실 비용(Total Misclassification Cost) 산출
    Total Cost = FP * Cost_FP + FN * Cost_FN
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total_cost = (fp * cost_fp) + (fn * cost_fn)
    avg_cost_per_customer = total_cost / len(y_true)

    return {
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "PR-AUC (AP)": average_precision_score(y_true, y_prob),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "Balanced Acc": balanced_accuracy_score(y_true, y_pred),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "총 오분류 비용": round(total_cost, 1),
        "고객당 평균비용": round(avg_cost_per_customer, 3)
    }

def build_preprocessor(numeric_cols, categorical_cols):
    """
    데이터 누수 방지 전처리 파이프라인
    """
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric_cols),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", sparse_output=False)), categorical_cols)
    ])

def run_cost_sensitive_experiment(args):
    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print(" 1. Bank Marketing 데이터 로드 및 2단계 분할 (Train 80% vs Test 20%)")
    print("=" * 85)

    df = pd.read_csv(data_path, sep=";")
    print(f"▶ 데이터 파일: {data_path.resolve()}")
    print(f"▶ 전체 데이터 크기: {len(df):,}행 x {df.shape[1]}열")

    # 타깃 변수 수치화
    y = df["y"].eq("yes").astype(int).to_numpy()

    # 사전 예측을 위해 duration 제외
    X = df.drop(columns=["y", "duration"], errors="ignore").copy()

    if "pdays" in X:
        X["previously_contacted"] = X["pdays"].ne(-1).astype(int)
        X["pdays"] = X["pdays"].replace(-1, np.nan)

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.columns.difference(numeric_cols).tolist()

    # 독립 Test Set 분리 (불균형 보존)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    imbalance_ratio = round(n_neg / n_pos, 2)
    
    # 비즈니스 오분류 비용 정의
    cost_fp = 1.0   # 전화 마케팅 실패 비용 (단위 비용: 1)
    cost_fn = imbalance_ratio  # 가입 고객을 놓쳤을 때의 기회 손실 비용 (약 7.55)
    theoretical_cost_threshold = round(cost_fp / (cost_fp + cost_fn), 4)

    print(f"▶ Train Set (80%): {len(X_train_df):,}건 (No: {n_neg:,}건, Yes: {n_pos:,}건 / 가입률 {y_train.mean():.2%})")
    print(f"▶ Test Set  (20%): {len(X_test_df):,}건 (No: {(y_test == 0).sum():,}건, Yes: {(y_test == 1).sum():,}건 / 가입률 {y_test.mean():.2%})")
    print(f"▶ 비즈니스 비용 정의: FP 비용 = {cost_fp}, FN 기회비용 = {cost_fn} (이론적 최적 임계값: {theoretical_cost_threshold:.4f})")

    # 전처리 변환
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_train_proc = preprocessor.fit_transform(X_train_df)
    X_test_proc = preprocessor.transform(X_test_df)

    # -------------------------------------------------------------------------
    # 2단계: 4가지 학습 및 의사결정 전략 설정
    # -------------------------------------------------------------------------
    strategies = [
        {
            "name": "1. Baseline (No Weight/Cost)",
            "type": "standard",
            "scale_pos_weight": 1.0,
            "threshold": 0.5,
            "desc": "가중치/비용 미부여 표준 XGBoost (th=0.5)"
        },
        {
            "name": "2. Penalty 강화 (scale_pos_weight)",
            "type": "scale_pos_weight",
            "scale_pos_weight": cost_fn,
            "threshold": 0.5,
            "desc": f"소수 클래스 손실에 {cost_fn}배 페널티 부여 (th=0.5)"
        },
        {
            "name": "3. Cost-sensitive Learning (Obj)",
            "type": "custom_obj",
            "scale_pos_weight": 1.0,
            "threshold": 0.5,
            "desc": f"비용 민감 맞춤형 목적함수(FN={cost_fn}, FP={cost_fp}) 최적화 (th=0.5)"
        },
        {
            "name": "4. Cost-sensitive (Threshold 튜닝)",
            "type": "standard",
            "scale_pos_weight": 1.0,
            "threshold": theoretical_cost_threshold,
            "desc": f"표준 모델 학습 후 비용 이론 임계값(th={theoretical_cost_threshold}) 적용"
        }
    ]

    print("\n" + "=" * 85)
    print(" 2. 기법별 모델 학습 및 독립 Test Set 평가")
    print("=" * 85)

    test_results = []
    test_probs = {}

    for strat in strategies:
        strat_name = strat["name"]
        stype = strat["type"]
        th = strat["threshold"]
        spw = strat["scale_pos_weight"]

        print(f"\n▶ [{strat_name}] 학습 진행 중...")
        t_start = perf_counter()

        if stype == "custom_obj":
            # 비용 민감 목적함수 적용
            custom_obj = make_cost_sensitive_objective(cost_fn=cost_fn, cost_fp=cost_fp)
            xgb = XGBClassifier(
                n_estimators=args.trees,
                learning_rate=args.lr,
                max_depth=args.depth,
                objective=custom_obj,
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
        test_probs[strat_name] = prob_test

        metrics = evaluate_with_cost(y_test, prob_test, threshold=th, cost_fn=cost_fn, cost_fp=cost_fp)
        
        row = {
            "전략": strat_name,
            "임계값(th)": th,
            "정밀도(Precision)": metrics["Precision"],
            "재현율(Recall)": metrics["Recall"],
            "F1-Score": metrics["F1-Score"],
            "PR-AUC (AP)": metrics["PR-AUC (AP)"],
            "ROC-AUC": metrics["ROC-AUC"],
            "FP(오탐)": metrics["FP"],
            "FN(미탐)": metrics["FN"],
            "총 오분류 비용": metrics["총 오분류 비용"],
            "고객당 비용": metrics["고객당 평균비용"],
            "학습시간(초)": round(fit_time, 2)
        }
        test_results.append(row)
        print(f"  → 완료: Recall: {metrics['Recall']:.4f} | F1: {metrics['F1-Score']:.4f} | 총 비용: {metrics['총 오분류 비용']:,} (FP={metrics['FP']}, FN={metrics['FN']})")

    df_results = pd.DataFrame(test_results)

    # -------------------------------------------------------------------------
    # 3단계: 결과 대조표 출력 및 저장
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print(" 3. Cost-sensitive Learning vs Penalty 강화 종합 비교표")
    print("=" * 85)
    display_cols = ["전략", "임계값(th)", "정밀도(Precision)", "재현율(Recall)", "F1-Score", "총 오분류 비용", "고객당 비용"]
    print(df_results[display_cols].to_string(index=False))

    df_results.to_csv(output_dir / "cost_sensitive_vs_penalty_comparison.csv", index=False, encoding="utf-8-sig")

    # -------------------------------------------------------------------------
    # 4단계: 시각화 차트 생성
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = ["1. Baseline", "2. Penalty 강화", "3. Cost-Obj", "4. Cost-Threshold"]
    x_indices = np.arange(len(labels))

    # 차트 1: 총 오분류 비용 (비즈니스 임팩트 비교)
    ax1 = axes[0]
    bars1 = ax1.bar(x_indices, df_results["총 오분류 비용"], color=["#78909C", "#FFA726", "#42A5F5", "#AB47BC"], alpha=0.9)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(labels, fontsize=9, fontweight='bold', rotation=15)
    ax1.set_title("1. 비즈니스 총 오분류 비용 (낮을수록 우수)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("총 비용 (FP*1 + FN*7.55)")
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    base_cost = df_results["총 오분류 비용"].iloc[0]
    for i, v in enumerate(df_results["총 오분류 비용"]):
        reduction = (1 - v / base_cost) * 100
        text = f"{v:,.0f}\n({reduction:+.1f}%)" if i > 0 else f"{v:,.0f}\n(기준)"
        ax1.text(i, v + 200, text, ha='center', fontsize=8, fontweight='bold')

    # 차트 2: Precision vs Recall 비교
    ax2 = axes[1]
    bw = 0.35
    ax2.bar(x_indices - bw/2, df_results["정밀도(Precision)"], width=bw, label="Precision", color="#26A69A", alpha=0.9)
    ax2.bar(x_indices + bw/2, df_results["재현율(Recall)"], width=bw, label="Recall", color="#EF5350", alpha=0.9)
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(labels, fontsize=9, fontweight='bold', rotation=15)
    ax2.set_ylim(0, 0.85)
    ax2.set_title("2. Precision vs Recall 트레이드오프", fontsize=11, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.legend()
    for i in range(len(labels)):
        ax2.text(i - bw/2, df_results["정밀도(Precision)"].iloc[i] + 0.015, f"{df_results['정밀도(Precision)'].iloc[i]:.2f}", ha='center', fontsize=8)
        ax2.text(i + bw/2, df_results["재현율(Recall)"].iloc[i] + 0.015, f"{df_results['재현율(Recall)'].iloc[i]:.2f}", ha='center', fontsize=8, fontweight='bold')

    # 차트 3: 비용 곡선 (임계값에 따른 총 비용 변화)
    ax3 = axes[2]
    # Baseline 모델 예측 확률 기준 비용 곡선
    base_prob = test_probs["1. Baseline (No Weight/Cost)"]
    thresholds = np.linspace(0.01, 0.99, 100)
    costs = []
    for t in thresholds:
        pred_t = (base_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred_t, labels=[0, 1]).ravel()
        c = (fp * cost_fp) + (fn * cost_fn)
        costs.append(c)
    
    ax3.plot(thresholds, costs, color="#3F51B5", linewidth=2.2, label="임계값별 총 오분류 비용")
    min_cost_idx = np.argmin(costs)
    opt_th = thresholds[min_cost_idx]
    ax3.axvline(opt_th, linestyle="--", color="red", label=f"최적 비용 임계값 ({opt_th:.2f})")
    ax3.axvline(0.5, linestyle=":", color="gray", label="기본 임계값 (0.50)")
    ax3.set_xlabel("분류 임계값 (Threshold)")
    ax3.set_ylabel("총 비용")
    ax3.set_title("3. 임계값(Threshold)에 따른 총 오분류 비용 곡선", fontsize=11, fontweight='bold')
    ax3.grid(linestyle='--', alpha=0.4)
    ax3.legend(fontsize=8)

    plt.suptitle("Bank Marketing: Cost-sensitive Learning vs Penalty 강화 비교 대시보드", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    chart_file = output_dir / "cost_sensitive_dashboard.png"
    plt.savefig(chart_file, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"\n▶ 비교 시각화 대시보드 저장 완료: {chart_file}")
    print(f"▶ 결과 CSV 파일 저장 완료: {output_dir / 'cost_sensitive_vs_penalty_comparison.csv'}")

def main():
    parser = argparse.ArgumentParser(description="Bank Marketing Cost-sensitive Learning vs Penalty Comparison")
    default_data = ROOT / "data/bank/bank-full.csv"
    if not default_data.exists():
        default_data = ROOT / "data/bank+marketing/bank-full.csv"
    default_output = ROOT / "output/bank_cost_sensitive"

    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=-1)
    run_cost_sensitive_experiment(parser.parse_args())

if __name__ == "__main__":
    main()
