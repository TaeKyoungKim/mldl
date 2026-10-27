"""
Bank Marketing 데이터셋: XGBoost 클래스 가중치(Class Weighting) 비교 파이프라인

비교 조건:
1. XGBoost (No Weight: scale_pos_weight = 1.0) - 가중치 미부여 베이스라인
2. XGBoost (Balanced Weight: scale_pos_weight = N_neg / N_pos ≈ 7.55) - 불균형 비율 역수 페널티
3. XGBoost (Moderate Weight: scale_pos_weight = sqrt(N_neg / N_pos) ≈ 2.75) - 중간 가중치 완충 실험

평가 설계 원칙:
- 전체 데이터를 Train(80%)과 독립 Test(20%)로 분할 (Stratified).
- 독립 Test 셋은 원본 불균형 상태(가입률 11.7%) 그대로 유지하여 현실 일반화 성능 측정.
- 사후 편향 정보인 duration 컬럼은 통화 전 마케팅 예측 관점에서 제외.
- 평가 지표: Precision, Recall, F1-Score, ROC-AUC, PR-AUC(Average Precision), Confusion Matrix.
- 5-Fold Stratified Cross-Validation을 통한 검증 점수와 최종 독립 Test 점수 대조(과대적합 점검).
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
    confusion_matrix, precision_recall_curve, classification_report
)
from xgboost import XGBClassifier

# 한글 폰트 및 마이너스 기호 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parent

def evaluate_predictions(y_true, y_prob, threshold=0.5):
    """
    불균형 분류 평가지표 산출
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC (AP)": average_precision_score(y_true, y_prob),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Balanced Acc": balanced_accuracy_score(y_true, y_pred),
        "TN": int(tn), "FP": int(fp), "fn": int(fn), "tp": int(tp)
    }

def build_preprocessor(numeric_cols, categorical_cols):
    """
    scikit-learn 전처리 파이프라인
    """
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric_cols),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", sparse_output=False)), categorical_cols)
    ])

def run_xgboost_weight_comparison(args):
    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print(" 1. Bank Marketing 데이터셋 로드 및 전처리 분할")
    print("=" * 85)

    df = pd.read_csv(data_path, sep=";")
    print(f"▶ 데이터 파일: {data_path.resolve()}")
    print(f"▶ 원본 데이터 크기: {len(df):,}행 x {df.shape[1]}열")

    # 타깃 변수 수치화 (no: 0, yes: 1)
    y = df["y"].eq("yes").astype(int).to_numpy()

    # 사전 예측을 위해 duration 제외
    X = df.drop(columns=["y", "duration"], errors="ignore").copy()

    # pdays 특성 전처리
    if "pdays" in X:
        X["previously_contacted"] = X["pdays"].ne(-1).astype(int)
        X["pdays"] = X["pdays"].replace(-1, np.nan)

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.columns.difference(numeric_cols).tolist()

    print(f"▶ 수치형 특성({len(numeric_cols)}개): {numeric_cols}")
    print(f"▶ 범주형 특성({len(categorical_cols)}개): {categorical_cols}")

    # Train (80%) vs Test (20%) 분할
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    balanced_weight = round(n_neg / n_pos, 4)
    moderate_weight = round(np.sqrt(n_neg / n_pos), 4)

    print(f"\n▶ Train Set (학습용 80%): {len(X_train_df):,}건 (No: {n_neg:,}건, Yes: {n_pos:,}건 / 가입률 {y_train.mean():.2%})")
    print(f"▶ Test Set  (평가용 20%): {len(X_test_df):,}건 (No: {(y_test == 0).sum():,}건, Yes: {(y_test == 1).sum():,}건 / 가입률 {y_test.mean():.2%}) - 격리 보관")
    print(f"▶ 계산된 클래스 불균형 비율 (N_neg / N_pos): {balanced_weight:.2f} : 1")

    # 실험 대상 정의
    experiments = [
        {"name": "1. XGBoost (No Weight)", "scale_pos_weight": 1.0, "desc": "가중치 미부여 (기본 1.0)"},
        {"name": "2. XGBoost (Balanced Weight)", "scale_pos_weight": balanced_weight, "desc": f"불균형 역수 가중치 ({balanced_weight:.2f})"},
        {"name": "3. XGBoost (Moderate Weight)", "scale_pos_weight": moderate_weight, "desc": f"완충 제곱근 가중치 ({moderate_weight:.2f})"},
    ]

    # 전처리 변환
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_train_proc = preprocessor.fit_transform(X_train_df)
    X_test_proc = preprocessor.transform(X_test_df)

    # -------------------------------------------------------------------------
    # 2단계: 5-Fold Stratified Cross-Validation (검증 점수 산출)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print(" 2. Train Set 내부 5-Fold Stratified Cross-Validation (누수 방지 검증)")
    print("=" * 85)

    cv_records = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    for exp in experiments:
        exp_name = exp["name"]
        spw = exp["scale_pos_weight"]
        print(f"\n▶ [{exp_name}] 5-Fold CV 진행 중... (scale_pos_weight={spw})")

        fold_f1s, fold_aps, fold_recalls, fold_precisions, fold_aucs = [], [], [], [], []

        for fold, (t_idx, v_idx) in enumerate(skf.split(X_train_df, y_train), 1):
            fold_preproc = build_preprocessor(numeric_cols, categorical_cols)
            f_X_tr = fold_preproc.fit_transform(X_train_df.iloc[t_idx])
            f_X_val = fold_preproc.transform(X_train_df.iloc[v_idx])
            f_y_tr, f_y_val = y_train[t_idx], y_train[v_idx]

            xgb = XGBClassifier(
                n_estimators=args.trees,
                learning_rate=args.lr,
                max_depth=args.depth,
                scale_pos_weight=spw,
                random_state=args.seed + fold,
                eval_metric="logloss",
                n_jobs=args.jobs
            )
            xgb.fit(f_X_tr, f_y_tr)
            val_prob = xgb.predict_proba(f_X_val)[:, 1]
            m = evaluate_predictions(f_y_val, val_prob, threshold=0.5)

            fold_f1s.append(m["F1-Score"])
            fold_aps.append(m["PR-AUC (AP)"])
            fold_recalls.append(m["Recall"])
            fold_precisions.append(m["Precision"])
            fold_aucs.append(m["ROC-AUC"])

        cv_records.append({
            "실험": exp_name,
            "CV AP (평균)": np.mean(fold_aps),
            "CV F1 (평균)": np.mean(fold_f1s),
            "CV Precision (평균)": np.mean(fold_precisions),
            "CV Recall (평균)": np.mean(fold_recalls),
            "CV ROC-AUC (평균)": np.mean(fold_aucs),
        })

    df_cv = pd.DataFrame(cv_records)
    print(df_cv.round(4).to_string(index=False))

    # -------------------------------------------------------------------------
    # 3단계: Train 전체 학습 및 독립 Test Set 평가
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print(" 3. Train 전체(80%) 학습 후 독립 Test Set(20%) 최종 평가")
    print("=" * 85)

    test_results = []
    test_probs = {}

    for exp in experiments:
        exp_name = exp["name"]
        spw = exp["scale_pos_weight"]

        t_start = perf_counter()
        xgb_final = XGBClassifier(
            n_estimators=args.trees,
            learning_rate=args.lr,
            max_depth=args.depth,
            scale_pos_weight=spw,
            random_state=args.seed,
            eval_metric="logloss",
            n_jobs=args.jobs
        )
        xgb_final.fit(X_train_proc, y_train)
        fit_time = perf_counter() - t_start

        prob_test = xgb_final.predict_proba(X_test_proc)[:, 1]
        test_probs[exp_name] = prob_test
        m_test = evaluate_predictions(y_test, prob_test, threshold=0.5)

        test_results.append({
            "실험": exp_name,
            "scale_pos_weight": spw,
            "Precision": m_test["Precision"],
            "Recall": m_test["Recall"],
            "F1-Score": m_test["F1-Score"],
            "PR-AUC (AP)": m_test["PR-AUC (AP)"],
            "ROC-AUC": m_test["ROC-AUC"],
            "Balanced Acc": m_test["Balanced Acc"],
            "학습시간(초)": round(fit_time, 2)
        })

    df_test = pd.DataFrame(test_results)
    
    # 종합 대조표 (CV 검증 점수 vs 최종 Test 점수)
    df_comparison = pd.DataFrame({
        "실험": df_test["실험"],
        "scale_pos_weight": df_test["scale_pos_weight"],
        "CV Val F1": df_cv["CV F1 (평균)"],
        "Test F1": df_test["F1-Score"],
        "CV Val AP": df_cv["CV AP (평균)"],
        "Test AP": df_test["PR-AUC (AP)"],
        "Test Precision": df_test["Precision"],
        "Test Recall": df_test["Recall"],
        "Test ROC-AUC": df_test["ROC-AUC"],
    })

    print("\n[최종 성능 및 과적합 점검 대조표]")
    print(df_comparison.round(4).to_string(index=False))

    # CSV 저장
    df_comparison.to_csv(output_dir / "xgboost_class_weight_comparison.csv", index=False, encoding="utf-8-sig")

    # -------------------------------------------------------------------------
    # 4단계: 시각화 차트 생성
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = ["No Weight (1.0)", f"Balanced ({balanced_weight:.1f})", f"Moderate ({moderate_weight:.1f})"]
    x_indices = np.arange(len(labels))

    # 차트 1: Precision vs Recall 트레이드오프
    ax1 = axes[0]
    bw = 0.35
    ax1.bar(x_indices - bw/2, df_test["Precision"], width=bw, label="Precision (정밀도)", color="#26A69A", alpha=0.9)
    ax1.bar(x_indices + bw/2, df_test["Recall"], width=bw, label="Recall (재현율)", color="#FFA726", alpha=0.9)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax1.set_ylim(0, 0.85)
    ax1.set_title("1. Precision vs Recall 트레이드오프", fontsize=11, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.legend()
    for i in range(len(labels)):
        ax1.text(i - bw/2, df_test["Precision"].iloc[i] + 0.015, f"{df_test['Precision'].iloc[i]:.3f}", ha='center', fontsize=8, fontweight='bold')
        ax1.text(i + bw/2, df_test["Recall"].iloc[i] + 0.015, f"{df_test['Recall'].iloc[i]:.3f}", ha='center', fontsize=8, fontweight='bold')

    # 차트 2: CV Val F1 vs Test F1 (과대적합 점검)
    ax2 = axes[1]
    ax2.bar(x_indices - bw/2, df_comparison["CV Val F1"], width=bw, label="CV 검증 F1", color="#5C6BC0", alpha=0.9)
    ax2.bar(x_indices + bw/2, df_comparison["Test F1"], width=bw, label="독립 Test F1", color="#AB47BC", alpha=0.9)
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax2.set_ylim(0, 0.60)
    ax2.set_title("2. CV 검증 F1 vs Test F1 (과대적합 점검)", fontsize=11, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.legend()
    for i in range(len(labels)):
        ax2.text(i + bw/2, df_comparison["Test F1"].iloc[i] + 0.015, f"{df_comparison['Test F1'].iloc[i]:.3f}", ha='center', fontsize=8, fontweight='bold')

    # 차트 3: Test Set PR 곡선
    ax3 = axes[2]
    colors = ["#78909C", "#EF5350", "#42A5F5"]
    for i, exp in enumerate(experiments):
        name = exp["name"]
        prob = test_probs[name]
        p, r, _ = precision_recall_curve(y_test, prob)
        ap = df_test.loc[df_test["실험"] == name, "PR-AUC (AP)"].values[0]
        ax3.plot(r, p, label=f"{labels[i]} (AP={ap:.3f})", color=colors[i], linewidth=2.0)

    ax3.axhline(y_test.mean(), linestyle="--", color="gray", label=f"기준선 ({y_test.mean():.1%})")
    ax3.set_xlabel("Recall (재현율)", fontsize=10)
    ax3.set_ylabel("Precision (정밀도)", fontsize=10)
    ax3.set_title("3. 독립 Test Set PR (정밀도-재현율) 곡선", fontsize=11, fontweight='bold')
    ax3.grid(linestyle='--', alpha=0.4)
    ax3.legend(fontsize=8, loc='upper right')

    plt.suptitle("Bank Marketing: XGBoost 클래스 가중치(scale_pos_weight) 적용 효과 비교", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    chart_file = output_dir / "xgboost_class_weight_dashboard.png"
    plt.savefig(chart_file, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"\n▶ 비교 시각화 대시보드 저장 완료: {chart_file}")
    print(f"▶ 결과 CSV 파일 저장 완료: {output_dir / 'xgboost_class_weight_comparison.csv'}")

def main():
    parser = argparse.ArgumentParser(description="Bank Marketing XGBoost Class Weighting Comparison")
    default_data = ROOT / "data/bank/bank-full.csv"
    if not default_data.exists():
        default_data = ROOT / "data/bank+marketing/bank-full.csv"
    default_output = ROOT / "output/bank_xgb_weight"

    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=-1)
    run_xgboost_weight_comparison(parser.parse_args())

if __name__ == "__main__":
    main()
