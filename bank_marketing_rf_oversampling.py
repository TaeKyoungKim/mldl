"""
Bank Marketing 데이터셋: Random Forest + 오버샘플링 비율별 비교 실험

실험 조건:
1. 원본 데이터 (Baseline, 오버샘플링 없음)
2. RandomOverSampler (비율 25% = 0.25)
3. SMOTENC (비율 25% = 0.25)
4. SMOTENC (비율 33% = 0.33)
5. SMOTENC (비율 50% = 0.50)

평가 설계 원칙:
- 전체 데이터를 Train(80%)과 독립 Test(20%)로 분할 (Stratified).
- 독립 Test 셋은 절대 오버샘플링하지 않고 원본 불균형 상태(가입률 11.7%) 그대로 유지하여 현실 일반화 성능 측정.
- 오버샘플링은 오직 Train Set에만 적용하여 데이터 누수(Data Leakage) 원천 차단.
- 범주형과 수치형이 혼합된 Bank 데이터셋 특성을 고려하여 SMOTENC(Synthetic Minority Over-sampling Technique for Nominal and Continuous) 적용.
- 불균형 평가 지표 산출: Precision, Recall, F1-Score, ROC-AUC, PR-AUC(Average Precision).
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

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, balanced_accuracy_score,
    confusion_matrix, precision_recall_curve, classification_report
)
from imblearn.over_sampling import RandomOverSampler, SMOTENC

# 한글 폰트 및 마이너스 기호 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parent

EXPERIMENTS = [
    {"id": 1, "name": "1. 원본 데이터", "method": "None", "ratio": None, "sampler": None},
    {"id": 2, "name": "2. RandomOverSampler (25%)", "method": "RandomOverSampler", "ratio": 0.25, "sampler": "ros"},
    {"id": 3, "name": "3. SMOTENC (25%)", "method": "SMOTENC", "ratio": 0.25, "sampler": "smotenc"},
    {"id": 4, "name": "4. SMOTENC (33%)", "method": "SMOTENC", "ratio": 0.33, "sampler": "smotenc"},
    {"id": 5, "name": "5. SMOTENC (50%)", "method": "SMOTENC", "ratio": 0.50, "sampler": "smotenc"},
]

def evaluate_predictions(y_true, y_prob, threshold=0.5):
    """
    불균형 데이터 평가 지표 산출
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
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    }

def run_oversampling_comparison(args):
    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print(" [1] Bank Marketing 데이터셋 로드 및 전처리 분할")
    print("=" * 85)

    df = pd.read_csv(data_path, sep=";")
    print(f"▶ 데이터 파일: {data_path.resolve()}")
    print(f"▶ 원본 데이터 크기: {len(df):,}행 x {df.shape[1]}열")

    # 타깃 변수 수치화 (no: 0, yes: 1)
    y = df["y"].eq("yes").astype(int).to_numpy()

    # 현실적 사전 마케팅 예측을 위해 사후 접촉 정보인 'duration' 제외
    X = df.drop(columns=["y", "duration"], errors="ignore").copy()

    # pdays 특성 처리: -1(미접촉)을 플래그 컬럼으로 분리하고 결측치로 전환
    if "pdays" in X:
        X["previously_contacted"] = X["pdays"].ne(-1).astype(int)
        X["pdays"] = X["pdays"].replace(-1, np.nan)

    # 수치형 및 범주형 컬럼 식별
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.columns.difference(numeric_cols).tolist()
    cat_mask = [col in categorical_cols for col in X.columns]

    print(f"▶ 수치형 특성({len(numeric_cols)}개): {numeric_cols}")
    print(f"▶ 범주형 특성({len(categorical_cols)}개): {categorical_cols}")

    # Train (80%) vs Test (20%) 분할 (불균형 계층화 유지)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    print(f"\n▶ Train Set (학습용 80%): {len(X_train_raw):,}건 (No: {(y_train == 0).sum():,}건, Yes: {(y_train == 1).sum():,}건, 가입률: {y_train.mean():.2%})")
    print(f"▶ Test Set  (평가용 20%): {len(X_test_raw):,}건 (No: {(y_test == 0).sum():,}건, Yes: {(y_test == 1).sum():,}건, 가입률: {y_test.mean():.2%}) - 절대 불균형 보존!")

    print("\n" + "=" * 85)
    print(" [2] 5가지 오버샘플링 실험 및 RandomForest 학습 진행")
    print("=" * 85)

    results = []
    test_probs = {}

    for exp in EXPERIMENTS:
        exp_name = exp["name"]
        sampler_type = exp["sampler"]
        ratio = exp["ratio"]

        print(f"\n▶ [{exp['id']}/5] {exp_name} 실행 중...")
        t_start = perf_counter()

        # 1. 오버샘플링 적용 (Train 데이터에만 적용)
        if sampler_type is None:
            # 원본 데이터 유지
            X_train_res = X_train_raw.copy()
            y_train_res = y_train.copy()
            sample_time = 0.0
        elif sampler_type == "ros":
            # RandomOverSampler
            ros = RandomOverSampler(sampling_strategy=ratio, random_state=args.seed)
            X_train_res, y_train_res = ros.fit_resample(X_train_raw, y_train)
            sample_time = perf_counter() - t_start
        elif sampler_type == "smotenc":
            # SMOTENC (범주형 마스크 필수)
            # 결측치를 임시 보정한 후 SMOTENC 수행
            imputer_num = SimpleImputer(strategy="median")
            imputer_cat = SimpleImputer(strategy="most_frequent")
            
            X_tr_imp = X_train_raw.copy()
            X_tr_imp[numeric_cols] = imputer_num.fit_transform(X_tr_imp[numeric_cols])
            X_tr_imp[categorical_cols] = imputer_cat.fit_transform(X_tr_imp[categorical_cols])

            smotenc = SMOTENC(categorical_features=cat_mask, sampling_strategy=ratio, random_state=args.seed)
            X_train_res, y_train_res = smotenc.fit_resample(X_tr_imp, y_train)
            sample_time = perf_counter() - t_start

        # 2. 전처리 파이프라인 (인코딩 & 스케일링) - 오버샘플링된 Train으로 Fit
        preprocessor = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric_cols),
            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                  OneHotEncoder(handle_unknown="ignore", sparse_output=False)), categorical_cols)
        ])

        X_train_proc = preprocessor.fit_transform(X_train_res)
        X_test_proc = preprocessor.transform(X_test_raw)

        # 3. RandomForest 학습
        t_fit_start = perf_counter()
        rf = RandomForestClassifier(
            n_estimators=args.trees,
            max_features="sqrt",
            min_samples_leaf=1,
            random_state=args.seed,
            n_jobs=args.jobs
        )
        rf.fit(X_train_proc, y_train_res)
        fit_time = perf_counter() - t_fit_start

        # 4. 독립 Test Set 예측 (임계값 0.5 기준)
        prob_test = rf.predict_proba(X_test_proc)[:, 1]
        test_probs[exp_name] = prob_test

        metrics = evaluate_predictions(y_test, prob_test, threshold=0.5)
        
        # 학습 데이터 클래스 분포 기록
        n_no = int((y_train_res == 0).sum())
        n_yes = int((y_train_res == 1).sum())
        current_ratio = round(n_yes / n_no * 100, 1)

        result_row = {
            "실험": exp_name,
            "학습 샘플수": len(y_train_res),
            "No(0)": n_no,
            "Yes(1)": n_yes,
            "Yes비율(%)": f"{current_ratio}%",
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1-Score": metrics["F1-Score"],
            "PR-AUC (AP)": metrics["PR-AUC (AP)"],
            "ROC-AUC": metrics["ROC-AUC"],
            "Balanced Acc": metrics["Balanced Acc"],
            "샘플링(초)": round(sample_time, 2),
            "학습(초)": round(fit_time, 2)
        }
        results.append(result_row)
        print(f"  → 완료: 샘플 수 {len(y_train_res):,}건 (Yes: {n_yes:,}건, {current_ratio}%) | F1: {metrics['F1-Score']:.4f} | Recall: {metrics['Recall']:.4f} | AP: {metrics['PR-AUC (AP)']:.4f}")

    df_results = pd.DataFrame(results)

    # -------------------------------------------------------------------------
    # [3] 결과 출력 및 상세 분석
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print(" [3] 오버샘플링 실험 결과 종합 비교표 (독립 Test Set 평가)")
    print("=" * 85)
    display_cols = ["실험", "학습 샘플수", "Yes비율(%)", "Precision", "Recall", "F1-Score", "PR-AUC (AP)", "ROC-AUC"]
    print(df_results[display_cols].to_string(index=False))

    # CSV 저장
    df_results.to_csv(output_dir / "oversampling_rf_comparison.csv", index=False, encoding="utf-8-sig")

    # -------------------------------------------------------------------------
    # [4] 시각화 대시보드 생성
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    exp_labels = [e["name"].split(". ")[-1] for e in EXPERIMENTS]
    x_indices = np.arange(len(exp_labels))

    # 차트 1: Precision vs Recall Trade-off
    ax1 = axes[0]
    bar_width = 0.35
    ax1.bar(x_indices - bar_width/2, df_results["Precision"], width=bar_width, label="Precision (정밀도)", color="#26A69A", alpha=0.9)
    ax1.bar(x_indices + bar_width/2, df_results["Recall"], width=bar_width, label="Recall (재현율)", color="#FFA726", alpha=0.9)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(exp_labels, rotation=25, ha='right', fontsize=9, fontweight='bold')
    ax1.set_ylim(0, 0.75)
    ax1.set_title("1. Precision vs Recall 트레이드오프", fontsize=11, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.legend()

    # 차트 2: F1-Score 및 PR-AUC (Average Precision)
    ax2 = axes[1]
    ax2.bar(x_indices - bar_width/2, df_results["F1-Score"], width=bar_width, label="F1-Score", color="#AB47BC", alpha=0.9)
    ax2.bar(x_indices + bar_width/2, df_results["PR-AUC (AP)"], width=bar_width, label="PR-AUC (AP)", color="#42A5F5", alpha=0.9)
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(exp_labels, rotation=25, ha='right', fontsize=9, fontweight='bold')
    ax2.set_ylim(0, 0.60)
    ax2.set_title("2. F1-Score 및 PR-AUC 비교", fontsize=11, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.legend()
    for i in range(len(exp_labels)):
        ax2.text(i - bar_width/2, df_results["F1-Score"].iloc[i] + 0.01, f"{df_results['F1-Score'].iloc[i]:.3f}", ha='center', fontsize=8, fontweight='bold')

    # 차트 3: 독립 Test Set PR 곡선
    ax3 = axes[2]
    colors = ["#78909C", "#EF5350", "#26A69A", "#42A5F5", "#AB47BC"]
    for i, exp in enumerate(EXPERIMENTS):
        name = exp["name"]
        prob = test_probs[name]
        p, r, _ = precision_recall_curve(y_test, prob)
        ap = df_results.loc[df_results["실험"] == name, "PR-AUC (AP)"].values[0]
        ax3.plot(r, p, label=f"{name.split('. ')[-1]} (AP={ap:.3f})", color=colors[i], linewidth=1.8)

    ax3.axhline(y_test.mean(), linestyle="--", color="gray", label=f"기준선 ({y_test.mean():.1%})")
    ax3.set_xlabel("Recall (재현율)", fontsize=10)
    ax3.set_ylabel("Precision (정밀도)", fontsize=10)
    ax3.set_title("3. 독립 Test Set PR (정밀도-재현율) 곡선", fontsize=11, fontweight='bold')
    ax3.grid(linestyle='--', alpha=0.4)
    ax3.legend(fontsize=8, loc='upper right')

    plt.suptitle("Bank Marketing: RandomForest 오버샘플링 기법 및 비율별 성능 비교", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    chart_file = output_dir / "oversampling_comparison_dashboard.png"
    plt.savefig(chart_file, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"\n▶ 비교 시각화 대시보드 저장 완료: {chart_file}")
    print(f"▶ 결과 CSV 파일 저장 완료: {output_dir / 'oversampling_rf_comparison.csv'}")

def main():
    parser = argparse.ArgumentParser(description="Bank Marketing RF Oversampling Comparison")
    default_data = ROOT / "data/bank/bank-full.csv"
    if not default_data.exists():
        default_data = ROOT / "data/bank+marketing/bank-full.csv"
    default_output = ROOT / "output/bank_rf_oversampling"
    
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=-1)
    run_oversampling_comparison(parser.parse_args())

if __name__ == "__main__":
    main()
