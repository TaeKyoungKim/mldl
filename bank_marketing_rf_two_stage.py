"""
Bank Marketing 데이터셋: 2단계 검증(Train/Val 5-Fold CV) 및 독립 Test 평가 파이프라인

데이터 분할 및 평가 구조:
1. 전체 데이터 (45,211건)
   ├─▶ [독립 Test Set (20%, 9,043건)]: 불균형 원본 비율 유지, 최종 모델 성능 평가 전까지 완전 격리 (금고 보관)
   └─▶ [Train Set (80%, 36,168건)]: 기법 비교 및 하이퍼파라미터 튜닝용 데이터
         │
         ▼ [5-Fold Stratified Cross-Validation]
         - Original (미적용)
         - Random Undersampling (1:1 축소)
         - Tomek Links (경계면 모호한 쌍 제거)
         - Edited Nearest Neighbours (ENN, 이웃 기반 노이즈 정제)
         ※ 불균형 데이터 핵심 지표(F1-Score, PR-AUC / Average Precision)로 최적 기법 자동 선정
         │
         ▼ [최적 기법으로 Train Set 전체(80%) 재학습 및 최적 분류 임계값(Threshold) 탐색]
         │
         ▼ [격리되었던 Test Set(20%)에서 최종 일반화 성능 및 과적합 여부 검증]
"""

import os
import sys
import json
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, balanced_accuracy_score,
    confusion_matrix, precision_recall_curve, classification_report
)
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours

# 한글 폰트 및 스타일
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parent
METHODS = ["original", "random", "tomek", "enn"]
METRICS = ["average_precision", "roc_auc", "f1", "precision", "recall", "balanced_accuracy"]

def evaluate_metrics(y_true, y_prob, threshold=0.5):
    """
    확률값과 임계값을 바탕으로 불균형 분류 평가지표 산출
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "average_precision": average_precision_score(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }

def build_preprocessor(numeric_cols, categorical_cols):
    """
    데이터 누수를 방지하는 scikit-learn 전처리 파이프라인
    """
    return ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()), numeric_cols),
        ("category", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                   OneHotEncoder(handle_unknown="ignore", sparse_output=False)), categorical_cols)
    ])

def run_pipeline(args):
    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print(" 1. Bank Marketing 데이터 로드 및 2단계 분할 (Train 80% vs Test 20%)")
    print("=" * 85)

    data = pd.read_csv(data_path, sep=";")
    print(f"▶ 원본 데이터셋 크기: {len(data):,}건 x {data.shape[1]}열")

    # 타깃 변수 수치화
    y_raw = data["y"].eq("yes").astype(int).to_numpy()

    # 현실적 사전 예측을 위해 사후 정보인 duration 제외
    X_raw = data.drop(columns=["y", "duration"], errors="ignore").copy()

    # pdays 처리 (-1: 이전 접촉 없음 플래그 생성 후 NaN 대체)
    if "pdays" in X_raw:
        X_raw["previously_contacted"] = X_raw["pdays"].ne(-1).astype(int)
        X_raw["pdays"] = X_raw["pdays"].replace(-1, np.nan)

    numeric_cols = X_raw.select_dtypes(include="number").columns.tolist()
    categorical_cols = X_raw.columns.difference(numeric_cols).tolist()

    # [1단계 핵심] 최종 독립 Test Set 20% 분리 (금고 격리: 모델 비교/학습에 절대 관여 안 함)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=args.seed, stratify=y_raw
    )

    print(f"▶ [Train Set] {len(X_train_df):,}건 (No: {(y_train == 0).sum():,}건, Yes: {(y_train == 1).sum():,}건 / 가입률 {y_train.mean():.2%})")
    print(f"▶ [Test Set]  {len(X_test_df):,}건 (No: {(y_test == 0).sum():,}건, Yes: {(y_test == 1).sum():,}건 / 가입률 {y_test.mean():.2%}) - 격리 보관")

    # -------------------------------------------------------------------------
    # 2단계: Train Set(80%) 내부에서 5-Fold CV로 언더샘플링 기법 비교 및 과적합 검증
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print(" 2. Train Set 내부 5-Fold Stratified Cross-Validation (누수 방지 검증)")
    print("=" * 85)

    cv_results = []
    oof_predictions = {m: np.zeros(len(y_train)) for m in METHODS}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    for fold, (t_idx, v_idx) in enumerate(skf.split(X_train_df, y_train), 1):
        print(f"\n[Fold {fold}/5 검증]")
        
        # Fold별 데이터 분할
        fold_X_train_raw = X_train_df.iloc[t_idx]
        fold_y_train = y_train[t_idx]
        fold_X_val_raw = X_train_df.iloc[v_idx]
        fold_y_val = y_train[v_idx]

        # Fold Train 데이터에서만 fit_transform (누수 차단)
        fold_preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        fold_X_train_proc = fold_preprocessor.fit_transform(fold_X_train_raw)
        fold_X_val_proc = fold_preprocessor.transform(fold_X_val_raw)

        samplers = {
            "original": None,
            "random": RandomUnderSampler(random_state=args.seed + fold, sampling_strategy=1.0),
            "tomek": TomekLinks(sampling_strategy="majority", n_jobs=args.jobs),
            "enn": EditedNearestNeighbours(sampling_strategy="majority", n_neighbors=3, kind_sel="all", n_jobs=args.jobs),
        }

        for method, sampler in samplers.items():
            if sampler is None:
                s_X_train, s_y_train = fold_X_train_proc, fold_y_train
            else:
                s_X_train, s_y_train = sampler.fit_resample(fold_X_train_proc, fold_y_train)

            rf = RandomForestClassifier(
                n_estimators=args.trees, max_features="sqrt", min_samples_leaf=1,
                random_state=args.seed + fold, n_jobs=args.jobs
            )
            rf.fit(s_X_train, s_y_train)

            # 검증 데이터(Validation)는 언더샘플링하지 않은 원래 불균형 분포 그대로 평가
            val_prob = rf.predict_proba(fold_X_val_proc)[:, 1]
            oof_predictions[method][v_idx] = val_prob

            eval_res = evaluate_metrics(fold_y_val, val_prob, threshold=0.5)
            eval_res.update({"fold": fold, "method": method})
            cv_results.append(eval_res)
            print(f"  - {method:8s} | Val AP: {eval_res['average_precision']:.4f} | Val F1: {eval_res['f1']:.4f} | Recall: {eval_res['recall']:.4f}")

    df_cv = pd.DataFrame(cv_results)
    df_cv_summary = df_cv.groupby("method")[METRICS].agg(["mean", "std"]).reindex(METHODS)
    df_cv_summary.columns = [f"{m}_{s}" for m, s in df_cv_summary.columns]

    print("\n" + "=" * 85)
    print(" 3. 5-Fold 교차 검증(CV) 성능 요약 (검증 점수)")
    print("=" * 85)
    display_cols = ["average_precision_mean", "f1_mean", "precision_mean", "recall_mean", "roc_auc_mean"]
    print(df_cv_summary[display_cols].round(4).to_string())

    # 불균형 데이터 챔피언 기법 선정 (Primary Metric: F1-Score 및 AP 종합)
    best_method = df_cv_summary["f1_mean"].idxmax()
    print(f"\n★ [최적 기법 선정]: 5-Fold CV 검증 결과 평균 F1-Score 1위인 '{best_method}' 선정!")

    # -------------------------------------------------------------------------
    # 3단계: 최적 기법으로 Train Set 전체(80%) 학습 및 최적 Threshold 탐색
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print(f" 4. 최적 기법({best_method})으로 Train Set(80%) 전체 학습 및 임계값 튜닝")
    print("=" * 85)

    full_train_preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_train_proc = full_train_preprocessor.fit_transform(X_train_df)
    X_test_proc = full_train_preprocessor.transform(X_test_df)

    # Train 전체에 샘플러 적용
    best_sampler = None
    if best_method == "random":
        best_sampler = RandomUnderSampler(random_state=args.seed, sampling_strategy=1.0)
    elif best_method == "tomek":
        best_sampler = TomekLinks(sampling_strategy="majority", n_jobs=args.jobs)
    elif best_method == "enn":
        best_sampler = EditedNearestNeighbours(sampling_strategy="majority", n_neighbors=3, kind_sel="all", n_jobs=args.jobs)

    if best_sampler is not None:
        X_train_final, y_train_final = best_sampler.fit_resample(X_train_proc, y_train)
    else:
        X_train_final, y_train_final = X_train_proc, y_train

    print(f"▶ 최종 학습 데이터 수: {len(X_train_final):,}건 (Class 0: {(y_train_final == 0).sum():,}건, Class 1: {(y_train_final == 1).sum():,}건)")

    final_rf = RandomForestClassifier(
        n_estimators=args.trees, max_features="sqrt", min_samples_leaf=1,
        random_state=args.seed, n_jobs=args.jobs
    )
    final_rf.fit(X_train_final, y_train_final)

    # OOF 예측값을 바탕으로 F1-Score를 극대화하는 최적의 Threshold 탐색
    oof_best_prob = oof_predictions[best_method]
    precisions, recalls, thresholds = precision_recall_curve(y_train, oof_best_prob)
    f1_scores = np.divide(2 * (precisions * recalls), (precisions + recalls), 
                          out=np.zeros_like(precisions), where=(precisions + recalls) != 0)
    best_thresh_idx = np.argmax(f1_scores)
    best_threshold = thresholds[min(best_thresh_idx, len(thresholds)-1)]
    print(f"▶ OOF 검증 기반 F1 최적화 임계값: {best_threshold:.4f} (기본 0.5 대비 튜닝)")

    # -------------------------------------------------------------------------
    # 4단계: 격리해둔 최종 Test Set(20%)에서 최종 성적표 산출 (과적합 검증)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print(" 5. 최종 독립 Test Set(20%) 평가 (과적합 및 일반화 성능 검증)")
    print("=" * 85)

    test_prob = final_rf.predict_proba(X_test_proc)[:, 1]

    # 기본 임계값 0.5 및 최적 임계값 적용 평가
    test_eval_default = evaluate_metrics(y_test, test_prob, threshold=0.5)
    test_eval_tuned = evaluate_metrics(y_test, test_prob, threshold=best_threshold)

    # 4가지 기법 모두 Test Set 성능 산출 (비교용)
    test_comparison = []
    for m in METHODS:
        # 각 기법별로 Train 전체 학습 후 Test 평가
        m_sampler = None
        if m == "random":
            m_sampler = RandomUnderSampler(random_state=args.seed, sampling_strategy=1.0)
        elif m == "tomek":
            m_sampler = TomekLinks(sampling_strategy="majority", n_jobs=args.jobs)
        elif m == "enn":
            m_sampler = EditedNearestNeighbours(sampling_strategy="majority", n_neighbors=3, kind_sel="all", n_jobs=args.jobs)

        if m_sampler is not None:
            m_X_tr, m_y_tr = m_sampler.fit_resample(X_train_proc, y_train)
        else:
            m_X_tr, m_y_tr = X_train_proc, y_train

        m_rf = RandomForestClassifier(n_estimators=args.trees, max_features="sqrt", min_samples_leaf=1,
                                      random_state=args.seed, n_jobs=args.jobs)
        m_rf.fit(m_X_tr, m_y_tr)
        m_test_prob = m_rf.predict_proba(X_test_proc)[:, 1]
        m_eval = evaluate_metrics(y_test, m_test_prob, threshold=0.5)
        m_eval["method"] = m
        test_comparison.append(m_eval)

    df_test_comp = pd.DataFrame(test_comparison).set_index("method")[METRICS]

    # 검증(Validation) 점수와 최종 평가(Test) 점수 대조표 (과적합 확인)
    df_overfit_check = pd.DataFrame({
        "CV Val AP": df_cv_summary["average_precision_mean"],
        "Test AP": df_test_comp["average_precision"],
        "CV Val F1": df_cv_summary["f1_mean"],
        "Test F1 (th=0.5)": df_test_comp["f1"],
        "Test F1 (th-tuned)": [test_eval_tuned["f1"] if m == best_method else np.nan for m in METHODS],
        "CV Val Recall": df_cv_summary["recall_mean"],
        "Test Recall": df_test_comp["recall"],
    })

    print(df_overfit_check.round(4).to_string())

    print("\n[최종 모델 혼동 행렬 (Test Set / Tuned Threshold)]")
    cm = confusion_matrix(y_test, (test_prob >= best_threshold).astype(int))
    print(f"TN: {cm[0,0]:,}, FP: {cm[0,1]:,} | FN: {cm[1,0]:,}, TP: {cm[1,1]:,}")

    # -------------------------------------------------------------------------
    # 5단계: 결과 시각화 및 리포트 저장
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. CV vs Test 과적합 점검 차트
    ax1 = axes[0]
    x_pos = np.arange(len(METHODS))
    w = 0.35
    ax1.bar(x_pos - w/2, df_overfit_check["CV Val F1"], width=w, label="CV 검증 F1", color="#5C6BC0", alpha=0.85)
    ax1.bar(x_pos + w/2, df_overfit_check["Test F1 (th=0.5)"], width=w, label="최종 Test F1", color="#26A69A", alpha=0.85)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(METHODS, fontweight='bold')
    ax1.set_ylim(0, 0.65)
    ax1.set_title("1. 과적합 점검 (CV 검증 F1 vs Test F1)", fontsize=11, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.legend()

    # 2. Test Set 정밀도-재현율(PR) 곡선
    ax2 = axes[1]
    for m in METHODS:
        # 각 기법 Test PR 곡선
        m_prob = oof_predictions[m]  # 단순 표기를 위해
        p, r, _ = precision_recall_curve(y_test, test_prob if m == best_method else m_prob[:len(y_test)])
        # 정확한 test_prob 계산
    # test용 곡선 다시 그리기
    for m in METHODS:
        # 각 기법의 test 예측 확률
        m_sampler = None
        if m == "random":
            m_sampler = RandomUnderSampler(random_state=args.seed, sampling_strategy=1.0)
        elif m == "tomek":
            m_sampler = TomekLinks(sampling_strategy="majority", n_jobs=args.jobs)
        elif m == "enn":
            m_sampler = EditedNearestNeighbours(sampling_strategy="majority", n_neighbors=3, kind_sel="all", n_jobs=args.jobs)
        m_X_tr, m_y_tr = m_sampler.fit_resample(X_train_proc, y_train) if m_sampler else (X_train_proc, y_train)
        m_rf = RandomForestClassifier(n_estimators=args.trees, max_features="sqrt", min_samples_leaf=1,
                                      random_state=args.seed, n_jobs=args.jobs).fit(m_X_tr, m_y_tr)
        p, r, _ = precision_recall_curve(y_test, m_rf.predict_proba(X_test_proc)[:, 1])
        ax2.plot(r, p, label=f"{m} (AP={average_precision_score(y_test, m_rf.predict_proba(X_test_proc)[:, 1]):.3f})")
    ax2.axhline(y_test.mean(), linestyle="--", color="gray", label=f"기준선 ({y_test.mean():.1%})")
    ax2.set_xlabel("Recall (재현율)")
    ax2.set_ylabel("Precision (정밀도)")
    ax2.set_title("2. 최종 Test Set 정밀도-재현율(PR) 곡선", fontsize=11, fontweight='bold')
    ax2.grid(linestyle='--', alpha=0.4)
    ax2.legend(fontsize=8)

    # 3. 최적 모델의 임계값에 따른 Precision / Recall / F1 변화
    ax3 = axes[2]
    pr_test, rec_test, th_test = precision_recall_curve(y_test, test_prob)
    f1_test = np.divide(2 * (pr_test * rec_test), (pr_test + rec_test),
                        out=np.zeros_like(pr_test), where=(pr_test + rec_test) != 0)
    ax3.plot(th_test, pr_test[:-1], label="Precision", color="#26A69A")
    ax3.plot(th_test, rec_test[:-1], label="Recall", color="#FFA726")
    ax3.plot(th_test, f1_test[:-1], label="F1-Score", color="#AB47BC", linewidth=2)
    ax3.axvline(best_threshold, linestyle="--", color="red", label=f"튜닝 임계값 ({best_threshold:.2f})")
    ax3.set_xlabel("분류 임계값 (Threshold)")
    ax3.set_ylabel("Score")
    ax3.set_title(f"3. 최적 모델({best_method}) 임계값 튜닝 곡선", fontsize=11, fontweight='bold')
    ax3.grid(linestyle='--', alpha=0.4)
    ax3.legend(fontsize=8)

    plt.suptitle("Bank Marketing 언더샘플링: 2단계(5-Fold CV + 독립 Test) 파이프라인 검증", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    chart_path = output_dir / "two_stage_evaluation_summary.png"
    plt.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close()

    # 결과 CSV 및 JSON 저장
    df_overfit_check.to_csv(output_dir / "overfit_check_comparison.csv", encoding="utf-8-sig")
    df_cv_summary.to_csv(output_dir / "cv_summary.csv", encoding="utf-8-sig")
    df_test_comp.to_csv(output_dir / "test_summary.csv", encoding="utf-8-sig")

    print(f"\n▶ 시각화 차트 저장 완료: {chart_path}")
    print(f"▶ 상세 요약 파일 저장 완료: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Bank Marketing RF 2-Stage Train/Val/Test Pipeline")
    default_data = ROOT / "data/bank/bank-full.csv"
    if not default_data.exists():
        default_data = ROOT / "data/bank+marketing/bank-full.csv"
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--output", type=Path, default=ROOT / "output/bank_rf_two_stage")
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=-1)
    run_pipeline(parser.parse_args())

if __name__ == "__main__":
    main()
