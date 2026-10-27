"""Random Forest + undersampling: leakage-safe Stratified 5-Fold comparison.

Run: python bank_marketing_rf_comparison.py
Dependencies: requirements-bank-undersampling.txt
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
                             recall_score, f1_score, balanced_accuracy_score,
                             confusion_matrix, precision_recall_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


ROOT = Path(__file__).resolve().parent
METHODS = ["original", "random", "tomek", "enn"]
SCORES = ["average_precision", "roc_auc", "precision", "recall", "f1", "balanced_accuracy"]


def evaluate(y, probability):
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return dict(average_precision=average_precision_score(y, probability),
                roc_auc=roc_auc_score(y, probability),
                precision=precision_score(y, predicted, zero_division=0),
                recall=recall_score(y, predicted, zero_division=0),
                f1=f1_score(y, predicted, zero_division=0),
                balanced_accuracy=balanced_accuracy_score(y, predicted),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def run(args):
    data = pd.read_csv(args.data, sep=";")
    if "y" not in data or data.empty or not data.y.isin(["yes", "no"]).all():
        raise ValueError("y가 yes/no인 비어 있지 않은 데이터가 필요합니다.")
    y = data.y.eq("yes").astype(int).to_numpy()
    if min(np.bincount(y, minlength=2)) < 5:
        raise ValueError("각 클래스에 최소 5개 샘플이 필요합니다.")
    x = data.drop(columns=["y", "duration"], errors="ignore").copy()
    if "pdays" in x:
        x["previously_contacted"] = x.pdays.ne(-1).astype(int)
        x["pdays"] = x.pdays.replace(-1, np.nan)
    numeric = x.select_dtypes(include="number").columns.tolist()
    categorical = x.columns.difference(numeric).tolist()
    args.output.mkdir(parents=True, exist_ok=True)
    def save(frame, name):
        frame.to_csv(args.output / f"{name}.csv", index=False, encoding="utf-8-sig")

    config = {"data": str(args.data.resolve()), "seed": args.seed, "folds": 5,
              "shuffle": True, "n_estimators": args.trees, "min_samples_leaf": 1,
              "max_features": "sqrt", "class_weight": None, "threshold": 0.5,
              "random_ratio": 1.0, "enn_neighbors": 3, "enn_kind_sel": "all",
              "duration_excluded": True, "sampling_strategy_cleaners": "majority"}
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    oof = {method: np.full(len(y), np.nan) for method in METHODS}
    fold_ids = np.zeros(len(y), dtype=int)
    fold_rows, sampling_rows = [], []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x, y), 1):
        fold_ids[valid_idx] = fold
        preprocessor = ColumnTransformer([
            ("numeric", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()), numeric),
            ("category", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                       OneHotEncoder(handle_unknown="ignore", sparse_output=False)), categorical),
        ])
        # 모든 기법이 동일한 학습 폴드에서 fit한 전처리를 공유합니다.
        x_train = preprocessor.fit_transform(x.iloc[train_idx])
        x_valid = preprocessor.transform(x.iloc[valid_idx])
        y_train, y_valid = y[train_idx], y[valid_idx]
        samplers = {
            "original": None,
            "random": RandomUnderSampler(random_state=args.seed + fold, sampling_strategy=1.0),
            "tomek": TomekLinks(sampling_strategy="majority", n_jobs=args.jobs),
            "enn": EditedNearestNeighbours(sampling_strategy="majority", n_neighbors=3,
                                           kind_sel="all", n_jobs=args.jobs),
        }
        for method, sampler in samplers.items():
            print(f"Fold {fold}/5: {method}", flush=True)
            start = perf_counter()
            if sampler is None:
                sampled_x, sampled_y = x_train, y_train
                selected = np.arange(len(train_idx))
            else:
                sampled_x, sampled_y = sampler.fit_resample(x_train, y_train)
                selected = sampler.sample_indices_
            sampling_seconds = perf_counter() - start
            assert int(sampled_y.sum()) == int(y_train.sum()), "소수 클래스 수가 변경되었습니다."
            assert not np.intersect1d(train_idx[selected], valid_idx).size, "학습/검증 누수"
            save(pd.DataFrame({"source_row_index": train_idx[selected]}), f"selected_fold{fold}_{method}")
            model = RandomForestClassifier(n_estimators=args.trees, max_features="sqrt",
                                           min_samples_leaf=1, class_weight=None,
                                           random_state=args.seed + fold, n_jobs=args.jobs)
            start = perf_counter()
            model.fit(sampled_x, sampled_y)
            fit_seconds = perf_counter() - start
            probability = model.predict_proba(x_valid)[:, 1]
            oof[method][valid_idx] = probability
            fold_rows.append(dict(fold=fold, method=method, **evaluate(y_valid, probability)))
            sampling_rows.append(dict(fold=fold, method=method, train_before=len(train_idx),
                                      no_before=int((y_train == 0).sum()), yes_before=int(y_train.sum()),
                                      train_after=len(sampled_y), no_after=int((sampled_y == 0).sum()),
                                      yes_after=int(sampled_y.sum()), removed=len(train_idx)-len(sampled_y),
                                      validation_size=len(valid_idx), validation_yes=int(y_valid.sum()),
                                      sampling_seconds=sampling_seconds, fit_seconds=fit_seconds))
    scores = pd.DataFrame(fold_rows)
    sampling = pd.DataFrame(sampling_rows)
    save(scores, "fold_scores")
    save(sampling, "fold_sampling")
    grouped = scores.groupby("method")[SCORES].agg(["mean", "std"]).reindex(METHODS)
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    save(grouped.reset_index(), "cv_summary")
    oof_frame = pd.DataFrame({"source_row_index": np.arange(len(y)), "fold": fold_ids, "y": y})
    pooled_rows = []
    for method in METHODS:
        assert np.isfinite(oof[method]).all(), "OOF 예측 누락"
        oof_frame[f"{method}_probability"] = oof[method]
        pooled_rows.append(dict(method=method, **evaluate(y, oof[method])))
    save(oof_frame, "oof_predictions")
    save(pd.DataFrame(pooled_rows), "oof_scores")
    # 같은 fold끼리 차이를 비교하여 분할 차이의 영향을 통제합니다.
    paired = []
    for metric in SCORES:
        pivot = scores.pivot(index="fold", columns="method", values=metric)
        for method in METHODS[1:]:
            delta = pivot[method] - pivot.original
            paired.append(dict(metric=metric, method=method, mean_difference=delta.mean(),
                               std_difference=delta.std(), improved_folds=int((delta > 0).sum())))
    save(pd.DataFrame(paired), "paired_difference_vs_original")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    positions = np.arange(len(METHODS))
    palette = ["#4A90E2", "#E94E77", "#50E3C2", "#F5A623"]
    for offset, (metric, color) in enumerate(zip(["precision", "recall", "f1", "average_precision"], palette)):
        axes[0].bar(positions + (offset - 1.5)*0.2, grouped[f"{metric}_mean"],
                    yerr=grouped[f"{metric}_std"], width=0.2, label=metric, capsize=3, color=color, alpha=0.9)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(METHODS, fontsize=10, fontweight='bold')
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Stratified 5-Fold: 평균 ± 표준편차 (Mean ± SD)", fontsize=11, fontweight='bold')
    axes[0].grid(axis='y', linestyle='--', alpha=0.4)
    axes[0].legend(fontsize=9, loc='upper left')

    for method in METHODS:
        precision, recall, _ = precision_recall_curve(y, oof[method])
        axes[1].plot(recall, precision, label=method, linewidth=1.8)
    axes[1].axhline(y.mean(), linestyle="--", color="gray", label=f"Prevalence ({y.mean():.1%})")
    axes[1].set_xlabel("Recall (재현율)", fontsize=10)
    axes[1].set_ylabel("Precision (정밀도)", fontsize=10)
    axes[1].set_title("Pooled Out-Of-Fold PR 곡선", fontsize=11, fontweight='bold')
    axes[1].grid(linestyle='--', alpha=0.4)
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(args.output / "rf_comparison.png", dpi=170)
    plt.close(fig)
    lines = []
    for method in METHODS:
        row = grouped.loc[method]
        lines.append(f"| {method} | " + " | ".join(
            f"{row[f'{metric}_mean']:.4f} ± {row[f'{metric}_std']:.4f}" for metric in SCORES) + " |")
    best_ap = grouped.average_precision_mean.idxmax()
    best_f1 = grouped.f1_mean.idxmax()
    report = f"""# Random Forest와 언더샘플링: Stratified 5-Fold

데이터 {len(y):,}행, 가입률 {y.mean():.2%}. StratifiedKFold(5, shuffle=True, random_state={args.seed}).
각 행은 정확히 한 번 검증에 사용하며, 네 방법 모두 동일한 분할을 사용합니다.

| 방법 | AP | ROC-AUC | Precision | Recall | F1 | Balanced accuracy |
|---|---|---|---|---|---|---|
{chr(10).join(lines)}

숫자는 5개 폴드 평균 ± 표본 표준편차입니다. 표준편차는 신뢰구간이 아닙니다.
평균 AP가 가장 높은 방법은 **{best_ap}**, 평균 F1이 가장 높은 방법은 **{best_f1}**입니다.
AP는 average precision이며 PR 곡선 사다리꼴 적분과 구별됩니다.

## 비교 조건

- RandomForestClassifier: 나무 {args.trees}개, max_features='sqrt', min_samples_leaf=1, class_weight=None, 나머지 기본값.
- 임계값은 모든 방법에서 0.5로 고정합니다. 튜닝은 수행하지 않았습니다.
- 원본 기준 모델도 포함합니다. 샘플링을 추가하는 것이 실제 이득인지 확인하기 위한 비교입니다.
- Random은 1:1로 축소합니다. Tomek/ENN은 다수 클래스만 정제하므로 균형 비율을 강제하지 않습니다.
- ENN은 이웃 3개, kind_sel='all'입니다. 제거된 행을 오라벨로 확정하지 않습니다.
- 학습 폴드에만 전처리 fit 및 샘플링을 적용합니다. 검증 폴드는 원래 비율을 유지합니다.
- 수치형 중앙값 대체/표준화, 범주형 최빈값 대체/원핫인코딩을 사용합니다. 표준화는 거리 기반 샘플링을 위한 것입니다.
- pdays=-1은 이전 접촉 여부로 분리하고 일수는 결측 처리 후 학습 중앙값으로 대체합니다. unknown은 범주로 유지합니다.
- 통화 전 예측을 가정하여 duration을 제외합니다. 범주 인코딩과 수치 스케일 선택은 이웃 관계에 영향을 줍니다.

## 결과를 읽는 방법

- Precision은 예측 가입자 중 실제 가입 비율, Recall은 실제 가입자를 찾아낸 비율입니다. Random은 Recall 증가와 오탐 증가가 함께 나타날 수 있습니다.
- AP/ROC-AUC는 확률 순위 평가이며, Precision/Recall/F1은 임계값 0.5에서의 평가입니다. 둘을 구분하여 비교하세요.
- OOF PR 곡선과 oof_scores는 폴드 예측을 합친 결과이며 폴드 점수 평균과 정확히 일치하지 않습니다.
- paired_difference_vs_original.csv는 동일 폴드 기준 점수 차이와 개선된 폴드 수를 제공합니다. 작은 차이를 통계적으로 유의하다고 단정하지 않습니다.
- 언더샘플링으로 학습 클래스 비율이 바뀌므로 예측 확률을 실제 가입 확률로 곧바로 해석하지 마세요.
- 이번 결과로 방법을 선택하면 그 CV 점수에는 선택 편향이 생길 수 있습니다. 최종 성능 주장은 별도 홀드아웃 또는 중첩 CV가 필요합니다.
- 무작위 층화 검증 결과이며 미래 시점 성능을 검증하는 시간 순서 분할은 아닙니다.

## 파일

cv_summary.csv: 평균/표준편차, fold_scores.csv: 폴드별 지표, fold_sampling.csv: 클래스 수 및 시간.
oof_predictions.csv: 원본 행별 검증 예측, oof_scores.csv: 합친 OOF 지표 및 혼동행렬.
selected_fold*.csv: 폴드별 선택된 학습 원본 행 인덱스. config.json: 실험 조건.
rf_comparison.png: 지표 비교 및 OOF PR 곡선.
"""
    (args.output / "README.md").write_text(report, encoding="utf-8")
    print(grouped.round(4).to_string(), flush=True)
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RF + Random/Tomek/ENN, Stratified 5-Fold")
    # C:/apps/mldl/data/bank/bank-full.csv 우선 탐색 및 fallback 지원
    default_data = ROOT / "data/bank/bank-full.csv"
    if not default_data.exists():
        default_data = ROOT / "data/bank+marketing/bank-full.csv"
    default_output = ROOT / "output/bank_rf_undersampling_5fold"
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=-1)
    run(parser.parse_args())
