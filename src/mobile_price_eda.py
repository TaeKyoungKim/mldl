"""스마트폰 가격 등급 EDA. 실행: uv run python mobile_price_eda.py

원본을 변경하거나 모델을 학습하지 않는다. Markdown 보고서와 표/차트를 생성한다.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent if ROOT.name == "src" else ROOT
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output" / "mobile_price_eda"

if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    plt.rc("font", family="Malgun Gothic")
    plt.rcParams["axes.unicode_minus"] = False

TARGET = "price_range"
BINARY = ["blue", "dual_sim", "four_g", "three_g", "touch_screen", "wifi"]
FEATURES = ["battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g",
            "int_memory", "m_dep", "mobile_wt", "n_cores", "pc", "px_height",
            "px_width", "ram", "sc_h", "sc_w", "talk_time", "three_g", "touch_screen", "wifi"]


def read_data(path, training):
    df = pd.read_csv(path)
    expected = set(FEATURES + ([TARGET] if training else ["id"]))
    if set(df.columns) != expected or df.empty:
        raise ValueError(f"{path}: 컬럼 구성 또는 행 개수를 확인하세요.")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if training and (df[TARGET].isna().any() or not df[TARGET].isin([0, 1, 2, 3]).all()):
        raise ValueError("price_range는 결측 없는 0, 1, 2, 3이어야 합니다.")
    return df


def inspect_ranges(df, split):
    """확실한 범위 오류와 의미 확인이 필요한 의심값을 구분한다."""
    checks = {}
    for col in FEATURES:
        checks[(col, "missing", "결측")] = df[col].isna()
        checks[(col, "nonfinite", "무한값")] = df[col].notna() & ~np.isfinite(df[col])
    for col in BINARY:
        checks[(col, "invalid", "0/1 이외 값")] = df[col].notna() & ~df[col].isin([0, 1])
    for col in set(FEATURES) - set(BINARY):
        checks[(col, "invalid", "음수")] = df[col].lt(0)
    for col in ["battery_power", "clock_speed", "int_memory", "m_dep", "mobile_wt",
                "ram", "sc_h", "talk_time"]:
        checks[(col, "review", "0값 의미 확인")] = df[col].eq(0)
    for col in ["sc_w", "px_height", "px_width"]:
        checks[(col, "review", "화면 크기/해상도 0값")] = df[col].eq(0)
    checks[("n_cores", "invalid", "양의 정수 아님")] = df.n_cores.notna() & (
        df.n_cores.le(0) | df.n_cores.mod(1).ne(0))
    checks[("fc,pc", "review", "전면 카메라 값이 후면보다 큼")] = df.fc.gt(df.pc)
    checks[("four_g,three_g", "review", "4G=1, 3G=0 조합")] = df.four_g.eq(1) & df.three_g.eq(0)
    rows = []
    for (column, severity, reason), mask in checks.items():
        for idx in df.index[mask]:
            rows.append({"split": split, "csv_row": int(idx) + 2,
                         "id": df.at[idx, "id"] if "id" in df else None,
                         "column": column, "severity": severity, "reason": reason,
                         "value": ", ".join(str(df.at[idx, col]) for col in column.split(","))})
    return pd.DataFrame(rows, columns=["split", "csv_row", "id", "column", "severity", "reason", "value"])


def md_table(df):
    # tabulate를 추가 설치하지 않고 작은 Markdown 표를 만든다.
    def cell(value):
        if pd.isna(value):
            return "—"
        return f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |",
                       "| " + " | ".join(["---"] * len(df.columns)) + " |"] +
                      ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False, name=None)])


def self_check():
    sample = pd.DataFrame({col: [1., 1.] for col in FEATURES})
    sample.loc[0, "sc_w"] = 0
    sample.loc[1, "blue"] = 2
    sample.loc[1, "ram"] = -1
    sample.loc[0, "fc"] = 0  # 카메라 0은 미지원일 수 있어 오류로 보지 않는다.
    findings = inspect_ranges(sample, "train")
    assert ((findings.column == "sc_w") & (findings.severity == "review")).any()
    assert ((findings.column == "blue") & (findings.severity == "invalid")).any()
    assert ((findings.column == "ram") & (findings.severity == "invalid")).any()
    assert not (findings.column == "fc").any()
    assert "| count |" in md_table(pd.DataFrame({"count": [2]}))
    print("Self-check passed: invalid values, review values, camera zero, Markdown.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    train = read_data(DATA_DIR / "mobile_price_category_train.csv", True)
    test = read_data(DATA_DIR / "mobile_price_category_test.csv", False)
    output = OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    numeric_train = train.replace([np.inf, -np.inf], np.nan)
    numeric_test = test.replace([np.inf, -np.inf], np.nan)
    quality = pd.DataFrame([{
        "split": name, "rows": len(df), "columns": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "infinite_cells": int(np.isinf(df.to_numpy()).sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_features": int(df.duplicated(subset=FEATURES).sum()),
        "constant_features": ", ".join(col for col in FEATURES if df[col].nunique() <= 1),
    } for name, df in [("train", train), ("test", test)]])
    overlap = train[FEATURES].drop_duplicates().merge(test[FEATURES].drop_duplicates(), on=FEATURES)
    conflicts = int(train.groupby(FEATURES, dropna=False)[TARGET].nunique().gt(1).sum())
    issues = pd.concat([inspect_ranges(train, "train"), inspect_ranges(test, "test")], ignore_index=True)
    counts = train[TARGET].value_counts().sort_index().rename_axis(TARGET).reset_index(name="count")
    counts["percent"] = counts["count"] / len(train) * 100
    correlation = numeric_train[FEATURES + [TARGET]].corr(method="spearman")
    ranking = correlation[TARGET].drop(TARGET).rename("spearman").to_frame()
    ranking["absolute"] = ranking.spearman.abs()
    ranking = ranking.sort_values("absolute", ascending=False).reset_index(names="feature")
    drift, outliers = [], []
    for col in FEATURES:
        a, b = numeric_train[col].dropna(), numeric_test[col].dropna()
        q1, q3 = a.quantile([0.25, 0.75])
        low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        outside = b.lt(a.min()) | b.gt(a.max())
        drift.append({"feature": col, "train_mean": a.mean(), "test_mean": b.mean(),
                      "train_median": a.median(), "test_median": b.median(),
                      "ks_statistic": ks_2samp(a, b).statistic if len(a) and len(b) else np.nan,
                      "test_outside_train_range": int(outside.sum())})
        if col not in BINARY:
            outliers.append({"feature": col, "train_IQR_low": low, "train_IQR_high": high,
                             "train_outside": int((a.lt(low) | a.gt(high)).sum()),
                             "test_outside": int((b.lt(low) | b.gt(high)).sum())})
    drift = pd.DataFrame(drift).sort_values("ks_statistic", ascending=False)
    issue_counts = issues.groupby(["split", "column", "severity", "reason"]).size().reset_index(name="count")
    grouped = numeric_train.groupby(TARGET)[FEATURES].agg(["mean", "median"])
    tables = {"quality": quality, "class_balance": counts, "issues": issues,
              "issue_counts": issue_counts, "target_associations": ranking,
              "distribution_shift": drift, "iqr_review": pd.DataFrame(outliers)}
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    grouped.to_csv(output / "class_feature_summary.csv")
    correlation.to_csv(output / "spearman_correlation.csv")
    pd.concat({"train": numeric_train.describe().T, "test": numeric_test.describe().T}).to_csv(output / "describe.csv")
    pd.DataFrame({"dtype": train.dtypes.astype(str), "train_unique": train.nunique(),
                  "test_unique": test.nunique()}).to_csv(output / "schema.csv", index_label="column")
    sns.set_theme(style="whitegrid")

    def save(name):
        plt.tight_layout()
        plt.savefig(output / name, dpi=140, bbox_inches="tight")
        plt.close()

    counts.plot.bar(x=TARGET, y="count", legend=False, rot=0, title="Price category counts", figsize=(7, 4))
    save("class_balance.png")
    fig, axes = plt.subplots(5, 4, figsize=(16, 17))
    for col, ax in zip(FEATURES, axes.flat):
        bins = [-0.5, 0.5, 1.5] if col in BINARY else np.histogram_bin_edges(
            pd.concat([numeric_train[col], numeric_test[col]]).dropna(), bins=20)
        sns.histplot(numeric_train[col], ax=ax, bins=bins, stat="probability", color="steelblue", label="train")
        sns.histplot(numeric_test[col], ax=ax, bins=bins, stat="probability", color="orange", alpha=0.4, label="test")
        ax.set_title(col)
        ax.set_ylabel("Proportion")
    axes.flat[0].legend()
    save("feature_distributions.png")
    plt.figure(figsize=(15, 12))
    sns.heatmap(correlation, cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title("Spearman correlations (association, not causation)")
    save("correlation.png")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    top = ranking["feature"].head(6).tolist()
    for col, ax in zip(top, axes.flat):
        sns.boxplot(data=numeric_train, x=TARGET, y=col, ax=ax)
        ax.set_title(f"{col} by price category")
    save("features_by_category.png")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for col, ax in zip(BINARY, axes.flat):
        rates = pd.crosstab(train[TARGET], train[col], normalize="index").reindex(columns=[0, 1], fill_value=0)
        rates.plot.bar(stacked=True, ax=ax, rot=0, color=["lightgray", "steelblue"])
        ax.set_title(col)
        ax.set_ylabel("Within-category proportion")
    save("binary_by_category.png")

    report = f"""# 스마트폰 가격 등급 EDA 과정과 결과

## 1. 분석 목적과 문제 정의

휴대폰 사양 20개가 가격 등급 `price_range`와 어떤 관계를 가지는지 확인한다.
타깃은 실제 금액이 아닌 0~3의 네 등급이므로 다중 분류 문제로 준비한다.
숫자 순서를 활용한 상관분석은 보조 진단이며 등급 간 가격 차이가 같다는 뜻은 아니다.
등급의 실제 금액 경계와 사양 단위는 CSV에 없으므로 원본 설명을 확인해야 한다.

입력 파일은 `data/mobile_price_category_train.csv`, `data/mobile_price_category_test.csv`이다.
테스트에는 `price_range`가 없으므로 테스트 정확도나 등급별 분포를 계산할 수 없다.
`id`는 결과 식별용으로 보관하고 학습 입력과 상관분석에서 제외한다.

## 2. 구조와 정합성부터 확인

컬럼 일치, 숫자형 변환, 행 개수, 결측·무한값·중복·상수 컬럼을 확인한다.
학습·테스트에서 같은 사양 조합이 반복되는지도 조사한다.

{md_table(quality)}

- 학습·테스트 공통 사양 조합: {len(overlap)}개.
- 동일 사양인데 타깃이 다른 학습 조합: {conflicts}개.
- 테스트 id 결측: {int(test.id.isna().sum())}개, 중복: {int(test.id.duplicated().sum())}개.

특성에 결측이 있으면 대체값은 이후 학습 폴드에서만 계산한다. 중복은 출처를 확인하고
같은 기기가 검증 양쪽에 들어가지 않도록 처리한다. EDA 코드 자체는 원본을 수정하지 않는다.

## 3. 타깃 균형 확인

{md_table(counts)}

![가격 등급별 개수](output/mobile_price_eda/class_balance.png)

등급 불균형은 Accuracy 해석에 영향을 준다. 향후 검증에는 층화 분할과 Macro-F1,
클래스별 Recall, 혼동행렬을 함께 사용한다. 균형이 충분하면 가중치나 오버샘플링부터 추가하지 않는다.

## 4. 값의 의미와 범위 점검

이진 컬럼은 0/1인지, 연속형은 음수·무한값이 있는지, 코어 수는 양의 정수인지 확인한다.
화면 크기·해상도 0은 의심값이다. 카메라 0은 미지원일 수 있어 곧바로 결측으로 바꾸지 않는다.
전면/후면 카메라 비교와 4G/3G 조합은 사양 정의가 필요한 검토 항목이다.
높이·너비는 기기 방향에 따라 달라질 수 있어 대소관계만으로 오류를 단정하지 않는다.

{md_table(issue_counts) if len(issue_counts) else '검토 조건에 걸리는 값이 없다.'}

행별 위치와 값은 `issues.csv`에 기록한다. CSV 행 번호는 헤더를 포함한 1부터의 번호다.
IQR 1.5배 기준은 **통계적 검토 후보**일 뿐 물리적 오류가 아니다.
IQR 경계는 학습 데이터에서 계산하고 테스트에도 그대로 적용한다.
가격 등급이 높은 정상 고사양 제품을 일괄 제거하면 중요한 신호를 잃을 수 있다.

## 5. 단변량 분포 확인

각 특성의 범위, 중앙값, 비대칭, 극단값, 이진 비율을 확인한다.
학습·테스트 행 수가 달라 히스토그램은 건수가 아닌 비율로 비교한다.

![전체 특성 분포](output/mobile_price_eda/feature_distributions.png)

로그 변환·표준화 필요성은 향후 모델에 따라 판단한다. 트리 모델은 일반적으로
표준화가 필수는 아니며 선형 모델·거리 기반 모델은 검토할 필요가 있다.

## 6. 가격 등급과 특성의 관계

학습 데이터만으로 Spearman 순위 상관을 계산하고 등급별 박스플롯·평균·중앙값을 비교한다.
이진 특성은 등급 안에서 0/1 비율을 비교한다. 숫자로 코딩된 범주와 상관이 낮더라도
비선형 관계나 다른 특성과의 상호작용은 있을 수 있다.

{md_table(ranking.head(8))}

![등급별 주요 특성](output/mobile_price_eda/features_by_category.png)

![등급별 이진 사양 비율](output/mobile_price_eda/binary_by_category.png)

상관 순위는 모델의 특성 중요도가 아니며 인과관계를 뜻하지 않는다.
특성끼리의 중복 정보는 아래 행렬에서 확인한다.

![Spearman 상관행렬](output/mobile_price_eda/correlation.png)

## 7. 학습·테스트 분포 차이

{md_table(drift.head(8))}

KS 통계량은 두 경험적 누적분포의 최대 차이이며 0에 가까울수록 유사하다.
이진·이산 컬럼에서는 보조적으로 읽고 비율도 확인한다. 평균만 비슷하다고
전체 분포가 같은 것은 아니다. 표본 차이만으로 분포 이동을 확정하지 않는다.
테스트가 학습 최솟값·최댓값 밖에 있는 건수도 별도로 기록한다.
테스트 분석은 적용 가능성 진단에 사용하고, 전처리 학습이나 모델 선택에는 넣지 않는다.

## 8. 모델링으로 이어지는 결정

1. 학습 데이터 내부에서 `stratify=price_range`, 고정 난수로 검증 세트를 먼저 확보한다.
   날짜 컬럼은 없으므로 앞선 Delhi 기후 데이터의 시계열 분할을 그대로 적용하지 않는다.
   동일 기기 그룹이 확인되면 그룹 분할을 우선한다.
2. 검증 타깃을 반복해서 탐색하며 특성을 고르면 평가가 낙관적으로 변한다.
   이 보고서의 전체 학습 데이터 EDA는 탐색용으로 읽고 최종 성능은 독립 검증한다.
3. `id`를 제외한 원래 20개 사양으로 기준 모델을 먼저 만들고 Accuracy/Macro-F1을 비교한다.
4. 화면 픽셀 수(`px_height * px_width`)와 화면 면적(`sc_h * sc_w`)은 후보지만,
   0값 의미부터 확인하고 검증 성능이 개선되는 경우에만 채택한다.
5. 테스트는 정답이 없으므로 최종 예측 CSV만 생성할 수 있다. 이 EDA는 학습을 수행하지 않는다.

## 실행과 산출물

```powershell
uv run python mobile_price_eda.py
uv run python mobile_price_eda.py --self-check
```

실행하면 이 문서(`mobile_price_eda.md`)와 `output/mobile_price_eda`의 표·차트를 갱신한다.
`quality.csv`, `schema.csv`, `describe.csv`, `class_balance.csv`는 기본 구조와 분포,
`issues.csv`, `issue_counts.csv`, `iqr_review.csv`는 정제 검토 목록,
`target_associations.csv`, `class_feature_summary.csv`, `spearman_correlation.csv`는 관계 분석,
`distribution_shift.csv`는 학습·테스트 차이를 담는다.
"""
    report_file = PROJECT_ROOT / "mobile_price_eda.md"
    report_file.write_text(report, encoding="utf-8")
    (output / "mobile_price_eda.md").write_text(report, encoding="utf-8")
    print(quality.to_string(index=False))
    print(counts.to_string(index=False))
    print(ranking.head(5).to_string(index=False))
    print(issue_counts.to_string(index=False))
    print(f"Report: {report_file}\nCharts and tables: {output}")


if __name__ == "__main__":
    main()
