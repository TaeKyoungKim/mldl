"""
================================================================================
스마트폰 가격 등급(price_range) 분류 - 전처리 및 서포트 벡터 머신(SVC) 모델링 & 평가
================================================================================
- 데이터 경로:
  - 훈련 데이터: C:/apps/mldl/data/mobile_price_category_train.csv
  - 테스트 데이터: C:/apps/mldl/data/mobile_price_category_test.csv
- 적용된 핵심 전처리 및 엔지니어링:
  1. 물리적 이상치/0값 정제: sc_w=0(너비 0cm), px_height=0(높이 0px) 중앙값 보간
  2. 테스트 세트 식별자(id) 격리 및 누수 방지
  3. 다중공선성 및 거리 가중치 왜곡 방지: 중복 합성 변수 배제, 순수 20개 원본 특성 보존
  4. 거리 기반 알고리즘 맞춤형 StandardScaler 표준화
  5. 데이터 누수 없는 Pipeline 및 5-Fold StratifiedKFold 교차 검증
  6. 커널 비교: Linear 커널(하드웨어 원가 선형성) vs RBF 커널(비선형) 벤치마크 및 C 튜닝
- 산출물:
  - 혼동 행렬(Confusion Matrix) 및 커널별 비교 차트
  - 선형 결정 경계 가중치 계수(Feature Coefficients) 시각화
  - 테스트 세트 1,000개 기기 가격 등급 예측 CSV
================================================================================
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 윈도우 환경 한글 폰트 및 콘솔 인코딩 대응
if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    plt.rc("font", family="Malgun Gothic")
    plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "src" else Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output" / "mobile_price_svc"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "mobile_price_category_train.csv"
TEST_PATH = DATA_DIR / "mobile_price_category_test.csv"

CLASS_LABELS = {
    0: "0 (저가 / Low Cost)",
    1: "1 (중저가 / Medium Cost)",
    2: "2 (중고가 / High Cost)",
    3: "3 (프리미엄 / Very High Cost)"
}
CLASS_NAMES = ["0 (저가)", "1 (중저가)", "2 (중고가)", "3 (프리미엄)"]


def load_and_preprocess_data():
    """
    1) 데이터 로드 및 test id 분리
    2) 물리적 이상 0값(sc_w, px_height) 정제
    3) 도메인 특성 생성
    """
    print("\n[1] 데이터 로드 및 정제 전처리 파이프라인 가동...")
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(f"데이터셋을 찾을 수 없습니다: {TRAIN_PATH}")

    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)

    print(f"  * 훈련 데이터 크기 : {train_raw.shape[0]}행, {train_raw.shape[1]}열")
    print(f"  * 테스트 데이터 크기: {test_raw.shape[0]}행, {test_raw.shape[1]}열")

    # 1. 테스트 세트 식별자(id) 분리
    test_ids = test_raw["id"].copy()
    test_features = test_raw.drop(columns=["id"]).copy()
    train_features = train_raw.drop(columns=["price_range"]).copy()
    y_train = train_raw["price_range"].copy()

    # 2. 물리적 0값 이상치 진단 및 중앙값 보간
    zero_sc_w_train = (train_features["sc_w"] == 0).sum()
    zero_px_h_train = (train_features["px_height"] == 0).sum()
    print(f"  * 이상 0값 탐색: sc_w=0 (Train {zero_sc_w_train}건), px_height=0 (Train {zero_px_h_train}건)")

    # 훈련 세트 기준 중앙값 계산 (누수 방지: Train 통계량만 사용)
    med_sc_w = train_features["sc_w"].replace(0, np.nan).median()
    med_px_h = train_features["px_height"].replace(0, np.nan).median()

    # 다중공선성 및 거리 왜곡을 방지하기 위해 불필요한 합성 파생 특성을 배제하고,
    # 물리적 0값(sc_w, px_height)만 Train 중앙값으로 정제하여 순수 20개 특성 보존
    for df in [train_features, test_features]:
        df["sc_w"] = df["sc_w"].replace(0, np.nan).fillna(med_sc_w)
        df["px_height"] = df["px_height"].replace(0, np.nan).fillna(med_px_h)

    feature_cols = list(train_features.columns)
    print(f"  * 다중공선성 방지 조치 완료: 합성 파생 변수 배제, 순수 20개 원본 특성 유지")
    print(f"  * 전처리 완료 후 투입 피처 수: {len(feature_cols)}개")

    return train_features, y_train, test_features, test_ids, feature_cols


def run_svc_pipeline(X, y):
    """
    StandardScaler + SVC 파이프라인 구축 및 GridSearchCV 튜닝
    """
    print("\n[2] SVC 하이퍼파라미터 튜닝 및 5-Fold Stratified 교차 검증...")

    # 독립 평가용 20% 홀드아웃 분할
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 파이프라인 정의 (누수 방지: 각 폴드 내부에서만 fit_transform)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(random_state=42))
    ])

    # Linear 커널 vs RBF 커널 및 C, gamma 탐색 그리드
    param_grid = [
        {
            "svc__kernel": ["linear"],
            "svc__C": [0.1, 1.0, 10.0, 50.0]
        },
        {
            "svc__kernel": ["rbf"],
            "svc__C": [1.0, 10.0, 50.0],
            "svc__gamma": ["scale", 0.01, 0.005]
        }
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        return_train_score=True
    )

    t0 = time.time()
    grid_search.fit(X_train_split, y_train_split)
    search_time = time.time() - t0

    print(f"  >> 그리드 서치 완료 (소요 시간: {search_time:.2f}초)")
    print(f"  >> 최적 파라미터 조합: {grid_search.best_params_}")
    print(f"  >> 5-Fold 교차 검증 최고 정확도: {grid_search.best_score_ * 100:.2f}%")

    # 파라미터별 탐색 결과 요약
    results_list = []
    for mean_score, std_score, params in zip(
        grid_search.cv_results_["mean_test_score"],
        grid_search.cv_results_["std_test_score"],
        grid_search.cv_results_["params"]
    ):
        k = params["svc__kernel"]
        c = params["svc__C"]
        g = params.get("svc__gamma", "-")
        results_list.append({
            "Kernel": k,
            "C": c,
            "Gamma": g,
            "CV_Accuracy": mean_score,
            "CV_Std": std_score
        })

    cv_results_df = pd.DataFrame(results_list).sort_values("CV_Accuracy", ascending=False)
    print("\n[상위 6개 파라미터 조합 성능 요약]:")
    print(cv_results_df.head(6).to_string(index=False, formatters={"CV_Accuracy": "{:.4f}".format, "CV_Std": "{:.4f}".format}))

    # 검증 세트 최종 평가
    best_pipe = grid_search.best_estimator_
    val_preds = best_pipe.predict(X_val_split)

    val_acc = accuracy_score(y_val_split, val_preds)
    val_prec = precision_score(y_val_split, val_preds, average="macro")
    val_rec = recall_score(y_val_split, val_preds, average="macro")
    val_f1 = f1_score(y_val_split, val_preds, average="macro")

    print("\n" + "=" * 65)
    print("  [최적 SVC 모델 검증 세트(Holdout 400개) 최종 성능 지표]")
    print("=" * 65)
    print(f"  * 검증 정확도 (Accuracy)  : {val_acc * 100:.2f}% ({val_acc:.4f})")
    print(f"  * Macro 정밀도 (Precision) : {val_prec:.4f}")
    print(f"  * Macro 재현율 (Recall)    : {val_rec:.4f}")
    print(f"  * Macro F1-Score          : {val_f1:.4f}")
    print("=" * 65)

    print("\n[가격 등급별 상세 분류 리포트]:")
    print(classification_report(y_val_split, val_preds, target_names=CLASS_NAMES, digits=4))

    # 전체 데이터(2,000개)로 최종 재학습
    print("  >> 전체 훈련 세트(2,000개 샘플) 기반 최종 모델 재학습 중...")
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(**{k.replace("svc__", ""): v for k, v in grid_search.best_params_.items()}, random_state=42))
    ])
    final_pipeline.fit(X, y)

    return final_pipeline, best_pipe, X_val_split, y_val_split, val_preds, cv_results_df


def generate_visualizations(best_pipe, X_val, y_val, val_preds, cv_results_df, feature_cols):
    """
    혼동 행렬, 커널 성능 비교, 선형 모델 가중치 계수 차트 생성
    """
    print("\n[3] 결과 시각화 차트 생성 및 저장...")

    # (1) 혼동 행렬 (Confusion Matrix Heatmap)
    cm = confusion_matrix(y_val, val_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6.5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title("스마트폰 가격 등급 SVC 혼동 행렬 (Normalized Confusion Matrix)", fontsize=13, pad=15)
    plt.xlabel("예측 가격 등급 (Predicted Class)", fontsize=11)
    plt.ylabel("실제 가격 등급 (Actual Class)", fontsize=11)
    plt.tight_layout()
    cm_path = OUTPUT_DIR / "svc_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"  - 혼동 행렬 차트 저장 완료: {cm_path}")

    # (2) 커널 및 하이퍼파라미터 비교 막대 차트
    plt.figure(figsize=(10, 5))
    df_plot = cv_results_df.copy()
    df_plot["Param_Label"] = df_plot.apply(
        lambda r: f"{r['Kernel'].upper()} (C={r['C']}, γ={r['Gamma']})", axis=1
    )
    sns.barplot(
        data=df_plot.head(8),
        x="CV_Accuracy",
        y="Param_Label",
        hue="Kernel",
        palette={"linear": "#2E86AB", "rbf": "#F24236"},
        dodge=False
    )
    plt.xlim(0.85, 1.0)
    plt.title("SVC 커널(Linear vs RBF) 및 하이퍼파라미터별 교차 검증 정확도 비교", fontsize=13, pad=15)
    plt.xlabel("5-Fold 교차 검증 정확도 (Accuracy)", fontsize=11)
    plt.ylabel("모델 구성 (Kernel & Params)", fontsize=11)
    plt.legend(title="커널(Kernel)", loc="lower right")
    plt.tight_layout()
    comp_path = OUTPUT_DIR / "svc_kernel_comparison.png"
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"  - 커널 비교 차트 저장 완료: {comp_path}")

    # (3) 선형 커널 결정 경계 가중치 계수 (Feature Coefficients)
    svc_model = best_pipe.named_steps["svc"]
    if svc_model.kernel == "linear":
        # 4개 클래스 다중 분류 시 계수 절대값 평균 계산
        coef_mean = np.mean(np.abs(svc_model.coef_), axis=0)
        coef_df = pd.DataFrame({
            "Feature": feature_cols,
            "Abs_Coefficient": coef_mean
        }).sort_values("Abs_Coefficient", ascending=False)

        plt.figure(figsize=(10, 6.5))
        sns.barplot(
            data=coef_df.head(12),
            x="Abs_Coefficient",
            y="Feature",
            hue="Feature",
            palette="mako",
            legend=False
        )
        plt.title("선형 SVC 가격 등급 결정 경계 주요 특성 가중치 Top 12 (Mean |Coef|)", fontsize=13, pad=15)
        plt.xlabel("평균 절대 가중치 계수 (Feature Importance Proxy)", fontsize=11)
        plt.ylabel("스마트폰 사양 특성 (Feature)", fontsize=11)
        plt.tight_layout()
        coef_path = OUTPUT_DIR / "svc_feature_coefficients.png"
        plt.savefig(coef_path, dpi=300)
        plt.close()
        print(f"  - 특성 가중치 차트 저장 완료: {coef_path}")


def predict_and_save_test(final_pipeline, X_test, test_ids):
    """
    1,000개 테스트 기기 가격 등급 예측 및 CSV 저장
    """
    print("\n[4] 1,000개 테스트 기기 가격 등급 예측 및 결과 저장...")
    test_preds = final_pipeline.predict(X_test)

    pred_df = pd.DataFrame({
        "id": test_ids.values,
        "predicted_price_range": test_preds,
        "class_label": [CLASS_LABELS[p] for p in test_preds]
    })

    pred_csv_path = OUTPUT_DIR / "mobile_price_test_predictions.csv"
    pred_df.to_csv(pred_csv_path, index=False, encoding="utf-8-sig")
    print(f"  - 테스트 세트 예측 결과 CSV 저장 완료: {pred_csv_path}")

    # 예측 등급별 분포 요약
    dist = pred_df["predicted_price_range"].value_counts().sort_index()
    print("  * 테스트 세트 예측 등급 분포:")
    for cls_idx, cnt in dist.items():
        print(f"    - 클래스 {cls_idx} ({CLASS_NAMES[cls_idx]}): {cnt}개 ({cnt/len(test_preds)*100:.1f}%)")


def main():
    print("#" * 80)
    print("  MOBILE PRICE CLASSIFICATION - SVC PREPROCESSING & MODELING PIPELINE")
    print("#" * 80)

    # 1. 전처리 파이프라인
    X_train, y_train, X_test, test_ids, feature_cols = load_and_preprocess_data()

    # 2. SVC 모델 학습 및 교차 검증
    final_pipeline, best_pipe, X_val, y_val, val_preds, cv_results_df = run_svc_pipeline(X_train, y_train)

    # 3. 시각화 생성
    generate_visualizations(best_pipe, X_val, y_val, val_preds, cv_results_df, feature_cols)

    # 4. 테스트 세트 예측 및 CSV 저장
    predict_and_save_test(final_pipeline, X_test, test_ids)

    print("\n" + "#" * 80)
    print("  전처리 및 SVC 모델링, 평가, 예측 파일 생성이 성공적으로 완료되었습니다!")
    print("#" * 80)


if __name__ == "__main__":
    main()
