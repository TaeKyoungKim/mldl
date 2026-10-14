"""
================================================================================
UCI HAR PCA (har_features_pca_90.csv) 랜덤 포레스트 분류 및 SHAP 해석 파이프라인
================================================================================
- 데이터 경로:
  - 특성 데이터: C:/apps/mldl/data/har_features_pca_90.csv (65개 주성분 피처)
  - 타깃 데이터: C:/apps/mldl/data/har_targets.csv (6가지 인간 행동 레이블)
- 행동 클래스 (Activity Mapping):
  1: WALKING (걷기)
  2: WALKING_UPSTAIRS (계단 오르기)
  3: WALKING_DOWNSTAIRS (계단 내려가기)
  4: SITTING (앉아있기)
  5: STANDING (서있기)
  6: LAYING (누워있기)
- 주요 파이프라인 단계:
  1. 데이터 로드 및 학습셋/평가셋 분할 (train_test_split, stratify 적용)
  2. 랜덤 포레스트(RandomForestClassifier) 모델 학습 및 분류 성능 평가 (정확도, 분류 리포트, 혼동 행렬)
  3. SHAP TreeExplainer를 이용한 모델 해석:
     - 전역 특성 중요도 (Global Feature Importance - Summary Plot)
     - 정적(Laying/Sitting) vs 동적(Walking) 행동별 주성분 기여도 비교
     - 개별 샘플 단위 로컬 해석 (Waterfall / Force Plot)
  4. 시각화 산출물 저장 (output/har_shap/)
================================================================================
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap

# 윈도우 환경 한글 폰트 및 마이너스 부호 설정
if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    plt.rc("font", family="Malgun Gothic")
    plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "src" else Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output" / "har_shap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PATH = DATA_DIR / "har_features_pca_90.csv"
TARGETS_PATH = DATA_DIR / "har_targets.csv"

# UCI HAR 클래스 매핑
ACTIVITY_NAMES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}


def load_and_split_data(test_size=0.2, random_state=42):
    """데이터 로드 및 Train/Test 분할"""
    print("[1] 데이터셋 로드 및 분할 준비...")
    if not FEATURES_PATH.exists() or not TARGETS_PATH.exists():
        raise FileNotFoundError(f"필요한 데이터 파일이 존재하지 않습니다:\n- {FEATURES_PATH}\n- {TARGETS_PATH}")

    X = pd.read_csv(FEATURES_PATH)
    y_df = pd.read_csv(TARGETS_PATH)
    y = y_df["Activity"].values

    print(f"  - PCA 특성 크기: {X.shape} (샘플 수: {X.shape[0]}, 주성분 수: {X.shape[1]})")
    print(f"  - 타깃 크기: {y.shape}")

    # 레이블별 분포 확인
    label_counts = pd.Series(y).map(ACTIVITY_NAMES).value_counts()
    print("  - 클래스별 분포:")
    for name, cnt in label_counts.items():
        print(f"    * {name:18s}: {cnt:5d}개 ({cnt/len(y)*100:5.1f}%)")

    # 계층화 분할 (Stratified Split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"\n  >> 분할 완료: 학습셋 {X_train.shape[0]}개, 평가셋 {X_test.shape[0]}개")
    return X_train, X_test, y_train, y_test


def train_and_evaluate_rf(X_train, X_test, y_train, y_test):
    """랜덤 포레스트 모델 학습 및 평가"""
    print("\n[2] 랜덤 포레스트(Random Forest) 분류 모델 학습 중...")
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # 평가셋 예측
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  ★ 평가셋 최종 정확도 (Test Accuracy): {acc * 100:.2f}%")

    # 분류 리포트 출력
    target_names = [ACTIVITY_NAMES[i] for i in sorted(ACTIVITY_NAMES.keys())]
    print("\n" + "=" * 65)
    print("                 [분류 성능 평가 보고서]")
    print("=" * 65)
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))
    print("=" * 65)

    # 혼동 행렬 시각화 및 저장
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names
    )
    plt.title(f"Random Forest 혼동 행렬 (정확도: {acc*100:.2f}%)", fontsize=14, pad=15)
    plt.xlabel("예측 클래스 (Predicted)", fontsize=11)
    plt.ylabel("실제 클래스 (Actual)", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = OUTPUT_DIR / "rf_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"  - 혼동 행렬 차트 저장 완료: {cm_path}")

    return rf


def explain_with_shap(rf, X_train, X_test, y_test, num_explain_samples=300):
    """SHAP TreeExplainer를 이용한 모델 해석 및 플롯 생성"""
    print("\n[3] SHAP TreeExplainer 기반 모델 해석 시작...")
    print(f"  - 연산 효율 및 대표성을 위해 평가셋 중 {num_explain_samples}개 샘플에 대해 SHAP 계산을 수행합니다.")

    # 샘플링
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), size=min(num_explain_samples, len(X_test)), replace=False)
    X_sample = X_test.iloc[sample_indices]
    y_sample = y_test[sample_indices]

    # TreeExplainer 생성
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_sample)

    print(f"  - 계산 완료! SHAP 값 형태: {shap_values.shape if hasattr(shap_values, 'shape') else len(shap_values)}")
    target_names = [ACTIVITY_NAMES[i] for i in sorted(ACTIVITY_NAMES.keys())]

    # --- 1. 전역 특성 중요도 (Global Summary Bar Plot) ---
    print("  - [시각화 1/4] 전역 특성 중요도(Multi-class Summary Bar Plot) 생성...")
    plt.figure(figsize=(10, 8))
    # shap_values가 ndarray (N, features, classes)인 경우
    if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        # 각 클래스별 shap_values 리스트로 변환
        shap_list = [shap_values[:, :, c] for c in range(shap_values.shape[2])]
        shap.summary_plot(
            shap_list,
            X_sample,
            feature_names=X_test.columns,
            class_names=target_names,
            plot_type="bar",
            max_display=15,
            show=False
        )
    else:
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=X_test.columns,
            class_names=target_names,
            plot_type="bar",
            max_display=15,
            show=False
        )
    plt.title("전체 행동 분류에 대한 상위 주성분(PCA) SHAP 전역 중요도", fontsize=14, pad=15)
    plt.tight_layout()
    bar_path = OUTPUT_DIR / "shap_summary_bar.png"
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"    * 저장 완료: {bar_path}")

    # --- 2. 동적 행동(WALKING, 클래스 0) Beeswarm Summary Plot ---
    print("  - [시각화 2/4] 동적 행동 'WALKING'에 대한 SHAP Beeswarm 플롯 생성...")
    plt.figure(figsize=(10, 7))
    class_0_shap = shap_values[:, :, 0] if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3 else shap_values[0]
    shap.summary_plot(
        class_0_shap,
        X_sample,
        feature_names=X_test.columns,
        max_display=12,
        show=False
    )
    plt.title("동적 행동: WALKING(걷기) 예측에 기여하는 상위 주성분 SHAP 분포", fontsize=13, pad=15)
    plt.tight_layout()
    walking_path = OUTPUT_DIR / "shap_summary_walking.png"
    plt.savefig(walking_path, dpi=300)
    plt.close()
    print(f"    * 저장 완료: {walking_path}")

    # --- 3. 정적 행동(LAYING, 클래스 5) Beeswarm Summary Plot ---
    print("  - [시각화 3/4] 정적 행동 'LAYING'에 대한 SHAP Beeswarm 플롯 생성...")
    plt.figure(figsize=(10, 7))
    class_5_shap = shap_values[:, :, 5] if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3 else shap_values[5]
    shap.summary_plot(
        class_5_shap,
        X_sample,
        feature_names=X_test.columns,
        max_display=12,
        show=False
    )
    plt.title("정적 행동: LAYING(누워있기) 예측에 기여하는 상위 주성분 SHAP 분포", fontsize=13, pad=15)
    plt.tight_layout()
    laying_path = OUTPUT_DIR / "shap_summary_laying.png"
    plt.savefig(laying_path, dpi=300)
    plt.close()
    print(f"    * 저장 완료: {laying_path}")

    # --- 4. 개별 샘플 로컬 워터폴(Waterfall) 설명 (첫 번째 테스트 샘플) ---
    print("  - [시각화 4/4] 단일 샘플 로컬 결정 해석(Waterfall Plot) 생성...")
    sample_idx = 0
    actual_label = y_sample[sample_idx]
    actual_name = ACTIVITY_NAMES[actual_label]
    pred_label = rf.predict(X_sample.iloc[[sample_idx]])[0]
    pred_name = ACTIVITY_NAMES[pred_label]
    target_class_idx = pred_label - 1

    # Explanation 객체 생성
    base_val = explainer.expected_value[target_class_idx] if hasattr(explainer.expected_value, "__len__") else explainer.expected_value
    sample_shap = class_0_shap = shap_values[sample_idx, :, target_class_idx] if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3 else shap_values[target_class_idx][sample_idx]

    explanation = shap.Explanation(
        values=sample_shap,
        base_values=base_val,
        data=X_sample.iloc[sample_idx].values,
        feature_names=X_test.columns.tolist()
    )

    plt.figure(figsize=(10, 7))
    shap.waterfall_plot(explanation, max_display=10, show=False)
    plt.title(f"단일 샘플 로컬 해석 (실제: {actual_name} / 예측: {pred_name})", fontsize=13, pad=15)
    plt.tight_layout()
    waterfall_path = OUTPUT_DIR / "shap_local_waterfall.png"
    plt.savefig(waterfall_path, dpi=300)
    plt.close()
    print(f"    * 저장 완료: {waterfall_path}")

    # 상위 5대 핵심 주성분 요약 통계 출력
    mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2)) if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3 else np.mean([np.abs(s).mean(axis=0) for s in shap_values], axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:5]
    print("\n" + "=" * 65)
    print("        [SHAP 기반 행동 분류 기여도 최상위 5개 주성분]")
    print("=" * 65)
    for rank, idx in enumerate(top_indices, 1):
        feat_name = X_test.columns[idx]
        score = mean_abs_shap[idx]
        print(f"  {rank}위: {feat_name:15s} | 평균 |SHAP| 기여도: {score:.5f}")
    print("=" * 65)


def main():
    print("#" * 80)
    print("  HAR PCA FEATURES - RANDOM FOREST CLASSIFICATION & SHAP EXPLANATION")
    print("#" * 80)

    # 1. 데이터 로드 및 분할
    X_train, X_test, y_train, y_test = load_and_split_data(test_size=0.2, random_state=42)

    # 2. 랜덤 포레스트 학습 및 평가
    rf = train_and_evaluate_rf(X_train, X_test, y_train, y_test)

    # 3. SHAP 해석 수행 및 시각화 저장
    explain_with_shap(rf, X_train, X_test, y_test, num_explain_samples=300)

    print("\n" + "#" * 80)
    print("  모든 모델링 및 SHAP 해석 파이프라인이 성공적으로 완료되었습니다!")
    print(f"  결과 산출물 디렉터리: {OUTPUT_DIR}")
    print("#" * 80)


if __name__ == "__main__":
    main()
