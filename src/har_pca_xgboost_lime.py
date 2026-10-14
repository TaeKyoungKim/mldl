"""
================================================================================
UCI HAR PCA 데이터셋 XGBoost 모델링 및 LIME(Local Interpretable Model-agnostic Explanations)
================================================================================
- 데이터 경로:
  - 특성 데이터: C:/apps/mldl/data/har_features_pca_90.csv (65개 주성분)
  - 타깃 데이터: C:/apps/mldl/data/har_targets.csv (1~6 인간 행동 라벨)
- 행동 클래스 (Activity Mapping):
  0: WALKING (원래 1)
  1: WALKING_UPSTAIRS (원래 2)
  2: WALKING_DOWNSTAIRS (원래 3)
  3: SITTING (원래 4)
  4: STANDING (원래 5)
  5: LAYING (원래 6)
  * XGBoost는 0부터 시작하는 클래스 인덱스를 요구하므로 (y - 1) 변환 적용
- 파이프라인 단계:
  1. 학습셋/평가셋 분할 (train_test_split, stratify=y)
  2. XGBoost 분류기(XGBClassifier) 학습 및 평가셋 정확도/분류 리포트 산출
  3. 평가셋 중 임의의 테스트 샘플을 선택하여 예측 수행
  4. LIME Tabular Explainer를 통해 예측 결과에 대한 국소 대리 모델(Local Surrogate) 기반 XAI 설명 도출
  5. LIME 설명 결과 텍스트 보고 및 시각화 저장 (HTML / PNG)
================================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from lime import lime_tabular

# 윈도우 환경 한글 폰트 설정
if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    plt.rc("font", family="Malgun Gothic")
    plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "src" else Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output" / "har_lime"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PATH = DATA_DIR / "har_features_pca_90.csv"
TARGETS_PATH = DATA_DIR / "har_targets.csv"

# 0-indexed 클래스 명칭 매핑
CLASS_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]


def load_and_preprocess(test_size=0.2, random_state=42):
    """데이터 로드 및 0-indexed 전처리 후 분할"""
    print("[1] 데이터셋 로드 및 전처리...")
    if not FEATURES_PATH.exists() or not TARGETS_PATH.exists():
        raise FileNotFoundError("필요한 데이터셋 파일이 존재하지 않습니다.")

    X = pd.read_csv(FEATURES_PATH)
    y_raw = pd.read_csv(TARGETS_PATH)["Activity"].values
    # XGBoost 다중 분류를 위해 1~6 -> 0~5 변환
    y = y_raw - 1

    print(f"  - 특성 데이터 크기: {X.shape} (주성분 {X.shape[1]}개)")
    print(f"  - 클래스 수: {len(CLASS_NAMES)}개 ({', '.join(CLASS_NAMES)})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  - 분할 완료: 학습셋 {X_train.shape[0]}개, 평가셋 {X_test.shape[0]}개")
    return X_train, X_test, y_train, y_test


def train_xgboost(X_train, X_test, y_train, y_test):
    """XGBoost 분류 모델 학습 및 평가"""
    print("\n[2] XGBoost 분류 모델 학습 중...")
    model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=6,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  ★ XGBoost 평가셋 최종 정확도: {acc * 100:.2f}%")
    print("\n" + "=" * 65)
    print("                 [XGBoost 분류 성능 리포트]")
    print("=" * 65)
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))
    print("=" * 65)
    return model


def explain_single_prediction_with_lime(model, X_train, X_test, y_test, sample_index=None):
    """
    평가셋 중 임의의 테스트 샘플을 선택하여 LIME XAI 설명을 생성하고 해석
    """
    print("\n[3] LIME(Local Interpretable Model-agnostic Explanations) 설명기 생성...")

    # 1. LIME Tabular Explainer 초기화
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=CLASS_NAMES,
        mode="classification",
        kernel_width=None,
        verbose=False,
        random_state=42
    )

    # 2. 임의의 테스트 샘플 인덱스 선택 (지정되지 않은 경우 시드 기반 고정 또는 랜덤)
    if sample_index is None:
        np.random.seed(2026)
        sample_index = int(np.random.randint(0, len(X_test)))

    sample_x = X_test.iloc[sample_index]
    actual_label_idx = y_test[sample_index]
    actual_label_name = CLASS_NAMES[actual_label_idx]

    # 모델 예측 확률 및 예측 라벨 도출
    pred_probs = model.predict_proba(sample_x.values.reshape(1, -1))[0]
    pred_label_idx = int(np.argmax(pred_probs))
    pred_label_name = CLASS_NAMES[pred_label_idx]
    pred_confidence = pred_probs[pred_label_idx]

    print("\n" + "=" * 75)
    print(f"  [선택된 테스트 샘플 #{sample_index} 예측 현황]")
    print("=" * 75)
    print(f"  * 실제 행동 (Ground Truth) : {actual_label_name} (Class {actual_label_idx})")
    print(f"  * 모델 예측 (Prediction)   : {pred_label_name} (Class {pred_label_idx})")
    print(f"  * 예측 확신도 (Confidence) : {pred_confidence * 100:.2f}%")
    print("  * 전체 클래스별 예측 확률:")
    for c_idx, c_name in enumerate(CLASS_NAMES):
        bar = "█" * int(pred_probs[c_idx] * 30)
        print(f"    - {c_name:18s}: {pred_probs[c_idx]*100:5.2f}% {bar}")
    print("=" * 75)

    # 3. LIME 국소 설명 도출 (예측된 클래스 기준 상위 10개 핵심 특성)
    print(f"\n[4] 샘플 #{sample_index}의 예측({pred_label_name})에 대한 LIME 국소 기여도 계산 중...")
    exp = explainer.explain_instance(
        data_row=sample_x.values,
        predict_fn=model.predict_proba,
        labels=[pred_label_idx],
        num_features=10
    )

    # LIME 결과 추출
    exp_list = exp.as_list(label=pred_label_idx)

    print("\n" + "=" * 75)
    print(f"   [LIME 특성 기여도 분석: 모델이 '{pred_label_name}'(으)로 판단한 근거]")
    print("=" * 75)
    print(f"{'기여 순위':<6} | {'조건 (주성분 조건식)':<30} | {'LIME 가중치':<12} | {'영향 방향'}")
    print("-" * 75)

    pos_reasons = []
    neg_reasons = []

    for rank, (rule, weight) in enumerate(exp_list, 1):
        direction = "▲ 지지 (Support)" if weight > 0 else "▼ 억제 (Oppose)"
        print(f"{rank:^6} | {rule:<30} | {weight:^+11.5f} | {direction}")
        if weight > 0:
            pos_reasons.append((rule, weight))
        else:
            neg_reasons.append((rule, weight))
    print("=" * 75)

    # 4. 도출된 결과 해석 보고
    print("\n[5] LIME 기반 도출 결과 비즈니스/물리적 해석:")
    print(f"  1) 결정적 지지 요인 (Positive Drivers):")
    for rule, w in pos_reasons[:3]:
        print(f"     • {rule} (기여도: +{w:.4f}): 이 조건이 해당 샘플을 '{pred_label_name}'로 강력히 분류하도록 기여함.")
    if neg_reasons:
        print(f"  2) 잠재적 반박/경계 요인 (Negative Drivers):")
        for rule, w in neg_reasons[:2]:
            print(f"     • {rule} (기여도: {w:.4f}): 타 클래스로 오분류할 가능성을 시사하는 미세 감점 요인.")
    else:
        print(f"  2) 상위 10개 특성 모두가 일관되게 '{pred_label_name}' 방향으로 지지하여 매우 견고한 예측임.")

    # 5. 시각화 저장 (PNG 차트 및 인터랙티브 HTML)
    # (1) PNG 바 차트 저장
    plt.figure(figsize=(11, 6))
    features = [x[0] for x in exp_list][::-1]
    weights = [x[1] for x in exp_list][::-1]
    colors = ["#2A9D8F" if w > 0 else "#E76F51" for w in weights]

    bars = plt.barh(features, weights, color=colors, edgecolor="black", alpha=0.85)
    plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
    plt.title(f"LIME 설명 결과: 샘플 #{sample_index} 예측 ['{pred_label_name}'] 기여도 분석\n(실제: {actual_label_name}, 확신도: {pred_confidence*100:.2f}%)",
              fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("LIME 국소 가중치 (Feature Weight: 양수=예측 지지, 음수=예측 반대)", fontsize=11)
    plt.tight_layout()

    png_path = OUTPUT_DIR / f"lime_sample_{sample_index}_explanation.png"
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"\n  >> LIME 바 차트 저장 완료: {png_path}")

    # (2) HTML 인터랙티브 리포트 저장
    html_path = OUTPUT_DIR / f"lime_sample_{sample_index}_report.html"
    exp.save_to_file(str(html_path))
    print(f"  >> LIME 인터랙티브 HTML 리포트 저장 완료: {html_path}")

    return sample_index, pred_label_name, exp_list


def main():
    print("#" * 80)
    print("  UCI HAR PCA - XGBOOST MODELING & LIME LOCAL EXPLANATION (XAI)")
    print("#" * 80)

    # 1. 데이터 로드 및 분할
    X_train, X_test, y_train, y_test = load_and_preprocess(test_size=0.2, random_state=42)

    # 2. XGBoost 모델 학습
    model = train_xgboost(X_train, X_test, y_train, y_test)

    # 3. LIME 국소 해석 수행
    explain_single_prediction_with_lime(model, X_train, X_test, y_test)

    print("\n" + "#" * 80)
    print("  XGBoost 학습 및 LIME XAI 해석 파이프라인이 성공적으로 완료되었습니다!")
    print(f"  결과 산출물 폴더: {OUTPUT_DIR}")
    print("#" * 80)


if __name__ == "__main__":
    main()
