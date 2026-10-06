"""
================================================================================
UCI HAR Dataset - Random Forest 모델링, 성능 평가 및 개별 트리(Estimator) 시각화
================================================================================
- 데이터셋:
  - 특성 데이터: C:/apps/mldl/data/har_features.csv
  - 타깃 데이터: C:/apps/mldl/data/har_targets.csv
- EDA 기반 권장 하이퍼파라미터 적용:
  - n_estimators: 200 (고차원 공간 분산 감소 극대화)
  - max_features: 'sqrt' (약 23~24개, 트리 간 다양성 확보)
  - max_depth: 25 (SITTING vs STANDING 미세 분기 깊이 확보)
  - min_samples_leaf: 2 (센서 노이즈 스파이크 과적합 방지)
  - 전처리: 스케일링 생략 (트리 모델의 스케일 불변성)
- 핵심 시각화:
  - 무작위 선정한 3개 개별 트리 분류기(Decision Tree Estimators) 구조 시각화
  - 혼동 행렬(Confusion Matrix) 및 특성 중요도(Feature Importance)
================================================================================
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree, export_graphviz
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 윈도우 환경 Graphviz 바이너리 경로 자동 연동
GRAPHVIZ_BIN_DIR = r"C:\Program Files\Graphviz\bin"
if os.path.exists(GRAPHVIZ_BIN_DIR) and GRAPHVIZ_BIN_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] += os.pathsep + GRAPHVIZ_BIN_DIR

try:
    import graphviz
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False

# 윈도우 콘솔 한글 인코딩 대응
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# ==============================================================================
# 0. 경로 및 시각화 스타일 설정
# ==============================================================================
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='Malgun Gothic')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES_PATH = os.path.join(DATA_DIR, "har_features.csv")
TARGETS_PATH = os.path.join(DATA_DIR, "har_targets.csv")

ACTIVITY_MAP = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}
CLASS_NAMES = [ACTIVITY_MAP[i] for i in sorted(ACTIVITY_MAP.keys())]


def main():
    print("#" * 80)
    print("  HUMAN ACTIVITY RECOGNITION - RANDOM FOREST MODELING & TREE VISUALIZATION")
    print("#" * 80)

    # ==========================================================================
    # 1. 정제된 CSV 데이터 로드
    # ==========================================================================
    print("\n[1] 데이터셋 로드...")
    if not os.path.exists(FEATURES_PATH) or not os.path.exists(TARGETS_PATH):
        raise FileNotFoundError(
            f"데이터 파일이 존재하지 않습니다.\n"
            f"- Features: {FEATURES_PATH}\n"
            f"- Targets : {TARGETS_PATH}\n"
            f"먼저 download_har_dataset.py를 실행하여 CSV를 생성해주세요."
        )

    t0 = time.time()
    X = pd.read_csv(FEATURES_PATH)
    y = pd.read_csv(TARGETS_PATH)["Activity"]
    print(f"  - 로드 완료 ({time.time() - t0:.2f}초)")
    print(f"  - 특성 데이터(X) 크기 : {X.shape[0]}개 샘플, {X.shape[1]}개 특성")
    print(f"  - 타깃 데이터(y) 크기 : {y.shape[0]}개 샘플 (클래스 1~6)")

    # ==========================================================================
    # 2. 데이터 분할 (EDA 제언: 스케일링 생략)
    # ==========================================================================
    print("\n[2] 데이터 분할 (Train 70% / Test 30%, 층화 추출)...")
    # 트리 기반 모델이므로 별도의 StandardScaler 적용 없이 원본 스케일 유지
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"  - 학습용 데이터(Train): {X_train.shape[0]}개 샘플")
    print(f"  - 평가용 데이터(Test) : {X_test.shape[0]}개 샘플")
    print(f"  - [전처리 전략]: Random Forest의 스케일 불변성에 따라 추가 스케일링 생략")

    # ==========================================================================
    # 3. EDA 기반 최적 하이퍼파라미터 랜덤 포레스트 모델 생성 및 학습
    # ==========================================================================
    print("\n[3] Random Forest 모델 학습 진행...")
    print("  [설정된 하이퍼파라미터]:")
    print("   * n_estimators    : 200 (561개 고차원 분산 감소 극대화)")
    print("   * max_features    : 'sqrt' (트리 간 다양성 확보 및 고상관 피처 과적합 방지)")
    print("   * max_depth       : 25 (SITTING vs STANDING 미세 분기 깊이 확보)")
    print("   * min_samples_leaf: 2 (센서 노이즈 스파이크 과적합 방지)")
    print("   * random_state    : 42 (결과 재현성 보장)")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_features="sqrt",
        max_depth=25,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    t_fit_start = time.time()
    rf.fit(X_train, y_train)
    fit_duration = time.time() - t_fit_start
    print(f"  >> 모델 학습 완료! (소요 시간: {fit_duration:.2f}초, 총 {len(rf.estimators_)}개 트리 구축)")

    # ==========================================================================
    # 4. 모델 성능 평가 및 진단
    # ==========================================================================
    print("\n[4] 모델 성능 평가...")
    train_acc = rf.score(X_train, y_train)
    test_preds = rf.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    test_prec = precision_score(y_test, test_preds, average="weighted")
    test_rec = recall_score(y_test, test_preds, average="weighted")
    test_f1 = f1_score(y_test, test_preds, average="weighted")

    print("=" * 65)
    print("  [Random Forest 최종 성능 지표 요약]")
    print("=" * 65)
    print(f"  * 훈련 세트 정확도 (Train Accuracy) : {train_acc * 100:.2f}%")
    print(f"  * 테스트 세트 정확도 (Test Accuracy): {test_acc * 100:.2f}%")
    print(f"  * 가중 정밀도 (Weighted Precision)  : {test_prec:.4f}")
    print(f"  * 가중 재현율 (Weighted Recall)     : {test_rec:.4f}")
    print(f"  * 가중 F1-Score (Weighted F1)       : {test_f1:.4f}")
    print("=" * 65)

    print("\n[행동 클래스별 상세 분류 리포트 (Classification Report)]:")
    print(classification_report(y_test, test_preds, target_names=CLASS_NAMES, digits=4))

    # ==========================================================================
    # 5. 혼동 행렬(Confusion Matrix) 시각화
    # ==========================================================================
    print("\n[5] 혼동 행렬 시각화 생성...")
    cm = confusion_matrix(y_test, test_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="YlGnBu",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title("Random Forest 정규화 혼동 행렬 (Normalized Confusion Matrix)", fontsize=14, pad=15)
    plt.xlabel("예측 행동 클래스 (Predicted Activity)", fontsize=11)
    plt.ylabel("실제 행동 클래스 (Actual Activity)", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "rf_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"  - 혼동 행렬 차트 저장: {cm_path}")

    # ==========================================================================
    # 6. 특성 중요도(Feature Importance) Top 15 시각화
    # ==========================================================================
    print("\n[6] 특성 중요도(MDI) 분석 및 시각화...")
    importances = rf.feature_importances_
    feat_names = X.columns
    top15_idx = np.argsort(importances)[::-1][:15]

    top15_df = pd.DataFrame({
        "Feature": [feat_names[i] for i in top15_idx],
        "Importance": importances[top15_idx]
    })

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=top15_df,
        x="Importance",
        y="Feature",
        hue="Feature",
        palette="viridis",
        legend=False
    )
    plt.title("Random Forest 지니 중요도(MDI) Top 15 센서 특성", fontsize=13, pad=15)
    plt.xlabel("상대적 특성 중요도 (Feature Importance)", fontsize=11)
    plt.ylabel("센서 측정 특성 (Sensor Feature)", fontsize=11)
    plt.tight_layout()
    fi_path = os.path.join(OUTPUT_DIR, "rf_feature_importance.png")
    plt.savefig(fi_path, dpi=300)
    plt.close()
    print(f"  - 특성 중요도 차트 저장: {fi_path}")

    # ==========================================================================
    # 7. 무작위 선정한 3개 개별 분류기(Decision Tree Estimators) 시각화
    # ==========================================================================
    print("\n[7] 랜덤으로 선정한 3개 개별 결정트리(Estimator) 시각화...")
    np.random.seed(42)
    # 200개 트리 중 3개 트리 무작위 추출
    selected_tree_indices = np.random.choice(len(rf.estimators_), size=3, replace=False)
    selected_tree_indices.sort()
    print(f"  - 무작위 선정된 3개 트리 번호: {[int(i) for i in selected_tree_indices]}")

    # (1) 3개 트리를 비교하는 통합 3-패널 시각화
    fig, axes = plt.subplots(3, 1, figsize=(20, 24))
    fig.suptitle(
        "Random Forest 앙상블 내부: 무작위 선정된 3개 개별 결정트리(Estimators) 구조 비교\n(가독성을 위해 상위 3단계 깊이 시각화)",
        fontsize=16,
        fontweight="bold",
        y=0.99
    )

    for rank, (tree_idx, ax) in enumerate(zip(selected_tree_indices, axes), 1):
        estimator = rf.estimators_[tree_idx]
        root_feature_id = estimator.tree_.feature[0]
        root_feature_name = feat_names[root_feature_id]
        root_threshold = estimator.tree_.threshold[0]
        total_nodes = estimator.tree_.node_count
        actual_depth = estimator.tree_.max_depth

        print(f"\n  [선정 트리 {rank} - Estimator #{tree_idx} 정보]:")
        print(f"   * 총 노드 수: {total_nodes}개, 실제 최대 깊이: {actual_depth}")
        print(f"   * 최상위(Root) 분기 피처: '{root_feature_name}' (임계값: {root_threshold:.4f})")

        plot_tree(
            estimator,
            max_depth=3,  # 복잡한 561개 고차원 트리 노드 텍스트 가독성을 위해 상위 3단계 표현
            feature_names=feat_names,
            class_names=CLASS_NAMES,
            filled=True,
            rounded=True,
            proportion=False,
            precision=2,
            fontsize=7,
            ax=ax
        )
        ax.set_title(
            f"[트리 #{tree_idx}] 루트 분기 특성: {root_feature_name} (총 노드: {total_nodes}개, 최대 깊이: {actual_depth})",
            fontsize=12,
            pad=10,
            fontweight="bold"
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    combined_trees_path = os.path.join(OUTPUT_DIR, "rf_3_trees_comparison.png")
    plt.savefig(combined_trees_path, dpi=250)
    plt.close()
    print(f"\n  >> 3개 트리 비교 통합 차트 저장 완료: {combined_trees_path}")

    # (2) 3개 트리 각각의 단독 고해상도 개별 이미지 저장
    for rank, tree_idx in enumerate(selected_tree_indices, 1):
        estimator = rf.estimators_[tree_idx]
        single_fig, single_ax = plt.subplots(figsize=(18, 9))
        plot_tree(
            estimator,
            max_depth=3,
            feature_names=feat_names,
            class_names=CLASS_NAMES,
            filled=True,
            rounded=True,
            precision=2,
            fontsize=8,
            ax=single_ax
        )
        single_ax.set_title(
            f"Random Forest 개별 분류기 #{tree_idx} (Top 3 Depths)",
            fontsize=14,
            pad=12,
            fontweight="bold"
        )
        plt.tight_layout()
        single_tree_path = os.path.join(OUTPUT_DIR, f"rf_tree_sample_{rank}_estimator_{tree_idx}.png")
        plt.savefig(single_tree_path, dpi=300)
        plt.close()
        print(f"  >> 개별 트리 #{tree_idx} 단독 차트 저장 완료: {single_tree_path}")

        # (3) Graphviz가 설치된 경우 DOT 및 고화질 벡터형 파일로도 내보내기
        if HAS_GRAPHVIZ:
            dot_data = export_graphviz(
                estimator,
                max_depth=3,
                feature_names=feat_names,
                class_names=CLASS_NAMES,
                filled=True,
                rounded=True,
                special_characters=True
            )
            dot_path = os.path.join(OUTPUT_DIR, f"rf_tree_estimator_{tree_idx}")
            graph = graphviz.Source(dot_data)
            try:
                graph.render(filename=dot_path, format="png", cleanup=True)
                print(f"     [Graphviz] 고화질 PNG 렌더링 완료: {dot_path}.png")
            except Exception as e:
                print(f"     [Graphviz] 렌더링 건너뜀 ({e})")

    # ==========================================================================
    # 8. 최종 예측 결과 CSV 저장
    # ==========================================================================
    print("\n[8] 테스트 세트 예측 결과 CSV 파일 저장...")
    results_df = pd.DataFrame({
        "Actual_Code": y_test.values,
        "Actual_Label": [ACTIVITY_MAP[code] for code in y_test.values],
        "Predicted_Code": test_preds,
        "Predicted_Label": [ACTIVITY_MAP[code] for code in test_preds]
    })
    results_df["Correct"] = results_df["Actual_Code"] == results_df["Predicted_Code"]
    pred_path = os.path.join(OUTPUT_DIR, "rf_test_predictions.csv")
    results_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print(f"  - 예측 결과 CSV 저장 완료: {pred_path}")

    print("\n" + "#" * 80)
    print("  랜덤 포레스트 모델링 및 3개 개별 트리 시각화가 성공적으로 완료되었습니다!")
    print("#" * 80)


if __name__ == "__main__":
    main()
