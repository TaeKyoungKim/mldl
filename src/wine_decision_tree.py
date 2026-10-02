"""
================================================================================
Wine Dataset - 의사결정나무(Decision Tree) 모델 학습, 평가 및 Graphviz 시각화
Author: Senior Data Scientist & ML Engineer
Dataset: Scikit-learn load_wine (와인 품종 3종 분류)
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Windows 환경에서 Graphviz 바이너리 경로 자동 추가
GRAPHVIZ_BIN_DIR = r"C:\Program Files\Graphviz\bin"
if os.path.exists(GRAPHVIZ_BIN_DIR) and GRAPHVIZ_BIN_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] += os.pathsep + GRAPHVIZ_BIN_DIR

import graphviz
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 윈도우 콘솔 한글 인코딩 대응
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# ==============================================================================
# 0. 시각화 및 환경 설정
# ==============================================================================
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='Malgun Gothic')

OUTPUT_DIR = "C:/apps/mldl/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("#" * 80)
    print("  WINE CLASSIFICATION: DECISION TREE MODELING, EVALUATION & VISUALIZATION")
    print("#" * 80)

    # ==========================================================================
    # 1. 와인 데이터셋 로드 및 구조 탐색
    # ==========================================================================
    print("\n[1] Wine 데이터셋 로드...")
    wine = load_wine()
    X = wine.data
    y = wine.target
    feature_names = wine.feature_names
    target_names = wine.target_names

    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    print(f"- 전체 샘플 수: {X.shape[0]}개, 특성 수: {X.shape[1]}개")
    print(f"- 타깃 클래스 ({len(target_names)}종): {list(target_names)}")
    print(f"- 클래스별 샘플 수: {dict(pd.Series(y).value_counts().sort_index())}")

    # ==========================================================================
    # 2. 계층적 데이터 분할 (Stratified Train 80% / Test 20%)
    # ==========================================================================
    print("\n[2] 데이터셋 분할 (Stratified Split 8:2)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"- 학습셋(Train): {X_train.shape[0]}건 | 테스트셋(Test): {X_test.shape[0]}건")

    # ==========================================================================
    # 3. 의사결정나무 모델 학습 (DecisionTreeClassifier)
    # ==========================================================================
    print("\n[3] 의사결정나무 모델 학습 중...")
    # max_depth=3으로 설정하여 과적합을 방지하고 트리의 시각적 해석력을 극대화
    dt_clf = DecisionTreeClassifier(
        criterion='gini',
        max_depth=3,
        random_state=42
    )
    dt_clf.fit(X_train, y_train)

    # 5-Fold 교차 검증 점수 확인
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(dt_clf, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"- 5-Fold 교차검증 평균 정확도: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # ==========================================================================
    # 4. 모델 성능 평가 (Evaluation)
    # ==========================================================================
    print("\n[4] 테스트셋 성능 평가 (Evaluation)...")
    # y_train_pred = dt_clf.predict(X_train)
    y_test_pred = dt_clf.predict(X_test)

    # train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    # print(f"- 학습셋 정확도 (Train Accuracy) : {train_acc:.4f}")
    print(f"- 테스트셋 정확도 (Test Accuracy) : {test_acc:.4f}")

    print("\n[상세 Classification Report (Test Set)]:")
    print(classification_report(y_test, y_test_pred, target_names=target_names))

    # ==========================================================================
    # 5. Graphviz를 이용한 의사결정나무 트리 구조 시각화
    # ==========================================================================
    print("\n[5] Graphviz를 이용한 의사결정나무 트리 시각화...")
    dot_data = export_graphviz(
        dt_clf,
        out_file=None,
        feature_names=feature_names,
        class_names=target_names,
        filled=True,
        rounded=True,
        special_characters=True,
        precision=2
    )

    graph = graphviz.Source(dot_data)
    tree_out_path = os.path.join(OUTPUT_DIR, "wine_tree_graphviz")
    rendered_path = graph.render(tree_out_path, format="png", cleanup=True)
    print(f"  -> Graphviz 트리 시각화 이미지 저장 완료: {rendered_path}")

    # ==========================================================================
    # 6. 피처 중요도(Feature Importance) 분석 및 시각화
    # ==========================================================================
    print("\n[6] 피처 중요도(Feature Importance) 분석 및 시각화...")
    importances = dt_clf.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("\n[*] 특성별 중요도 (Gini Importance):")
    for _, row in fi_df.iterrows():
        if row['Importance'] > 0:
            print(f"  - {row['Feature']:<25}: {row['Importance']:.4f} ({row['Importance']*100:.1f}%)")

    # 혼동 행렬 및 피처 중요도 2분할 서브플롯 생성
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    plt.subplots_adjust(wspace=0.3)

    # (1) 혼동 행렬 시각화
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=target_names, yticklabels=target_names,
        ax=axes[0], cbar=False
    )
    axes[0].set_title(f"1. 혼동 행렬 (Confusion Matrix)\n테스트 정확도: {test_acc*100:.1f}%", fontsize=14, fontweight='bold', pad=12)
    axes[0].set_xlabel("예측 클래스 (Predicted Class)", fontsize=12)
    axes[0].set_ylabel("실제 클래스 (Actual Class)", fontsize=12)

    # (2) 피처 중요도 수평 막대그래프 시각화
    top_fi_df = fi_df.sort_values(by='Importance', ascending=True)
    # 0보다 큰 피처는 주황색 계열, 0인 피처는 회색으로 구분
    bar_colors = ['#1f77b4' if imp > 0 else '#cccccc' for imp in top_fi_df['Importance']]
    
    y_pos = np.arange(len(top_fi_df))
    axes[1].barh(y_pos, top_fi_df['Importance'], color=bar_colors, edgecolor='none', alpha=0.85)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(top_fi_df['Feature'], fontsize=11)
    axes[1].set_title("2. 와인 분류 의사결정나무 피처 중요도 (Feature Importance)", fontsize=14, fontweight='bold', pad=12)
    axes[1].set_xlabel("특성 중요도 (Gini Impurity 감소 기여도)", fontsize=12)
    axes[1].grid(True, linestyle=':', alpha=0.6)

    # 중요도 수치 텍스트 표기
    for i, imp in enumerate(top_fi_df['Importance']):
        if imp > 0:
            axes[1].annotate(f" {imp:.3f} ({imp*100:.1f}%)", (imp, i), va='center', fontsize=10, fontweight='bold')

    plt.suptitle("Wine 데이터셋 의사결정나무(Decision Tree) 평가 및 특성 중요도", fontsize=16, fontweight='bold', y=1.02)
    fi_plot_path = os.path.join(OUTPUT_DIR, "wine_feature_importance.png")
    plt.tight_layout()
    plt.savefig(fi_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> 피처 중요도 & 혼동행렬 시각화 차트 저장 완료: {fi_plot_path}")

    print("\n" + "=" * 80)
    print("  의사결정나무 모델 학습, 평가 및 시각화가 성공적으로 완료되었습니다.")
    print(f"  산출물 저장 위치: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
