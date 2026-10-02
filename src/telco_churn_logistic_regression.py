"""
================================================================================
Telco Customer Churn - 로지스틱 회귀(Logistic Regression) 모델링 및 성능 평가
Author: Senior Data Scientist & ML Engineer
Based on: C:/apps/mldl/output/telco_churn_eda_report.md
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report
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
DATA_PATH = os.path.join(OUTPUT_DIR, "telco_churn_featured.csv")


def main():
    print("#" * 80)
    print("  TELCO CUSTOMER CHURN: LOGISTIC REGRESSION MODELING & EVALUATION")
    print("#" * 80)

    # ==========================================================================
    # 1. 데이터 로드 및 피처 정제 (EDA 보고서 권고안 반영)
    # ==========================================================================
    print("\n[1] 피처 엔지니어링 완료 데이터셋 로드...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"데이터셋을 찾을 수 없습니다: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"- 로드된 데이터셋 형상: {df.shape[0]:,}행 × {df.shape[1]}열")

    # 타깃 분리 (Yes: 1, No: 0)
    y = (df['Churn'] == 'Yes').astype(int)

    # [EDA 보고서 반영]: 다중공선성(VIF 9.5+)을 유발하는 TotalCharges는 제거하고,
    # 파생 변수인 Avg_Monthly_Ratio, Service_Count, Tenure_Group을 포함
    drop_cols = ['Churn', 'TotalCharges']
    X = df.drop(columns=drop_cols)

    # 특성 유형 분리
    numeric_features = ['tenure', 'MonthlyCharges', 'Service_Count', 'Avg_Monthly_Ratio']
    categorical_features = [col for col in X.columns if col not in numeric_features]

    print(f"- 수치형 특성 ({len(numeric_features)}개): {numeric_features}")
    print(f"- 범주형 특성 ({len(categorical_features)}개): {categorical_features}")

    # ==========================================================================
    # 2. 계층적 데이터 분할 (Train 80% / Test 20%)
    # ==========================================================================
    print("\n[2] 데이터 분할 (Stratified Train 80% / Test 20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"- 학습셋 크기: {X_train.shape[0]:,}건 (이탈률: {y_train.mean()*100:.2f}%)")
    print(f"- 테스트셋 크기: {X_test.shape[0]:,}건 (이탈률: {y_test.mean()*100:.2f}%)")

    # ==========================================================================
    # 3. 전처리 파이프라인 구성 (StandardScaler + OneHotEncoder)
    # ==========================================================================
    print("\n[3] 전처리 파이프라인(ColumnTransformer) 구성 중...")
    
    # 로지스틱 회귀의 다중공선성 방지를 위해 drop='first' 적용
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ]
    )

    # ==========================================================================
    # 4. 모델 학습 및 하이퍼파라미터 튜닝 (GridSearchCV)
    # ==========================================================================
    print("\n[4] 로지스틱 회귀 모델 최적화 (GridSearchCV with StratifiedKFold)...")
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs'))
    ])

    param_grid = {
        'classifier__C': [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        'classifier__class_weight': ['balanced', None]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    print(f"- 최적 하이퍼파라미터: C={best_params['classifier__C']}, class_weight={best_params['classifier__class_weight']}")
    print(f"- 5-Fold CV 최고 ROC-AUC 점수: {best_cv_score:.4f}")

    # ==========================================================================
    # 5. 테스트셋 평가 및 최적 임계값(Threshold) 분석
    # ==========================================================================
    print("\n[5] 테스트셋(Test Set) 종합 평가...")

    # 기본 임계값(0.5) 기준 예측
    y_pred_default = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # 평가지표 산출
    acc = accuracy_score(y_test, y_pred_default)
    prec = precision_score(y_test, y_pred_default)
    rec = recall_score(y_test, y_pred_default)
    f1 = f1_score(y_test, y_pred_default)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    print("\n[기본 임계값(Threshold = 0.5) 기준 성능 지표]:")
    print(f"  - 정확도 (Accuracy)       : {acc:.4f}")
    print(f"  - 정밀도 (Precision)      : {prec:.4f}")
    print(f"  - 재현율 (Recall)         : {rec:.4f}  (실제 이탈 고객 감지율)")
    print(f"  - F1-Score (조화평균)     : {f1:.4f}")
    print(f"  - ROC-AUC                 : {roc_auc:.4f}")
    print(f"  - PR-AUC (Average Prec.)  : {pr_auc:.4f}")

    print("\n[상세 Classification Report]:")
    print(classification_report(y_test, y_pred_default, target_names=['유지 (No)', '이탈 (Yes)']))

    # 최적 임계값 탐색 (F1-score 극대화 기준)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]

    y_pred_opt = (y_prob >= best_threshold).astype(int)
    print(f"\n[비즈니스 최적 임계값 튜닝 (F1-Score 극대화)]: Threshold = {best_threshold:.4f}")
    print(f"  - 최적 임계값 적용 시 F1-Score : {best_f1:.4f} (정밀도: {precisions[best_idx]:.4f}, 재현율: {recalls[best_idx]:.4f})")

    # ==========================================================================
    # 6. 회귀계수(Coefficients) 및 승산비(Odds Ratio) 분석
    # ==========================================================================
    print("\n[6] 로지스틱 회귀 계수(Feature Importance & Odds Ratio) 추출...")
    
    # 피처명 추출
    fitted_preprocessor = best_model.named_steps['preprocessor']
    ohe_cols = fitted_preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
    all_feature_names = numeric_features + ohe_cols

    # 계수 및 승산비(Odds Ratio = exp(beta))
    coef = best_model.named_steps['classifier'].coef_[0]
    odds_ratio = np.exp(coef)

    coef_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Coefficient': coef,
        'Odds_Ratio': odds_ratio,
        'Abs_Coef': np.abs(coef)
    }).sort_values(by='Coefficient', ascending=False)

    print("\n[*] 이탈 위험 증가 상위 5개 요인 (Positive Coefficients / Odds Ratio > 1):")
    for _, row in coef_df.head(5).iterrows():
        print(f"  - {row['Feature']:<35}: 계수 {row['Coefficient']:+.4f} | 승산비(Odds Ratio) {row['Odds_Ratio']:.2f}배")

    print("\n[*] 이탈 방어/유지 기여 상위 5개 요인 (Negative Coefficients / Odds Ratio < 1):")
    for _, row in coef_df.tail(5).iloc[::-1].iterrows():
        print(f"  - {row['Feature']:<35}: 계수 {row['Coefficient']:+.4f} | 승산비(Odds Ratio) {row['Odds_Ratio']:.2f}배")

    # ==========================================================================
    # 7. 종합 평가 시각화 (4분할 서브플롯)
    # ==========================================================================
    print("\n[7] 종합 평가 시각화 차트 생성 중...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    plt.subplots_adjust(hspace=0.32, wspace=0.25)

    # 1. 혼동 행렬 (Confusion Matrix)
    cm = confusion_matrix(y_test, y_pred_default)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], cbar=False,
                xticklabels=['예측: 유지(No)', '예측: 이탈(Yes)'],
                yticklabels=['실제: 유지(No)', '실제: 이탈(Yes)'])
    axes[0, 0].set_title(f"1. 혼동 행렬 (Confusion Matrix, Th=0.5)\n정확도: {acc*100:.1f}%, 재현율: {rec*100:.1f}%", fontsize=13, fontweight='bold')
    axes[0, 0].set_ylabel("실제 라벨 (Actual)", fontsize=11)
    axes[0, 0].set_xlabel("예측 라벨 (Predicted)", fontsize=11)

    # 2. ROC 커브
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[0, 1].plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'Logistic Regression (AUC = {roc_auc:.4f})')
    axes[0, 1].plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.2)
    axes[0, 1].set_title("2. ROC Curve (수신자 조작 특성 곡선)", fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel("위양성률 (False Positive Rate)", fontsize=11)
    axes[0, 1].set_ylabel("진양성률 (True Positive Rate / Recall)", fontsize=11)
    axes[0, 1].legend(loc="lower right", fontsize=11)
    axes[0, 1].grid(True, linestyle=':', alpha=0.6)

    # 3. Precision-Recall 커브
    axes[1, 0].plot(recalls, precisions, color='#2ca02c', lw=2.5, label=f'PR Curve (AP / PR-AUC = {pr_auc:.4f})')
    axes[1, 0].scatter(recalls[best_idx], precisions[best_idx], color='red', s=80, zorder=5,
                       label=f'최적 F1 점 ({best_threshold:.2f})')
    axes[1, 0].set_title("3. Precision-Recall Curve (불균형 데이터 핵심 지표)", fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel("재현율 (Recall)", fontsize=11)
    axes[1, 0].set_ylabel("정밀도 (Precision)", fontsize=11)
    axes[1, 0].legend(loc="lower left", fontsize=11)
    axes[1, 0].grid(True, linestyle=':', alpha=0.6)

    # 4. 상위 회귀계수 (특성 영향력 막대 그래프)
    top_pos = coef_df.head(8)
    top_neg = coef_df.tail(8)
    top_features = pd.concat([top_pos, top_neg]).sort_values(by='Coefficient', ascending=True)

    colors = ['#2b5c8f' if c < 0 else '#d9534f' for c in top_features['Coefficient']]
    y_pos = np.arange(len(top_features))

    axes[1, 1].barh(y_pos, top_features['Coefficient'], color=colors, alpha=0.85)
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels(top_features['Feature'], fontsize=10)
    axes[1, 1].axvline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[1, 1].set_title("4. 주요 특성 회귀계수 Top 16 (빨강: 이탈 촉진 / 파랑: 유지 기여)", fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel("회귀 계수 (Log Odds)", fontsize=11)
    axes[1, 1].grid(True, linestyle=':', alpha=0.6)

    plt.suptitle("Telco Churn 로지스틱 회귀 모델 학습 및 종합 성능 평가", fontsize=16, fontweight='bold', y=0.99)
    save_path = os.path.join(OUTPUT_DIR, "telco_logistic_regression_eval.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"- 평가 차트 저장 완료: {save_path}")

    print("\n" + "=" * 80)
    print("  로지스틱 회귀 모델 학습 및 평가가 성공적으로 완료되었습니다.")
    print("=" * 80)


if __name__ == "__main__":
    main()
