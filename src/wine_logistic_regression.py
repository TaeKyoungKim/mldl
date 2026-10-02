import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss
)

# ==========================================
# 0. 한글 폰트 및 시각화 스타일 설정 (Windows 환경)
# ==========================================
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='Malgun Gothic')

OUTPUT_DIR = "C:/apps/mldl/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 1. 와인 데이터셋 로드 및 특성 스케일 확인
# ==========================================
print("=" * 60)
print("[1] Wine 데이터셋 로드 및 구조 탐색")
print("=" * 60)
wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names
target_names = wine.target_names

df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print(f"- 샘플 수: {df.shape[0]}개, 특성 수: {len(feature_names)}개")
print(f"- 타깃 클래스: {target_names.tolist()} (0, 1, 2)")
print("\n[주요 특성별 스케일 차이 확인 (평균 및 표준편차)]:")
scale_summary = df[feature_names].agg(['mean', 'std', 'min', 'max']).T[['mean', 'std', 'min', 'max']]
print(scale_summary.head(7).to_string())
print("  ... (예: proline은 1000단위인 반면 nonflavanoid_phenols는 0.3단위로 극심한 스케일 차이 존재)")

# ==========================================
# 2. 데이터 분할 (Train 80% / Test 20%)
# ==========================================
# 다중 클래스 비율을 유지하기 위해 stratify=y 지정
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 3. StandardScaler 적용 전/후 로지스틱 회귀 비교
# ==========================================
print("\n" + "=" * 60)
print("[2] StandardScaler 전처리 필요성 실습 비교")
print("=" * 60)

# (A) 스케일링 미적용 모델
clf_unscaled = LogisticRegression(
    max_iter=500,
    random_state=42
)
clf_unscaled.fit(X_train, y_train)
y_pred_unscaled = clf_unscaled.predict(X_test)
acc_unscaled = accuracy_score(y_test, y_pred_unscaled)
loss_unscaled = log_loss(y_test, clf_unscaled.predict_proba(X_test))

print(f"[스케일링 미적용] 정확도: {acc_unscaled:.4f} | 다중 클래스 교차 엔트로피 손실(Log Loss): {loss_unscaled:.4f}")

# (B) StandardScaler 적용 모델
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf_scaled = LogisticRegression(
    max_iter=500,
    random_state=42
)
clf_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = clf_scaled.predict(X_test_scaled)
y_prob_scaled = clf_scaled.predict_proba(X_test_scaled)
acc_scaled = accuracy_score(y_test, y_pred_scaled)
loss_scaled = log_loss(y_test, y_prob_scaled)

print(f"[StandardScaler 적용] 정확도: {acc_scaled:.4f} | 다중 클래스 교차 엔트로피 손실(Log Loss): {loss_scaled:.4f}")

# ==========================================
# 4. 상세 분류 성능 평가 (Classification Report & Confusion Matrix)
# ==========================================
print("\n" + "=" * 60)
print("[3] 최종 모델(StandardScaler 적용) 상세 평가 결과")
print("=" * 60)
print(classification_report(y_test, y_pred_scaled, target_names=target_names))

# ==========================================
# 5. 시각화 (혼동 행렬 & 클래스별 회귀계수 가중치)
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# (1) 혼동 행렬 시각화
cm = confusion_matrix(y_test, y_pred_scaled)
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=target_names,
    yticklabels=target_names,
    ax=axes[0],
    cbar=False
)
axes[0].set_title("혼동 행렬 (Confusion Matrix)", fontsize=14, fontweight='bold', pad=12)
axes[0].set_xlabel("예측 클래스 (Predicted)", fontsize=12)
axes[0].set_ylabel("실제 클래스 (Actual)", fontsize=12)

# (2) 클래스별 회귀계수(Weights) 히트맵 시각화
# 다중 클래스 로지스틱 회귀에서 각 클래스에 대한 피처의 영향력 확인
coef_df = pd.DataFrame(clf_scaled.coef_, index=target_names, columns=feature_names)
sns.heatmap(
    coef_df,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    center=0,
    cbar_kws={'label': '회귀 계수 (가중치 w)'},
    ax=axes[1]
)
axes[1].set_title("클래스별 특성 가중치 (Feature Coefficients)", fontsize=14, fontweight='bold', pad=12)
axes[1].set_xlabel("화학 성분 특성 (Features)", fontsize=12)
axes[1].set_ylabel("와인 품종 (Class)", fontsize=12)
axes[1].tick_params(axis='x', rotation=45)

plt.suptitle(
    f"Wine 데이터셋 로지스틱 회귀 다중 분류 결과 (정확도: {acc_scaled*100:.1f}%, Log Loss: {loss_scaled:.4f})",
    fontsize=16,
    fontweight='bold',
    y=1.02
)
plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, "wine_logistic_regression.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"- 시각화 결과 저장 완료: {save_path}")

# ==========================================
# 6. 클래스별 상위 기여 특성 분석 요약 출력
# ==========================================
print("\n" + "=" * 60)
print("[4] 품종별 주요 화학 성분 영향력 분석 (가장 큰 양의 가중치)")
print("=" * 60)
for idx, target in enumerate(target_names):
    top_pos_feature = coef_df.iloc[idx].idxmax()
    top_pos_val = coef_df.iloc[idx].max()
    top_neg_feature = coef_df.iloc[idx].idxmin()
    top_neg_val = coef_df.iloc[idx].min()
    print(f"[{target}]")
    print(f"  - 긍정 영향 요인(+): '{top_pos_feature}' (가중치: {top_pos_val:+.3f})")
    print(f"  - 부정 영향 요인(-): '{top_neg_feature}' (가중치: {top_neg_val:+.3f})")
