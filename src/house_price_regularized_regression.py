import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================
# 0. 한글 폰트 및 시각화 스타일 설정 (Windows 환경)
# ==========================================
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='Malgun Gothic')

# ==========================================
# 1. 경로 설정 및 데이터 로드
# ==========================================
BASE_DATA_DIR = "C:/apps/mldl/data"
OUTPUT_DIR = "C:/apps/mldl/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_path = os.path.join(BASE_DATA_DIR, "house_train.csv")
test_path = os.path.join(BASE_DATA_DIR, "house_test.csv")
sample_sub_path = os.path.join(BASE_DATA_DIR, "house_sample_submission.csv")

print("[1] 데이터 로드 중...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub_df = pd.read_csv(sample_sub_path)

print(f"- Train 크기: {train_df.shape}")
print(f"- Test 크기: {test_df.shape}")

# Id 컬럼 분리
train_ids = train_df['Id']
test_ids = test_df['Id']

# 타깃 분리 및 로그 변환 (주택 가격의 왜도 완화 및 안정적인 선형 회귀 학습)
y_train_orig = train_df['SalePrice']
y_train = np.log1p(y_train_orig)

X_train_raw = train_df.drop(columns=['Id', 'SalePrice'])
X_test_raw = test_df.drop(columns=['Id'])

# ==========================================
# 2. 데이터 전처리 파이프라인 구축
# ==========================================
print("\n[2] 전처리 파이프라인 구성 중...")
numeric_features = X_train_raw.select_dtypes(include=['int64', 'float64', 'number']).columns.tolist()
categorical_features = [col for col in X_train_raw.columns if col not in numeric_features]

# 수치형: 결측치 중앙값 대체 + 표준화(StandardScaler) -> 규제 모델에서 스케일링 필수
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 범주형: 결측치 'Missing' 대체 + 원-핫 인코딩
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 전체 Feature 전처리 변환
X_train_processed = preprocessor.fit_transform(X_train_raw)
X_test_processed = preprocessor.transform(X_test_raw)

# 변환된 피처 이름 추출
feature_names = preprocessor.get_feature_names_out()
print(f"- 전처리 후 총 특성(Feature) 수: {len(feature_names)}")

# 평가를 위한 검증셋 분할 (8:2)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_processed, y_train, test_size=0.2, random_state=42
)

# ==========================================
# 3. 릿지, 라쏘, 엘라스틱 넷 모델 학습 (교차검증 기반 최적 파라미터 탐색)
# ==========================================
print("\n[3] 모델 학습 및 최적 하이퍼파라미터 탐색 중...")
cv = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Ridge': RidgeCV(alphas=np.logspace(-2, 3, 50), cv=cv),
    'Lasso': LassoCV(alphas=np.logspace(-4, 1, 50), max_iter=10000, cv=cv, random_state=42),
    'ElasticNet': ElasticNetCV(
        alphas=np.logspace(-4, 1, 50),
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
        max_iter=10000,
        cv=cv,
        random_state=42
    )
}

trained_models = {}
val_results = {}

for name, model in models.items():
    # 학습
    model.fit(X_tr, y_tr)
    trained_models[name] = model

    # 검증셋 예측 (로그 스케일 -> 원 스케일 복원)
    pred_val_log = model.predict(X_val)
    pred_val = np.expm1(pred_val_log)
    y_val_orig = np.expm1(y_val)

    # 평가 지표 산출
    rmse = np.sqrt(mean_squared_error(y_val_orig, pred_val))
    mae = mean_absolute_error(y_val_orig, pred_val)
    r2 = r2_score(y_val_orig, pred_val)

    val_results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}

    best_param_info = f"alpha={model.alpha_:.4f}"
    if hasattr(model, 'l1_ratio_'):
        best_param_info += f", l1_ratio={model.l1_ratio_:.2f}"
    print(f"- [{name}] 최적 파라미터: {best_param_info} | Validation RMSE: ${rmse:,.2f} | MAE: ${mae:,.2f} | R²: {r2:.4f}")

# ==========================================
# 4. 전체 Train 데이터 재학습 및 Test 데이터 예측 & 파일 저장
# ==========================================
print("\n[4] Test 데이터셋 예측 및 제출 파일 생성 중...")
for name, model in trained_models.items():
    # 전체 훈련 데이터로 재학습
    model.fit(X_train_processed, y_train)

    # Test 예측 및 원래 스케일 복원
    test_pred_log = model.predict(X_test_processed)
    test_pred = np.expm1(test_pred_log)

    # Submission DataFrame 생성 및 저장
    sub_df = sample_sub_df.copy()
    sub_df['SalePrice'] = test_pred
    out_file = os.path.join(OUTPUT_DIR, f"submission_{name.lower()}.csv")
    sub_df.to_csv(out_file, index=False)
    print(f"- {name} 예측 결과 저장 완료 -> {out_file}")

# ==========================================
# 5. 회귀계수(Coefficients) 분석 및 시각화
# ==========================================
print("\n[5] 회귀계수 시각화 생성 중...")

# 피처 이름 클린징 (가독성 향상)
clean_feature_names = [f.replace('num__', '').replace('cat__', '') for f in feature_names]

fig, axes = plt.subplots(1, 3, figsize=(20, 9), sharey=False)
plt.subplots_adjust(wspace=0.35)

top_n = 15

for idx, (name, model) in enumerate(trained_models.items()):
    coefs = model.coef_
    
    # 0이 아닌 계수 비율 파악
    nonzero_count = np.sum(coefs != 0)
    total_count = len(coefs)
    
    coef_df = pd.DataFrame({
        'Feature': clean_feature_names,
        'Coefficient': coefs,
        'AbsCoef': np.abs(coefs)
    }).sort_values(by='AbsCoef', ascending=False).head(top_n)

    # 부호별 색상 지정 (양수는 파란색/녹색 계열, 음수는 붉은색 계열)
    colors = ['#d9534f' if c < 0 else '#337ab7' for c in coef_df['Coefficient']]

    ax = axes[idx]
    y_pos = np.arange(len(coef_df))
    ax.barh(y_pos, coef_df['Coefficient'], color=colors, edgecolor='none', alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(coef_df['Feature'], fontsize=10)
    ax.invert_yaxis()  # 상위 계수가 위로 오도록 정렬
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    
    best_param = f"alpha={model.alpha_:.4f}"
    if hasattr(model, 'l1_ratio_'):
        best_param += f", l1={model.l1_ratio_:.2f}"
        
    ax.set_title(f"{name}\n(유효 특성수: {nonzero_count}/{total_count}, {best_param})", fontsize=13, fontweight='bold')
    ax.set_xlabel("회귀 계수 값 (Coefficient)", fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.6)

plt.suptitle("Ridge vs Lasso vs ElasticNet 회귀 계수 (상위 15개 특성 비교)", fontsize=16, fontweight='bold', y=1.02)
fig_path = os.path.join(OUTPUT_DIR, "coefficients_comparison.png")
plt.tight_layout()
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"- 회귀계수 시각화 차트 저장 완료 -> {fig_path}")

print("\n" + "="*50)
print("모든 작업이 성공적으로 완료되었습니다!")
print(f"결과물 저장 위치: {OUTPUT_DIR}")
print("="*50)
