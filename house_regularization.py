import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 한글 폰트 설정 (Windows 기준 Malgun Gothic)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data(train_path: str, test_path: str, submission_path: str = None):
    """
    데이터 로드 및 train/test 전처리 준비
    house_test.csv에 SalePrice가 없는 경우 house_sample_submission.csv를 병합하여 평가용 정답으로 사용합니다.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # test 데이터에 SalePrice가 없는 경우 submission 파일에서 병합
    if 'SalePrice' not in test_df.columns:
        if submission_path and os.path.exists(submission_path):
            sub_df = pd.read_csv(submission_path)
            test_df = test_df.merge(sub_df[['Id', 'SalePrice']], on='Id', how='left')
            print(f"[*] '{submission_path}'로부터 test 데이터의 'SalePrice' 정답을 병합했습니다.")
        else:
            print("[!] test 데이터셋에 'SalePrice' 컬럼이 없어 실제 정답 평가는 제한될 수 있습니다.")

    # Id 컬럼 및 타깃 분리
    X_train_full = train_df.drop(columns=['Id', 'SalePrice'])
    y_train_full = train_df['SalePrice']

    # 타깃 변수 로그 변환 (주택 가격의 왜도 완화)
    y_train_full_log = np.log1p(y_train_full)

    if 'SalePrice' in test_df.columns and not test_df['SalePrice'].isnull().all():
        X_test_final = test_df.drop(columns=['Id', 'SalePrice'])
        y_test_final = test_df['SalePrice']
        y_test_final_log = np.log1p(y_test_final)
    else:
        X_test_final = test_df.drop(columns=['Id'])
        y_test_final = None
        y_test_final_log = None

    # 수치형 / 범주형 컬럼 분류
    num_cols = X_train_full.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_train_full.select_dtypes(include=['object', 'category']).columns.tolist()

    # 전처리 파이프라인 구성
    # 수치형: 결측치 중앙값 대체 + 표준화(StandardScaler)
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 범주형: 결측치 최빈값 대체 + 원-핫 인코딩(OneHotEncoder)
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )

    # 전처리기 학습 및 변환
    X_train_trans = preprocessor.fit_transform(X_train_full)
    X_test_trans = preprocessor.transform(X_test_final)

    # 원-핫 인코딩 후 피처 이름 추출
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    encoded_cat_cols = cat_encoder.get_feature_names_out(cat_cols).tolist()
    all_feature_names = num_cols + encoded_cat_cols

    return (X_train_trans, y_train_full, y_train_full_log,
            X_test_trans, y_test_final, y_test_final_log,
            all_feature_names, test_df['Id'])

def train_and_evaluate_models(X_train, y_train_log, X_test, y_test, y_test_log):
    """
    릿지(Ridge), 라쏘(Lasso), 엘라스틱 넷(ElasticNet) 학습 및 평가
    """
    alphas = np.logspace(-3, 3, 50)
    
    models = {
        'Ridge': RidgeCV(alphas=alphas, cv=5),
        'Lasso': LassoCV(alphas=np.logspace(-4, 1, 50), max_iter=10000, random_state=42, cv=5),
        'ElasticNet': ElasticNetCV(alphas=np.logspace(-4, 1, 30), l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                                   max_iter=10000, random_state=42, cv=5)
    }

    results = {}
    fitted_models = {}

    print("\n" + "=" * 60)
    print("           모델 학습 및 테스트 데이터 평가 결과           ")
    print("=" * 60)

    for name, model in models.items():
        # 모델 학습 (로그 변환된 y_train_log 사용)
        model.fit(X_train, y_train_log)
        fitted_models[name] = model

        # 최적 파라미터 확인
        best_alpha = getattr(model, 'alpha_', None)
        l1_ratio = getattr(model, 'l1_ratio_', None)
        param_str = f"alpha={best_alpha:.4f}" + (f", l1_ratio={l1_ratio:.2f}" if l1_ratio else "")

        # 예측 (로그 스케일 -> 원래 스케일 복원)
        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)

        # 평가 (y_test가 존재하는 경우)
        if y_test is not None:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'Best Params': param_str,
                'Predictions': y_pred
            }
            print(f"[{name}] ({param_str})")
            print(f"  - Test RMSE : ${rmse:,.2f}")
            print(f"  - Test MAE  : ${mae:,.2f}")
            print(f"  - Test R²   : {r2:.4f}")
        else:
            results[name] = {
                'Best Params': param_str,
                'Predictions': y_pred
            }
            print(f"[{name}] ({param_str}) - 예측 완료")

    print("=" * 60)
    return fitted_models, results

def visualize_coefficients(fitted_models, feature_names, top_n: int = 20, save_path: str = 'regularized_regression_coefs.png'):
    """
    각 모델(Ridge, Lasso, ElasticNet)의 상위 회귀계수를 막대그래프로 시각화
    """
    model_names = list(fitted_models.keys())
    fig, axes = plt.subplots(1, 3, figsize=(20, 10), sharey=False)

    summary_info = []

    for idx, (name, model) in enumerate(fitted_models.items()):
        coef = model.coef_
        non_zero_count = np.sum(coef != 0)
        zero_count = np.sum(coef == 0)
        summary_info.append(f"{name}: 0이 아닌 계수 {non_zero_count}개 / 제거된 특성 {zero_count}개")

        # 계수 절대값 기준 상위 top_n개 선택
        coef_series = pd.Series(coef, index=feature_names)
        top_coefs = coef_series.reindex(coef_series.abs().sort_values(ascending=False).index).head(top_n)
        top_coefs = top_coefs.sort_values(ascending=True)  # 그래프 아래에서 위로 정렬

        colors = ['#e74c3c' if c < 0 else '#3498db' for c in top_coefs.values]
        
        ax = axes[idx]
        top_coefs.plot(kind='barh', ax=ax, color=colors, edgecolor='black', alpha=0.85)
        ax.set_title(f'{name} 상위 {top_n}개 회귀 계수\n(유효 특성: {non_zero_count}/{len(feature_names)})', fontsize=13, fontweight='bold')
        ax.set_xlabel('회귀 계수 (Coefficient Value)', fontsize=11)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.grid(axis='x', linestyle=':', alpha=0.6)

    plt.suptitle("House Prices: 릿지(Ridge), 라쏘(Lasso), 엘라스틱 넷(ElasticNet) 회귀 계수 비교", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[*] 회귀 계수 비교 그래프가 '{save_path}'에 저장되었습니다.")
    
    print("\n[피처 선택(Feature Selection) 요약]")
    for info in summary_info:
        print(f"  - {info}")

    plt.show()

def main():
    train_path = r"C:/apps/mldl/data/house_train.csv"
    test_path = r"C:/apps/mldl/data/house_test.csv"
    submission_path = r"C:/apps/mldl/data/house_sample_submission.csv"

    # 1. 데이터 로드 및 전처리
    (X_train, y_train, y_train_log,
     X_test, y_test, y_test_log,
     feature_names, test_ids) = load_and_preprocess_data(train_path, test_path, submission_path)

    print(f"[*] 전처리 완료: 학습 데이터 특성 수 = {X_train.shape[1]}개, 샘플 수 = {X_train.shape[0]}개")

    # 2. 모델 학습 및 평가
    fitted_models, results = train_and_evaluate_models(X_train, y_train_log, X_test, y_test, y_test_log)

    # 3. 테스트 예측 결과 저장 (선택 사항)
    preds_df = pd.DataFrame({'Id': test_ids})
    for name in fitted_models.keys():
        preds_df[f'SalePrice_Pred_{name}'] = results[name]['Predictions']
    preds_df.to_csv('house_test_predictions.csv', index=False)
    print(f"[*] 테스트 데이터 예측 결과가 'house_test_predictions.csv'에 저장되었습니다.")

    # 4. 회귀 계수 시각화
    visualize_coefficients(fitted_models, feature_names, top_n=20)

if __name__ == '__main__':
    main()
