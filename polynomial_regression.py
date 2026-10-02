import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

def generate_sample_data(n_samples: int = 1000, random_state: int = 42):
    """
    다항 회귀를 위한 1000개의 샘플 데이터 생성
    관계식: y = 0.5 * x^3 - 1.2 * x^2 + 0.8 * x + 3 + noise
    """
    rng = np.random.RandomState(random_state)
    
    # -3에서 3 사이의 X 데이터 1000개 생성
    X = rng.uniform(-3, 3, size=(n_samples, 1))
    
    # 3차 다항식 기반의 참값 계산
    y_true = 0.5 * (X ** 3) - 1.2 * (X ** 2) + 0.8 * X + 3
    
    # 가우시안 노이즈(오차) 추가
    noise = rng.normal(0, 2.0, size=(n_samples, 1))
    y = y_true + noise
    
    return X, y.ravel(), y_true.ravel()

def train_and_visualize(degree: int = 3):
    # 1. 1000개의 샘플 데이터 생성
    X, y, _ = generate_sample_data(n_samples=1000)

    # 2. 다항 특성 변환 및 선형 회귀 파이프라인 구성 및 학습
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )
    model.fit(X, y)

    # 3. 모델 평가
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"=== 다항 회귀 (Degree: {degree}) 학습 결과 ===")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # 회귀 계수 출력
    lin_reg = model.named_steps['linearregression']
    print(f"절편(Intercept): {lin_reg.intercept_:.4f}")
    print(f"가중치(Coefficients): {lin_reg.coef_}")

    # 4. 시각화를 위한 정렬된 X_test 생성
    X_test = np.linspace(-3, 3, 500).reshape(-1, 1)
    y_test_pred = model.predict(X_test)
    y_test_true = 0.5 * (X_test ** 3) - 1.2 * (X_test ** 2) + 0.8 * X_test + 3

    # 5. 그래프 시각화
    plt.figure(figsize=(10, 6))
    
    # 생성된 1000개의 샘플 데이터 산점도
    plt.scatter(X, y, color='steelblue', alpha=0.35, edgecolors='none', label='Sample Data (N=1000)')
    
    # 다항 회귀 예측 곡선
    plt.plot(X_test, y_test_pred, color='crimson', linewidth=2.5, label=f'Polynomial Fit (degree={degree})')
    
    # 실제 원본 함수 곡선
    plt.plot(X_test, y_test_true, color='black', linestyle='--', linewidth=1.5, label='True Function (Ground Truth)')

    plt.title(f'Polynomial Regression (Degree = {degree}, $R^2$ = {r2:.3f})', fontsize=14, fontweight='bold')
    plt.xlabel('X', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    # 이미지 저장 및 화면 출력
    plt.savefig('polynomial_regression_result.png', dpi=300)
    print("그래프가 'polynomial_regression_result.png' 파일로 저장되었습니다.")
    plt.show()

if __name__ == '__main__':
    train_and_visualize(degree=3)
