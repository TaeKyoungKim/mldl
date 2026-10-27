import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours

# Matplotlib 한글 및 스타일 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess(data_path="C:/apps/mldl/data/bank/bank-full.csv"):
    """
    Bank Marketing 데이터를 로드하고 인코딩 및 스케일링 전처리를 수행합니다.
    """
    print("=" * 80)
    print(" 1. Bank Marketing 데이터 로드 및 전처리")
    print("=" * 80)
    
    df = pd.read_csv(data_path, sep=';')
    print(f"▶ 원본 데이터 형태: {df.shape[0]:,}행 x {df.shape[1]}열")

    # 타깃 변수 'y'를 이진화 (no: 0, yes: 1)
    df['y'] = (df['y'] == 'yes').astype(int)

    # 특성과 타깃 분리
    X = df.drop(columns=['y'])
    y = df['y']

    # 범주형 변수 One-Hot 인코딩
    cat_cols = X.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"▶ 수치형 피처({len(num_cols)}개): {num_cols}")
    print(f"▶ 범주형 피처({len(cat_cols)}개): {cat_cols}")

    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    # 훈련셋 / 테스트셋 분리 (평가 데이터는 왜곡 없이 원본 분포 80:20 Stratified 분할 유지)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    # 거리 기반 언더샘플링(Tomek, ENN) 및 로지스틱 회귀를 위해 표준화 스케일링 수행
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 컬럼명 복원
    feature_names = X_encoded.columns.tolist()
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

    print(f"▶ 인코딩 후 피처 수: {X_train_scaled.shape[1]}개")
    print(f"▶ Train 데이터: {len(X_train):,}개 (Class 0: {(y_train == 0).sum():,}개, Class 1: {(y_train == 1).sum():,}개)")
    print(f"▶ Test 데이터: {len(X_test):,}개 (Class 0: {(y_test == 0).sum():,}개, Class 1: {(y_test == 1).sum():,}개)")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def evaluate_model(X_train, y_train, X_test, y_test, method_name=""):
    """
    동일한 베이스라인 로지스틱 회귀 분류기를 학습하고 테스트셋 성능을 산출합니다.
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    return {
        'Method': method_name,
        'Train Total': len(y_train),
        'Class 0 (No)': (y_train == 0).sum(),
        'Class 1 (Yes)': (y_train == 1).sum(),
        'Imbalance Ratio': round((y_train == 0).sum() / max(1, (y_train == 1).sum()), 2),
        'Precision (Yes)': report['1']['precision'],
        'Recall (Yes)': report['1']['recall'],
        'F1-Score (Yes)': f1,
        'ROC-AUC': roc_auc
    }

def run_undersampling_experiments(X_train, X_test, y_train, y_test):
    """
    3가지 언더샘플링 기법 적용 및 성능 비교:
    1. Baseline (원형 유지)
    2. Random Undersampling (무작위 1:1 다운샘플링)
    3. Tomek Links (경계면 노이즈 정제)
    4. Edited Nearest Neighbours (ENN, 이웃 기반 노이즈 제거)
    """
    print("\n" + "=" * 80)
    print(" 2. 언더샘플링 기법 적용 및 비교 실험")
    print("=" * 80)

    results = []

    # 1. Baseline (불균형 원본)
    print("\n[1] Baseline (원형 유지 학습 중...)")
    res_base = evaluate_model(X_train, y_train, X_test, y_test, method_name="1. Baseline (No Sampling)")
    results.append(res_base)

    # 2. Random Undersampling
    print("[2] Random Undersampling (1:1 비율 균형화 중...)")
    rus = RandomUnderSampler(random_state=42)
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
    res_rus = evaluate_model(X_train_rus, y_train_rus, X_test, y_test, method_name="2. Random Undersampling")
    results.append(res_rus)

    # 3. Tomek Links (경계면 노이즈 정제)
    print("[3] Tomek Links (경계면 모호한 쌍 제거 중...)")
    # sampling_strategy='auto' (다수 클래스에서만 쌍 제거)
    tl = TomekLinks(sampling_strategy='auto')
    X_train_tl, y_train_tl = tl.fit_resample(X_train, y_train)
    res_tl = evaluate_model(X_train_tl, y_train_tl, X_test, y_test, method_name="3. Tomek Links")
    results.append(res_tl)

    # 4. Edited Nearest Neighbours (ENN)
    print("[4] Edited Nearest Neighbours (ENN, k=3 이웃 불일치 샘플 제거 중...)")
    enn = EditedNearestNeighbours(n_neighbors=3, sampling_strategy='auto')
    X_train_enn, y_train_enn = enn.fit_resample(X_train, y_train)
    res_enn = evaluate_model(X_train_enn, y_train_enn, X_test, y_test, method_name="4. Edited Nearest Neighbours (ENN)")
    results.append(res_enn)

    df_results = pd.DataFrame(results)
    return df_results

def print_explanation(df_results):
    """
    언더샘플링 결과 상세 비교표 및 메커니즘 해설 출력
    """
    print("\n" + "=" * 80)
    print(" 3. 언더샘플링 실험 결과 비교표")
    print("=" * 80)
    print(df_results.to_string(index=False))

    print("\n" + "=" * 80)
    print(" 4. 기법별 원리 및 결과 심층 분석")
    print("=" * 80)
    explanation = """
1) Random Undersampling (무작위 언더샘플링):
   - 원리: 다수 클래스(0: No)를 소수 클래스(1: Yes)의 개수(4,231개)와 동일하게 무작위 추출하여 1:1로 맞춤.
   - 데이터 변화: 36,168개 -> 8,462개로 약 76.6% 데이터 축소 (학습 속도 대폭 향상).
   - 성능 특성:
     * 재현율(Recall)이 급상승: 모델이 소수 클래스('Yes') 예측에 적극적으로 변함.
     * 정밀도(Precision)는 하락: 다수 클래스의 대다수 유용한 정보가 버려져 오탐(False Positive)이 증가.
     * F1-Score 및 ROC-AUC가 균형을 이루며, 소수 클래스를 반드시 놓치지 않아야 하는 비즈니스 상황에 적합.

2) Tomek Links (토멕 링크):
   - 원리: 서로 다른 클래스에 속하면서 서로에게 가장 가까운 최근접 이웃 쌍(Tomek Link)을 찾아 다수 클래스 샘플을 제거.
   - 데이터 변화: 다수 클래스에서 경계면에 엉켜있는 모호한 샘플만 핀포인트로 제거됨 (소수 클래스 개수는 보존).
   - 성능 특성:
     * 전체 데이터 크기는 거의 줄어들지 않으나, 결정 경계(Decision Boundary)가 명확해짐.
     * 정밀도(Precision)를 보존하면서 미세하게 경계면 분류 신뢰도를 높임.
     * 실무에서는 단독 언더샘플링보다는 SMOTE 등의 오버샘플링 후 경계면 노이즈 제거용(SMOTE+Tomek)으로 강력하게 권장됨.

3) Edited Nearest Neighbours (ENN):
   - 원리: k개(기본 3개)의 최근접 이웃과 라벨을 비교하여, 이웃들의 다수 의견과 자신의 라벨이 일치하지 않는 다수 클래스 샘플을 제거.
   - 데이터 변화: 경계면에 침범한 노이즈 데이터를 Tomek Links보다 훨씬 공격적으로 다량 제거함.
   - 성능 특성:
     * Tomek Links보다 더 많은 경계면 노이즈를 솎아내어 재현율(Recall) 향상에 더 크게 기여함.
     * SMOTE+ENN 복합 기법(오버샘플링 후 노이즈 필터링)의 구성 요소로 널리 활용됨.
"""
    print(explanation)

def plot_comparisons(df_results):
    """
    결과를 시각화하여 PNG 이미지로 저장
    """
    output_dir = "C:/apps/mldl/output/bank_eda"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "undersampling_comparison.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. 샘플 수 및 불균형 비율
    ax1 = axes[0]
    methods = df_results['Method'].str.split('. ').str[-1]
    ax1.bar(methods, df_results['Train Total'], color='#5C6BC0', alpha=0.85, label='총 학습 데이터 수')
    ax1.set_title("1. 기법별 학습 데이터 크기 비교", fontsize=12, fontweight='bold')
    ax1.set_ylabel("데이터 수 (건)")
    ax1.tick_params(axis='x', rotation=20)
    for i, v in enumerate(df_results['Train Total']):
        ax1.text(i, v + 500, f"{v:,}건", ha='center', fontsize=9, fontweight='bold')

    # 2. Precision vs Recall Trade-off
    ax2 = axes[1]
    x_indices = np.arange(len(methods))
    bar_width = 0.35
    ax2.bar(x_indices - bar_width/2, df_results['Precision (Yes)'], width=bar_width, label='Precision (정밀도)', color='#26A69A')
    ax2.bar(x_indices + bar_width/2, df_results['Recall (Yes)'], width=bar_width, label='Recall (재현율)', color='#FFA726')
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(methods, rotation=20)
    ax2.set_title("2. 소수 클래스(Yes) Precision vs Recall", fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1.0)
    ax2.legend()

    # 3. F1-Score & ROC-AUC
    ax3 = axes[2]
    ax3.bar(x_indices - bar_width/2, df_results['F1-Score (Yes)'], width=bar_width, label='F1-Score', color='#AB47BC')
    ax3.bar(x_indices + bar_width/2, df_results['ROC-AUC'], width=bar_width, label='ROC-AUC', color='#42A5F5')
    ax3.set_xticks(x_indices)
    ax3.set_xticklabels(methods, rotation=20)
    ax3.set_title("3. F1-Score 및 ROC-AUC 비교", fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 1.0)
    ax3.legend()

    plt.suptitle("Bank Marketing 데이터셋 언더샘플링 기법별 효과 비교", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n▶ 비교 시각화 차트 저장 완료: {save_path}")

def main():
    data_path = "C:/apps/mldl/data/bank/bank-full.csv"
    X_train, X_test, y_train, y_test = load_and_preprocess(data_path)
    df_results = run_undersampling_experiments(X_train, X_test, y_train, y_test)
    print_explanation(df_results)
    plot_comparisons(df_results)

if __name__ == "__main__":
    main()
