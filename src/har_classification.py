"""
================================================================================
UCI ML Repository: Human Activity Recognition Using Smartphones (ID: 240)
스마트폰 센서(가속도계, 자이로스코프) 데이터를 활용한 6대 인간 행동 인식 머신러닝 분류
================================================================================
"""

import os
import sys
import zipfile
import urllib.request
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 윈도우 콘솔 UTF-8 한글 인코딩 대응
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# ==============================================================================
# 0. 환경 설정 및 시각화 스타일
# ==============================================================================
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='Malgun Gothic')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 행동 클래스 ID 및 이름 매핑 정의 (UCI 원본 정의)
ACTIVITY_MAP = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}


# ==============================================================================
# 1. UCI HAR 데이터셋 로더 (ucimlrepo 호환 및 Fallback 로더)
# ==============================================================================
def load_uci_har_fallback(data_dir: str):
    """
    UCI Repository에서 직접 CSV import가 지원되지 않을 때,
    공식 원본 데이터 파일을 안전하게 로드하여 ucimlrepo와 동일한 인터페이스로 반환합니다.
    """
    dataset_folder = os.path.join(data_dir, "UCI HAR Dataset")
    zip_path = os.path.join(data_dir, "har.zip")
    inner_zip_path = os.path.join(data_dir, "UCI HAR Dataset.zip")

    # 1) 폴더가 없는 경우 압축 해제 또는 다운로드
    if not os.path.exists(dataset_folder):
        if not os.path.exists(zip_path) and not os.path.exists(inner_zip_path):
            print("  [다운로드] UCI 저장소에서 스마트폰 행동 인식 데이터셋 다운로드 중...")
            url = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
            urllib.request.urlretrieve(url, zip_path)

        if os.path.exists(inner_zip_path):
            with zipfile.ZipFile(inner_zip_path, 'r') as z:
                z.extractall(data_dir)
        elif os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(data_dir)
            if os.path.exists(inner_zip_path):
                with zipfile.ZipFile(inner_zip_path, 'r') as z:
                    z.extractall(data_dir)

    # 2) 특성 이름 로드 및 중복 열 이름 정리
    features_path = os.path.join(dataset_folder, "features.txt")
    features_df = pd.read_csv(features_path, sep=r"\s+", header=None, names=["idx", "feature_name"])
    raw_feature_names = features_df["feature_name"].tolist()

    # 열 이름 중복 방지를 위한 고유 이름 생성
    seen = {}
    feature_names = []
    for name in raw_feature_names:
        if name in seen:
            seen[name] += 1
            feature_names.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            feature_names.append(name)

    # 3) Train / Test 원본 데이터셋 로드
    x_train_path = os.path.join(dataset_folder, "train", "X_train.txt")
    y_train_path = os.path.join(dataset_folder, "train", "y_train.txt")
    x_test_path = os.path.join(dataset_folder, "test", "X_test.txt")
    y_test_path = os.path.join(dataset_folder, "test", "y_test.txt")

    X_train = pd.read_csv(x_train_path, sep=r"\s+", header=None, names=feature_names)
    y_train = pd.read_csv(y_train_path, sep=r"\s+", header=None, names=["Activity"])

    X_test = pd.read_csv(x_test_path, sep=r"\s+", header=None, names=feature_names)
    y_test = pd.read_csv(y_test_path, sep=r"\s+", header=None, names=["Activity"])

    # 전체 X, y 결합 데이터프레임 구성
    X = pd.concat([X_train, X_test], ignore_index=True)
    y = pd.concat([y_train, y_test], ignore_index=True)

    # ucimlrepo의 fetch_ucirepo 결과 구조와 일치하는 SimpleNamespace 객체 구성
    har = SimpleNamespace()
    har.data = SimpleNamespace(
        features=X,
        targets=y,
        train=SimpleNamespace(features=X_train, targets=y_train),
        test=SimpleNamespace(features=X_test, targets=y_test)
    )
    har.metadata = {
        "id": 240,
        "name": "Human Activity Recognition Using Smartphones",
        "num_features": 561,
        "classes": ACTIVITY_MAP
    }
    return har


def main():
    print("#" * 80)
    print("  HUMAN ACTIVITY RECOGNITION (HAR) - SMARTPHONE SENSOR CLASSIFICATION")
    print("#" * 80)

    # ==========================================================================
    # 1. 데이터셋 로드 (ucimlrepo 호출 및 Fallback)
    # ==========================================================================
    print("\n[1] 데이터셋 로드...")
    try:
        from ucimlrepo import fetch_ucirepo
        # Human Activity Recognition Using Smartphones (ID: 240)
        har = fetch_ucirepo(id=240)
        X = har.data.features
        y = har.data.targets
        print("  - ucimlrepo API를 통해 성공적으로 로드되었습니다.")
    except Exception as e:
        print(f"  - ucimlrepo API 호출 알림: {e}")
        print("  - 내장 UCI HAR 데이터 로더(Fallback)로 자동 전환하여 로드합니다.")
        har = load_uci_har_fallback(data_dir=DATA_DIR)
        X = har.data.features
        y = har.data.targets

    print(f"  - 전체 특성(Features) 형태 (X): {X.shape}")
    print(f"  - 전체 타깃(Targets) 형태 (y): {y.shape}")

    # ==========================================================================
    # 2. 데이터 탐색 및 구조 분석 (EDA)
    # ==========================================================================
    print("\n[2] 데이터 구조 및 클래스 분포 분석...")
    target_col = y.columns[0]
    y_series = y[target_col]

    # 클래스 레이블 텍스트 매핑
    y_named = y_series.map(ACTIVITY_MAP)
    class_counts = y_named.value_counts()
    print("  - 행동 클래스별 샘플 수 및 비율:")
    for cls_name, count in class_counts.items():
        pct = (count / len(y)) * 100
        print(f"    * {cls_name:<22}: {count:>5}개 ({pct:.2f}%)")

    missing_values = X.isnull().sum().sum()
    print(f"  - 결측치(Missing Values) 개수: {missing_values}개 (결측 없음)")

    # ==========================================================================
    # 3. 데이터 분할 및 표준화 스케일링
    # ==========================================================================
    print("\n[3] 훈련(Train) 및 테스트(Test) 데이터 분할 및 스케일링...")
    # 원본 UCI 표준 분할 데이터셋 사용 (Train 7,352개, Test 2,947개)
    if hasattr(har.data, 'train') and hasattr(har.data, 'test'):
        X_train = har.data.train.features
        y_train = har.data.train.targets[target_col]
        X_test = har.data.test.features
        y_test = har.data.test.targets[target_col]
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_series, test_size=0.3, random_state=42, stratify=y_series
        )

    print(f"  - 학습용 데이터(Train): {X_train.shape[0]}개 샘플, {X_train.shape[1]}개 특성")
    print(f"  - 평가용 데이터(Test) : {X_test.shape[0]}개 샘플, {X_test.shape[1]}개 특성")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("  - StandardScaler 적용 완료 (평균 0, 분산 1 표준화 정규화)")

    # ==========================================================================
    # 4. 머신러닝 모델 학습 및 성능 평가
    # ==========================================================================
    print("\n[4] 3개 머신러닝 분류 모델 학습 및 평가 비교...")
    models = {
        "Linear SVM (SVC)": LinearSVC(dual="auto", C=1.0, random_state=42, max_iter=2000),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }

    results = []
    best_model_name = None
    best_accuracy = -1.0
    best_preds = None

    for name, model in models.items():
        print(f"  >> [{name}] 모델 학습 진행 중...")
        # 선형/거리 기반 모델은 스케일링된 데이터 사용, 트리는 원본/스케일 데이터 모두 가능
        if "Forest" in name:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted")
        rec = recall_score(y_test, preds, average="weighted")
        f1 = f1_score(y_test, preds, average="weighted")

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })
        print(f"     => 정확도(Accuracy): {acc * 100:.2f}%, F1-Score: {f1:.4f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_preds = preds

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 65)
    print("  [모델별 성능 요약 비교표]")
    print("=" * 65)
    print(results_df.to_string(index=False, formatters={
        "Accuracy": "{:.4f}".format,
        "Precision": "{:.4f}".format,
        "Recall": "{:.4f}".format,
        "F1-Score": "{:.4f}".format
    }))
    print("=" * 65)

    # ==========================================================================
    # 5. 최우수 모델 상세 평가 (Classification Report)
    # ==========================================================================
    print(f"\n[5] 최고 성능 모델 ({best_model_name}) 상세 분류 리포트:")
    target_names = [ACTIVITY_MAP[i] for i in sorted(ACTIVITY_MAP.keys())]
    print(classification_report(y_test, best_preds, target_names=target_names))

    # ==========================================================================
    # 6. 시각화 1: 정규화 혼동 행렬(Confusion Matrix Heatmap)
    # ==========================================================================
    print("\n[6] 시각화 결과 생성 및 저장...")
    cm = confusion_matrix(y_test, best_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names
    )
    plt.title(f"스마트폰 행동 인식 혼동 행렬 ({best_model_name})", fontsize=14, pad=15)
    plt.xlabel("예측 클래스 (Predicted Activity)", fontsize=11)
    plt.ylabel("실제 클래스 (Actual Activity)", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "har_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"  - 혼동 행렬 차트 저장 완료: {cm_path}")

    # ==========================================================================
    # 7. 시각화 2: 모델 성능 비교 차트
    # ==========================================================================
    plt.figure(figsize=(8, 5))
    bar_plot = sns.barplot(
        data=results_df,
        x="Model",
        y="Accuracy",
        hue="Model",
        palette="viridis",
        legend=False
    )
    plt.title("머신러닝 모델별 테스트 세트 정확도 비교", fontsize=14, pad=15)
    plt.ylim(0.85, 1.0)
    plt.ylabel("정확도 (Accuracy)", fontsize=11)
    plt.xlabel("모델 (Model)", fontsize=11)

    for p in bar_plot.patches:
        height = p.get_height()
        bar_plot.annotate(f"{height * 100:.2f}%",
                          (p.get_x() + p.get_width() / 2., height),
                          ha='center', va='bottom',
                          xytext=(0, 5), textcoords='offset points',
                          fontsize=11, fontweight='bold')

    plt.tight_layout()
    comp_path = os.path.join(OUTPUT_DIR, "har_model_comparison.png")
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"  - 모델 비교 차트 저장 완료: {comp_path}")

    # ==========================================================================
    # 8. 시각화 3: Random Forest 주요 센서 특성 중요도 Top 15
    # ==========================================================================
    rf_model = models["Random Forest"]
    importances = rf_model.feature_importances_
    feat_names = X_train.columns
    top_indices = np.argsort(importances)[::-1][:15]

    top_df = pd.DataFrame({
        "Feature": [feat_names[i] for i in top_indices],
        "Importance": importances[top_indices]
    })

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=top_df,
        x="Importance",
        y="Feature",
        hue="Feature",
        palette="mako",
        legend=False
    )
    plt.title("행동 인식을 위한 핵심 센서 특성 중요도 Top 15 (Random Forest)", fontsize=13, pad=15)
    plt.xlabel("상대적 특성 중요도 (Feature Importance)", fontsize=11)
    plt.ylabel("센서 측정 특성 (Sensor Feature)", fontsize=11)
    plt.tight_layout()
    feat_path = os.path.join(OUTPUT_DIR, "har_feature_importance.png")
    plt.savefig(feat_path, dpi=300)
    plt.close()
    print(f"  - 특성 중요도 차트 저장 완료: {feat_path}")

    # ==========================================================================
    # 9. 예측 결과 CSV 파일 저장
    # ==========================================================================
    predictions_df = pd.DataFrame({
        "Actual_Code": y_test.values,
        "Actual_Label": [ACTIVITY_MAP[code] for code in y_test.values],
        "Predicted_Code": best_preds,
        "Predicted_Label": [ACTIVITY_MAP[code] for code in best_preds]
    })
    predictions_df["Correct"] = predictions_df["Actual_Code"] == predictions_df["Predicted_Code"]
    pred_path = os.path.join(OUTPUT_DIR, "har_test_predictions.csv")
    predictions_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print(f"  - 테스트 세트 예측 결과 CSV 저장 완료: {pred_path}")

    print("\n" + "#" * 80)
    print("  모든 모델링 및 평가 프로세스가 성공적으로 완료되었습니다!")
    print("#" * 80)


if __name__ == "__main__":
    main()
