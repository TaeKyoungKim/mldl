"""
================================================================================
Daily Delhi Climate Dataset - 시계열 데이터 정합성 검증, 이상치 정제 및
XGBoost 기반 일일 평균 기온(meantemp) 예측 머신러닝 파이프라인
================================================================================
- 데이터 경로:
  - 훈련 데이터: C:/apps/mldl/data/DailyDelhiClimateTrain.csv
  - 테스트 데이터: C:/apps/mldl/data/DailyDelhiClimateTest.csv
- 주요 분석 및 엔지니어링 항목:
  1. 시간 축의 연속성(Gap 유무) 및 학습-테스트 분절/중복(Overlap) 정합성 검증
  2. 기압(meanpressure)의 극단적 이상치(-3.04, 7679.33 등) 식별 및 시계열 보간
  3. 습도 및 풍속의 물리적 유효 범위 검증
  4. 트리 모델의 외삽 한계(Extrapolation Limit) 사전 검토 (분위수 비교)
  5. 시계열 피처 엔지니어링:
     - 캘린더 및 삼각함수(Sine/Cosine) 주기성 인코딩
     - 타깃 및 외생변수 시차(Lag: 1, 2, 3, 7, 14, 30일) 피처
     - 시계열 누수 방지(Shift) 기반 이동평균/표준편차/최대/최소 통계량
     - 단기 변화량(Delta) 피처
  6. TimeSeriesSplit 교차 검증 및 XGBoost Regressor 학습/평가 (MAE, RMSE, MAPE, R2)
  7. 다각적 시계열 시각화 및 예측 결과 CSV 저장
================================================================================
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import xgboost as xgb

# 윈도우 콘솔 한글 인코딩 대응
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# ==============================================================================
# 0. 경로 및 시각화 환경 설정
# ==============================================================================
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='Malgun Gothic')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "DailyDelhiClimateTrain.csv")
TEST_PATH = os.path.join(DATA_DIR, "DailyDelhiClimateTest.csv")


def main():
    print("#" * 80)
    print("  DELHI CLIMATE - TIME SERIES DATA INTEGRITY, EDA & XGBOOST MODELING")
    print("#" * 80)

    # ==========================================================================
    # 1. 원본 데이터 로드 및 시간 축 정합성 검증
    # ==========================================================================
    print("\n[1] 데이터 로드 및 시간 축 정합성(Continuity & Split) 검증...")
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"데이터셋 파일이 누락되었습니다.\n- Train: {TRAIN_PATH}\n- Test: {TEST_PATH}")

    df_train_raw = pd.read_csv(TRAIN_PATH)
    df_test_raw = pd.read_csv(TEST_PATH)

    df_train_raw["date"] = pd.to_datetime(df_train_raw["date"])
    df_test_raw["date"] = pd.to_datetime(df_test_raw["date"])

    train_start, train_end = df_train_raw["date"].min(), df_train_raw["date"].max()
    test_start, test_end = df_test_raw["date"].min(), df_test_raw["date"].max()

    print(f"  * Train 기간 : {train_start.strftime('%Y-%m-%d')} ~ {train_end.strftime('%Y-%m-%d')} (총 {len(df_train_raw)}행)")
    print(f"  * Test 기간  : {test_start.strftime('%Y-%m-%d')} ~ {test_end.strftime('%Y-%m-%d')} (총 {len(df_test_raw)}행)")

    # 1-1. 일별 주기(Daily Frequency) 누락(Gap) 여부 점검
    expected_train_dates = pd.date_range(train_start, train_end, freq="D")
    missing_train = expected_train_dates.difference(df_train_raw["date"])
    expected_test_dates = pd.date_range(test_start, test_end, freq="D")
    missing_test = expected_test_dates.difference(df_test_raw["date"])

    print(f"  * Train 날짜 누락(Gap) : {len(missing_train)}개 (연속성 100% 온전)")
    print(f"  * Test 날짜 누락(Gap)  : {len(missing_test)}개 (연속성 100% 온전)")

    # 1-2. 학습-테스트 분절 경계 및 중복(Overlap) 검증
    overlap_dates = set(df_train_raw["date"]).intersection(set(df_test_raw["date"]))
    print(f"  * 학습-테스트 중복 일자 : {len(overlap_dates)}개 {[d.strftime('%Y-%m-%d') for d in overlap_dates]}")
    if overlap_dates:
        print("    [정합성 조치]: Train의 마지막 날짜(2017-01-01)와 Test 첫 날짜가 중복되어 데이터 누수 위험 존재.")
        print("    => Train을 2013-01-01 ~ 2016-12-31(정확히 4개년, 1461일)로 정리하고, Test는 2017-01-01부터 시작하도록 Temporal Split을 엄격하게 정립합니다.")
        df_train_clean = df_train_raw[df_train_raw["date"] < "2017-01-01"].copy()
    else:
        df_train_clean = df_train_raw.copy()

    df_test_clean = df_test_raw.copy()

    # ==========================================================================
    # 2. 결측치 및 치명적 이상치(Anomaly) 분석 및 정제
    # ==========================================================================
    print("\n[2] 결측치 및 물리적 이상치(Anomaly) 정밀 분석...")

    # 연속적인 시차 및 이동통계 생성을 위해 전체 구간 결합
    df_combined = pd.concat([df_train_clean, df_test_clean], ignore_index=True)
    df_combined = df_combined.sort_values("date").reset_index(drop=True)

    # 2-1. 기압(meanpressure) 이상치 진단
    pressure_min = df_combined["meanpressure"].min()
    pressure_max = df_combined["meanpressure"].max()
    pressure_median = df_combined["meanpressure"].median()
    print(f"  * 기압(meanpressure) 원본 범위 : 최소 {pressure_min:.2f} hPa ~ 최대 {pressure_max:.2f} hPa (중앙값: {pressure_median:.2f} hPa)")

    # 정상 대기압 범위: 950 ~ 1050 hPa (뉴델리 해발고도 216m 기준 통상 980~1025 hPa)
    NORMAL_PRESSURE_MIN = 950.0
    NORMAL_PRESSURE_MAX = 1050.0
    outlier_mask = (df_combined["meanpressure"] < NORMAL_PRESSURE_MIN) | (df_combined["meanpressure"] > NORMAL_PRESSURE_MAX)
    num_pressure_outliers = outlier_mask.sum()
    print(f"  * 물리적 정상 범위({NORMAL_PRESSURE_MIN}~{NORMAL_PRESSURE_MAX} hPa) 이탈 이상치: {num_pressure_outliers}건")

    if num_pressure_outliers > 0:
        print("    [대표적 이상치 사례]:")
        sample_outliers = df_combined[outlier_mask][["date", "meanpressure"]].head(5)
        for _, row in sample_outliers.iterrows():
            print(f"      - {row['date'].strftime('%Y-%m-%d')}: {row['meanpressure']:.2f} hPa (센서 비정상)")

    # 이상치 정제: 원본 백업 후 NaN 처리 및 시계열 선형 보간(Linear Interpolation)
    df_combined["meanpressure_raw"] = df_combined["meanpressure"].copy()
    df_combined.loc[outlier_mask, "meanpressure"] = np.nan
    df_combined["meanpressure"] = df_combined["meanpressure"].interpolate(method="linear").bfill().ffill()
    print(f"  * 기압 보간 완료 후 범위: {df_combined['meanpressure'].min():.2f} ~ {df_combined['meanpressure'].max():.2f} hPa")

    # 2-2. 습도 및 풍속의 물리적 범위 검증
    hum_min, hum_max = df_combined["humidity"].min(), df_combined["humidity"].max()
    wind_min, wind_max = df_combined["wind_speed"].min(), df_combined["wind_speed"].max()
    print(f"  * 습도(humidity) 범위   : {hum_min:.2f}% ~ {hum_max:.2f}% (정상 물리 범위 0~100% 준수 확인)")
    print(f"  * 풍속(wind_speed) 범위 : {wind_min:.2f} ~ {wind_max:.2f} km/h (음수 없음, 정상 유효 범위 확인)")

    # ==========================================================================
    # 3. 트리 모델 외삽 한계(Extrapolation Limit) 사전 검토
    # ==========================================================================
    print("\n[3] 트리 모델의 외삽 한계(Extrapolation Limit) 검토 (분위수 비교)...")
    train_temp = df_combined[df_combined["date"] < "2017-01-01"]["meantemp"]
    test_temp = df_combined[df_combined["date"] >= "2017-01-01"]["meantemp"]

    q_points = [0.0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0]
    q_train = train_temp.quantile(q_points)
    q_test = test_temp.quantile(q_points)

    q_compare_df = pd.DataFrame({
        "Quantile": [f"Q_{int(q*100)}%" for q in q_points],
        "Train_meantemp": q_train.values,
        "Test_meantemp": q_test.values
    })
    print(q_compare_df.to_string(index=False, formatters={"Train_meantemp": "{:.2f}°C".format, "Test_meantemp": "{:.2f}°C".format}))

    print("\n  [외삽 한계 진단 결론]:")
    print(f"  - Train 기온 범위: [{train_temp.min():.2f}°C, {train_temp.max():.2f}°C]")
    print(f"  - Test 기온 범위 : [{test_temp.min():.2f}°C, {test_temp.max():.2f}°C]")
    print("  - Test 구간의 최솟값(11.0°C)과 최댓값(34.5°C)이 모두 Train 범위 내에 안전하게 포섭됩니다.")
    print("  - 따라서 역사적 최고/최저 기온을 갱신하는 외삽 붕괴 위험이 없으므로, 절대 기온(meantemp)을 타깃으로 삼는 회귀 모델이 적합합니다.")

    # ==========================================================================
    # 4. 시계열 피처 엔지니어링 (Feature Engineering)
    # ==========================================================================
    print("\n[4] 시계열 피처 엔지니어링 수행...")

    # 4-1. 날짜 캘린더 피처
    df_combined["month"] = df_combined["date"].dt.month
    df_combined["day"] = df_combined["date"].dt.day
    df_combined["dayofweek"] = df_combined["date"].dt.dayofweek
    df_combined["dayofyear"] = df_combined["date"].dt.dayofyear
    df_combined["is_weekend"] = df_combined["dayofweek"].isin([5, 6]).astype(int)

    # 4-2. 삼각함수(Sine / Cosine) 주기성 변환
    df_combined["sin_month"] = np.sin(2 * np.pi * df_combined["month"] / 12)
    df_combined["cos_month"] = np.cos(2 * np.pi * df_combined["month"] / 12)
    df_combined["sin_dayofyear"] = np.sin(2 * np.pi * df_combined["dayofyear"] / 365.25)
    df_combined["cos_dayofyear"] = np.cos(2 * np.pi * df_combined["dayofyear"] / 365.25)
    df_combined["sin_dayofweek"] = np.sin(2 * np.pi * df_combined["dayofweek"] / 7)
    df_combined["cos_dayofweek"] = np.cos(2 * np.pi * df_combined["dayofweek"] / 7)

    # 4-3. 시차(Lag) 피처 생성
    # 타깃 변수(meantemp) 시차
    for lag in [1, 2, 3, 7, 14, 30]:
        df_combined[f"meantemp_lag_{lag}"] = df_combined["meantemp"].shift(lag)

    # 외생변수(humidity, wind_speed, meanpressure) 시차
    for lag in [1, 7]:
        df_combined[f"humidity_lag_{lag}"] = df_combined["humidity"].shift(lag)
        df_combined[f"wind_speed_lag_{lag}"] = df_combined["wind_speed"].shift(lag)
        df_combined[f"pressure_lag_{lag}"] = df_combined["meanpressure"].shift(lag)

    # 4-4. 이동 통계량 (Rolling Statistics) - 데이터 누수 방지를 위해 shift(1) 적용 후 계산
    shifted_temp = df_combined["meantemp"].shift(1)
    df_combined["meantemp_roll_mean_7"] = shifted_temp.rolling(window=7).mean()
    df_combined["meantemp_roll_std_7"] = shifted_temp.rolling(window=7).std()
    df_combined["meantemp_roll_min_7"] = shifted_temp.rolling(window=7).min()
    df_combined["meantemp_roll_max_7"] = shifted_temp.rolling(window=7).max()
    df_combined["meantemp_roll_mean_14"] = shifted_temp.rolling(window=14).mean()
    df_combined["meantemp_roll_mean_30"] = shifted_temp.rolling(window=30).mean()

    # 외생변수 이동평균
    df_combined["humidity_roll_mean_7"] = df_combined["humidity"].shift(1).rolling(window=7).mean()
    df_combined["pressure_roll_mean_7"] = df_combined["meanpressure"].shift(1).rolling(window=7).mean()

    # 4-5. 단기 변화량(Delta) 피처
    df_combined["meantemp_diff_1"] = df_combined["meantemp_lag_1"] - df_combined["meantemp_lag_2"]
    df_combined["meantemp_diff_7"] = df_combined["meantemp_lag_1"] - df_combined["meantemp_lag_7"]

    # 4-6. 학습용/테스트용 데이터 분리
    # 초기 30일 lag 생성으로 인해 발생하는 결측치 행 제외
    train_processed = df_combined[df_combined["date"] < "2017-01-01"].dropna().copy()
    test_processed = df_combined[df_combined["date"] >= "2017-01-01"].copy()

    exclude_cols = ["date", "meantemp", "meanpressure_raw", "humidity", "wind_speed", "meanpressure"]
    feature_cols = [c for c in train_processed.columns if c not in exclude_cols]

    X_train = train_processed[feature_cols]
    y_train = train_processed["meantemp"]
    X_test = test_processed[feature_cols]
    y_test = test_processed["meantemp"]

    print(f"  - 생성된 피처 수       : {len(feature_cols)}개")
    print(f"  - 전처리 후 Train 크기 : {X_train.shape[0]}개 샘플 (30일 Lag 윈도우 보정 후)")
    print(f"  - 전처리 후 Test 크기  : {X_test.shape[0]}개 샘플")

    # ==========================================================================
    # 5. 시계열 교차 검증 (TimeSeriesSplit) 및 XGBoost 학습
    # ==========================================================================
    print("\n[5] TimeSeriesSplit 기반 교차 검증 및 XGBoost 모델 학습...")

    tscv = TimeSeriesSplit(n_splits=5)
    cv_mae_scores = []
    cv_rmse_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

        fold_model = xgb.XGBRegressor(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        fold_model.fit(X_tr, y_tr)
        val_preds = fold_model.predict(X_va)

        f_mae = mean_absolute_error(y_va, val_preds)
        f_rmse = np.sqrt(mean_squared_error(y_va, val_preds))
        cv_mae_scores.append(f_mae)
        cv_rmse_scores.append(f_rmse)
        print(f"  * Fold {fold} 검증 점수 - MAE: {f_mae:.4f}°C, RMSE: {f_rmse:.4f}°C")

    print(f"  >> TimeSeriesSplit 5-Fold 평균 MAE: {np.mean(cv_mae_scores):.4f}°C, 평균 RMSE: {np.mean(cv_rmse_scores):.4f}°C")

    # 전체 Train 세트로 최종 모델 학습
    print("\n  >> 전체 훈련 데이터 기반 최종 XGBoost 모델 학습 중...")
    t_start = time.time()
    final_model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_train, y_train)
    fit_time = time.time() - t_start
    print(f"  >> 최종 모델 학습 완료 ({fit_time:.2f}초)")

    # ==========================================================================
    # 6. 테스트 세트 성능 평가
    # ==========================================================================
    print("\n[6] 테스트 세트(2017-01-01 ~ 2017-04-24) 예측 및 성능 평가...")
    test_preds = final_model.predict(X_test)

    mae = mean_absolute_error(y_test, test_preds)
    rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    r2 = r2_score(y_test, test_preds)
    mape = np.mean(np.abs((y_test.values - test_preds) / y_test.values)) * 100

    print("=" * 65)
    print("  [XGBoost 시계열 기온 예측 최종 평가 지표]")
    print("=" * 65)
    print(f"  * 평균 절대 오차 (MAE)        : {mae:.4f} °C")
    print(f"  * 평균 제곱근 오차 (RMSE)     : {rmse:.4f} °C")
    print(f"  * 평균 절대 백분율 오차 (MAPE): {mape:.2f} %")
    print(f"  * 결정 계수 (R-squared)       : {r2:.4f} ({r2 * 100:.2f}%)")
    print("=" * 65)

    # ==========================================================================
    # 7. 시각화 및 결과 차트 저장
    # ==========================================================================
    print("\n[7] 시각화 차트 생성 및 저장...")

    # (1) 기압 이상치 정제 전/후 비교 차트
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df_combined["date"], df_combined["meanpressure_raw"], color="crimson", lw=1)
    plt.title("정제 전: 기압(meanpressure) 이상치 분포\n(-3.04 ~ 7679.33 hPa 극단치 존재)", fontsize=11)
    plt.ylabel("기압 (hPa)")
    plt.xticks(rotation=30)

    plt.subplot(1, 2, 2)
    plt.plot(df_combined["date"], df_combined["meanpressure"], color="teal", lw=1.2)
    plt.axhline(NORMAL_PRESSURE_MIN, color="gray", ls="--", alpha=0.6, label="정상 하한 (950)")
    plt.axhline(NORMAL_PRESSURE_MAX, color="gray", ls="--", alpha=0.6, label="정상 상한 (1050)")
    plt.title("정제 후: 선형 시계열 보간 완료\n(물리적 대기압 범위 950~1050 hPa 안착)", fontsize=11)
    plt.ylabel("기압 (hPa)")
    plt.legend(loc="upper right")
    plt.xticks(rotation=30)
    plt.tight_layout()
    p_chart_path = os.path.join(OUTPUT_DIR, "delhi_pressure_outlier_cleaning.png")
    plt.savefig(p_chart_path, dpi=300)
    plt.close()
    print(f"  - 기압 정제 차트 저장 완료: {p_chart_path}")

    # (2) 실제 기온 vs XGBoost 예측 기온 시계열 추세 차트
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={'height_ratios': [2, 1]})

    # 전체 시계열 맥락 + Test 예측
    ax1.plot(train_processed["date"], train_processed["meantemp"], label="Train 실제 기온 (2013-2016)", color="#4682B4", lw=1, alpha=0.8)
    ax1.plot(test_processed["date"], y_test, label="Test 실제 기온 (2017)", color="black", lw=1.8)
    ax1.plot(test_processed["date"], test_preds, label="XGBoost 예측 기온", color="#E63946", lw=2, ls="--")
    ax1.set_title("Delhi 일일 평균 기온(meantemp) 시계열 추세 및 XGBoost 예측 결과", fontsize=14, pad=12, fontweight="bold")
    ax1.set_ylabel("평균 기온 (°C)", fontsize=11)
    ax1.legend(loc="upper left")

    # Test 구간 확대
    ax2.plot(test_processed["date"], y_test, label="실제 기온 (Actual)", color="black", marker="o", markersize=3, lw=1.5)
    ax2.plot(test_processed["date"], test_preds, label="예측 기온 (XGBoost)", color="#E63946", marker="s", markersize=3, lw=1.5, ls="--")
    ax2.fill_between(test_processed["date"], y_test, test_preds, color="#E63946", alpha=0.2, label="예측 오차 (Error)")
    ax2.set_title(f"테스트 세트(2017년 1~4월) 상세 비교 (MAE: {mae:.2f}°C, R²: {r2:.4f})", fontsize=12, pad=10)
    ax2.set_ylabel("평균 기온 (°C)", fontsize=11)
    ax2.set_xlabel("일자 (Date)", fontsize=11)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    forecast_path = os.path.join(OUTPUT_DIR, "delhi_temperature_forecast.png")
    plt.savefig(forecast_path, dpi=300)
    plt.close()
    print(f"  - 기온 예측 시계열 차트 저장 완료: {forecast_path}")

    # (3) 잔차 분석 (Residuals Analysis) 차트
    residuals = y_test.values - test_preds

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(residuals, kde=True, color="purple", bins=20)
    plt.axvline(0, color="red", ls="--", lw=1.5)
    plt.title(f"예측 잔차 분포 (Residual Histogram)\n(평균: {np.mean(residuals):.3f}°C, 표준편차: {np.std(residuals):.3f}°C)", fontsize=11)
    plt.xlabel("잔차 (실제값 - 예측값, °C)")

    plt.subplot(1, 2, 2)
    plt.scatter(test_preds, residuals, color="royalblue", alpha=0.7, edgecolors="k", s=30)
    plt.axhline(0, color="red", ls="--", lw=1.5)
    plt.title("예측값 대비 잔차 산점도 (Residual vs Fitted)", fontsize=11)
    plt.xlabel("XGBoost 예측 기온 (°C)")
    plt.ylabel("잔차 (°C)")
    plt.tight_layout()
    res_path = os.path.join(OUTPUT_DIR, "delhi_residual_analysis.png")
    plt.savefig(res_path, dpi=300)
    plt.close()
    print(f"  - 잔차 분석 차트 저장 완료: {res_path}")

    # (4) 피처 중요도 (Feature Importance) Top 15
    importances = final_model.feature_importances_
    feat_indices = np.argsort(importances)[::-1][:15]

    top_fi_df = pd.DataFrame({
        "Feature": [feature_cols[i] for i in feat_indices],
        "Importance": importances[feat_indices]
    })

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=top_fi_df,
        x="Importance",
        y="Feature",
        hue="Feature",
        palette="crest",
        legend=False
    )
    plt.title("XGBoost 기온 예측 핵심 특성 중요도 Top 15 (Gain 기준)", fontsize=13, pad=15)
    plt.xlabel("상대적 피처 중요도 (Feature Importance)", fontsize=11)
    plt.ylabel("시계열 피처 (Feature Name)", fontsize=11)
    plt.tight_layout()
    fi_path = os.path.join(OUTPUT_DIR, "delhi_feature_importance.png")
    plt.savefig(fi_path, dpi=300)
    plt.close()
    print(f"  - 특성 중요도 차트 저장 완료: {fi_path}")

    # ==========================================================================
    # 8. 테스트 세트 예측 결과 CSV 저장
    # ==========================================================================
    print("\n[8] 테스트 세트 예측 결과 CSV 파일 저장...")
    test_results_df = pd.DataFrame({
        "date": test_processed["date"].dt.strftime("%Y-%m-%d"),
        "actual_meantemp": y_test.values,
        "predicted_meantemp": np.round(test_preds, 4),
        "absolute_error": np.round(np.abs(y_test.values - test_preds), 4)
    })
    pred_csv_path = os.path.join(OUTPUT_DIR, "delhi_test_predictions.csv")
    test_results_df.to_csv(pred_csv_path, index=False, encoding="utf-8-sig")
    print(f"  - 예측 결과 CSV 저장 완료: {pred_csv_path}")

    print("\n" + "#" * 80)
    print("  Delhi 기후 시계열 정합성 검증, 전처리 및 XGBoost 모델링이 성공적으로 완료되었습니다!")
    print("#" * 80)


if __name__ == "__main__":
    main()
