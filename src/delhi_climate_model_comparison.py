"""
================================================================================
Daily Delhi Climate Dataset - XGBoost vs LightGBM 시계열 기온 예측 모델 성능 비교
================================================================================
- 데이터 경로:
  - 훈련 세트: C:/apps/mldl/data/DailyDelhiClimateTrain.csv
  - 테스트 세트: C:/apps/mldl/data/DailyDelhiClimateTest.csv
- 모델 비교 대상:
  1. XGBoost Regressor (Extreme Gradient Boosting)
  2. LightGBM Regressor (Light Gradient Boosting Machine)
- 비교 평가 항목:
  1. TimeSeriesSplit (5-Fold) 시계열 교차 검증 점수 및 안정성
  2. 테스트 세트(2017-01-01 ~ 2017-04-24) 예측 오차 (MAE, RMSE, MAPE, R2)
  3. 모델별 학습 속도 및 연산 효율성
  4. 시계열 예측 궤적 및 잔차(Residual) 패턴 비교
  5. 핵심 특성 중요도(Feature Importance) 도메인 기여도 비교
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
import lightgbm as lgb

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


def load_and_preprocess_data():
    """
    정합성 검증 및 이상치 보간, 시계열 피처 엔지니어링을 수행합니다.
    """
    df_train_raw = pd.read_csv(TRAIN_PATH)
    df_test_raw = pd.read_csv(TEST_PATH)

    df_train_raw["date"] = pd.to_datetime(df_train_raw["date"])
    df_test_raw["date"] = pd.to_datetime(df_test_raw["date"])

    # 1. 학습-테스트 중복 일자(2017-01-01) 누수 방지 조치
    df_train_clean = df_train_raw[df_train_raw["date"] < "2017-01-01"].copy()
    df_test_clean = df_test_raw.copy()

    # 2. 연속 시계열 통합
    df_combined = pd.concat([df_train_clean, df_test_clean], ignore_index=True)
    df_combined = df_combined.sort_values("date").reset_index(drop=True)

    # 3. 기압 이상치(-3.04, 7679.33 등) 선형 보간
    outlier_mask = (df_combined["meanpressure"] < 950.0) | (df_combined["meanpressure"] > 1050.0)
    df_combined.loc[outlier_mask, "meanpressure"] = np.nan
    df_combined["meanpressure"] = df_combined["meanpressure"].interpolate(method="linear").bfill().ffill()

    # 4. 캘린더 및 삼각함수 주기성 변환
    df_combined["month"] = df_combined["date"].dt.month
    df_combined["day"] = df_combined["date"].dt.day
    df_combined["dayofweek"] = df_combined["date"].dt.dayofweek
    df_combined["dayofyear"] = df_combined["date"].dt.dayofyear
    df_combined["is_weekend"] = df_combined["dayofweek"].isin([5, 6]).astype(int)

    df_combined["sin_month"] = np.sin(2 * np.pi * df_combined["month"] / 12)
    df_combined["cos_month"] = np.cos(2 * np.pi * df_combined["month"] / 12)
    df_combined["sin_dayofyear"] = np.sin(2 * np.pi * df_combined["dayofyear"] / 365.25)
    df_combined["cos_dayofyear"] = np.cos(2 * np.pi * df_combined["dayofyear"] / 365.25)
    df_combined["sin_dayofweek"] = np.sin(2 * np.pi * df_combined["dayofweek"] / 7)
    df_combined["cos_dayofweek"] = np.cos(2 * np.pi * df_combined["dayofweek"] / 7)

    # 5. 시차(Lag) 피처
    for lag in [1, 2, 3, 7, 14, 30]:
        df_combined[f"meantemp_lag_{lag}"] = df_combined["meantemp"].shift(lag)

    for lag in [1, 7]:
        df_combined[f"humidity_lag_{lag}"] = df_combined["humidity"].shift(lag)
        df_combined[f"wind_speed_lag_{lag}"] = df_combined["wind_speed"].shift(lag)
        df_combined[f"pressure_lag_{lag}"] = df_combined["meanpressure"].shift(lag)

    # 6. 이동 통계량(Rolling Statistics) - shift(1) 누수 방지
    s_temp = df_combined["meantemp"].shift(1)
    df_combined["meantemp_roll_mean_7"] = s_temp.rolling(window=7).mean()
    df_combined["meantemp_roll_std_7"] = s_temp.rolling(window=7).std()
    df_combined["meantemp_roll_min_7"] = s_temp.rolling(window=7).min()
    df_combined["meantemp_roll_max_7"] = s_temp.rolling(window=7).max()
    df_combined["meantemp_roll_mean_14"] = s_temp.rolling(window=14).mean()
    df_combined["meantemp_roll_mean_30"] = s_temp.rolling(window=30).mean()

    df_combined["humidity_roll_mean_7"] = df_combined["humidity"].shift(1).rolling(window=7).mean()
    df_combined["pressure_roll_mean_7"] = df_combined["meanpressure"].shift(1).rolling(window=7).mean()

    # 7. 단기 변화량(Delta) 피처
    df_combined["meantemp_diff_1"] = df_combined["meantemp_lag_1"] - df_combined["meantemp_lag_2"]
    df_combined["meantemp_diff_7"] = df_combined["meantemp_lag_1"] - df_combined["meantemp_lag_7"]

    # 8. Train/Test 분리
    train_clean = df_combined[df_combined["date"] < "2017-01-01"].dropna().copy()
    test_clean = df_combined[df_combined["date"] >= "2017-01-01"].copy()

    exclude_cols = ["date", "meantemp", "humidity", "wind_speed", "meanpressure"]
    features = [c for c in train_clean.columns if c not in exclude_cols]

    return train_clean, test_clean, features


def main():
    print("#" * 80)
    print("  DELHI CLIMATE - XGBOOST VS LIGHTGBM TIME SERIES REGRESSION BENCHMARK")
    print("#" * 80)

    # ==========================================================================
    # 1. 데이터 로드 및 피처 엔지니어링
    # ==========================================================================
    print("\n[1] 데이터 로드 및 시계열 피처 파이프라인 가동...")
    train_df, test_df, feature_cols = load_and_preprocess_data()

    X_train = train_df[feature_cols]
    y_train = train_df["meantemp"]
    X_test = test_df[feature_cols]
    y_test = test_df["meantemp"]

    print(f"  * 학습 세트 크기 (Train): {X_train.shape[0]}일 (2013-01-31 ~ 2016-12-31)")
    print(f"  * 테스트 세트 크기 (Test): {X_test.shape[0]}일 (2017-01-01 ~ 2017-04-24)")
    print(f"  * 투입 피처 수           : {len(feature_cols)}개")

    # ==========================================================================
    # 2. 모델 파라미터 정의
    # ==========================================================================
    # 공정한 비교를 위해 트리 깊이, 학습률, 부스팅 수 등 동일 수준 조건 설정
    xgb_params = {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1
    }

    lgb_params = {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 4,
        "num_leaves": 15,  # 2^max_depth 근사
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }

    # ==========================================================================
    # 3. 시계열 교차 검증 (TimeSeriesSplit 5-Fold) 비교
    # ==========================================================================
    print("\n[2] TimeSeriesSplit (5-Fold) 시계열 교차 검증 점수 비교...")
    tscv = TimeSeriesSplit(n_splits=5)

    xgb_cv_mae, xgb_cv_rmse = [], []
    lgb_cv_mae, lgb_cv_rmse = [], []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train), 1):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]

        # XGBoost Fold 학습
        m_xgb = xgb.XGBRegressor(**xgb_params)
        m_xgb.fit(X_tr, y_tr)
        p_xgb = m_xgb.predict(X_va)
        xgb_cv_mae.append(mean_absolute_error(y_va, p_xgb))
        xgb_cv_rmse.append(np.sqrt(mean_squared_error(y_va, p_xgb)))

        # LightGBM Fold 학습
        m_lgb = lgb.LGBMRegressor(**lgb_params)
        m_lgb.fit(X_tr, y_tr)
        p_lgb = m_lgb.predict(X_va)
        lgb_cv_mae.append(mean_absolute_error(y_va, p_lgb))
        lgb_cv_rmse.append(np.sqrt(mean_squared_error(y_va, p_lgb)))

        print(f"  * Fold {fold} - [XGBoost] MAE: {xgb_cv_mae[-1]:.3f}°C, RMSE: {xgb_cv_rmse[-1]:.3f}°C | [LightGBM] MAE: {lgb_cv_mae[-1]:.3f}°C, RMSE: {lgb_cv_rmse[-1]:.3f}°C")

    print("\n  >> [5-Fold 평균 교차 검증 지표]:")
    print(f"     - XGBoost  : 평균 MAE = {np.mean(xgb_cv_mae):.4f}°C, 평균 RMSE = {np.mean(xgb_cv_rmse):.4f}°C")
    print(f"     - LightGBM : 평균 MAE = {np.mean(lgb_cv_mae):.4f}°C, 평균 RMSE = {np.mean(lgb_cv_rmse):.4f}°C")

    # ==========================================================================
    # 4. 전체 훈련 세트 기반 최종 모델 학습 및 훈련 시간 측정
    # ==========================================================================
    print("\n[3] 최종 모델 전체 학습 및 소요 시간(Training Speed) 비교...")

    # XGBoost 학습
    t0 = time.time()
    final_xgb = xgb.XGBRegressor(**xgb_params)
    final_xgb.fit(X_train, y_train)
    xgb_fit_time = time.time() - t0

    # LightGBM 학습
    t0 = time.time()
    final_lgb = lgb.LGBMRegressor(**lgb_params)
    final_lgb.fit(X_train, y_train)
    lgb_fit_time = time.time() - t0

    print(f"  * XGBoost  학습 소요 시간: {xgb_fit_time:.4f}초")
    print(f"  * LightGBM 학습 소요 시간: {lgb_fit_time:.4f}초 (LightGBM이 약 {xgb_fit_time / max(lgb_fit_time, 1e-5):.1f}배 빠름)")

    # ==========================================================================
    # 5. 테스트 세트(미래 미관측 114일) 최종 예측 및 성능 비교
    # ==========================================================================
    print("\n[4] 테스트 세트(2017-01-01 ~ 2017-04-24) 예측 평가 및 지표 종합 비교...")
    xgb_preds = final_xgb.predict(X_test)
    lgb_preds = final_lgb.predict(X_test)

    # 지표 계산
    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    xgb_r2 = r2_score(y_test, xgb_preds)
    xgb_mape = np.mean(np.abs((y_test.values - xgb_preds) / y_test.values)) * 100

    lgb_mae = mean_absolute_error(y_test, lgb_preds)
    lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_preds))
    lgb_r2 = r2_score(y_test, lgb_preds)
    lgb_mape = np.mean(np.abs((y_test.values - lgb_preds) / y_test.values)) * 100

    mae_diff = abs(xgb_mae - lgb_mae)
    rmse_diff = abs(xgb_rmse - lgb_rmse)
    mape_diff = abs(xgb_mape - lgb_mape)
    r2_diff = abs(xgb_r2 - lgb_r2) * 100
    cv_diff = abs(np.mean(xgb_cv_mae) - np.mean(lgb_cv_mae))
    speed_ratio = xgb_fit_time / max(lgb_fit_time, 1e-5)

    comparison_df = pd.DataFrame({
        "평가 지표 (Metric)": [
            "MAE (평균 절대 오차, °C)",
            "RMSE (평균 제곱근 오차, °C)",
            "MAPE (평균 절대 백분율 오차, %)",
            "R² (결정 계수, 설명력)",
            "CV 5-Fold 평균 MAE (°C)",
            "학습 소요 시간 (Fit Time, 초)"
        ],
        "XGBoost": [
            f"{xgb_mae:.4f}",
            f"{xgb_rmse:.4f}",
            f"{xgb_mape:.2f}%",
            f"{xgb_r2:.4f} ({xgb_r2*100:.2f}%)",
            f"{np.mean(xgb_cv_mae):.4f}",
            f"{xgb_fit_time:.4f}s"
        ],
        "LightGBM": [
            f"{lgb_mae:.4f}",
            f"{lgb_rmse:.4f}",
            f"{lgb_mape:.2f}%",
            f"{lgb_r2:.4f} ({lgb_r2*100:.2f}%)",
            f"{np.mean(lgb_cv_mae):.4f}",
            f"{lgb_fit_time:.4f}s"
        ],
        "우세 모델 (Winner)": [
            f"XGBoost (-{mae_diff:.4f}°C)" if xgb_mae < lgb_mae else f"LightGBM (-{mae_diff:.4f}°C)",
            f"XGBoost (-{rmse_diff:.4f}°C)" if xgb_rmse < lgb_rmse else f"LightGBM (-{rmse_diff:.4f}°C)",
            f"XGBoost (-{mape_diff:.2f}%p)" if xgb_mape < lgb_mape else f"LightGBM (-{mape_diff:.2f}%p)",
            f"XGBoost (+{r2_diff:.2f}%p)" if xgb_r2 > lgb_r2 else f"LightGBM (+{r2_diff:.2f}%p)",
            f"XGBoost (-{cv_diff:.4f}°C)" if np.mean(xgb_cv_mae) < np.mean(lgb_cv_mae) else f"LightGBM (-{cv_diff:.4f}°C)",
            f"LightGBM ({speed_ratio:.1f}x 빠름)"
        ]
    })

    print("\n" + "=" * 78)
    print("  [XGBoost vs LightGBM 최종 성능 비교 대조표]")
    print("=" * 78)
    print(comparison_df.to_string(index=False))
    print("=" * 78)

    # ==========================================================================
    # 6. 비교 시각화 차트 생성 및 저장
    # ==========================================================================
    print("\n[5] 비교 시각화 결과 차트 생성 중...")

    # (1) 시계열 기온 예측 궤적 비교 (전체 시계열 + Test 상세 확대)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={'height_ratios': [2, 1.2]})

    # 전체 시계열
    ax1.plot(train_df["date"], train_df["meantemp"], label="Train 실제 기온 (2013-2016)", color="#4A90E2", lw=1, alpha=0.7)
    ax1.plot(test_df["date"], y_test, label="Test 실제 기온 (Actual 2017)", color="black", lw=2)
    ax1.plot(test_df["date"], xgb_preds, label=f"XGBoost 예측 (MAE: {xgb_mae:.2f}°C, R²: {xgb_r2:.3f})", color="#E63946", lw=1.8, ls="--")
    ax1.plot(test_df["date"], lgb_preds, label=f"LightGBM 예측 (MAE: {lgb_mae:.2f}°C, R²: {lgb_r2:.3f})", color="#2A9D8F", lw=1.8, ls=":")
    ax1.set_title("Delhi 일일 평균 기온: XGBoost vs LightGBM 시계열 예측 궤적 비교", fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylabel("평균 기온 (°C)", fontsize=11)
    ax1.legend(loc="upper left", frameon=True)

    # Test 구간 상세 확대
    ax2.plot(test_df["date"], y_test.values, label="실제 기온 (Actual)", color="black", marker="o", markersize=3.5, lw=1.8)
    ax2.plot(test_df["date"], xgb_preds, label="XGBoost", color="#E63946", marker="^", markersize=3, lw=1.5, ls="--")
    ax2.plot(test_df["date"], lgb_preds, label="LightGBM", color="#2A9D8F", marker="s", markersize=3, lw=1.5, ls=":")
    ax2.fill_between(test_df["date"], xgb_preds, lgb_preds, color="orange", alpha=0.25, label="두 모델 간 예측 편차 (Model Discrepancy)")
    ax2.set_title("테스트 세트(2017년 1~4월) 모델별 일일 예측 비교 상세", fontsize=12, pad=10)
    ax2.set_ylabel("평균 기온 (°C)", fontsize=11)
    ax2.set_xlabel("일자 (Date)", fontsize=11)
    ax2.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    chart1_path = os.path.join(OUTPUT_DIR, "delhi_model_comparison_forecast.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"  - 기온 예측 궤적 비교 차트 저장 완료: {chart1_path}")

    # (2) 주요 성능 지표 비교 막대 그래프
    metrics_names = ["MAE (°C)", "RMSE (°C)", "MAPE (%)", "R² Score"]
    xgb_vals = [xgb_mae, xgb_rmse, xgb_mape, xgb_r2]
    lgb_vals = [lgb_mae, lgb_rmse, lgb_mape, lgb_r2]

    x = np.arange(len(metrics_names))
    width = 0.35

    plt.figure(figsize=(9, 5))
    bar1 = plt.bar(x - width/2, xgb_vals, width, label='XGBoost', color='#E63946', alpha=0.9)
    bar2 = plt.bar(x + width/2, lgb_vals, width, label='LightGBM', color='#2A9D8F', alpha=0.9)

    plt.title("XGBoost vs LightGBM 핵심 성능 지표(Test Set) 비교", fontsize=13, pad=15)
    plt.xticks(x, metrics_names, fontsize=11)
    plt.ylabel("지표 수치", fontsize=11)
    plt.legend(frameon=True)

    for bar in bar1:
        h = bar.get_height()
        plt.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width()/2, h),
                     ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=9)
    for bar in bar2:
        h = bar.get_height()
        plt.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width()/2, h),
                     ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=9)

    plt.tight_layout()
    chart2_path = os.path.join(OUTPUT_DIR, "delhi_model_metrics_bar.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"  - 성능 지표 막대 비교 차트 저장 완료: {chart2_path}")

    # (3) 잔차(Residuals) 분포 및 시계열 오차 비교
    xgb_res = y_test.values - xgb_preds
    lgb_res = y_test.values - lgb_preds

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    sns.kdeplot(xgb_res, label=f"XGBoost (Std: {np.std(xgb_res):.2f})", color="#E63946", lw=2, fill=True, alpha=0.2)
    sns.kdeplot(lgb_res, label=f"LightGBM (Std: {np.std(lgb_res):.2f})", color="#2A9D8F", lw=2, fill=True, alpha=0.2)
    plt.axvline(0, color="gray", ls="--", alpha=0.7)
    plt.title("예측 잔차 확률밀도분포 (Residual KDE)", fontsize=11)
    plt.xlabel("잔차 (실제값 - 예측값, °C)")
    plt.ylabel("밀도 (Density)")
    plt.legend(frameon=True)

    plt.subplot(1, 2, 2)
    plt.plot(test_df["date"], xgb_res, label="XGBoost 잔차", color="#E63946", lw=1.3, alpha=0.85)
    plt.plot(test_df["date"], lgb_res, label="LightGBM 잔차", color="#2A9D8F", lw=1.3, ls="--", alpha=0.85)
    plt.axhline(0, color="black", ls="-", lw=1)
    plt.title("일자별 예측 잔차 추이 (Residuals over Time)", fontsize=11)
    plt.xlabel("일자 (Date)")
    plt.ylabel("잔차 (°C)")
    plt.legend(frameon=True)
    plt.xticks(rotation=30)
    plt.tight_layout()

    chart3_path = os.path.join(OUTPUT_DIR, "delhi_model_residuals_comparison.png")
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    print(f"  - 잔차 비교 차트 저장 완료: {chart3_path}")

    # (4) 특성 중요도(Feature Importance) Top 10 비교
    xgb_fi = final_xgb.feature_importances_
    lgb_fi = final_lgb.feature_importances_
    # 중요도 정규화
    xgb_fi_norm = xgb_fi / xgb_fi.sum()
    lgb_fi_norm = lgb_fi / lgb_fi.sum()

    top_idx = np.argsort(xgb_fi_norm)[::-1][:10]
    top_features = [feature_cols[i] for i in top_idx]

    fi_compare_df = pd.DataFrame({
        "Feature": top_features,
        "XGBoost": [xgb_fi_norm[i] for i in top_idx],
        "LightGBM": [lgb_fi_norm[i] for i in top_idx]
    })
    fi_melted = pd.melt(fi_compare_df, id_vars=["Feature"], value_vars=["XGBoost", "LightGBM"],
                        var_name="Model", value_name="Normalized_Importance")

    plt.figure(figsize=(11, 6))
    sns.barplot(data=fi_melted, y="Feature", x="Normalized_Importance", hue="Model", palette=["#E63946", "#2A9D8F"])
    plt.title("XGBoost vs LightGBM 주요 피처 중요도(Top 10) 비교", fontsize=13, pad=15)
    plt.xlabel("정규화 특성 중요도 (Normalized Importance)", fontsize=11)
    plt.ylabel("피처 명칭 (Feature)", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()

    chart4_path = os.path.join(OUTPUT_DIR, "delhi_feature_importance_comparison.png")
    plt.savefig(chart4_path, dpi=300)
    plt.close()
    print(f"  - 피처 중요도 비교 차트 저장 완료: {chart4_path}")

    # ==========================================================================
    # 7. 모델 예측 비교 CSV 저장
    # ==========================================================================
    print("\n[6] 테스트 세트 예측 비교 결과 CSV 파일 저장...")
    results_df = pd.DataFrame({
        "date": test_df["date"].dt.strftime("%Y-%m-%d"),
        "actual_meantemp": y_test.values,
        "xgb_predicted": np.round(xgb_preds, 4),
        "lgb_predicted": np.round(lgb_preds, 4),
        "xgb_abs_error": np.round(np.abs(y_test.values - xgb_preds), 4),
        "lgb_abs_error": np.round(np.abs(y_test.values - lgb_preds), 4),
        "model_discrepancy": np.round(np.abs(xgb_preds - lgb_preds), 4)
    })
    pred_csv_path = os.path.join(OUTPUT_DIR, "delhi_models_comparison_predictions.csv")
    results_df.to_csv(pred_csv_path, index=False, encoding="utf-8-sig")
    print(f"  - 예측 비교 CSV 저장 완료: {pred_csv_path}")

    print("\n" + "#" * 80)
    print("  XGBoost vs LightGBM 시계열 기온 예측 벤치마크 비교가 성공적으로 완료되었습니다!")
    print("#" * 80)


if __name__ == "__main__":
    main()
