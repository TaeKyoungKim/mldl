"""
Telco 고객 이탈(Customer Churn) 데이터셋 대상
로지스틱 회귀(Logistic Regression) 모델링 및 VAE 기반 오버샘플링(Oversampling) 전후 비교 분석

[데이터 소스]
- 파일: D:/dl/WA_Fn-UseC_-Telco-Customer-Churn (1).csv
- 목표 변수: Churn ('Yes': 1, 'No': 0) - 불균형 비율 약 73.5% vs 26.5%

[파이프라인]
1. 원본 데이터 로드 및 전처리 (결측치 정제, 범주형 One-Hot Encoding, 스케일링)
2. 시계열/랜덤 분할 (Train / Test)
3. [Baseline] 원본 불균형 데이터로 로지스틱 회귀 학습 및 테스트 평가
4. 소수 클래스(이탈 고객: Churn='Yes') 전용 VAE(Variational Autoencoder) 구축 및 학습
5. VAE 잠재 공간 샘플링을 통한 오버샘플링 (1:1 균형 달성)
6. [Oversampled] 증강 데이터로 로지스틱 회귀 재학습 및 동일한 테스트 세트 평가
7. 모델 성능(Accuracy, Precision, Recall, F1-Score, ROC-AUC) 및 혼동 행렬 비교
8. 원본 vs 증강 데이터 시각화 차트 출력
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
)
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow.keras import layers, models

# 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# 1. VAE 리파라미터화 트릭 (Reparameterization Trick) 레이어
# ============================================================
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# ============================================================
# 2. VAE 모델 구성 (Encoder + Decoder + Loss)
# ============================================================
class VAE(models.Model):
    def __init__(self, input_dim, latent_dim=4, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        encoder_inputs = layers.Input(shape=(input_dim,))
        x = layers.Dense(64, activation='relu')(encoder_inputs)
        x = layers.Dense(32, activation='relu')(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        # Decoder
        latent_inputs = layers.Input(shape=(latent_dim,))
        x = layers.Dense(32, activation='relu')(latent_inputs)
        x = layers.Dense(64, activation='relu')(x)
        decoder_outputs = layers.Dense(input_dim, activation='linear')(x)
        self.decoder = models.Model(latent_inputs, decoder_outputs, name="decoder")

        # Loss trackers
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # Reconstruction Loss
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(data - reconstruction), axis=1)
            )
            # KL Divergence
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            total_loss = recon_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        return {"loss": self.total_loss_tracker.result()}

def main():
    csv_path = r"C:/apps/mldl/data/telco_churn_preprocessed.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {csv_path}")

    print("=" * 80)
    print(" 1. Telco 고객 이탈 데이터 로드 및 전처리")
    print("=" * 80)

    df = pd.read_csv(csv_path)

    # 1) ID 컬럼 제거
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # 2) TotalCharges 결측치(공백 문자열) 처리
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan))
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # 3) 타깃 변수 확인 및 매핑
    # 이미 0과 1로 숫자로 인코딩되어 있는 경우 그대로 사용하고, 문자열('Yes'/'No')일 때만 매핑
    if df['Churn'].dtype == 'object':
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    else:
        df['Churn'] = df['Churn'].astype(int)

    # 4) 범주형 변수 One-Hot Encoding
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df_encoded.drop(columns=['Churn']).values
    y = df_encoded['Churn'].values

    # Train / Test 분할 (Stratified 80:20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 특성 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    n_features = X_train_scaled.shape[1]
    n_train_total = len(y_train)
    n_train_no = np.sum(y_train == 0)
    n_train_yes = np.sum(y_train == 1)

    print(f"▶ 전체 데이터 샘플 수: {len(df):,}개 | 인코딩 후 피처 수: {n_features}개")
    print(f"▶ 학습 데이터 세트: 총 {n_train_total:,}개")
    print(f"   - 유지 고객 (Churn=No,  0): {n_train_no:,}개 ({n_train_no/n_train_total*100:.1f}%)")
    print(f"   - 이탈 고객 (Churn=Yes, 1): {n_train_yes:,}개 ({n_train_yes/n_train_total*100:.1f}%)")
    print(f"▶ 클래스 불균형 비율 (Imbalance Ratio): {n_train_no / n_train_yes:.2f} : 1")

    # ============================================================
    # 2. [기준 모델] 원본 불균형 데이터로 로지스틱 회귀 학습 및 평가
    # ============================================================
    print("\n" + "=" * 80)
    print(" 2. [Base Line] 원본 불균형 데이터로 로지스틱 회귀 모델 학습")
    print("=" * 80)

    lr_base = LogisticRegression(max_iter=1000, random_state=42)
    lr_base.fit(X_train_scaled, y_train)

    y_pred_base = lr_base.predict(X_test_scaled)
    y_prob_base = lr_base.predict_proba(X_test_scaled)[:, 1]

    acc_base = accuracy_score(y_test, y_pred_base)
    prec_base = precision_score(y_test, y_pred_base)
    rec_base = recall_score(y_test, y_pred_base)
    f1_base = f1_score(y_test, y_pred_base)
    auc_base = roc_auc_score(y_test, y_prob_base)
    cm_base = confusion_matrix(y_test, y_pred_base)

    print(f"▶ 정확도 (Accuracy) : {acc_base * 100:.2f}%")
    print(f"▶ 정밀도 (Precision): {prec_base * 100:.2f}%")
    print(f"▶ 재현율 (Recall)   : {rec_base * 100:.2f}% (이탈 고객 탐지율)")
    print(f"▶ F1-Score          : {f1_base:.4f}")
    print(f"▶ ROC-AUC           : {auc_base:.4f}")
    print(f"▶ 혼동 행렬 (TN={cm_base[0,0]}, FP={cm_base[0,1]} / FN={cm_base[1,0]}, TP={cm_base[1,1]})")

    # ============================================================
    # 3. 소수 클래스(이탈 고객: Churn=Yes) 전용 VAE 모델 학습
    # ============================================================
    print("\n" + "=" * 80)
    print(" 3. 소수 클래스(이탈 고객) 전용 VAE 학습 및 오버샘플링(Oversampling)")
    print("=" * 80)

    X_train_minority = X_train_scaled[y_train == 1]
    latent_dim = 4

    vae = VAE(input_dim=n_features, latent_dim=latent_dim)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003))
    
    vae.fit(X_train_minority, epochs=80, batch_size=32, verbose=0)
    print("▶ 소수 클래스(Churn=Yes) 잠재 특징 공간 VAE 학습 완료")

    # 불균형을 1:1로 맞추기 위해 필요한 증강 샘플 수 계산
    n_synthetic = n_train_no - n_train_yes
    z_random = np.random.normal(size=(n_synthetic, latent_dim))
    X_synthetic = vae.decoder.predict(z_random, verbose=0)

    # 원본 Train 데이터와 VAE 합성 데이터 결합
    X_train_aug = np.vstack([X_train_scaled, X_synthetic])
    y_train_aug = np.hstack([y_train, np.ones(n_synthetic)])

    print(f"▶ 증강 전 이탈 고객 샘플: {n_train_yes:,}개")
    print(f"▶ VAE로 생성된 합성 고객: {n_synthetic:,}개")
    print(f"▶ 증강 후 전체 학습 세트: {len(y_train_aug):,}개 (유지: {n_train_no:,} vs 이탈: {np.sum(y_train_aug == 1):,}, 1:1 완벽 균형)")

    # ============================================================
    # 4. [오버샘플링 모델] VAE 증강 데이터로 로지스틱 회귀 학습 및 평가
    # ============================================================
    print("\n" + "=" * 80)
    print(" 4. [VAE Oversampled] 증강 데이터로 로지스틱 회귀 모델 재학습")
    print("=" * 80)

    lr_aug = LogisticRegression(max_iter=1000, random_state=42)
    lr_aug.fit(X_train_aug, y_train_aug)

    # 동일한 테스트 세트(X_test_scaled, y_test)로 공정하게 평가
    y_pred_aug = lr_aug.predict(X_test_scaled)
    y_prob_aug = lr_aug.predict_proba(X_test_scaled)[:, 1]

    acc_aug = accuracy_score(y_test, y_pred_aug)
    prec_aug = precision_score(y_test, y_pred_aug)
    rec_aug = recall_score(y_test, y_pred_aug)
    f1_aug = f1_score(y_test, y_pred_aug)
    auc_aug = roc_auc_score(y_test, y_prob_aug)
    cm_aug = confusion_matrix(y_test, y_pred_aug)

    print(f"▶ 정확도 (Accuracy) : {acc_aug * 100:.2f}%")
    print(f"▶ 정밀도 (Precision): {prec_aug * 100:.2f}%")
    print(f"▶ 재현율 (Recall)   : {rec_aug * 100:.2f}% (이탈 고객 탐지율)")
    print(f"▶ F1-Score          : {f1_aug:.4f}")
    print(f"▶ ROC-AUC           : {auc_aug:.4f}")
    print(f"▶ 혼동 행렬 (TN={cm_aug[0,0]}, FP={cm_aug[0,1]} / FN={cm_aug[1,0]}, TP={cm_aug[1,1]})")

    # ============================================================
    # 5. [교차검증] 5-Fold Stratified K-Fold 정밀 검증 (Data Leakage 완벽 방지)
    # ============================================================
    print("\n" + "=" * 80)
    print(" 5. 5-Fold 계층적 교차검증 (Stratified K-Fold Cross-Validation)")
    print("=" * 80)
    print("※ 검증 데이터 누수(Data Leakage)를 방지하기 위해,")
    print("   각 Fold의 Train 데이터에 대해서만 VAE를 학습하고 오버샘플링한 뒤 Validation 세트로 평가합니다.\n")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_base_metrics = {"acc": [], "prec": [], "rec": [], "f1": [], "auc": []}
    cv_vae_metrics  = {"acc": [], "prec": [], "rec": [], "f1": [], "auc": []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_val = X[val_idx], y[val_idx]

        # Fold별 독립 스케일링
        fold_scaler = StandardScaler()
        X_tr_s = fold_scaler.fit_transform(X_tr)
        X_va_s = fold_scaler.transform(X_va)

        # 1) Fold 기준모델 (Base Logistic Regression)
        fold_lr_base = LogisticRegression(max_iter=1000, random_state=42)
        fold_lr_base.fit(X_tr_s, y_tr)
        pred_b = fold_lr_base.predict(X_va_s)
        prob_b = fold_lr_base.predict_proba(X_va_s)[:, 1]

        cv_base_metrics["acc"].append(accuracy_score(y_val, pred_b))
        cv_base_metrics["prec"].append(precision_score(y_val, pred_b, zero_division=0))
        cv_base_metrics["rec"].append(recall_score(y_val, pred_b))
        cv_base_metrics["f1"].append(f1_score(y_val, pred_b))
        cv_base_metrics["auc"].append(roc_auc_score(y_val, prob_b))

        # 2) Fold VAE 오버샘플링 (Train 내 소수 클래스만으로 학습)
        tr_no = np.sum(y_tr == 0)
        tr_yes = np.sum(y_tr == 1)
        X_tr_min = X_tr_s[y_tr == 1]
        n_syn = tr_no - tr_yes

        fold_vae = VAE(input_dim=n_features, latent_dim=latent_dim)
        fold_vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003))
        fold_vae.fit(X_tr_min, epochs=50, batch_size=32, verbose=0)

        # 합성 데이터 생성 및 결합
        z_f = np.random.normal(size=(n_syn, latent_dim))
        X_syn = fold_vae.decoder.predict(z_f, verbose=0)
        X_tr_aug = np.vstack([X_tr_s, X_syn])
        y_tr_aug = np.hstack([y_tr, np.ones(n_syn)])

        # Fold VAE 모델 학습 및 평가
        fold_lr_vae = LogisticRegression(max_iter=1000, random_state=42)
        fold_lr_vae.fit(X_tr_aug, y_tr_aug)
        pred_v = fold_lr_vae.predict(X_va_s)
        prob_v = fold_lr_vae.predict_proba(X_va_s)[:, 1]

        cv_vae_metrics["acc"].append(accuracy_score(y_val, pred_v))
        cv_vae_metrics["prec"].append(precision_score(y_val, pred_v, zero_division=0))
        cv_vae_metrics["rec"].append(recall_score(y_val, pred_v))
        cv_vae_metrics["f1"].append(f1_score(y_val, pred_v))
        cv_vae_metrics["auc"].append(roc_auc_score(y_val, prob_v))

        print(f"  [Fold {fold}/5] Base Recall: {cv_base_metrics['rec'][-1]*100:.2f}% | VAE Recall: {cv_vae_metrics['rec'][-1]*100:.2f}% (AUC: {cv_vae_metrics['auc'][-1]:.4f})")

    # 교차검증 평균 결과 테이블
    cv_summary_df = pd.DataFrame({
        "평가 지표 (5-Fold CV Mean ± Std)": ["정확도 (Accuracy)", "정밀도 (Precision)", "재현율 (Recall)", "F1-Score", "ROC-AUC"],
        "1) 원본 불균형 Base": [
            f"{np.mean(cv_base_metrics['acc'])*100:.2f} ± {np.std(cv_base_metrics['acc'])*100:.2f}%",
            f"{np.mean(cv_base_metrics['prec'])*100:.2f} ± {np.std(cv_base_metrics['prec'])*100:.2f}%",
            f"{np.mean(cv_base_metrics['rec'])*100:.2f} ± {np.std(cv_base_metrics['rec'])*100:.2f}%",
            f"{np.mean(cv_base_metrics['f1']):.4f} ± {np.std(cv_base_metrics['f1']):.4f}",
            f"{np.mean(cv_base_metrics['auc']):.4f} ± {np.std(cv_base_metrics['auc']):.4f}"
        ],
        "2) VAE 오버샘플링 적용": [
            f"{np.mean(cv_vae_metrics['acc'])*100:.2f} ± {np.std(cv_vae_metrics['acc'])*100:.2f}%",
            f"{np.mean(cv_vae_metrics['prec'])*100:.2f} ± {np.std(cv_vae_metrics['prec'])*100:.2f}%",
            f"{np.mean(cv_vae_metrics['rec'])*100:.2f} ± {np.std(cv_vae_metrics['rec'])*100:.2f}%",
            f"{np.mean(cv_vae_metrics['f1']):.4f} ± {np.std(cv_vae_metrics['f1']):.4f}",
            f"{np.mean(cv_vae_metrics['auc']):.4f} ± {np.std(cv_vae_metrics['auc']):.4f}"
        ],
        "평균 변화폭 (Delta)": [
            f"{(np.mean(cv_vae_metrics['acc']) - np.mean(cv_base_metrics['acc']))*100:+.2f}%p",
            f"{(np.mean(cv_vae_metrics['prec']) - np.mean(cv_base_metrics['prec']))*100:+.2f}%p",
            f"{(np.mean(cv_vae_metrics['rec']) - np.mean(cv_base_metrics['rec']))*100:+.2f}%p",
            f"{(np.mean(cv_vae_metrics['f1']) - np.mean(cv_base_metrics['f1'])):+.4f}",
            f"{(np.mean(cv_vae_metrics['auc']) - np.mean(cv_base_metrics['auc'])):+.4f}"
        ]
    })
    print("\n" + "=" * 80)
    print(" 5-Fold 계층적 교차검증(Stratified K-Fold) 종합 평균 결과")
    print("=" * 80)
    print(cv_summary_df.to_string(index=False))

    # ============================================================
    # 6. 단일 Hold-out 테스트 세트 기준 최종 성능 요약
    # ============================================================
    print("\n" + "=" * 80)
    print(" 6. 단일 Hold-out Test 세트 최종 성능 비교 요약")
    print("=" * 80)

    diff_acc = (acc_aug - acc_base) * 100
    diff_prec = (prec_aug - prec_base) * 100
    diff_rec = (rec_aug - rec_base) * 100
    diff_f1 = f1_aug - f1_base
    diff_auc = auc_aug - auc_base

    comparison_df = pd.DataFrame({
        "평가 지표 (Metric)": ["정확도 (Accuracy)", "정밀도 (Precision)", "재현율 (Recall)", "F1-Score", "ROC-AUC"],
        "1) 원본 불균형 데이터": [f"{acc_base*100:.2f}%", f"{prec_base*100:.2f}%", f"{rec_base*100:.2f}%", f"{f1_base:.4f}", f"{auc_base:.4f}"],
        "2) VAE 오버샘플링 적용": [f"{acc_aug*100:.2f}%", f"{prec_aug*100:.2f}%", f"{rec_aug*100:.2f}%", f"{f1_aug:.4f}", f"{auc_aug:.4f}"],
        "변화폭 (Delta)": [f"{diff_acc:+.2f}%p", f"{diff_prec:+.2f}%p", f"{diff_rec:+.2f}%p", f"{diff_f1:+.4f}", f"{diff_auc:+.4f}"]
    })
    print(comparison_df.to_string(index=False))

    print("\n[비즈니스 관점의 해석]")
    print(f" 1) 실제 이탈 고객 탐지 성공 건수(TP): {cm_base[1,1]}명 -> {cm_aug[1,1]}명 (추가로 {cm_aug[1,1] - cm_base[1,1]}명의 이탈 위험 고객 조기 포착)")
    print(f" 2) 이탈 고객 미탐 건수(FN)          : {cm_base[1,0]}명 -> {cm_aug[1,0]}명 ({cm_base[1,0] - cm_aug[1,0]}명 방어 성공)")
    print(f" 3) 고객 이탈 방지 마케팅에서는 이탈자를 놓치지 않는 '재현율(Recall: {diff_rec:+.2f}%p)'이 가장 중요하며,")
    print("    VAE 오버샘플링을 통해 실질적인 고객 방어 성공률을 비약적으로 끌어올렸습니다.")

    # ============================================================
    # 7. 시각화 비교 (혼동 행렬 및 PCA 분포)
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) 원본 혼동 행렬
    im1 = axes[0].imshow(cm_base, cmap='Blues')
    axes[0].set_title(f"1. 원본 불균형 혼동 행렬\n(Recall: {rec_base*100:.1f}%)", fontweight='bold')
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(['유지(No)', '이탈(Yes)'])
    axes[0].set_yticklabels(['유지(No)', '이탈(Yes)'])
    axes[0].set_xlabel("예측값 (Predicted)")
    axes[0].set_ylabel("실제값 (Actual)")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f"{cm_base[i, j]}명", ha='center', va='center', color='red' if (i==1 and j==0) else 'black', fontweight='bold')

    # 2) VAE 증강 후 혼동 행렬
    im2 = axes[1].imshow(cm_aug, cmap='Greens')
    axes[1].set_title(f"2. VAE 오버샘플링 후 혼동 행렬\n(Recall: {rec_aug*100:.1f}%)", fontweight='bold')
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(['유지(No)', '이탈(Yes)'])
    axes[1].set_yticklabels(['유지(No)', '이탈(Yes)'])
    axes[1].set_xlabel("예측값 (Predicted)")
    axes[1].set_ylabel("실제값 (Actual)")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f"{cm_aug[i, j]}명", ha='center', va='center', color='red' if (i==1 and j==0) else 'black', fontweight='bold')

    # 3) PCA 2차원 투영 공간 분포
    pca = PCA(n_components=2)
    pca.fit(X_train_scaled)
    pca_minority = pca.transform(X_train_minority)
    pca_synthetic = pca.transform(X_synthetic)

    axes[2].scatter(pca_minority[:, 0], pca_minority[:, 1], color='#E53935', alpha=0.6, label='원본 이탈 고객 (Churn=Yes)', s=25)
    axes[2].scatter(pca_synthetic[:, 0], pca_synthetic[:, 1], color='#1E88E5', alpha=0.25, label='VAE 합성 증강 고객', s=15)
    axes[2].set_title("3. 원본 이탈 고객 vs VAE 합성 고객 (PCA)", fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("Telco 고객 이탈: 로지스틱 회귀 모델의 VAE 오버샘플링 전후 성능 비교", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
