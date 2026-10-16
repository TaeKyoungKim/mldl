import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

RAW_PATH = "C:/apps/mldl/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUT_PATH = "C:/apps/mldl/data/telco_churn_preprocessed.csv"

# 입력 파일 존재 여부 확인
if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"원본 데이터 파일을 찾을 수 없습니다: {RAW_PATH}")

df = pd.read_csv(RAW_PATH)

# 1. 식별자 컬럼 제거 (예측에 불필요)
df = df.drop(columns=["customerID"])

# 2. TotalCharges: 공백 문자열이 섞여있어 object 타입 -> 숫자로 변환 (공백은 결측으로 처리 후 평균값으로 채움)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())

# 3. 타겟(Churn) 라벨 인코딩: No -> 0, Yes -> 1
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# 4. 나머지 범주형 컬럼 라벨 인코딩
categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# 5. 수치형 컬럼 스케일링 (CNN 입력에 적합하도록 0 근처로 정규화)
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("전처리 후 shape:", df.shape)
print("타겟(Churn) 분포:\n", df["Churn"].value_counts())
print("\n전처리 결과 샘플:")
print(df.head())

# 6. 새 CSV로 저장
df.to_csv(OUT_PATH, index=False)
print(f"\n전처리된 데이터 저장: {OUT_PATH}")
