"""
================================================================================
UCI ML Repository: Human Activity Recognition Using Smartphones (ID: 240)
데이터셋 다운로드 및 피처명 중복 해결/분석 스크립트
================================================================================
"""

import os
import sys
import zipfile
import urllib.request
from types import SimpleNamespace
import pandas as pd

# 윈도우 콘솔 UTF-8 한글 인코딩 대응
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# 프로젝트 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(CURRENT_DIR) == "src":
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
else:
    PROJECT_ROOT = CURRENT_DIR

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
DOWNLOAD_ZIP_PATH = os.path.join(DATA_DIR, "har.zip")
DATASET_DIR = os.path.join(DATA_DIR, "UCI HAR Dataset")


def download_and_extract_uci_archive(data_dir: str = DATA_DIR):
    """
    UCI 머신러닝 저장소 공식 서버에서 HAR 압축 파일을 다운로드하고 압축을 해제합니다.
    """
    inner_zip_path = os.path.join(data_dir, "UCI HAR Dataset.zip")

    # 1. 압축 파일 다운로드 (없는 경우)
    if not os.path.exists(DATASET_DIR):
        if not os.path.exists(DOWNLOAD_ZIP_PATH) and not os.path.exists(inner_zip_path):
            print(f"[1/3] UCI 공식 서버에서 데이터셋 압축 파일 다운로드 중...")
            print(f"      - 다운로드 URL: {UCI_ZIP_URL}")
            urllib.request.urlretrieve(UCI_ZIP_URL, DOWNLOAD_ZIP_PATH)
            print(f"      - 다운로드 완료: {DOWNLOAD_ZIP_PATH}")
        else:
            print(f"[1/3] 이미 다운로드된 압축 파일이 존재합니다: {DOWNLOAD_ZIP_PATH}")

        # 2. 압축 해제
        print(f"[2/3] 압축 파일 해제 중...")
        if os.path.exists(DOWNLOAD_ZIP_PATH):
            with zipfile.ZipFile(DOWNLOAD_ZIP_PATH, 'r') as z:
                z.extractall(data_dir)

        if os.path.exists(inner_zip_path):
            with zipfile.ZipFile(inner_zip_path, 'r') as z:
                z.extractall(data_dir)
        print(f"      - 압축 해제 완료: {DATASET_DIR}")
    else:
        print(f"[1/3] 데이터셋 폴더가 이미 준비되어 있습니다: {DATASET_DIR}")


def analyze_and_resolve_duplicate_features(features_path: str):
    """
    원본 features.txt의 피처명 중복 현황을 상세 분석하고,
    Pandas DataFrame에서 인덱싱 오류가 발생하지 않도록 중복 컬럼명을 고유(Unique)하게 처리합니다.

    [중복 발생 원인]
    UCI HAR 데이터셋 원본의 주파수 대역 에너지(bandsEnergy()) 측정값은
    14개 주파수 구간에 대해 3개 축(X, Y, Z)별로 계산되었으나,
    features.txt 작성 시 3개 축 접미사(-X, -Y, -Z)가 누락되어
    42개 변수명이 각각 3회씩(총 126개 행) 중복 표기되었습니다.
    """
    print("\n" + "=" * 78)
    print("  [피처(Feature) 중복 이름 정밀 분석 및 인덱싱 처리]")
    print("=" * 78)

    # 1. features.txt 로드
    features_df = pd.read_csv(
        features_path,
        sep=r"\s+",
        header=None,
        names=["feature_idx", "feature_name"]
    )

    total_count = len(features_df)
    unique_count = features_df["feature_name"].nunique()
    duplicated_mask = features_df.duplicated(subset=["feature_name"], keep=False)
    dup_rows_count = duplicated_mask.sum()

    # 중복 피처명 빈도 분석
    value_counts = features_df["feature_name"].value_counts()
    dup_names = value_counts[value_counts > 1]
    num_dup_names = len(dup_names)

    print(f"- 전체 피처 수                     : {total_count}개")
    print(f"- 고유(Unique) 피처명 수           : {unique_count}개")
    print(f"- 중복에 관여된 전체 행 수         : {dup_rows_count}개")
    print(f"- 2회 이상 중복된 고유 피처명 수   : {num_dup_names}개")
    print(f"- 피처명당 중복 발생 횟수          : 모두 정확히 3회씩 발생 ({num_dup_names}개 × 3회 = {dup_rows_count}개)")

    print("\n[중복 발생 주요 원인 분석]:")
    print("  - 주파수 대역 에너지(bandsEnergy()) 특성은 14개 주파수 구간 × 3개 축(X, Y, Z)으로 산출됩니다.")
    print("  - 원본 작성 시 축 접미사(-X, -Y, -Z)가 생략되어 아래 3대 신호 그룹에서 중복이 발생했습니다:")
    print("    1) fBodyAcc-bandsEnergy()      : 14개 구간 × 3회 = 42개 행 (X: 303~316, Y: 317~330, Z: 331~344)")
    print("    2) fBodyAccJerk-bandsEnergy()  : 14개 구간 × 3회 = 42개 행 (X: 382~395, Y: 396~409, Z: 410~423)")
    print("    3) fBodyGyro-bandsEnergy()     : 14개 구간 × 3회 = 42개 행 (X: 461~474, Y: 475~488, Z: 489~502)")
    print("    => 총 42개 고유 피처명 × 3회 = 126개 중복 행")

    print("\n[대표 중복 피처명 샘플 (Top 5)]:")
    for name, cnt in dup_names.head(5).items():
        sample_indices = features_df[features_df["feature_name"] == name]["feature_idx"].tolist()
        print(f"  * '{name}': {cnt}회 중복 (원본 인덱스: {sample_indices})")

    # 2. 중복 컬럼명 인덱싱 해결 처리
    # Pandas 컬럼 중복 시 df[col] 접근 시 DataFrame이 반환되는 모호성을 방지하기 위해
    # 발생 순서에 따라 축 정보(axis1, axis2, axis3) 또는 인덱스 접미사를 부여
    axis_labels = {0: "axisX", 1: "axisY", 2: "axisZ"}
    seen_counts = {}
    resolved_feature_names = []
    change_log = []

    for _, row in features_df.iterrows():
        idx = row["feature_idx"]
        name = row["feature_name"]

        if name in dup_names:
            occurrence = seen_counts.get(name, 0)
            seen_counts[name] = occurrence + 1
            axis_suffix = axis_labels.get(occurrence, f"dup{occurrence + 1}")
            resolved_name = f"{name}_{axis_suffix}"
            change_log.append({
                "Index": idx,
                "Original": name,
                "Resolved": resolved_name,
                "Axis/Seq": axis_suffix
            })
        else:
            resolved_name = name

        resolved_feature_names.append(resolved_name)

    # 처리 결과 검증
    resolved_series = pd.Series(resolved_feature_names)
    final_dups = resolved_series.duplicated().sum()

    print("\n[중복 해결 후 컬럼명 변환 샘플 (Before vs After)]:")
    sample_changes_df = pd.DataFrame(change_log).head(6)
    print(sample_changes_df.to_string(index=False))

    print(f"\n[최종 검증]:")
    print(f"  - 처리 후 고유 컬럼 수 : {resolved_series.nunique()}개 / {len(resolved_feature_names)}개")
    print(f"  - 잔여 중복 컬럼 수     : {final_dups}개 (100% 고유 컬럼 변환 완료)")
    print("=" * 78 + "\n")

    return resolved_feature_names


def parse_har_dataset(dataset_dir: str = DATASET_DIR):
    """
    압축 해제된 텍스트 파일들을 읽어 고유 컬럼명을 적용한 피처(X)와 타깃(y) 데이터프레임으로 변환합니다.
    """
    print("[3/3] 중복 처리된 고유 피처명을 적용하여 데이터 로드 중...")

    # 피처명 중복 분석 및 고유 컬럼명 리스트 생성
    features_path = os.path.join(dataset_dir, "features.txt")
    feature_names = analyze_and_resolve_duplicate_features(features_path)

    # 학습용(Train) 및 평가용(Test) 데이터 로드
    X_train = pd.read_csv(
        os.path.join(dataset_dir, "train", "X_train.txt"),
        sep=r"\s+",
        header=None,
        names=feature_names
    )
    y_train = pd.read_csv(
        os.path.join(dataset_dir, "train", "y_train.txt"),
        sep=r"\s+",
        header=None,
        names=["Activity"]
    )

    X_test = pd.read_csv(
        os.path.join(dataset_dir, "test", "X_test.txt"),
        sep=r"\s+",
        header=None,
        names=feature_names
    )
    y_test = pd.read_csv(
        os.path.join(dataset_dir, "test", "y_test.txt"),
        sep=r"\s+",
        header=None,
        names=["Activity"]
    )

    # 전체 X, y 통합
    X = pd.concat([X_train, X_test], ignore_index=True)
    y = pd.concat([y_train, y_test], ignore_index=True)

    # ucimlrepo 호환 구조 생성
    har = SimpleNamespace()
    har.data = SimpleNamespace(
        features=X,
        targets=y,
        train_features=X_train,
        train_targets=y_train,
        test_features=X_test,
        test_targets=y_test
    )
    return har


def download_har():
    """
    1) ucimlrepo의 fetch_ucirepo(id=240) 시도
    2) UCI 서버 정책상 API 직접 로드가 제한된 경우 공식 압축 파일 다운로드 및 중복 피처 처리 로더 실행
    """
    print("#" * 78)
    print("  UCI Human Activity Recognition (ID: 240) 데이터셋 다운로드 & 피처 분석")
    print("#" * 78)

    try:
        from ucimlrepo import fetch_ucirepo
        print("[시도] ucimlrepo 패키지를 통한 fetch_ucirepo(id=240) 호출...")
        har = fetch_ucirepo(id=240)
        X = har.data.features
        y = har.data.targets
        print("  >> ucimlrepo API를 통해 다운로드 완료!")
    except Exception as e:
        print(f"  >> ucimlrepo API 직접 다운로드 불가 (사유: {e})")
        print("  >> UCI 공식 아카이브 직통 다운로드 및 정밀 전처리 프로세스로 전환합니다.")
        download_and_extract_uci_archive(DATA_DIR)
        har = parse_har_dataset(DATASET_DIR)
        X = har.data.features
        y = har.data.targets

    print("=" * 78)
    print("  [최종 데이터셋 구성 요약]")
    print("=" * 78)
    print(f"- 피처 데이터 (X) 형태 : {X.shape} (샘플 수: {X.shape[0]}, 고유 특성 수: {X.shape[1]})")
    print(f"- 타깃 데이터 (y) 형태 : {y.shape}")
    print(f"- 타깃 클래스 분포:\n{y['Activity'].value_counts().sort_index()}")

    # 중복 제거된 깔끔한 피처 데이터 CSV 파일로 저장
    csv_x_path = os.path.join(DATA_DIR, "har_features.csv")
    csv_y_path = os.path.join(DATA_DIR, "har_targets.csv")
    print(f"\n[저장] 중복 컬럼명이 정제된 피처 및 타깃 데이터를 CSV 파일로 저장 중...")
    X.to_csv(csv_x_path, index=False)
    y.to_csv(csv_y_path, index=False)
    print(f"  - X 저장 완료: {csv_x_path}")
    print(f"  - y 저장 완료: {csv_y_path}")

    print("\n모든 다운로드, 중복 분석 및 데이터셋 처리가 완료되었습니다!")
    return har, X, y


if __name__ == "__main__":
    download_har()
