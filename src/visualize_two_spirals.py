"""
================================================================================
Two Spirals Noisy Dense 데이터셋 군집 시각화
================================================================================
- 데이터 경로: C:/apps/mldl/data/two_spirals_noisy_dense.csv
- 열 구성: x, y, cluster
- 주요 기능:
  1. cluster 컬럼값을 기준으로 색상을 구분한 2D 산점도 시각화
  2. 군집별 데이터 개수 및 요약 정보 출력
  3. 고해상도 이미지(PNG) 저장 (output/two_spirals_visualization.png)
  4. plt.show()를 통한 대화형 윈도우 창 팝업 표시
================================================================================
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 윈도우 콘솔 및 한글 폰트 설정
if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    plt.rc("font", family="Malgun Gothic")
    plt.rcParams["axes.unicode_minus"] = False

# 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "src" else Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "two_spirals_noisy_dense.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH = OUTPUT_DIR / "two_spirals_visualization.png"


def visualize_spirals():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"데이터 파일이 존재하지 않습니다: {DATA_PATH}")

    # 1. 데이터 로드
    df = pd.read_csv(DATA_PATH)
    print(f"[1] 데이터 로드 완료: 총 {len(df)}개 샘플")
    print("\n[군집별 샘플 수]:")
    print(df["cluster"].value_counts().sort_index().to_string())

    # 2. 시각화 생성
    plt.figure(figsize=(9, 8))

    # 명확한 구분을 위한 커스텀 팔레트 (Spiral 0: 오렌지/레드 계열, Spiral 1: 블루/틸 계열 등)
    unique_clusters = sorted(df["cluster"].unique())
    palette = sns.color_palette("bright", len(unique_clusters))

    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="cluster",
        palette=palette,
        style="cluster",
        s=45,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5
    )

    plt.title("Two Spirals Noisy Dense 데이터 군집 시각화 (by Cluster)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("X 좌표", fontsize=12)
    plt.ylabel("Y 좌표", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(title="Cluster", loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()

    # 3. 이미지 저장
    plt.savefig(SAVE_PATH, dpi=300)
    print(f"\n[2] 그래프 이미지 저장 완료: {SAVE_PATH}")

    # 4. 화면 표시
    print("[3] 시각화 창(Window)을 표시합니다.")
    plt.show()


if __name__ == "__main__":
    visualize_spirals()
