"""
Keras IMDB 영화 리뷰 데이터셋 (https://keras.io/api/datasets/imdb/)
다운로드, 전체 데이터 구조 파악(EDA), 단어 사전 디코딩 및 시각화 코드
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import imdb

# 한글 폰트 및 마이너스 기호 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("=" * 80)
    print(" 1. Keras IMDB 영화 리뷰 데이터셋 다운로드 및 로드")
    print("=" * 80)

    # 가장 자주 사용되는 상위 10,000개 단어로 제한하여 로드
    NUM_WORDS = 10000
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=NUM_WORDS)

    # 1. 기본 데이터 구조 파악
    print(f"학습 데이터 (Train) 리뷰 개수 : {len(x_train):,}개")
    print(f"테스트 데이터 (Test) 리뷰 개수: {len(x_test):,}개")
    print(f"타깃 레이블 형태 (0: 부정, 1: 긍정): {np.unique(y_train)}")

    # 2. 클래스 균형(라벨 분포) 확인
    train_pos = np.sum(y_train == 1)
    train_neg = np.sum(y_train == 0)
    print(f"\n[학습 데이터 라벨 분포]")
    print(f" - 긍정(Positive, 1): {train_pos:,}개 ({train_pos / len(y_train) * 100:.1f}%)")
    print(f" - 부정(Negative, 0): {train_neg:,}개 ({train_neg / len(y_train) * 100:.1f}%)")

    # 3. 리뷰 길이(단어 수) 통계량 분석
    review_lens = [len(review) for review in x_train]
    min_len = np.min(review_lens)
    max_len = np.max(review_lens)
    mean_len = np.mean(review_lens)
    median_len = np.median(review_lens)
    percentile_90 = np.percentile(review_lens, 90)

    print("\n" + "=" * 80)
    print(" 2. 리뷰 텍스트 길이 (단어 수) 통계량")
    print("=" * 80)
    print(f"최소 길이: {min_len} 단어")
    print(f"최대 길이: {max_len} 단어")
    print(f"평균 길이: {mean_len:.2f} 단어")
    print(f"중앙값(Median): {median_len:.1f} 단어")
    print(f"상위 90% 백분위수: {percentile_90:.1f} 단어 (시퀀스 고정 길이(maxlen) 결정 시 주요 지표)")

    # 4. 정수 인덱스 -> 실제 영문 텍스트 디코딩 (Word Index 복원)
    print("\n" + "=" * 80)
    print(" 3. 리뷰 샘플 디코딩 (정수 인덱스 -> 실제 영문 텍스트)")
    print("=" * 80)

    word_index = imdb.get_word_index()
    # Keras IMDB 예약 인덱스:
    # 0: <PAD>, 1: <START>, 2: <UNK>, 3: <UNUSED>
    # 따라서 실제 단어 인덱스는 3씩 오프셋(shift)되어 있습니다.
    reverse_word_index = {value + 3: key for key, value in word_index.items()}
    reverse_word_index[0] = "<PAD>"
    reverse_word_index[1] = "<START>"
    reverse_word_index[2] = "<UNK>"
    reverse_word_index[3] = "<UNUSED>"

    def decode_review(token_list):
        return ' '.join([reverse_word_index.get(i, '?') for i in token_list])

    # 1번째 리뷰 (인덱스: 0) 디코딩
    print("[1번째 학습 리뷰 (인덱스: 0)]")
    print(f" - 실제 라벨: {'긍정 (Positive: 1)' if y_train[0] == 1 else '부정 (Negative: 0)'}")
    print(f" - 정수 토큰 (앞 15개): {x_train[0][:15]} ...")
    print(f" - 디코딩 텍스트 (앞부분 200자): {decode_review(x_train[0])[:200]} ...\n")

    # 10번째 리뷰 (인덱스: 9) 디코딩
    target_idx = 9  # 10번째 샘플 (0-based indexing: 10 - 1)
    print("-" * 80)
    print(f"[{target_idx + 1}번째 학습 리뷰 (인덱스: {target_idx}) 디코딩 결과]")
    print("-" * 80)
    print(f" - 실제 라벨       : {'긍정 (Positive: 1)' if y_train[target_idx] == 1 else '부정 (Negative: 0)'}")
    print(f" - 리뷰 단어 길이  : {len(x_train[target_idx])}개 단어")
    print(f" - 정수 토큰 배열  : {x_train[target_idx][:20]} ... (앞 20개)")
    print("\n[10번째 리뷰 전체 텍스트 원문]:")
    print(decode_review(x_train[target_idx]))
    print("-" * 80 + "\n")

    # ============================================================
    # 5. 데이터 파악을 위한 시각화 (라벨 분포 & 리뷰 길이 히스토그램)
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (1) 라벨 분포 막대 그래프
    axes[0].bar(['부정 (0)', '긍정 (1)'], [train_neg, train_pos], color=['#E57373', '#81C784'], width=0.5)
    axes[0].set_title("1. 클래스 라벨 분포 (학습 데이터)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("리뷰 개수")
    axes[0].text(0, train_neg / 2, f"{train_neg:,}개\n(50%)", ha='center', va='center', color='white', fontweight='bold')
    axes[0].text(1, train_pos / 2, f"{train_pos:,}개\n(50%)", ha='center', va='center', color='white', fontweight='bold')
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)

    # (2) 리뷰 길이 히스토그램 & 분포
    axes[1].hist(review_lens, bins=60, color='#64B5F6', edgecolor='white')
    axes[1].axvline(mean_len, color='red', linestyle='--', linewidth=1.5, label=f'평균 ({mean_len:.0f} 단어)')
    axes[1].axvline(percentile_90, color='darkorange', linestyle='--', linewidth=1.5, label=f'상위 90% ({percentile_90:.0f} 단어)')
    axes[1].set_title("2. 리뷰 단어 수 길이 분포 (Histogram)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("단어 수 (Sequence Length)")
    axes[1].set_ylabel("리뷰 빈도 수")
    axes[1].set_xlim(0, 1200)
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)

    plt.suptitle("Keras IMDB Dataset 전체 데이터 구조 및 특성 EDA", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
