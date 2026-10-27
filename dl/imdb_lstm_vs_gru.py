"""Keras IMDB 데이터를 활용해 LSTM과 GRU 모델을 각각 학습하고,
학습 소요 시간(Training Time) 측정 및 테스트 데이터의 100번째 리뷰 예측 결과를 비교한다.

실행:
    python imdb_lstm_vs_gru.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KERAS_HOME = ROOT / "data" / "imdb"
os.environ["KERAS_HOME"] = str(KERAS_HOME)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, Embedding, GRU, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences

TEST_POSITION = 100
TEST_INDEX = TEST_POSITION - 1
MOVIE_TITLE = "제공되지 않음 (Keras IMDB 데이터셋에 영화 제목 필드가 없음)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-words", type=int, default=10_000, help="사용할 최대 단어 수")
    parser.add_argument("--maxlen", type=int, default=300, help="리뷰의 고정 토큰 길이")
    parser.add_argument("--embedding-dim", type=int, default=64, help="임베딩 차원")
    parser.add_argument("--units", type=int, default=64, help="LSTM/GRU 은닉 유닛 수")
    parser.add_argument("--epochs", type=int, default=3, help="학습 epoch 수")
    parser.add_argument("--batch-size", type=int, default=128, help="미니배치 크기")
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="학습 데이터 일부만 사용할 때의 개수(생략 시 전체 25,000개)",
    )
    parser.add_argument("--seed", type=int, default=42, help="난수 시드")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def decode_review(tokens: list[int]) -> str:
    word_index = imdb.get_word_index()
    reverse_index = {index + 3: word for word, index in word_index.items()}
    reverse_index.update({0: "[PAD]", 1: "[START]", 2: "[OOV]", 3: "[UNUSED]"})
    return " ".join(reverse_index.get(token, "[UNKNOWN]") for token in tokens)


def build_recurrent_model(model_type: str, args: argparse.Namespace) -> tf.keras.Model:
    """model_type에 따라 LSTM 또는 GRU 모델을 생성"""
    rnn_layer = (
        LSTM(args.units, name="lstm_layer")
        if model_type.upper() == "LSTM"
        else GRU(args.units, name="gru_layer")
    )

    model = Sequential(
        [
            tf.keras.Input(shape=(args.maxlen,), dtype="int32"),
            Embedding(input_dim=args.num_words, output_dim=args.embedding_dim, mask_zero=True),
            rnn_layer,
            Dense(1, activation="sigmoid", name="output"),
        ],
        name=f"imdb_{model_type.lower()}",
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def sentiment_name(label: int) -> str:
    return "긍정" if label == 1 else "부정"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = ROOT / "output" / "imdb_lstm_vs_gru"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(" 1. IMDB 영화 리뷰 데이터셋 로드 및 패딩 전처리")
    print("=" * 75)
    print(f"IMDB 데이터 캐시 위치: {KERAS_HOME / 'datasets'}")
    
    (x_train, y_train), (x_test, y_test) = imdb.load_data(
        num_words=args.num_words,
        seed=113,
        start_char=1,
        oov_char=2,
        index_from=3,
    )

    if args.train_limit is not None:
        limit = min(args.train_limit, len(x_train))
        x_train, y_train = x_train[:limit], y_train[:limit]

    original_test_review = x_test[TEST_INDEX]
    decoded_test_review = decode_review(original_test_review)
    actual_label = int(y_test[TEST_INDEX])

    # 길이 고정 (Zero-padding)
    x_train_padded = pad_sequences(
        x_train, maxlen=args.maxlen, padding="post", truncating="post"
    )
    x_test_padded = pad_sequences(
        x_test, maxlen=args.maxlen, padding="post", truncating="post"
    )

    results = {}
    target_sample = x_test_padded[TEST_INDEX : TEST_INDEX + 1]

    # ============================================================
    # 2. LSTM vs GRU 학습 및 수행시간 측정
    # ============================================================
    for model_name in ["LSTM", "GRU"]:
        print("\n" + "=" * 75)
        print(f" [{model_name} 모델] 아키텍처 구축 및 학습 시작")
        print("=" * 75)
        
        # 모델 간 공정한 비교를 위해 시드 재설정
        set_seed(args.seed)
        model = build_recurrent_model(model_name, args)
        model.summary()

        # 학습 수행시간 측정 시작
        start_time = time.perf_counter()
        history = model.fit(
            x_train_padded,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=0.2,
            shuffle=True,
            verbose=2,
        )
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # 테스트 데이터셋 전체 평가
        test_loss, test_accuracy = model.evaluate(
            x_test_padded, y_test, batch_size=args.batch_size, verbose=0
        )

        # 100번째 테스트 샘플 예측
        pred_prob = float(model.predict(target_sample, verbose=0)[0, 0])
        pred_label = int(pred_prob >= 0.5)

        # 결과 저장
        results[model_name] = {
            "elapsed_time_sec": elapsed_time,
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "predicted_probability": pred_prob,
            "predicted_label": pred_label,
            "predicted_sentiment": sentiment_name(pred_label),
            "is_correct": pred_label == actual_label,
            "params_count": model.count_params(),
        }

        # 모델 가중치 저장
        model.save(output_dir / f"imdb_{model_name.lower()}.keras")

    # ============================================================
    # 3. 수행시간 및 100번째 샘플 예측 결과 종합 비교 출력
    # ============================================================
    print("\n" + "=" * 78)
    print("                    [LSTM vs GRU 수행 결과 종합 비교]")
    print("=" * 78)
    print(f"테스트 데이터 대상 : {TEST_POSITION}번째 리뷰 (Python index = {TEST_INDEX})")
    print(f"영화 제목          : {MOVIE_TITLE}")
    print(f"실제 정답 (Actual) : [{actual_label}] {sentiment_name(actual_label)}")
    print("-" * 78)
    print(f"{'지표 (Metric)':<26} | {'LSTM':<22} | {'GRU':<22}")
    print("-" * 78)
    print(f"{'총 파라미터 수':<22} | {results['LSTM']['params_count']:<22,} | {results['GRU']['params_count']:<22,}")
    print(f"{'학습 소요 시간 (초)':<20} | {results['LSTM']['elapsed_time_sec']:<22.2f} | {results['GRU']['elapsed_time_sec']:<22.2f}")
    print(f"{'테스트 정확도 (Test Acc)':<19} | {results['LSTM']['test_accuracy'] * 100:<20.2f}% | {results['GRU']['test_accuracy'] * 100:<20.2f}%")
    print(f"{'테스트 손실 (Test Loss)':<19} | {results['LSTM']['test_loss']:<22.4f} | {results['GRU']['test_loss']:<22.4f}")
    print(f"{'100번째 긍정 예측 확률':<20} | {results['LSTM']['predicted_probability'] * 100:<20.2f}% | {results['GRU']['predicted_probability'] * 100:<20.2f}%")
    print(f"{'100번째 예측 라벨':<22} | {results['LSTM']['predicted_sentiment']:<22} | {results['GRU']['predicted_sentiment']:<22}")
    print(f"{'정답 일치 여부':<23} | {'일치 (O)' if results['LSTM']['is_correct'] else '불일치 (X)':<22} | {'일치 (O)' if results['GRU']['is_correct'] else '불일치 (X)':<22}")
    print("=" * 78)

    # 시간 차이 분석 요약
    lstm_time = results['LSTM']['elapsed_time_sec']
    gru_time = results['GRU']['elapsed_time_sec']
    time_diff = abs(lstm_time - gru_time)
    faster_model = "GRU" if gru_time < lstm_time else "LSTM"
    percent_faster = (time_diff / max(lstm_time, gru_time)) * 100

    print(f"\n[시간 분석 요약] {faster_model}가 약 {time_diff:.2f}초 ({percent_faster:.1f}%) 더 빠르게 학습을 마쳤습니다.")
    print(" (GRU는 Cell State와 Hidden State를 하나로 통합하고 게이트 수를 줄여 구조가 더 단순하기 때문입니다.)\n")

    print("-" * 78)
    print(f"100번째 리뷰 본문 일부:\n{decoded_test_review[:350]} ...")
    print("-" * 78)

    # 결과 JSON 저장
    final_output = {
        "test_position": TEST_POSITION,
        "movie_title": MOVIE_TITLE,
        "actual_label": actual_label,
        "actual_sentiment": sentiment_name(actual_label),
        "comparison": results,
    }
    (output_dir / "lstm_vs_gru_comparison.json").write_text(
        json.dumps(final_output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n비교 결과 저장 완료: {output_dir / 'lstm_vs_gru_comparison.json'}")


if __name__ == "__main__":
    main()
