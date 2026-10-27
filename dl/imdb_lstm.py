"""Keras IMDB 데이터로 LSTM을 학습하고 100번째 테스트 리뷰를 예측한다.

실행:
    uv run python imdb_lstm.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
KERAS_HOME = ROOT / "data" / "imdb"
os.environ["KERAS_HOME"] = str(KERAS_HOME)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences


TEST_POSITION = 100
TEST_INDEX = TEST_POSITION - 1
MOVIE_TITLE = "제공되지 않음 (Keras IMDB 데이터셋에 영화 제목 필드가 없음)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-words", type=int, default=10_000, help="사용할 최대 단어 수")
    parser.add_argument("--maxlen", type=int, default=300, help="리뷰의 고정 토큰 길이")
    parser.add_argument("--embedding-dim", type=int, default=64, help="임베딩 차원")
    parser.add_argument("--lstm-units", type=int, default=64, help="LSTM 은닉 유닛 수")
    parser.add_argument("--epochs", type=int, default=5, help="학습 epoch 수")
    parser.add_argument("--batch-size", type=int, default=128, help="미니배치 크기")
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="학습 데이터 일부만 사용할 때의 개수(생략 시 전체 25,000개)",
    )
    parser.add_argument("--seed", type=int, default=42, help="난수 시드")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_words < 5:
        raise ValueError("--num-words는 5 이상이어야 합니다.")
    if args.maxlen < 1 or args.embedding_dim < 1 or args.lstm_units < 1:
        raise ValueError("--maxlen, --embedding-dim, --lstm-units는 1 이상이어야 합니다.")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs와 --batch-size는 1 이상이어야 합니다.")
    if args.train_limit is not None and args.train_limit < 2:
        raise ValueError("--train-limit는 2 이상이어야 합니다.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def decode_review(tokens: list[int]) -> str:
    word_index = imdb.get_word_index()
    reverse_index = {index + 3: word for word, index in word_index.items()}
    reverse_index.update({0: "[PAD]", 1: "[START]", 2: "[OOV]", 3: "[UNUSED]"})
    return " ".join(reverse_index.get(token, "[UNKNOWN]") for token in tokens)


def build_model(args: argparse.Namespace) -> tf.keras.Model:
    model = Sequential(
        [
            tf.keras.Input(shape=(args.maxlen,), dtype="int32"),
            Embedding(input_dim=args.num_words, output_dim=args.embedding_dim, mask_zero=True),
            LSTM(args.lstm_units),
            Dense(1, activation="sigmoid"),
        ],
        name="imdb_lstm",
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
    validate_args(args)
    set_seed(args.seed)

    output_dir = ROOT / "output" / "imdb_lstm"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"IMDB 데이터 위치: {KERAS_HOME / 'datasets'}")
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

    x_train_padded = pad_sequences(
        x_train, maxlen=args.maxlen, padding="post", truncating="post"
    )
    x_test_padded = pad_sequences(
        x_test, maxlen=args.maxlen, padding="post", truncating="post"
    )

    model = build_model(args)
    model.summary()
    history = model.fit(
        x_train_padded,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        shuffle=True,
        verbose=2,
    )

    test_loss, test_accuracy = model.evaluate(
        x_test_padded, y_test, batch_size=args.batch_size, verbose=0
    )
    positive_probability = float(
        model.predict(x_test_padded[TEST_INDEX : TEST_INDEX + 1], verbose=0)[0, 0]
    )
    predicted_label = int(positive_probability >= 0.5)
    actual_label = int(y_test[TEST_INDEX])

    result = {
        "model": "LSTM",
        "test_position": TEST_POSITION,
        "python_index": TEST_INDEX,
        "movie_title": MOVIE_TITLE,
        "predicted_probability_positive": positive_probability,
        "predicted_label": predicted_label,
        "predicted_sentiment": sentiment_name(predicted_label),
        "actual_label": actual_label,
        "actual_sentiment": sentiment_name(actual_label),
        "is_correct": predicted_label == actual_label,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "review_token_count": len(original_test_review),
        "review_preview": decoded_test_review[:500],
    }

    model.save(output_dir / "imdb_lstm.keras")
    (output_dir / "history.json").write_text(
        json.dumps(history.history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "test_100th_prediction.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 64)
    print(f"테스트 데이터의 {TEST_POSITION}번째 샘플 (Python index={TEST_INDEX})")
    print(f"영화 제목 : {MOVIE_TITLE}")
    print(f"예측값     : {predicted_label} ({sentiment_name(predicted_label)})")
    print(f"긍정 확률  : {positive_probability:.6f}")
    print(f"실제값     : {actual_label} ({sentiment_name(actual_label)})")
    print(f"비교 결과  : {'일치' if predicted_label == actual_label else '불일치'}")
    print(f"전체 정확도: {test_accuracy:.4f} / 전체 손실: {test_loss:.4f}")
    print(f"리뷰 일부  : {decoded_test_review[:300]}...")
    print("=" * 64)
    print(f"모델 및 결과 저장 위치: {output_dir}")


if __name__ == "__main__":
    main()
