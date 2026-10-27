"""
IMDB 영화 리뷰 데이터셋 (또는 Fashion-MNIST 데이터셋)
순수 SimpleRNN 알고리즘을 이용한 학습 및 테스트 데이터 100번째 예측/비교 스크립트

- 사용 알고리즘: 순수 RNN (layers.SimpleRNN)
- 테스트 데이터 100번째 샘플(Index: 99)에 대한 예측 확률, 긍정/부정 판별 및 실제 라벨 비교
- 디코딩된 리뷰 텍스트 및 언급된 영화 정보 출력
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences

def main():
    print("=" * 75)
    print(" 1. IMDB 영화 리뷰 데이터 로드 및 전처리")
    print("=" * 75)

    VOCAB_SIZE = 10000      # 가장 빈번하게 등장하는 상위 10,000개 단어 사용
    MAX_LEN = 200           # 각 리뷰의 길이를 200개 단어로 통일 (Padding/Truncating)
    EMBEDDING_DIM = 64      # 단어 임베딩 차원

    # 데이터 다운로드 및 로드
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

    print(f"학습 데이터 개수  : {len(x_train):,}개")
    print(f"테스트 데이터 개수: {len(x_test):,}개")

    # 시퀀스 길이 맞추기 (Zero-padding)
    x_train_pad = pad_sequences(x_train, maxlen=MAX_LEN, padding='post', truncating='post')
    x_test_pad = pad_sequences(x_test, maxlen=MAX_LEN, padding='post', truncating='post')

    print(f"전처리 후 입력 Shape: {x_train_pad.shape}")

    # ============================================================
    # 2. 순수 SimpleRNN 기반 딥러닝 모델 구성
    # ============================================================
    print("\n" + "=" * 75)
    print(" 2. 순수 SimpleRNN 신경망 모델 구축")
    print("=" * 75)

    model = models.Sequential([
        # 1) Embedding 레이어: 단어 정수 ID -> 64차원 밀집 벡터 변환
        layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LEN, name='embedding'),
        
        # 2) 순수 RNN (SimpleRNN) 레이어: 은닉 상태 노드 수 64
        layers.SimpleRNN(64, activation='tanh', name='simple_rnn'),
        
        # 3) 과적합 방지를 위한 Dropout
        layers.Dropout(0.3, name='dropout'),
        
        # 4) 이진 분류를 위한 Dense 레이어 (Sigmoid: 0.0 ~ 1.0)
        layers.Dense(1, activation='sigmoid', name='output_classifier')
    ])

    model.summary()

    # 모델 컴파일
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # ============================================================
    # 3. 모델 학습 진행
    # ============================================================
    print("\n" + "=" * 75)
    print(" 3. 순수 SimpleRNN 모델 학습 진행 (3 Epochs)")
    print("=" * 75)

    history = model.fit(
        x_train_pad, y_train,
        epochs=3,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

    # 전체 테스트 데이터셋 평가
    test_loss, test_acc = model.evaluate(x_test_pad, y_test, verbose=0)
    print(f"\n[전체 테스트 데이터셋 성능] Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc * 100:.2f}%")

    # ============================================================
    # 4. 테스트 데이터 100번째 값(인덱스: 99) 예측 및 비교
    # ============================================================
    target_idx = 99  # 100번째 샘플 (0-based indexing: 100 - 1)
    sample_input = np.expand_dims(x_test_pad[target_idx], axis=0)
    actual_label = y_test[target_idx]

    # 모델 예측 수행 (0.0 ~ 1.0의 긍정 확률)
    prediction_prob = model.predict(sample_input, verbose=0)[0][0]
    predicted_label = 1 if prediction_prob >= 0.5 else 0
    confidence = prediction_prob * 100 if predicted_label == 1 else (1 - prediction_prob) * 100

    actual_text = "긍정 (Positive: 1)" if actual_label == 1 else "부정 (Negative: 0)"
    pred_text = "긍정 (Positive: 1)" if predicted_label == 1 else "부정 (Negative: 0)"

    # ============================================================
    # 5. 리뷰 텍스트 디코딩 및 영화 정보 확인
    # ============================================================
    word_index = imdb.get_word_index()
    reverse_word_index = {v + 3: k for k, v in word_index.items()}
    reverse_word_index[0] = "<PAD>"
    reverse_word_index[1] = "<START>"
    reverse_word_index[2] = "<UNK>"
    reverse_word_index[3] = "<UNUSED>"

    decoded_review = ' '.join([reverse_word_index.get(i, '?') for i in x_test[target_idx]])

    print("\n" + "=" * 75)
    print(f"       테스트 데이터 [100번째] (Index: {target_idx}) 리뷰 예측 및 분석 결과")
    print("=" * 75)
    print(f" ▶ 실제 정답 (Actual)    : [{actual_label}] {actual_text}")
    print(f" ▶ 예측 결과 (Predicted) : [{predicted_label}] {pred_text} (신뢰도: {confidence:.2f}%)")
    print(f" ▶ 예측된 긍정 확률      : {prediction_prob * 100:.2f}%")
    print(f" ▶ 예측 일치 여부        : {'정답 (O)' if actual_label == predicted_label else '오답 (X)'}")
    print("-" * 75)
    print(f" ▶ 100번째 리뷰의 본문 텍스트:")
    print(decoded_review)
    print("=" * 75)

if __name__ == "__main__":
    main()
