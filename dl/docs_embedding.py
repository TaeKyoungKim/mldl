import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def main():
    docs = [
        '너무 재밌네요',
        '최고예요',
        '참 잘 만든 영화예요',
        '추천하고 싶은 영화입니다.',
        '한 번 더 보고싶네요',
        '글쎄요',
        '별로예요',
        '생각보다 지루하네요',
        '연기가 어색해요',
        '재미없어요'
    ]

    print("=" * 70)
    print(" 1. 텍스트 토큰화 및 정수 인코딩 (TextVectorization)")
    print("=" * 70)

    # 문장들 중 가장 긴 문장의 단어 수가 5개이므로 길이를 5로 고정
    MAX_SEQUENCE_LENGTH = 5

    # 1) 텍스트 정수 시퀀스 변환 레이어
    vectorizer = layers.TextVectorization(
        output_mode='int',
        output_sequence_length=MAX_SEQUENCE_LENGTH
    )
    # 어휘집(Vocabulary) 학습
    vectorizer.adapt(docs)
    vocab = vectorizer.get_vocabulary()
    vocab_size = len(vocab)

    print(f"총 단어 사전 크기 (Vocab Size): {vocab_size}개")
    print(f"단어 사전: {vocab}\n")

    # 문장들을 정수 시퀀스 텐서로 변환
    int_sequences = vectorizer(docs)
    print("[정수 인코딩 결과 (Shape: [10, 5])]")
    for doc, seq in zip(docs, int_sequences.numpy()):
        print(f"'{doc:20s}' -> {seq}")

    # ============================================================
    # 2. TensorFlow Embedding 레이어 적용
    # ============================================================
    print("\n" + "=" * 70)
    print(" 2. Embedding 레이어를 통한 밀집 벡터(Dense Vector) 변환")
    print("=" * 70)

    # 각 단어 정수를 4차원의 연속형 실수 벡터로 임베딩 (output_dim = 4)
    EMBEDDING_DIM = 4

    embedding_layer = layers.Embedding(
        input_dim=vocab_size,          # 어휘 수 (사전 크기)
        output_dim=EMBEDDING_DIM,      # 각 단어를 표현할 임베딩 차원 수
        input_length=MAX_SEQUENCE_LENGTH,
        name='embedding_layer'
    )

    # Embedding 레이어 통과
    # 입력 shape: (10 문장, 5 단어) -> 출력 shape: (10 문장, 5 단어, 4 임베딩차원)
    embedded_vectors = embedding_layer(int_sequences)

    print(f"임베딩 출력 텐서 Shape: {embedded_vectors.shape}")
    print(f"(배치 크기: {embedded_vectors.shape[0]}개 문장, 시퀀스 길이: {embedded_vectors.shape[1]}개 단어, 임베딩 차원: {embedded_vectors.shape[2]}차원)\n")

    # ============================================================
    # 3. 각 문장별 임베딩 벡터 결과 출력
    # ============================================================
    print("[첫 번째 문장 임베딩 벡터 세부 예시]")
    first_doc = docs[0]
    first_seq = int_sequences[0].numpy()
    first_emb = embedded_vectors[0].numpy()

    print(f"문장: '{first_doc}'")
    for word_idx, token_id in enumerate(first_seq):
        word_text = vocab[token_id] if token_id < len(vocab) else "[UNK]"
        vec_str = np.array2string(first_emb[word_idx], precision=4, suppress_small=True)
        print(f"  단어 [{word_text:6s}] (ID: {token_id}) -> 임베딩 벡터: {vec_str}")

    print("\n" + "-" * 70)
    print("문장 단위 요약 벡터 (Global Average Pooling 적용 시 문장 대표 벡터):")
    # 각 단어 임베딩의 평균을 내어 문장 1개를 4차원 벡터 1개로 요약 표현
    doc_vectors = layers.GlobalAveragePooling1D()(embedded_vectors).numpy()
    for doc, vec in zip(docs, doc_vectors):
        vec_str = np.array2string(vec, precision=4, suppress_small=True)
        print(f"'{doc:20s}' -> {vec_str}")
    print("=" * 70)

if __name__ == "__main__":
    main()
