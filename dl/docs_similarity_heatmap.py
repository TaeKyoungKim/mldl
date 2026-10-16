import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

import tensorflow as tf
from tensorflow.keras import layers

# 한글 폰트 및 마이너스 기호 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 재현성을 위한 시드 고정
tf.random.set_seed(42)
np.random.seed(42)

def main():
    # 1. 10개 영화 리뷰 문장 정의
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
    print(" 1. 텍스트 토큰화 및 정수 인코딩")
    print("=" * 70)

    # 시퀀스 길이를 5로 고정
    MAX_SEQUENCE_LENGTH = 5
    vectorizer = layers.TextVectorization(
        output_mode='int',
        output_sequence_length=MAX_SEQUENCE_LENGTH
    )
    vectorizer.adapt(docs)
    vocab = vectorizer.get_vocabulary()
    int_sequences = vectorizer(docs)

    print(f"총 단어 사전 크기: {len(vocab)}개")
    print(f"단어 사전: {vocab}")

    # ============================================================
    # 2. Embedding + GlobalAveragePooling1D 로 문장 벡터 추출
    # ============================================================
    print("\n" + "=" * 70)
    print(" 2. Embedding -> GlobalAveragePooling1D (문장 벡터 생성)")
    print("=" * 70)

    EMBEDDING_DIM = 4

    # 1) Embedding 레이어 (각 단어 정수 -> 4차원 밀집 벡터)
    # mask_zero=True: 0번 패딩 토큰이 평균 계산에 영향을 주지 않도록 마스킹
    embedding_layer = layers.Embedding(
        input_dim=len(vocab),
        output_dim=EMBEDDING_DIM,
        mask_zero=True,
        name='embedding'
    )
    
    # 2) GlobalAveragePooling1D 레이어 (5개 단어 벡터 -> 1개 문장 대표 4차원 벡터)
    gap_layer = layers.GlobalAveragePooling1D(name='global_avg_pool')

    # 순전파 수행
    # (10, 5) -> (10, 5, 4) -> (10, 4)
    embedded = embedding_layer(int_sequences)
    sentence_vectors = gap_layer(embedded).numpy()

    print(f"단어별 임베딩 텐서 Shape : {embedded.shape} (10 문장 × 5 단어 × 4 차원)")
    print(f"GlobalAveragePooling1D 통과 후 Shape: {sentence_vectors.shape} (10 문장 × 4 차원 문장벡터)")

    # ============================================================
    # 3. 문장 간 10x10 코사인 유사도(Cosine Similarity) 계산
    # ============================================================
    print("\n" + "=" * 70)
    print(" 3. 10x10 코사인 유사도 매트릭스 계산")
    print("=" * 70)

    similarity_matrix = cosine_similarity(sentence_vectors)

    # 소수점 3자리로 콘솔에 매트릭스 출력
    print(np.round(similarity_matrix, 3))

    # ============================================================
    # 4. 10x10 유사도 히트맵(Heatmap) 시각화
    # ============================================================
    plt.figure(figsize=(11, 9))
    
    # 라벨 표시용 문장 (인덱스 + 문장)
    labels = [f"[{i+1}] {doc}" for i, doc in enumerate(docs)]

    # Seaborn 히트맵
    sns.heatmap(
        similarity_matrix,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,          # 각 셀에 유사도 수치 표시
        fmt=".2f",
        cmap="coolwarm",     # 유사도가 높으면 붉은색, 낮으면 푸른색
        vmin=-1.0,
        vmax=1.0,
        linewidths=0.5,
        cbar_kws={'label': '코사인 유사도 (Cosine Similarity)'}
    )

    plt.title("영화 리뷰 문장 간 코사인 유사도 매트릭스 (10×10 히트맵)\n[Embedding + GlobalAveragePooling1D]",
              fontsize=13, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=9.5)
    plt.yticks(rotation=0, fontsize=9.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
