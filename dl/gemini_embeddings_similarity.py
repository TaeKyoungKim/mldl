"""
Google AI Studio (gemini-embedding-001) 임베딩 벡터를 기반으로
10개 영화 리뷰 문장 간의 코사인 유사도(Cosine Similarity) 및
유클리디안 거리/유사도(Euclidean Similarity)를 계산하고
비교 분석 및 시각화하는 스크립트
"""

import os
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# 한글 폰트 및 마이너스 기호 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Google 공식 문서 기준 임베딩 모델
MODEL_NAME = "gemini-embedding-001"

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

def embed_texts(texts, api_key):
    """
    Google AI Studio batchEmbedContents REST API를 호출하여 문장 임베딩 벡터 리스트 반환
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:batchEmbedContents?key={api_key}"
    payload = {
        "requests": [
            {
                "model": f"models/{MODEL_NAME}",
                "content": {"parts": [{"text": text}]}
            }
            for text in texts
        ]
    }
    response = requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    if response.status_code != 200:
        raise RuntimeError(f"API 호출 실패 ({response.status_code}): {response.text}")

    data = response.json()
    embeddings = [item["values"] for item in data.get("embeddings", [])]
    return np.array(embeddings)

def main():
    print("=" * 75)
    print(" Google Gemini 임베딩 기반 문장 간 코사인 유사도(Cosine Similarity) 분석")
    print("=" * 75)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[안내] GEMINI_API_KEY 환경변수가 설정되어 있지 않습니다.")
        api_key = input("Google AI Studio API Key를 입력하세요: ").strip()
        if not api_key:
            print("API Key가 입력되지 않아 종료합니다.")
            return

    print(f"\n1. {len(docs)}개 문장에 대해 '{MODEL_NAME}' 임베딩 벡터 추출 중...")
    embeddings = embed_texts(docs, api_key)
    print(f"   -> 임베딩 완료! 텐서 Shape: {embeddings.shape} (10개 문장 × {embeddings.shape[1]}차원)")

    # ============================================================
    # 2. 코사인 유사도 (Cosine Similarity) 10x10 계산
    # ============================================================
    # ============================================================
    # 2. 유사도 계산 (코사인 유사도 vs 유클리디안 거리/유사도)
    # ============================================================
    print("\n2. 코사인 유사도 및 유클리디안 유사도 계산 중...")
    
    # 1) 코사인 유사도 (방향성 비교, 범위: -1 ~ 1, 1에 가까울수록 유사)
    cos_sim_matrix = cosine_similarity(embeddings)

    # 2) 유클리디안 거리 (공간상 직선 거리, 0에 가까울수록 유사)
    euclidean_dist_matrix = euclidean_distances(embeddings)
    
    # 3) 유클리디안 유사도 변환: 1 / (1 + distance) (범위: 0 ~ 1, 1에 가까울수록 유사)
    euclidean_sim_matrix = 1.0 / (1.0 + euclidean_dist_matrix)

    # ============================================================
    # 3. 콘솔 결과 출력 (두 가지 유사도 매트릭스 비교)
    # ============================================================
    print("\n" + "=" * 80)
    print(" [1] 10 × 10 코사인 유사도 매트릭스 (Cosine Similarity, 1.00 = 완전 일치)")
    print("=" * 80)
    print("     " + " ".join([f"[{i+1:2d}]" for i in range(len(docs))]))
    for i, row in enumerate(cos_sim_matrix):
        row_str = " ".join([f"{val:5.2f}" for val in row])
        print(f"[{i+1:2d}] {row_str}  | {docs[i]}")

    print("\n" + "=" * 80)
    print(" [2] 10 × 10 유클리디안 유사도 매트릭스 (1 / (1 + dist), 1.00 = 거리 0)")
    print("=" * 80)
    print("     " + " ".join([f"[{i+1:2d}]" for i in range(len(docs))]))
    for i, row in enumerate(euclidean_sim_matrix):
        row_str = " ".join([f"{val:5.2f}" for val in row])
        print(f"[{i+1:2d}] {row_str}  | {docs[i]}")

    # ============================================================
    # 4. 주요 유사도 문장 쌍 비교 분석
    # ============================================================
    print("\n" + "-" * 80)
    print(" [주요 문장 쌍 비교 분석]")
    print("-" * 80)

    # 코사인 기준 최상위/최하위
    np.fill_diagonal(cos_sim_matrix, -np.inf)
    cos_max = np.unravel_index(np.argmax(cos_sim_matrix), cos_sim_matrix.shape)
    cos_min = np.unravel_index(np.argmin(cos_sim_matrix), cos_sim_matrix.shape)
    print(f"▶ [코사인 유사도] 가장 유사한 쌍: '{docs[cos_max[0]]}' <-> '{docs[cos_max[1]]}' ({cos_sim_matrix[cos_max]:.4f})")
    print(f"▶ [코사인 유사도] 가장 먼 쌍    : '{docs[cos_min[0]]}' <-> '{docs[cos_min[1]]}' ({cos_sim_matrix[cos_min]:.4f})")

    # 유클리디안 기준 최상위/최하위
    np.fill_diagonal(euclidean_sim_matrix, -np.inf)
    euc_max = np.unravel_index(np.argmax(euclidean_sim_matrix), euclidean_sim_matrix.shape)
    euc_min = np.unravel_index(np.argmin(euclidean_sim_matrix), euclidean_sim_matrix.shape)
    print(f"▶ [유클리디안 유사도] 가장 가까운 쌍: '{docs[euc_max[0]]}' <-> '{docs[euc_max[1]]}' (유사도: {euclidean_sim_matrix[euc_max]:.4f}, 거리: {euclidean_dist_matrix[euc_max]:.4f})")
    print(f"▶ [유클리디안 유사도] 가장 먼 쌍    : '{docs[euc_min[0]]}' <-> '{docs[euc_min[1]]}' (유사도: {euclidean_sim_matrix[euc_min]:.4f}, 거리: {euclidean_dist_matrix[euc_min]:.4f})")
    print("-" * 80 + "\n")

    # 대각선 원상 복구 (히트맵용)
    np.fill_diagonal(cos_sim_matrix, 1.0)
    np.fill_diagonal(euclidean_sim_matrix, 1.0)

    # ============================================================
    # 5. 코사인 vs 유클리디안 유사도 히트맵(Heatmap) 나란히 시각화
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(20, 8.5))
    labels = [f"[{i+1}] {doc}" for i, doc in enumerate(docs)]

    # (좌측) 코사인 유사도 히트맵
    sns.heatmap(
        cos_sim_matrix,
        ax=axes[0],
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        cbar_kws={'label': '코사인 유사도'}
    )
    axes[0].set_title(f"1. 코사인 유사도 (Cosine Similarity)\n[방향/각도 기반]", fontsize=12, fontweight='bold', pad=12)
    axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    axes[0].set_yticklabels(labels, rotation=0, fontsize=9)

    # (우측) 유클리디안 유사도 히트맵
    sns.heatmap(
        euclidean_sim_matrix,
        ax=axes[1],
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        cbar_kws={'label': '유클리디안 유사도 (1 / (1 + Dist))'}
    )
    axes[1].set_title(f"2. 유클리디안 유사도 (Euclidean Similarity)\n[직선 거리 기반: 1 / (1 + distance)]", fontsize=12, fontweight='bold', pad=12)
    axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    axes[1].set_yticklabels(labels, rotation=0, fontsize=9)

    plt.suptitle(f"Google Gemini ({MODEL_NAME}) 임베딩 기반 문장 유사도 비교 분석", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
