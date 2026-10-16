"""
Google AI Studio Embeddings API를 사용하여 영화 리뷰 문장들을 임베딩하는 스크립트

1) 공식 Google GenAI SDK (google-genai) 사용 방식
2) requests 라이브러리를 통한 REST API 호출 방식 (SDK 미설치 환경에서도 즉시 실행 가능)
두 가지 방식을 모두 지원하도록 구현되었습니다.
"""

import os
import requests
import numpy as np

# Google 공식 임베딩 문서(ai.google.dev/gemini-api/docs/embeddings) 기준 모델
# - gemini-embedding-001: 텍스트 임베딩 표준 모델
# - gemini-embedding-2: 최신 멀티모달 임베딩 모델
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

def embed_texts_with_rest_api(texts, api_key):
    """
    Google AI Studio REST API(batchEmbedContents) 엔드포인트를 직접 호출하여 임베딩 추출
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:batchEmbedContents?key={api_key}"
    
    # batchEmbedContents 페이로드 구성
    requests_payload = [
        {
            "model": f"models/{MODEL_NAME}",
            "content": {"parts": [{"text": text}]}
        }
        for text in texts
    ]

    response = requests.post(
        url,
        json={"requests": requests_payload},
        headers={"Content-Type": "application/json"}
    )

    if response.status_code != 200:
        raise RuntimeError(f"API 호출 실패 (Status {response.status_code}): {response.text}")

    data = response.json()
    embeddings = [item["values"] for item in data.get("embeddings", [])]
    return embeddings

def embed_texts_with_sdk(texts, api_key):
    """
    신규 공식 google-genai SDK를 사용한 임베딩 호출 방식
    """
    from google import genai
    client = genai.Client(api_key=api_key)
    response = client.models.embed_content(
        model=MODEL_NAME,
        contents=texts
    )
    embeddings = [e.values for e in response.embeddings]
    return embeddings

def main():
    print("=" * 75)
    print(" Google AI Studio (Gemini text-embedding-004) 문장 임베딩")
    print("=" * 75)

    # 환경 변수 GEMINI_API_KEY 확인
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[안내] GEMINI_API_KEY 환경변수가 설정되어 있지 않습니다.")
        api_key = input("Google AI Studio API Key를 입력하세요: ").strip()
        if not api_key:
            print("API Key가 입력되지 않아 종료합니다.")
            return

    print(f"\n대상 문장 수: {len(docs)}개")
    print(f"사용 모델: {MODEL_NAME}")
    print("임베딩 요청 전송 중...")

    # SDK 설치 여부에 따라 유연하게 호출
    try:
        embeddings = embed_texts_with_sdk(docs, api_key)
        print("(구글 공식 google-genai SDK로 호출 성공)")
    except ImportError:
        embeddings = embed_texts_with_rest_api(docs, api_key)
        print("(REST API batchEmbedContents 엔드포인트로 호출 성공)")

    embeddings = np.array(embeddings)

    print("\n" + "=" * 75)
    print(f" 임베딩 결과 요약 (전체 Shape: {embeddings.shape})")
    print(f" - 문장 개수: {embeddings.shape[0]}개")
    print(f" - 임베딩 차원: {embeddings.shape[1]}차원 (text-embedding-004 표준)")
    print("=" * 75)

    # 각 문장별 임베딩 벡터 요약 출력
    for idx, (doc, vec) in enumerate(zip(docs, embeddings)):
        # 768차원 중 앞쪽 5개, 뒤쪽 3개 값만 미리보기
        head_vals = ", ".join([f"{v:+.4f}" for v in vec[:5]])
        tail_vals = ", ".join([f"{v:+.4f}" for v in vec[-3:]])
        vec_preview = f"[{head_vals}, ..., {tail_vals}]"
        
        # L2 Norm(벡터 크기) 출력 (대체로 1.0으로 정규화되어 있음)
        norm = np.linalg.norm(vec)
        print(f"\n[{idx+1:2d}] 문장: '{doc}'")
        print(f"     벡터 미리보기 (768차원): {vec_preview}")
        print(f"     벡터 L2 크기 (Norm)    : {norm:.4f}")

    print("\n" + "=" * 75)
    print("모든 문장의 임베딩 벡터 추출이 완료되었습니다.")
    print("=" * 75)

if __name__ == "__main__":
    main()
