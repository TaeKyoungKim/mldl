import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def main():
    # 1. 대상 기사 텍스트
    article = """
    배우 류혜영이 16년지기 절친 배우 고경표와의 관계를 둘러싼 열애 추측에 선을 그었다.
    류혜영은 지난 14일 서울 종로구 삼청로의 한 카페에서 영화 ‘킬링타임’ 개봉을 앞두고 인터뷰를 가졌다. 이날 류혜영은 고경표와 출연한 영상에 달린 댓글에 대해 “남녀 사이를 이렇게 생각하는 사람도 있구나 싶었다. 인간관계에 대해 공부가 됐다”고 밝혔고, 두 사람 사이에 대해서는 “청첩장 받을 일은 절대 없다”고 강조했다.
    """

    # 2. 문장 부호 및 특수기호 정제
    cleaned_text = re.sub(r"[^\w\s]", " ", article)

    # 3. CountVectorizer를 이용한 Bag of Words (BoW) 벡터화
    # (?u)\b\w+\b: 한 글자 단어(한, 그, 두 등)도 놓치지 않고 토큰화
    cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    bow_matrix = cv.fit_transform([cleaned_text])

    vocab = cv.vocabulary_
    bow_vector = bow_matrix.toarray()[0]
    feature_names = cv.get_feature_names_out()

    # 4. 단어 빈도 데이터프레임 구성
    df_bow = pd.DataFrame({
        '단어(Word)': feature_names,
        '출현 빈도(Frequency)': bow_vector
    }).sort_values(by='출현 빈도(Frequency)', ascending=False).reset_index(drop=True)

    # 5. 결과 출력
    print("=" * 70)
    print(" [배우 류혜영 인터뷰 기사 Bag of Words (BoW) 분석 결과]")
    print("=" * 70)
    print(f"총 고유 단어 수 (어휘 사전 크기): {len(vocab)}개")
    print(f"기사 내 총 단어 토큰 수: {bow_vector.sum()}개\n")

    print("[주요 다빈도 단어 (2회 이상 등장)]")
    multiple_counts = df_bow[df_bow['출현 빈도(Frequency)'] >= 2]
    print(multiple_counts.to_string(index=False))

    print("\n" + "-" * 70)
    print(f"Bag of Words 최종 수치 벡터 (1 × {len(bow_vector)} 차원):")
    print(bow_vector)

    print("-" * 70)
    print("\n[단어별 사전 인덱스 및 빈도수 상위 15개]")
    print(df_bow.head(15).to_string(index=False))
    print("=" * 70)

if __name__ == "__main__":
    main()
