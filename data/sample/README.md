# 샘플 TAGbank 데이터

이 디렉토리는 TAGbank 형식으로 작성된 샘플 데이터를 포함하고 있습니다. TAGbank는 Tree-Adjoining Grammar(TAG) 유도 트리(derivation tree)를 기반으로 하는 말뭉치로, 문장의 구문적 구조를 TAG 형식으로 표현합니다.

## 파일 설명

- `sample.tag`: 기본 TAGbank 형식의 샘플 데이터
- `sample_mwe.tag`: 다중어 표현(MWE)을 포함한 TAGbank 형식의 샘플 데이터

## TAGbank 형식

TAGbank 형식은 다음 필드를 포함합니다:

- IDX: 토큰 인덱스
- LEX: 표면 어휘 항목
- POS: Penn Treebank POS 태그
- HD: 의존 규칙에 따른 구문적 헤드
- ELEM: TAG 기본 트리 타입 (α: 초기 트리, β: 보조 트리)
- RHS, LHS: 괄호로 표시된 PTB 스타일의 구성 요소 정보

## 예시

```
IDX LEX POS HD ELEM RHS LHS
1 pierre nnp 2 beta (S (S (NP-SBJ (NP-SBJ (NP (NP _
2 viken nnp 9 alpha _ )NP
3 , punct 2 beta _ )NP
...
```

## 사용 방법

이 샘플 데이터는 TAG 구문 구조를 활용한 Transformer 모델 학습에 사용될 수 있습니다. 데이터 로드 및 처리 방법은 `src/data` 디렉토리의 스크립트를 참조하세요.
