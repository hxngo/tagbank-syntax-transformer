# TAGbank for Syntax-aware Transformer

이 저장소는 TAGbank를 활용한 Syntax-aware Transformer 모델의 성능 향상 연구를 위한 데이터셋과 코드를 포함하고 있습니다.

## 개요

Tree-Adjoining Grammar(TAG)는 확장된 로컬리티 도메인을 제공하여 자연어의 문법적 구조를 효과적으로 표현할 수 있는 형식 문법입니다. 이 프로젝트는 TAGbank 형식의 데이터를 생성하고, 이를 활용하여 Transformer 모델이 문장의 구문적 정보를 더 잘 학습할 수 있도록 하는 방법을 연구합니다.

## 데이터셋

본 저장소에는 다음과 같은 데이터셋이 포함되어 있습니다:

1. `data/sample` - 샘플 TAGbank 형식 데이터
2. `data/mini_tagbank` - 소규모 TAGbank 데이터셋 (연구 목적)
3. `data/penn_to_tag` - Penn Treebank에서 변환된 TAGbank 형식 데이터 예시

## 모델

Syntax-aware Transformer는 기존 Transformer 모델에 문법적 구조 정보를 주입하여 자연어 이해 능력을 향상시키는 모델입니다. 이 저장소의 코드는 다음 기능을 포함합니다:

1. TAG 유도 트리 처리 및 변환 코드
2. 구문 정보를 인코딩하는 임베딩 레이어
3. 구문 구조를 활용한 어텐션 메커니즘 
4. 훈련 및 평가 스크립트

## 사용 방법

자세한 사용 방법은 각 디렉토리의 README를 참조하세요.

## 인용

이 연구를 인용할 때는 다음 형식을 사용해 주세요:

```
홍성관. (2025). TAGbank를 활용한 Syntax-aware Transformer 모델의 성능 향상 연구. 인공지능공학과, 제주한라대학교.
```

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.
