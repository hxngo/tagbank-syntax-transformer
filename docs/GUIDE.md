# TAGbank 활용 Syntax-aware Transformer 사용 가이드

이 문서는 TAGbank 데이터를 활용한 Syntax-aware Transformer 모델의 사용 방법을 설명합니다.

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [설치 방법](#설치-방법)
3. [데이터 준비](#데이터-준비)
4. [모델 학습](#모델-학습)
5. [모델 평가](#모델-평가)
6. [API 사용법](#api-사용법)
7. [연구 확장 방향](#연구-확장-방향)

## 프로젝트 개요

Tree-Adjoining Grammar(TAG)는 자연어의 구문 구조를 표현하는 강력한 형식 문법입니다. TAGbank는 기존 트리뱅크(예: Penn Treebank)에서 추출한 TAG 유도 트리(derivation tree) 말뭉치입니다. 이 프로젝트는 TAG 구문 정보를 Transformer 모델에 주입하여 자연어 이해 태스크의 성능을 향상시키는 방법을 연구합니다.

주요 특징:
- TAG 구문 구조를 활용한 구문 인식 어텐션 메커니즘
- 기본 트리 유형(초기 트리/보조 트리) 정보 활용
- 의존 관계와 구문 깊이 정보 임베딩

## 설치 방법

### 요구 사항
- Python 3.8 이상
- PyTorch 1.10 이상
- NLTK
- scikit-learn
- matplotlib
- tqdm

### 설치 과정

```bash
# 저장소 클론
git clone https://github.com/hxngo/tagbank-syntax-transformer.git
cd tagbank-syntax-transformer

# 가상 환경 생성 (선택 사항)
python -m venv venv
source venv/bin/activate  # Unix/Linux
# venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

## 데이터 준비

### 샘플 데이터 사용

프로젝트에 포함된 샘플 데이터를 사용하려면:

```bash
# 샘플 데이터 경로
export TRAIN_DATA=data/sample/sample.tag
export VALID_DATA=data/sample/sample.tag
```

### Penn Treebank에서 TAGbank 형식으로 변환

Penn Treebank 형식의 데이터를 TAGbank 형식으로 변환하려면:

```bash
# 변환 스크립트 실행
python scripts/convert_ptb_to_tag.py --input path/to/ptb_file.txt --output path/to/output.tag
```

### 데이터 형식

TAGbank 데이터 형식은 다음과 같은 필드로 구성됩니다:

- IDX: 토큰 인덱스
- LEX: 표면 어휘 항목
- POS: Penn Treebank POS 태그
- HD: 의존 관계에 따른 구문적 헤드 인덱스
- ELEM: TAG 기본 트리 타입 (α: 초기 트리, β: 보조 트리)
- RHS, LHS: 괄호로 표시된 PTB 스타일의 구성 요소 정보

예시:
```
IDX LEX POS HD ELEM RHS LHS
1 the dt 2 alpha (S (S (NP-SBJ (NP (DT _
2 researchers nns 3 alpha _ )NP )NP-SBJ
3 published vbd 0 alpha (VP _
4 their prp$ 5 alpha (NP (NP (PRP$ _
5 findings nns 3 alpha _ )NP
```

## 모델 학습

Syntax-aware Transformer 모델을 학습하려면:

```bash
# 기본 학습
python src/train.py --train_data $TRAIN_DATA --valid_data $VALID_DATA \
    --output_dir ./output --num_epochs 10

# 상세 옵션 지정
python src/train.py --train_data $TRAIN_DATA --valid_data $VALID_DATA \
    --output_dir ./output --num_epochs 20 --batch_size 16 \
    --d_model 768 --n_layers 8 --n_heads 12 --learning_rate 2e-5
```

주요 학습 옵션:
- `--train_data`: 학습 데이터 경로
- `--valid_data`: 검증 데이터 경로
- `--output_dir`: 출력 디렉토리
- `--num_epochs`: 학습 에폭 수
- `--batch_size`: 배치 크기
- `--d_model`: 모델 차원
- `--n_layers`: 인코더 레이어 수
- `--n_heads`: 어텐션 헤드 수
- `--learning_rate`: 학습률
- `--use_mwe`: 다중어 표현(MWE) 사용 여부

전체 옵션 목록은 `python src/train.py --help` 명령으로 확인할 수 있습니다.

## 모델 평가

학습된 모델을 평가하려면:

```bash
# 기본 평가
python src/evaluate.py --model_path ./output/best_model.pt \
    --test_data $TEST_DATA --output_dir ./evaluation

# 어텐션 분석 수행
python src/evaluate.py --model_path ./output/best_model.pt \
    --test_data $TEST_DATA --output_dir ./evaluation \
    --analyze_attention --plot_cm
```

주요 평가 옵션:
- `--model_path`: 모델 체크포인트 경로
- `--test_data`: 테스트 데이터 경로
- `--output_dir`: 출력 디렉토리
- `--analyze_attention`: 어텐션 패턴 분석 수행
- `--plot_cm`: 혼동 행렬 시각화

## API 사용법

학습된 모델을 Python 코드에서 사용하려면:

```python
import torch
from src.models.syntax_transformer import create_syntax_aware_transformer, create_pad_mask
from src.data.tag_dataset import TAGDataset

# 모델 로드
checkpoint = torch.load("path/to/model.pt", map_location=torch.device('cpu'))
vocab = checkpoint['vocab']
model = create_syntax_aware_transformer(
    vocab_size=len(vocab),
    num_classes=2  # 분류 클래스 수
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 데이터 로드 및 추론
dataset = TAGDataset("path/to/data.tag", vocab=vocab)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

for batch in dataloader:
    # 데이터 준비
    token_ids = batch['token_ids']
    pos_tags = batch['pos_tags']
    elem_types = batch['elem_types']
    syntax_features = batch['syntax_features']
    heads = batch['heads']
    mask = create_pad_mask(token_ids)
    
    # 추론
    with torch.no_grad():
        logits = model(token_ids, pos_tags, elem_types, syntax_features, heads, mask)
        predictions = torch.argmax(logits, dim=1)
        
    print(f"예측: {predictions.item()}")
```

## 연구 확장 방향

이 프로젝트는 다음과 같은 방향으로 확장할 수 있습니다:

1. **다국어 지원**: 한국어, 중국어 등 다양한 언어에 대한 TAGbank 구축 및 모델 훈련
2. **사전학습 모델 통합**: BERT, RoBERTa 등의 사전학습 모델과 TAG 구문 정보 결합
3. **의미 분석 강화**: 의미역 라벨링과 같은 의미 분석 태스크에 적용
4. **생성 모델 확장**: 텍스트 생성 모델에 TAG 구문 정보 주입
5. **다양한 NLP 태스크 적용**: 기계 번역, 요약, 대화 시스템 등에 적용

## 추가 자료

- [TAGbank 논문](https://arxiv.org/abs/2504.05226)
- [Tree-Adjoining Grammar 소개](https://en.wikipedia.org/wiki/Tree_adjoining_grammar)
- [Penn Treebank 문서](https://catalog.ldc.upenn.edu/LDC99T42)
