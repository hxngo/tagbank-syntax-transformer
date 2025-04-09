"""
Syntax-aware Transformer 모델

이 모듈은 TAG 구문 정보를 활용하는 Transformer 모델을 구현합니다.
기존 Transformer 아키텍처를 확장하여 TAG 유도 트리의 구조적 정보를 주입합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Optional, Tuple, List, Union


class SyntaxAwareEmbedding(nn.Module):
    """TAG 구문 정보를 포함하는 임베딩 레이어"""
    
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int = 512, 
                 max_length: int = 512,
                 dropout: float = 0.1):
        """
        구문 정보를 포함하는 임베딩 레이어 초기화
        
        Args:
            vocab_size: 어휘 크기
            d_model: 모델 차원
            max_length: 최대 시퀀스 길이
            dropout: 드롭아웃 비율
        """
        super().__init__()
        self.d_model = d_model
        
        # 토큰 임베딩
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 위치 인코딩
        self.positional_encoding = nn.Parameter(
            self._create_positional_encoding(max_length, d_model),
            requires_grad=False
        )
        
        # TAG 요소 타입 임베딩 (alpha/beta)
        self.elem_type_embedding = nn.Embedding(3, d_model)  # <pad>, alpha, beta
        
        # POS 태그 임베딩
        self.pos_tag_embedding = nn.Embedding(vocab_size, d_model)
        
        # 구문 깊이 임베딩
        self.syntax_depth_embedding = nn.Embedding(20, d_model // 4)  # 최대 구문 깊이 20
        
        # 구문 유형 임베딩 (NP/VP)
        self.np_embedding = nn.Embedding(2, d_model // 4)  # 0: not NP, 1: NP
        self.vp_embedding = nn.Embedding(2, d_model // 4)  # 0: not VP, 1: VP
        
        # 구문 특성 투영 레이어
        self.syntax_projection = nn.Linear(d_model // 2, d_model)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def _create_positional_encoding(self, max_length, d_model):
        """위치 인코딩 생성"""
        pos_enc = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        
        return pos_enc.unsqueeze(0)
    
    def forward(self, 
                token_ids: torch.Tensor, 
                pos_tags: torch.Tensor, 
                elem_types: torch.Tensor, 
                syntax_features: torch.Tensor):
        """
        임베딩 전방 패스
        
        Args:
            token_ids: 토큰 ID 텐서 [batch_size, seq_len]
            pos_tags: POS 태그 ID 텐서 [batch_size, seq_len]
            elem_types: 요소 타입 텐서 [batch_size, seq_len]
            syntax_features: 구문 특성 텐서 [batch_size, seq_len, 3]
                - 첫 번째 차원: 구문 트리 깊이
                - 두 번째 차원: NP 여부 (0/1)
                - 세 번째 차원: VP 여부 (0/1)
                
        Returns:
            구문 정보가 주입된 임베딩 텐서 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = token_ids.shape
        
        # 토큰 임베딩
        token_emb = self.token_embedding(token_ids) * math.sqrt(self.d_model)
        
        # 위치 인코딩
        position_enc = self.positional_encoding[:, :seq_len, :]
        
        # POS 태그 임베딩
        pos_emb = self.pos_tag_embedding(pos_tags)
        
        # 요소 타입 임베딩
        elem_emb = self.elem_type_embedding(elem_types)
        
        # 구문 특성 임베딩
        depth_emb = self.syntax_depth_embedding(syntax_features[:, :, 0])
        np_emb = self.np_embedding(syntax_features[:, :, 1])
        vp_emb = self.vp_embedding(syntax_features[:, :, 2])
        
        # 구문 특성 결합
        syntax_emb = torch.cat([depth_emb, np_emb, vp_emb], dim=-1)
        syntax_emb = self.syntax_projection(syntax_emb)
        
        # 모든 임베딩 결합
        combined_emb = token_emb + position_enc + pos_emb + elem_emb + syntax_emb
        
        return self.dropout(self.layer_norm(combined_emb))


class TAGAttention(nn.Module):
    """TAG 구조 정보를 활용한 어텐션 메커니즘"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        TAG 어텐션 초기화
        
        Args:
            d_model: 모델 차원
            n_heads: 어텐션 헤드 수
            dropout: 드롭아웃 비율
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"
        
        # 표준 멀티헤드 어텐션 레이어
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        # TAG 구조 정보를 위한 추가 가중치
        self.tag_biases = nn.Parameter(torch.zeros(n_heads, 1, 1))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None, 
                heads: Optional[torch.Tensor] = None):
        """
        TAG 어텐션 전방 패스
        
        Args:
            query: 쿼리 텐서 [batch_size, q_len, d_model]
            key: 키 텐서 [batch_size, k_len, d_model]
            value: 값 텐서 [batch_size, v_len, d_model]
            mask: 어텐션 마스크 [batch_size, q_len, k_len]
            heads: 의존 관계 헤드 정보 [batch_size, seq_len]
                
        Returns:
            어텐션 출력과 어텐션 가중치
        """
        batch_size, q_len, _ = query.shape
        k_len = key.shape[1]
        
        # 선형 변환
        q = self.q_proj(query).view(batch_size, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, k_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, k_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 스케일드 닷-프로덕트 어텐션
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # TAG 구조 기반 어텐션 바이어스 추가
        if heads is not None:
            # 의존 관계 헤드 정보를 바탕으로 어텐션 바이어스 생성
            tag_bias = self._create_tag_bias(heads, q_len, k_len, batch_size)
            scores = scores + tag_bias.unsqueeze(1)  # 헤드 차원 추가
        
        # 마스킹 적용
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        # 어텐션 가중치 계산
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 어텐션 적용
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).reshape(batch_size, q_len, self.d_model)
        
        return self.o_proj(output), attn_weights
    
    def _create_tag_bias(self, heads, q_len, k_len, batch_size):
        """TAG 의존 관계 정보를 바탕으로 어텐션 바이어스 생성"""
        device = heads.device
        tag_bias = torch.zeros(batch_size, q_len, k_len, device=device)
        
        # 각 토큰의 헤드(의존 관계 대상) 토큰에 바이어스 추가
        for b in range(batch_size):
            for i in range(q_len):
                if i < heads.shape[1]:  # 패딩 방지
                    head_idx = heads[b, i].item()
                    if 0 < head_idx < k_len:  # 유효한 헤드 인덱스인 경우
                        tag_bias[b, i, head_idx] = self.tag_biases[0, 0, 0]  # 모든 헤드에 동일 바이어스 적용
        
        return tag_bias


class SyntaxTransformerEncoderLayer(nn.Module):
    """TAG 구문 정보를 활용하는 Transformer 인코더 레이어"""
    
    def __init__(self, 
                 d_model: int, 
                 n_heads: int, 
                 d_ff: int = 2048, 
                 dropout: float = 0.1):
        """
        인코더 레이어 초기화
        
        Args:
            d_model: 모델 차원
            n_heads: 어텐션 헤드 수
            d_ff: 피드포워드 네트워크 차원
            dropout: 드롭아웃 비율
        """
        super().__init__()
        
        # TAG 어텐션 레이어
        self.tag_attn = TAGAttention(d_model, n_heads, dropout)
        
        # 피드포워드 네트워크
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # 정규화 및 드롭아웃
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None, 
                heads: Optional[torch.Tensor] = None):
        """
        인코더 레이어 전방 패스
        
        Args:
            x: 입력 텐서 [batch_size, seq_len, d_model]
            mask: 어텐션 마스크 [batch_size, seq_len, seq_len]
            heads: 의존 관계 헤드 정보 [batch_size, seq_len]
                
        Returns:
            인코더 레이어 출력
        """
        # 셀프 어텐션
        attn_out, _ = self.tag_attn(
            query=self.norm1(x),
            key=self.norm1(x),
            value=self.norm1(x),
            mask=mask,
            heads=heads
        )
        
        # 잔차 연결 및 정규화
        x = x + self.dropout(attn_out)
        
        # 피드포워드 네트워크
        ff_out = self.ff(self.norm2(x))
        
        # 잔차 연결
        x = x + self.dropout(ff_out)
        
        return x


class SyntaxTransformerEncoder(nn.Module):
    """TAG 구문 정보를 활용하는 Transformer 인코더"""
    
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int = 512, 
                 n_layers: int = 6, 
                 n_heads: int = 8, 
                 d_ff: int = 2048, 
                 max_length: int = 512, 
                 dropout: float = 0.1):
        """
        인코더 초기화
        
        Args:
            vocab_size: 어휘 크기
            d_model: 모델 차원
            n_layers: 인코더 레이어 수
            n_heads: 어텐션 헤드 수
            d_ff: 피드포워드 네트워크 차원
            max_length: 최대 시퀀스 길이
            dropout: 드롭아웃 비율
        """
        super().__init__()
        
        # 구문 정보를 포함하는 임베딩 레이어
        self.embedding = SyntaxAwareEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_length=max_length,
            dropout=dropout
        )
        
        # 인코더 레이어 스택
        self.layers = nn.ModuleList([
            SyntaxTransformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, 
                token_ids: torch.Tensor, 
                pos_tags: torch.Tensor, 
                elem_types: torch.Tensor, 
                syntax_features: torch.Tensor, 
                heads: torch.Tensor, 
                mask: Optional[torch.Tensor] = None):
        """
        인코더 전방 패스
        
        Args:
            token_ids: 토큰 ID 텐서 [batch_size, seq_len]
            pos_tags: POS 태그 ID 텐서 [batch_size, seq_len]
            elem_types: 요소 타입 텐서 [batch_size, seq_len]
            syntax_features: 구문 특성 텐서 [batch_size, seq_len, 3]
            heads: 의존 관계 헤드 정보 [batch_size, seq_len]
            mask: 어텐션 마스크 [batch_size, seq_len, seq_len]
                
        Returns:
            인코더 출력
        """
        # 구문 정보를 포함한 임베딩
        x = self.embedding(token_ids, pos_tags, elem_types, syntax_features)
        
        # 인코더 레이어 스택 통과
        for layer in self.layers:
            x = layer(x, mask, heads)
        
        return self.norm(x)


class SyntaxAwareTransformer(nn.Module):
    """TAG 구문 정보를 활용하는 Transformer 모델"""
    
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int = 512, 
                 n_encoder_layers: int = 6, 
                 n_heads: int = 8, 
                 d_ff: int = 2048, 
                 max_length: int = 512, 
                 dropout: float = 0.1, 
                 num_classes: int = 2):
        """
        모델 초기화
        
        Args:
            vocab_size: 어휘 크기
            d_model: 모델 차원
            n_encoder_layers: 인코더 레이어 수
            n_heads: 어텐션 헤드 수
            d_ff: 피드포워드 네트워크 차원
            max_length: 최대 시퀀스 길이
            dropout: 드롭아웃 비율
            num_classes: 분류 클래스 수
        """
        super().__init__()
        
        # TAG 구문 정보를 활용하는 인코더
        self.encoder = SyntaxTransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_encoder_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_length=max_length,
            dropout=dropout
        )
        
        # 분류 헤드
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, 
                token_ids: torch.Tensor, 
                pos_tags: torch.Tensor, 
                elem_types: torch.Tensor, 
                syntax_features: torch.Tensor, 
                heads: torch.Tensor, 
                mask: Optional[torch.Tensor] = None):
        """
        모델 전방 패스
        
        Args:
            token_ids: 토큰 ID 텐서 [batch_size, seq_len]
            pos_tags: POS 태그 ID 텐서 [batch_size, seq_len]
            elem_types: 요소 타입 텐서 [batch_size, seq_len]
            syntax_features: 구문 특성 텐서 [batch_size, seq_len, 3]
            heads: 의존 관계 헤드 정보 [batch_size, seq_len]
            mask: 어텐션 마스크 [batch_size, seq_len, seq_len]
                
        Returns:
            클래스 로짓
        """
        # 인코더 출력
        enc_output = self.encoder(
            token_ids=token_ids,
            pos_tags=pos_tags,
            elem_types=elem_types,
            syntax_features=syntax_features,
            heads=heads,
            mask=mask
        )
        
        # [CLS] 토큰 위치(첫 번째 토큰)의 표현 사용
        cls_rep = enc_output[:, 0, :]
        
        # 분류
        logits = self.classifier(cls_rep)
        
        return logits


def create_pad_mask(seq, pad_idx=0):
    """패딩 마스크 생성"""
    return (seq != pad_idx).unsqueeze(-2)


def create_syntax_aware_transformer(
    vocab_size: int,
    d_model: int = 768,
    n_encoder_layers: int = 6,
    n_heads: int = 12,
    d_ff: int = 3072,
    max_length: int = 512,
    dropout: float = 0.1,
    num_classes: int = 2
) -> SyntaxAwareTransformer:
    """
    구문 정보를 활용하는 Transformer 모델 생성
    
    Args:
        vocab_size: 어휘 크기
        d_model: 모델 차원
        n_encoder_layers: 인코더 레이어 수
        n_heads: 어텐션 헤드 수
        d_ff: 피드포워드 네트워크 차원
        max_length: 최대 시퀀스 길이
        dropout: 드롭아웃 비율
        num_classes: 분류 클래스 수
        
    Returns:
        TAG 구문 정보를 활용하는 Transformer 모델
    """
    model = SyntaxAwareTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_encoder_layers=n_encoder_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_length=max_length,
        dropout=dropout,
        num_classes=num_classes
    )
    
    # 가중치 초기화
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return model


# 사용 예시
if __name__ == "__main__":
    # 모델 파라미터
    vocab_size = 10000
    batch_size = 2
    seq_len = 64
    d_model = 512
    
    # 샘플 입력 생성
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    pos_tags = torch.randint(0, vocab_size, (batch_size, seq_len))
    elem_types = torch.randint(0, 3, (batch_size, seq_len))  # <pad>, alpha, beta
    syntax_features = torch.randint(0, 10, (batch_size, seq_len, 3))
    heads = torch.randint(0, seq_len, (batch_size, seq_len))
    
    # 패딩 마스크
    mask = create_pad_mask(token_ids)
    
    # 모델 생성
    model = create_syntax_aware_transformer(
        vocab_size=vocab_size,
        num_classes=3  # 예: 감성 분석
    )
    
    # 모델 출력
    logits = model(token_ids, pos_tags, elem_types, syntax_features, heads, mask)
    print(f"출력 로짓 크기: {logits.shape}")  # [batch_size, num_classes]
