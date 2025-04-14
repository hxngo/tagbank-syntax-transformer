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
from collections import namedtuple
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.checkpoint import checkpoint



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
    
        # 구문 특성 처리를 위한 레이어들
        self.syntax_feature_proj = nn.Sequential(
            nn.Linear(3, d_model // 4),  # 3차원 특성을 d_model//4 차원으로 확장
            nn.LayerNorm(d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model)  # 최종적으로 d_model 차원으로 확장
        )
    
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
    
    def forward(self, token_ids, pos_tags=None, elem_types=None, syntax_features=None):
        """
        입력 텐서들을 임베딩으로 변환합니다.
        
        Args:
            token_ids (torch.Tensor): 토큰 ID 텐서 [batch_size, seq_len]
            pos_tags (torch.Tensor, optional): 품사 태그 텐서 [batch_size, seq_len]
            elem_types (torch.Tensor, optional): 구문 요소 타입 텐서 [batch_size, seq_len]
            syntax_features (torch.Tensor, optional): 구문 특성 텐서 [batch_size, seq_len, 3]
        """
        # 데이터 타입 변환
        token_ids = token_ids.long()
        if pos_tags is not None:
            pos_tags = pos_tags.long()
        if elem_types is not None:
            elem_types = elem_types.long()
        if syntax_features is not None:
            syntax_features = syntax_features.float()
        
        # 각 임베딩 계산
        token_emb = self.token_embedding(token_ids) * math.sqrt(self.d_model)
        pos_emb = self.pos_tag_embedding(pos_tags) if pos_tags is not None else 0
        elem_emb = self.elem_type_embedding(elem_types) if elem_types is not None else 0
        
        # 구문 특성 변환
        if syntax_features is not None:
            # [batch_size, seq_len, 3] -> [batch_size, seq_len, d_model]
            syntax_emb = self.syntax_feature_proj(syntax_features)
        else:
            syntax_emb = 0
        
        # 모든 임베딩 결합
        combined_emb = token_emb + pos_emb + elem_emb + syntax_emb
        
        # 위치 인코딩 추가
        seq_len = token_ids.size(1)
        combined_emb = combined_emb + self.positional_encoding[:, :seq_len, :]
        
        # 정규화 및 드롭아웃
        combined_emb = self.layer_norm(combined_emb)
        combined_emb = self.dropout(combined_emb)
        
        return combined_emb


class TAGTreeEncoding(nn.Module):
    """TAG 트리 구조를 인코딩하는 모듈"""
    
    def __init__(self, d_model, max_depth=20):
        super().__init__()
        self.d_model = d_model
        self.max_depth = max_depth
        
        # 대체(substitution) 노드 임베딩
        self.substitution_embedding = nn.Embedding(2, d_model // 4)  # 0: 아님, 1: 대체 노드
        
        # 접합(adjoining) 노드 임베딩
        self.adjoining_embedding = nn.Embedding(2, d_model // 4)  # 0: 아님, 1: 접합 노드
        
        # 상대적 위치 인코딩 (트리에서의 상대적 위치)
        self.relative_pos_embedding = nn.Embedding(11, d_model // 4)  # -5~+5 범위
        
        # 특성 결합 레이어
        self.feature_fusion = nn.Linear(d_model // 4 * 3, d_model)
        
    def forward(self, subst_nodes, adj_nodes, rel_positions):
        """
        TAG 트리 특성 인코딩
        
        Args:
            subst_nodes: 대체 노드 표시 [batch_size, seq_len]
            adj_nodes: 접합 노드 표시 [batch_size, seq_len]
            rel_positions: 상대적 트리 위치 [batch_size, seq_len]
            
        Returns:
            인코딩된 TAG 트리 특성 [batch_size, seq_len, d_model]
        """
        # 대체 노드 임베딩
        subst_emb = self.substitution_embedding(subst_nodes)
        
        # 접합 노드 임베딩
        adj_emb = self.adjoining_embedding(adj_nodes)
        
        # 상대적 위치 임베딩 (5를 더해 0~10 범위로 변환)
        rel_pos_shifted = (rel_positions + 5).clamp(0, 10)
        rel_pos_emb = self.relative_pos_embedding(rel_pos_shifted)
        
        # 특성 결합
        combined = torch.cat([subst_emb, adj_emb, rel_pos_emb], dim=-1)
        return self.feature_fusion(combined)


class EnhancedTAGAttention(nn.Module):
    """향상된 TAG 구조 정보를 활용한 어텐션 메커니즘"""
    
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
        
        # 가중치 초기화 스케일 조정
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # 선형 변환 레이어
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        # TAG 구조 정보를 위한 추가 가중치
        self.tag_biases = nn.Parameter(torch.zeros(n_heads, 1, 1))
        self.relation_type_weights = nn.Parameter(torch.zeros(4))  # 4가지 관계 타입
        self.relation_scale = nn.Parameter(torch.tensor(1.0))
        
        # 가중치 초기화
        nn.init.xavier_uniform_(self.tag_biases)
        nn.init.normal_(self.relation_type_weights, mean=0.0, std=0.02)
        
        # 레이어 정규화 추가
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None, 
                tag_relations: Optional[torch.Tensor] = None):
        """
        어텐션 계산 수행
        
        Args:
            query: 쿼리 텐서 [batch_size, q_len, d_model]
            key: 키 텐서 [batch_size, k_len, d_model]
            value: 값 텐서 [batch_size, k_len, d_model]
            mask: 어텐션 마스크 [batch_size, 1, 1, seq_len]
            tag_relations: TAG 관계 텐서 [batch_size, seq_len, seq_len, 4]
        """
        batch_size, q_len, _ = query.shape
        k_len = key.shape[1]

        # 입력 정규화
        query = self.layer_norm(query)
        key = self.layer_norm(key)
        value = self.layer_norm(value)

        # 선형 변환 및 헤드 분할
        q = self.q_proj(query).view(batch_size, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, k_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, k_len, self.n_heads, self.head_dim).transpose(1, 2)

        # 스케일드 닷-프로덕트 어텐션
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # TAG 구조 기반 어텐션 바이어스 추가
        if tag_relations is not None:
            # 관계 행렬 차원 확인
            assert tag_relations.dim() == 4, f"Expected 4D tensor for tag_relations, got {tag_relations.dim()}D"

            # 관계 가중치 정규화 및 스케일링
            relation_weights = F.softmax(self.relation_type_weights, dim=0)
            weighted_relations = torch.sum(tag_relations * relation_weights.view(1, 1, 1, -1), dim=-1)
            relation_bias = weighted_relations.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
            
            # 바이어스 스케일링 조정
            relation_scale = torch.sigmoid(self.relation_scale)  # 0~1 범위로 제한
            scores = scores + relation_bias * relation_scale * 0.1  # 영향력 감소

        # 마스킹 적용
        if mask is not None:
            # 마스크 차원 확인
            assert mask.dim() == 4, f"마스크 차원이 잘못되었습니다: {mask.shape}"
            # 마스크를 헤드 수만큼 확장 [batch_size, 1, 1, seq_len] -> [batch_size, n_heads, q_len, seq_len]
            mask = mask.expand(-1, self.n_heads, q_len, -1)
            scores = scores.masked_fill(mask == 0, -1e9)  # -inf 대신 큰 음수 사용

        # 어텐션 가중치 계산 및 드롭아웃
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 어텐션 출력 계산
        attn_output = torch.matmul(attn_weights, v)
        
        # 차원 변환 및 출력 투영
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        attn_output = self.o_proj(attn_output)
        
        # 출력 정규화
        attn_output = self.dropout(attn_output)

        return attn_output, attn_weights


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
        
        # 향상된 TAG 어텐션 레이어
        self.tag_attn = EnhancedTAGAttention(d_model, n_heads, dropout)
        
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
                attention_mask: Optional[torch.Tensor] = None, 
                tag_relations: Optional[torch.Tensor] = None):
        """
        인코더 레이어 전방 패스
        
        Args:
            x: 입력 텐서 [batch_size, seq_len, d_model]
            attention_mask: 어텐션 마스크 [batch_size, 1, 1, seq_len]
            tag_relations: TAG 관계 텐서 [batch_size, seq_len, seq_len, 4]
        """
        # Pre-LN 구조 적용
        # 1. 셀프 어텐션
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.tag_attn(
            query=x,
            key=x,
            value=x,
            mask=attention_mask,  # 여기서는 mask로 전달 (EnhancedTAGAttention에서 처리)
            tag_relations=tag_relations
        )
        x = residual + self.dropout(attn_out)
        
        # 2. 피드포워드
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ff(x))
        
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
                 dropout: float = 0.1,
                 use_checkpoint: bool = False):  # 체크포인팅 옵션 추가
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
            use_checkpoint: 체크포인팅 옵션
        """
        super().__init__()
        
        self.use_checkpoint = use_checkpoint
        
        # 구문 정보를 포함하는 임베딩 레이어
        self.embedding = SyntaxAwareEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_length=max_length,
            dropout=dropout
        )
        
        # TAG 트리 구조 인코더
        self.tree_encoder = TAGTreeEncoding(d_model)
        
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
                attention_mask: Optional[torch.Tensor] = None,
                pos_tags: Optional[torch.Tensor] = None, 
                elem_types: Optional[torch.Tensor] = None, 
                syntax_features: Optional[torch.Tensor] = None, 
                tag_relations: Optional[torch.Tensor] = None,
                subst_nodes: Optional[torch.Tensor] = None,
                adj_nodes: Optional[torch.Tensor] = None,
                rel_positions: Optional[torch.Tensor] = None):
        """
        인코더 전방 패스
        
        Args:
            token_ids: 토큰 ID 텐서 [batch_size, seq_len]
            attention_mask: 어텐션 마스크 [batch_size, seq_len]
            pos_tags: POS 태그 텐서 (선택적)
            elem_types: 구문 요소 타입 텐서 (선택적)
            syntax_features: 구문 특성 텐서 (선택적)
            tag_relations: TAG 관계 텐서 (선택적)
            subst_nodes: 대체 노드 표시 (선택적)
            adj_nodes: 접합 노드 표시 (선택적)
            rel_positions: 상대적 트리 위치 (선택적)
        """
        # 구문 정보를 포함한 임베딩
        x = self.embedding(token_ids, pos_tags, elem_types, syntax_features)
        
        # TAG 트리 구조 인코딩 (선택적)
        if subst_nodes is not None and adj_nodes is not None and rel_positions is not None:
            tree_encoding = self.tree_encoder(subst_nodes, adj_nodes, rel_positions)
            x = self.norm(x + tree_encoding)  # 잔차 연결 및 정규화
        
        # 어텐션 마스크 처리
        if attention_mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        
        # 체크포인팅을 사용한 인코더 레이어 처리
        for i, layer in enumerate(self.layers):
            if self.training and self.use_checkpoint and i > 0:
                # 체크포인팅 함수 정의
                def checkpoint_fn(x_in, attention_mask_in, tag_relations_in):
                    return layer(x_in, attention_mask=attention_mask_in, tag_relations=tag_relations_in)
                
                # 체크포인팅 적용
                layer_output = checkpoint(
                    checkpoint_fn,
                    x, attention_mask, tag_relations,
                    use_reentrant=False  # 재진입 비활성화
                )
            else:
                layer_output = layer(x, attention_mask=attention_mask, tag_relations=tag_relations)
            
            # 잔차 연결 및 정규화
            x = self.norm(x + layer_output)
        
        return x


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
                 num_classes: int = 2,
                 use_checkpoint: bool = False):
        super().__init__()
        
        self.encoder = SyntaxTransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_encoder_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_length=max_length,
            dropout=dropout,
            use_checkpoint=use_checkpoint
        )
        
        # 개선된 분류 헤드
        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, 
                token_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                pos_tags: Optional[torch.Tensor] = None, 
                elem_types: Optional[torch.Tensor] = None, 
                syntax_features: Optional[torch.Tensor] = None, 
                tag_relations: Optional[torch.Tensor] = None,
                subst_nodes: Optional[torch.Tensor] = None,
                adj_nodes: Optional[torch.Tensor] = None,
                rel_positions: Optional[torch.Tensor] = None):
        """
        모델의 순전파 수행
        
        Args:
            token_ids: 토큰 ID 텐서 [batch_size, seq_len]
            attention_mask: 어텐션 마스크 [batch_size, seq_len]
            pos_tags: POS 태그 텐서 (선택적)
            elem_types: 구문 요소 타입 텐서 (선택적)
            syntax_features: 구문 특성 텐서 (선택적)
            tag_relations: TAG 관계 텐서 (선택적)
            subst_nodes: 대체 노드 표시 (선택적)
            adj_nodes: 접합 노드 표시 (선택적)
            rel_positions: 상대적 트리 위치 (선택적)
        """
        # 인코더 출력
        enc_output = self.encoder(
            token_ids=token_ids,
            attention_mask=attention_mask,
            pos_tags=pos_tags,
            elem_types=elem_types,
            syntax_features=syntax_features,
            tag_relations=tag_relations,
            subst_nodes=subst_nodes,
            adj_nodes=adj_nodes,
            rel_positions=rel_positions
        )
        
        # 전체 시퀀스의 평균 표현 계산
        if attention_mask is not None:
            # 마스크를 사용하여 패딩되지 않은 토큰의 평균만 계산
            mask = attention_mask.float()
            pooled = (enc_output * mask.unsqueeze(-1)).sum(dim=1)
            pooled = pooled / mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        else:
            pooled = enc_output.mean(dim=1)
        
        # 풀링된 표현 변환
        pooled = self.pooler(pooled)
        
        # 정규화 및 분류
        pooled = self.norm(pooled)
        logits = self.classifier(pooled)
        
        return logits


def create_pad_mask(seq, pad_idx=0):
    """패딩 마스크 생성"""
    return (seq != pad_idx).unsqueeze(-2)


def init_weights(module):
    """가중치 초기화 함수"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def create_syntax_aware_transformer(
    vocab_size: int,
    d_model: int = 768,
    n_encoder_layers: int = 6,
    n_heads: int = 12,
    d_ff: int = 3072,
    max_length: int = 512,
    dropout: float = 0.1,
    num_classes: int = 2,
    use_checkpoint: bool = False  # 체크포인팅 옵션 추가
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
        use_checkpoint: 체크포인팅 옵션
        
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
        num_classes=num_classes,
        use_checkpoint=use_checkpoint
    )
    
    # 가중치 초기화 적용
    model.apply(init_weights)
    
    return model

def _parse_tag_tree(self, entry: str):
    """TAG 트리 파싱"""
    # 구현할 로직: TAG 형식 텍스트를 트리 구조로 파싱
    # 여기서는 실제 파싱 로직 대신 간단한 구조체 반환 (실제 구현 필요)
    TAGTree = namedtuple('TAGTree', ['tokens', 'pos_tags', 'elem_types', 'heads'])

def _extract_label_from_tag_entry(self, entry: str) -> int:
    """TAG 엔트리에서 레이블 추출"""
    # 구현할 로직: 엔트리에서 레이블 추출
    # 현재는 더미 값 반환 (실제 구현 필요)
    complexity_score = (
        0.3 * (max_depth / 10) +  # 최대 깊이 (0-1 범위로 정규화)
        0.3 * (avg_depth / 5) +   # 평균 깊이
        0.2 * (np_vp_count / len(lines)) +  # NP/VP 밀도
        0.2 * (clause_count / len(lines))   # 절 밀도
    )
    return complexity_score

def _build_vocab(self) -> Dict[str, int]:
    """어휘 사전 구축"""
    # 구현할 로직: 모든 문장에서 고유 토큰 추출하여 어휘 구축
    # 현재는 더미 어휘 반환 (실제 구현 필요)
    return {
        '<pad>': 0,
        '<unk>': 1
    }

def train_with_validation(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scheduler: optim.lr_scheduler._LRScheduler,
    device: torch.device,
    output_dir: str,
    epochs: int = 10  # 너무 적은 에포크
):
    # 구현할 로직: 학습 및 검증 과정 구현
    pass