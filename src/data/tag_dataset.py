"""
TAG Dataset Processing Module

이 모듈은 TAGbank 형식의 데이터를 읽고 처리하는 기능을 제공합니다.
Transformer 모델 훈련에 사용할 수 있는 형태로 데이터를 변환합니다.
"""

import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional, Union


class TAGDataset(Dataset):
    """TAGbank 형식 데이터를 처리하는 PyTorch 데이터셋 클래스"""
    
    def __init__(self, 
                 data_path: str, 
                 vocab=None, 
                 max_length: int = 128, 
                 use_mwe: bool = False):
        """
        TAGbank 데이터셋 초기화
        
        Args:
            data_path: TAGbank 파일 경로
            vocab: 사전 정의된 어휘 사전 (없을 경우 데이터에서 구축)
            max_length: 최대 문장 길이
            use_mwe: 다중어 표현(MWE) 사용 여부
        """
        self.data_path = data_path
        self.max_length = max_length
        self.use_mwe = use_mwe
        
        # 데이터 로드
        self.sentences = []
        self.tag_info = []
        self._load_data()
        
        # 어휘 사전 구축
        self.vocab = vocab if vocab else self._build_vocab()
        
        # 텐서로 변환
        self.encoded_data = self._encode_data()
    
    def _load_data(self):
        """TAGbank 파일에서 데이터 로드"""
        current_sentence = []
        current_tag_info = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                
                # 주석이나 빈 줄 건너뛰기
                if line.startswith('#') or not line:
                    if current_sentence and current_tag_info:
                        self.sentences.append(current_sentence)
                        self.tag_info.append(current_tag_info)
                        current_sentence = []
                        current_tag_info = []
                    continue
                
                # 헤더 행 건너뛰기
                if line.startswith('IDX'):
                    continue
                
                parts = line.split()
                if len(parts) < 7:
                    continue
                
                idx, lex, pos, hd, elem, rhs, lhs = parts[:7]
                
                # MWE 처리 (예: "5-6" 형태의 인덱스)
                if self.use_mwe and '-' in idx:
                    # MWE 정보를 저장하고 건너뛰기 (각 토큰은 개별적으로 처리됨)
                    continue
                
                # 일반 토큰 처리
                current_sentence.append(lex)
                current_tag_info.append({
                    'idx': int(idx) if idx.isdigit() else idx,
                    'pos': pos,
                    'head': int(hd) if hd.isdigit() else 0,
                    'elem_type': elem,  # alpha 또는 beta
                    'rhs': rhs if rhs != '_' else '',
                    'lhs': lhs if lhs != '_' else ''
                })
        
        # 마지막 문장 추가
        if current_sentence and current_tag_info:
            self.sentences.append(current_sentence)
            self.tag_info.append(current_tag_info)
    
    def _build_vocab(self):
        """데이터셋에서 어휘 사전 구축"""
        vocab = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
        token_idx = 4
        
        # 어휘 추출
        for sentence in self.sentences:
            for token in sentence:
                if token.lower() not in vocab:
                    vocab[token.lower()] = token_idx
                    token_idx += 1
        
        # POS 태그 추가
        for tag_info_list in self.tag_info:
            for tag_info in tag_info_list:
                pos_tag = f"POS_{tag_info['pos']}"
                if pos_tag not in vocab:
                    vocab[pos_tag] = token_idx
                    token_idx += 1
        
        # 요소 유형 추가 (alpha/beta)
        if 'ELEM_alpha' not in vocab:
            vocab['ELEM_alpha'] = token_idx
            token_idx += 1
        if 'ELEM_beta' not in vocab:
            vocab['ELEM_beta'] = token_idx
            token_idx += 1
            
        return vocab
    
    def _encode_data(self):
        """데이터를 텐서로 인코딩"""
        encoded_data = []
        
        for i, (sentence, tag_info) in enumerate(zip(self.sentences, self.tag_info)):
            # 토큰 인코딩
            token_ids = [self.vocab.get(token.lower(), self.vocab['<unk>']) for token in sentence]
            token_ids = token_ids[:self.max_length]
            
            # 패딩
            if len(token_ids) < self.max_length:
                token_ids = token_ids + [self.vocab['<pad>']] * (self.max_length - len(token_ids))
            
            # POS 태그 인코딩
            pos_tags = [self.vocab.get(f"POS_{info['pos']}", self.vocab['<unk>']) 
                        for info in tag_info]
            pos_tags = pos_tags[:self.max_length]
            if len(pos_tags) < self.max_length:
                pos_tags = pos_tags + [self.vocab['<pad>']] * (self.max_length - len(pos_tags))
            
            # 요소 유형 인코딩 (alpha/beta)
            elem_types = []
            for info in tag_info:
                if info['elem_type'] == 'alpha':
                    elem_types.append(self.vocab['ELEM_alpha'])
                elif info['elem_type'] == 'beta':
                    elem_types.append(self.vocab['ELEM_beta'])
                else:
                    elem_types.append(self.vocab['<unk>'])
            
            elem_types = elem_types[:self.max_length]
            if len(elem_types) < self.max_length:
                elem_types = elem_types + [self.vocab['<pad>']] * (self.max_length - len(elem_types))
            
            # 헤드 정보 인코딩 (의존 관계)
            heads = [info['head'] for info in tag_info]
            heads = heads[:self.max_length]
            if len(heads) < self.max_length:
                heads = heads + [0] * (self.max_length - len(heads))
            
            # 구문 구조 정보 추출
            # RHS/LHS 태그에서 구문 구조 특성 추출
            syntax_features = self._extract_syntax_features(tag_info)
            syntax_features = syntax_features[:self.max_length]
            if len(syntax_features) < self.max_length:
                syntax_features = syntax_features + [[0, 0, 0]] * (self.max_length - len(syntax_features))
            
            # 텐서로 변환
            encoded_data.append({
                'token_ids': torch.tensor(token_ids, dtype=torch.long),
                'pos_tags': torch.tensor(pos_tags, dtype=torch.long),
                'elem_types': torch.tensor(elem_types, dtype=torch.long),
                'heads': torch.tensor(heads, dtype=torch.long),
                'syntax_features': torch.tensor(syntax_features, dtype=torch.long),
                'length': min(len(sentence), self.max_length)
            })
        
        return encoded_data

    def _extract_syntax_features(self, tag_info_list):
        """구문 구조 정보에서 특성 추출"""
        features = []
        
        for info in tag_info_list:
            # RHS에서 구문 깊이 정보 추출
            rhs = info.get('rhs', '')
            depth = rhs.count('(')
            
            # 노드 타입 정보
            is_np = 1 if 'NP' in rhs else 0
            is_vp = 1 if 'VP' in rhs else 0
            
            features.append([depth, is_np, is_vp])
        
        return features
    
    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, idx):
        return self.encoded_data[idx]


def create_tagbank_dataloaders(
    train_path: str,
    valid_path: Optional[str] = None,
    test_path: Optional[str] = None,
    batch_size: int = 32,
    max_length: int = 128,
    use_mwe: bool = False
) -> Dict[str, DataLoader]:
    """
    TAGbank 데이터로 데이터 로더 생성
    
    Args:
        train_path: 훈련 데이터 경로
        valid_path: 검증 데이터 경로 (없을 경우 훈련 데이터의 10% 사용)
        test_path: 테스트 데이터 경로
        batch_size: 배치 크기
        max_length: 최대 문장 길이
        use_mwe: 다중어 표현(MWE) 사용 여부
        
    Returns:
        데이터 로더 딕셔너리 {'train', 'valid', 'test'}
    """
    # 훈련 데이터셋 로드
    train_dataset = TAGDataset(train_path, max_length=max_length, use_mwe=use_mwe)
    vocab = train_dataset.vocab
    
    dataloaders = {
        'train': DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
    }
    
    # 검증 데이터셋
    if valid_path:
        valid_dataset = TAGDataset(valid_path, vocab=vocab, max_length=max_length, use_mwe=use_mwe)
        dataloaders['valid'] = DataLoader(
            valid_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
    
    # 테스트 데이터셋
    if test_path:
        test_dataset = TAGDataset(test_path, vocab=vocab, max_length=max_length, use_mwe=use_mwe)
        dataloaders['test'] = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
    
    return dataloaders


# 사용 예시
if __name__ == "__main__":
    # 샘플 데이터 경로
    data_path = "../../data/sample/sample.tag"
    
    # 데이터셋 로드
    dataset = TAGDataset(data_path, max_length=64)
    
    # 첫 번째 샘플 확인
    sample = dataset[0]
    print(f"토큰 수: {sample['length']}")
    print(f"토큰 ID: {sample['token_ids'][:sample['length']]}")
    print(f"POS 태그: {sample['pos_tags'][:sample['length']]}")
    print(f"요소 타입: {sample['elem_types'][:sample['length']]}")
    print(f"헤드 정보: {sample['heads'][:sample['length']]}")
    
    # 데이터 로더 생성
    dataloaders = create_tagbank_dataloaders(data_path, batch_size=2)
    
    # 첫 번째 배치 확인
    for batch in dataloaders['train']:
        print(f"배치 크기: {batch['token_ids'].shape}")
        break
