"""
TAG 데이터셋 모듈

TAG 형식의 구문 분석 데이터를 처리하고 모델에서 사용할 수 있는 형태로 변환합니다.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
import os
from typing import Dict, List, Tuple, Optional, Union


class TAGDataProcessor:
    """TAG 데이터 처리 및 변환 클래스"""
    
    def __init__(self, max_seq_len=512):
        self.max_seq_len = max_seq_len
    
    def extract_tag_features(self, tag_tree):
        """TAG 트리에서 구문 특성 추출"""
        # 트리 구조 분석
        depth_features = self._compute_tree_depth(tag_tree)
        node_types = self._identify_node_types(tag_tree)
        subst_nodes = self._identify_substitution_nodes(tag_tree)
        adj_nodes = self._identify_adjoining_nodes(tag_tree)
        
        # 관계 행렬 생성 (대체, 접합, 부모-자식, 형제 관계)
        relation_matrix = self._create_relation_matrix(tag_tree)
        
        return {
            'depth': depth_features,
            'node_types': node_types,
            'subst_nodes': subst_nodes,
            'adj_nodes': adj_nodes,
            'relation_matrix': relation_matrix
        }
    
    def _compute_tree_depth(self, tag_tree):
        """트리 깊이 계산"""
        # 구현할 로직: 각 노드의 깊이 계산
        # 예시 구현 (실제 TAG 트리 구조에 맞게 수정 필요)
        depths = {}
        
        def _dfs(node, depth=0):
            depths[node.id] = depth
            for child in node.children:
                _dfs(child, depth + 1)
        
        _dfs(tag_tree.root)
        return depths
    
    def _identify_node_types(self, tag_tree):
        """노드 유형 파악 (NP, VP 등)"""
        # 구현할 로직: 각 노드의 구문 유형 식별
        # 예시 구현
        node_types = {}
        for node in tag_tree.all_nodes():
            if node.label.startswith('NP'):
                node_types[node.id] = {'NP': 1, 'VP': 0}
            elif node.label.startswith('VP'):
                node_types[node.id] = {'NP': 0, 'VP': 1}
            else:
                node_types[node.id] = {'NP': 0, 'VP': 0}
        return node_types
    
    def _identify_substitution_nodes(self, tag_tree):
        """대체 노드 식별"""
        # 구현할 로직: TAG의 대체 노드 식별
        # 예시 구현
        subst_nodes = {}
        for node in tag_tree.all_nodes():
            subst_nodes[node.id] = 1 if node.is_substitution_node() else 0
        return subst_nodes
    
    def _identify_adjoining_nodes(self, tag_tree):
        """접합 노드 식별"""
        # 구현할 로직: TAG의 접합 노드 식별
        # 예시 구현
        adj_nodes = {}
        for node in tag_tree.all_nodes():
            adj_nodes[node.id] = 1 if node.is_adjoining_node() else 0
        return adj_nodes
    
    def _create_relation_matrix(self, tag_tree):
        """관계 행렬 생성"""
        # 구현할 로직: 노드 간 관계를 표현하는 행렬 생성
        # 예시 구현 (4가지 관계 유형: 대체, 접합, 부모-자식, 형제)
        n_nodes = len(tag_tree.all_nodes())
        relation_matrix = torch.zeros(n_nodes, n_nodes, 4)
        
        # 각 관계 유형별로 행렬 채우기 
        # (실제 TAG 트리 구조에 맞게 수정 필요)
        for i, node_i in enumerate(tag_tree.all_nodes()):
            for j, node_j in enumerate(tag_tree.all_nodes()):
                # 대체 관계
                if node_i.is_substitution_site_for(node_j):
                    relation_matrix[i, j, 0] = 1.0
                
                # 접합 관계
                if node_i.is_adjoining_site_for(node_j):
                    relation_matrix[i, j, 1] = 1.0
                
                # 부모-자식 관계
                if node_j in node_i.children:
                    relation_matrix[i, j, 2] = 1.0
                
                # 형제 관계
                if node_i.parent == node_j.parent and i != j:
                    relation_matrix[i, j, 3] = 1.0
        
        return relation_matrix


class TAGDataset(Dataset):
    """TAG 형식 데이터를 처리하는 데이터셋 클래스"""
    
    def __init__(self, 
                 tag_file_path: str, 
                 vocab: Optional[Dict[str, int]] = None,
                 max_seq_len: int = 512):
        """
        TAG 데이터셋 초기화
        
        Args:
            tag_file_path: TAG 파일 경로
            vocab: 어휘 사전 (없으면 자동 생성)
            max_seq_len: 최대 시퀀스 길이
        """
        self.max_seq_len = max_seq_len
        self.processor = TAGDataProcessor(max_seq_len)
        
        # TAG 파일 로드 및 파싱
        self.tag_trees, self.sentences, self.labels = self._parse_tag_file(tag_file_path)
        
        # 어휘 구축
        self.vocab = vocab
        if self.vocab is None:
            self.vocab = self._build_vocab()
        
        # 토큰 -> ID 변환
        self.token_ids = self._tokenize_sentences()
        
        # TAG 구문 특성 추출
        self.syntax_features = self._extract_all_syntax_features()
        
        # 패딩 및 트리밍 적용
        self._apply_padding()
    
    def _parse_tag_file(self, file_path: str) -> Tuple[List, List[str], List[int]]:
        """TAG 파일 파싱"""
        tag_trees = []
        sentences = []
        labels = []
        
        # TAG 파일 형식에 맞는 파싱 로직 구현
        # 이 부분은 실제 TAG 파일 형식에 따라 구체적으로 구현해야 함
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 파일 내용을 문장 단위로 분리
            entries = re.split(r'\n\s*\n', content)
            
            for entry in entries:
                if not entry.strip():
                    continue
                
                # TAG 트리 파싱 및 추가 (실제 구현 필요)
                tag_tree = self._parse_tag_tree(entry)
                tag_trees.append(tag_tree)
                
                # 문장 추출
                sentence = self._extract_sentence_from_tag_tree(tag_tree)
                sentences.append(sentence)
                
                # 라벨 추출 (예: 감정 분석 라벨)
                label = self._extract_label_from_tag_entry(entry)
                labels.append(label)
        
        return tag_trees, sentences, labels
    
    def _parse_tag_tree(self, entry: str):
        """TAG 트리 파싱"""
        tokens = []
        pos_tags = []
        elem_types = []
        heads = []
        rhs_tags = []  # 오른쪽 구문 태그
        lhs_tags = []  # 왼쪽 구문 태그
        
        lines = entry.strip().split('\n')
        for line in lines:
            if line.startswith('#') or not line.strip():  # 주석이나 빈 줄 건너뛰기
                continue
            
            parts = line.strip().split()
            if len(parts) >= 7:  # IDX LEX POS HD ELEM RHS LHS 형식 확인
                tokens.append(parts[1].lower())  # 토큰을 소문자로 정규화
                pos_tags.append(parts[2])
                try:
                    heads.append(int(parts[3]))
                except ValueError:
                    heads.append(0)  # 파싱 실패시 기본값
                elem_types.append(1 if parts[4] == 'alpha' else 2)  # alpha=1, beta=2
                rhs_tags.append(parts[5])
                lhs_tags.append(parts[6])
        
        # 구문 깊이 계산
        depths = []
        for rhs in rhs_tags:
            depth = len([x for x in rhs.split() if x.startswith('(')])
            depths.append(depth)
        
        return {
            'tokens': tokens,
            'pos_tags': pos_tags,
            'elem_types': elem_types,
            'heads': heads,
            'depths': depths,
            'rhs_tags': rhs_tags,
            'lhs_tags': lhs_tags
        }
    
    def _extract_sentence_from_tag_tree(self, tag_tree):
        """TAG 트리에서 문장 추출"""
        return ' '.join(tag_tree['tokens'])
    
    def _extract_label_from_tag_entry(self, entry: str) -> int:
        """TAG 엔트리에서 레이블 추출 (문장 복잡도 기반)"""
        lines = [line for line in entry.split('\n') if not line.startswith('#') and line.strip()]
        if not lines:
            return 1  # 기본값: Medium
        
        # 구문 복잡도 지표 계산
        max_depth = 0
        total_depth = 0
        np_vp_count = 0
        clause_count = 0
        branching_factors = []
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 6:
                rhs_tags = parts[5]
                tags = rhs_tags.split()
                
                # 깊이 계산
                depth = len([x for x in tags if x.startswith('(')])
                max_depth = max(max_depth, depth)
                total_depth += depth
                
                # 구문 구조 분석
                np_vp = len([x for x in tags if x.startswith('(NP') or x.startswith('(VP')])
                np_vp_count += np_vp
                
                # 절 계산
                clauses = len([x for x in tags if x.startswith('(S') or x.startswith('(SBAR')])
                clause_count += clauses
                
                # 분기 계수 계산
                if depth > 0:
                    branching = (np_vp + clauses) / depth
                    branching_factors.append(branching)
        
        # 복잡도 점수 계산을 위한 정규화된 특성들
        avg_depth = total_depth / len(lines)
        avg_branching = sum(branching_factors) / len(branching_factors) if branching_factors else 0
        np_vp_density = np_vp_count / len(lines)
        clause_density = clause_count / len(lines)
        
        # 복잡도 임계값 설정
        depth_threshold = 5  # 중간 깊이 기준
        branching_threshold = 1.5  # 중간 분기 기준
        density_threshold = 0.8  # 중간 밀도 기준
        
        # 복잡도 분류 규칙
        if (max_depth <= 3 and avg_branching < 1.0 and 
            np_vp_density < 0.5 and clause_density < 0.3):
            return 0  # Simple
        elif (max_depth >= 7 or avg_branching > 2.0 or 
              np_vp_density > 1.2 or clause_density > 0.8):
            return 2  # Complex
        else:
            return 1  # Medium
    
    def _build_vocab(self) -> Dict[str, int]:
        """어휘 사전 구축"""
        vocab = {'<pad>': 0, '<unk>': 1}
        token_set = set()
        pos_set = set()
        
        # 토큰과 POS 태그 수집
        for tree in self.tag_trees:
            token_set.update(tree['tokens'])
            pos_set.update(tree['pos_tags'])
        
        # 토큰 사전 구축
        for i, token in enumerate(sorted(token_set)):
            vocab[token] = i + 2  # +2 for <pad> and <unk>
        
        # POS 태그 사전 구축
        for i, pos in enumerate(sorted(pos_set)):
            vocab[f'POS_{pos}'] = len(vocab)
        
        return vocab
    
    def _tokenize_sentences(self) -> List[List[int]]:
        """문장 토큰화"""
        tokenized = []
        for tree in self.tag_trees:
            # 토큰을 ID로 변환
            token_ids = []
            for token in tree['tokens']:
                token_ids.append(self.vocab.get(token, self.vocab['<unk>']))
            tokenized.append(token_ids)
        return tokenized
    
    def _extract_all_syntax_features(self) -> List[Dict]:
        """모든 문장의 구문 특성 추출"""
        features = []
        for tree in self.tag_trees:
            # 구문 깊이 정규화 (0-19 범위로)
            depths = [min(d, 19) for d in tree['depths']]
            
            # NP/VP 여부 확인
            np_flags = []
            vp_flags = []
            for rhs in tree['rhs_tags']:
                np_flags.append(1 if '(NP' in rhs else 0)
                vp_flags.append(1 if '(VP' in rhs else 0)
            
            # 구문 특성 텐서 생성 [seq_len, 3]
            syntax_tensor = torch.zeros(len(tree['tokens']), 3)
            syntax_tensor[:, 0] = torch.tensor(depths)
            syntax_tensor[:, 1] = torch.tensor(np_flags)
            syntax_tensor[:, 2] = torch.tensor(vp_flags)
            
            # 모든 필요한 특성을 포함하는 딕셔너리 생성
            features.append({
                'pos_tags': [self.vocab.get(f'POS_{pos}', self.vocab['<unk>']) for pos in tree['pos_tags']],
                'elem_types': tree['elem_types'],
                'heads': tree['heads'],
                'depths': depths,
                'np_flags': np_flags,
                'vp_flags': vp_flags,
                'syntax_features': syntax_tensor
            })
        
        return features
    
    def _apply_padding(self):
        """시퀀스에 패딩 적용"""
        for i in range(len(self.token_ids)):
            # 토큰 ID 패딩/트리밍
            if len(self.token_ids[i]) > self.max_seq_len:
                self.token_ids[i] = self.token_ids[i][:self.max_seq_len]
            else:
                self.token_ids[i] = self.token_ids[i] + [self.vocab['<pad>']] * (self.max_seq_len - len(self.token_ids[i]))
            
            # 구문 특성 패딩/트리밍
            for key in ['pos_tags', 'elem_types', 'heads']:
                if key in self.syntax_features[i]:
                    if len(self.syntax_features[i][key]) > self.max_seq_len:
                        self.syntax_features[i][key] = self.syntax_features[i][key][:self.max_seq_len]
                    else:
                        pad_len = self.max_seq_len - len(self.syntax_features[i][key])
                        if isinstance(self.syntax_features[i][key], list):
                            self.syntax_features[i][key] = self.syntax_features[i][key] + [0] * pad_len
                        else:  # torch.Tensor
                            pad_shape = list(self.syntax_features[i][key].shape)
                            pad_shape[0] = pad_len
                            padding = torch.zeros(pad_shape)
                            self.syntax_features[i][key] = torch.cat([self.syntax_features[i][key], padding], dim=0)
            
            # syntax_features 패딩/트리밍
            if 'syntax_features' in self.syntax_features[i]:
                if self.syntax_features[i]['syntax_features'].shape[0] > self.max_seq_len:
                    self.syntax_features[i]['syntax_features'] = self.syntax_features[i]['syntax_features'][:self.max_seq_len]
                else:
                    pad_len = self.max_seq_len - self.syntax_features[i]['syntax_features'].shape[0]
                    padding = torch.zeros(pad_len, self.syntax_features[i]['syntax_features'].shape[1])
                    self.syntax_features[i]['syntax_features'] = torch.cat([self.syntax_features[i]['syntax_features'], padding], dim=0)
            
            # 관계 행렬 패딩/트리밍
            if 'tag_relations' in self.syntax_features[i]:
                rel_matrix = self.syntax_features[i]['tag_relations']
                if rel_matrix.shape[0] > self.max_seq_len or rel_matrix.shape[1] > self.max_seq_len:
                    self.syntax_features[i]['tag_relations'] = rel_matrix[:self.max_seq_len, :self.max_seq_len]
                else:
                    pad_rows = self.max_seq_len - rel_matrix.shape[0]
                    pad_cols = self.max_seq_len - rel_matrix.shape[1]
                    
                    if pad_rows > 0 or pad_cols > 0:
                        padded_matrix = torch.zeros(self.max_seq_len, self.max_seq_len, rel_matrix.shape[2])
                        padded_matrix[:rel_matrix.shape[0], :rel_matrix.shape[1]] = rel_matrix
                        self.syntax_features[i]['tag_relations'] = padded_matrix
    
    def __len__(self) -> int:
        """데이터셋 길이 반환"""
        return len(self.sentences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """인덱스로 항목 접근"""
        # 토큰 ID 텐서
        token_ids = torch.tensor(self.token_ids[idx], dtype=torch.long)
        
        # 구문 특성 텐서
        pos_tags = torch.tensor(self.syntax_features[idx]['pos_tags'], dtype=torch.long)
        elem_types = torch.tensor(self.syntax_features[idx]['elem_types'], dtype=torch.long)
        syntax_features = self.syntax_features[idx].get('syntax_features', torch.zeros(self.max_seq_len, 3))
        heads = torch.tensor(self.syntax_features[idx]['heads'], dtype=torch.long)
        tag_relations = self.syntax_features[idx].get('tag_relations', torch.zeros(self.max_seq_len, self.max_seq_len, 4))
        
        # 어텐션 마스크 (패딩 토큰: 0, 일반 토큰: 1)
        attention_mask = (token_ids != self.vocab['<pad>']).float()
        
        # 라벨
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {
            'token_ids': token_ids,
            'pos_tags': pos_tags,
            'elem_types': elem_types,
            'syntax_features': syntax_features,
            'heads': heads,
            'tag_relations': tag_relations,
            'attention_mask': attention_mask,
            'labels': label
        }


def create_tag_dataloaders(
    tag_file_path: str,
    batch_size: int = 16,
    max_seq_len: int = 512,
    val_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    TAG 데이터로부터 데이터로더 생성
    
    Args:
        tag_file_path: TAG 파일 경로
        batch_size: 배치 크기
        max_seq_len: 최대 시퀀스 길이
        val_ratio: 검증 세트 비율
        random_seed: 랜덤 시드
        
    Returns:
        훈련 데이터로더, 검증 데이터로더, 어휘 사전
    """
    # 전체 데이터셋 생성
    dataset = TAGDataset(tag_file_path, max_seq_len=max_seq_len)
    
    # 훈련/검증 세트 분할
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size
    
    from torch.utils.data import random_split
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, dataset.vocab