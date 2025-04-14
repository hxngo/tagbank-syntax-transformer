"""
TAG 파싱 유틸리티

TAG(Tree Adjoining Grammar) 형식의 파일을 파싱하고 처리하기 위한 유틸리티 함수들을 제공합니다.
"""

import re
from typing import Dict, List, Tuple, Optional, Set, Union
from collections import namedtuple, defaultdict


# TAG 트리의 노드를 표현하는 클래스
class TAGNode:
    """TAG 트리의 노드"""
    
    def __init__(self, 
                 id: int, 
                 label: str, 
                 word: Optional[str] = None, 
                 pos: Optional[str] = None,
                 is_subst: bool = False,
                 is_adj: bool = False):
        """
        TAG 노드 초기화
        
        Args:
            id: 노드 ID
            label: 노드 레이블 (NP, VP 등)
            word: 단어 (단말 노드인 경우)
            pos: 품사 태그
            is_subst: 대체 노드 여부
            is_adj: 접합 노드 여부
        """
        self.id = id
        self.label = label
        self.word = word
        self.pos = pos
        self.is_subst = is_subst
        self.is_adj = is_adj
        
        self.parent = None
        self.children = []
        self.depth = 0
    
    def add_child(self, child):
        """자식 노드 추가"""
        self.children.append(child)
        child.parent = self
    
    def is_terminal(self) -> bool:
        """단말 노드 여부 확인"""
        return len(self.children) == 0 and self.word is not None
    
    def is_substitution_node(self) -> bool:
        """대체 노드 여부 확인"""
        return self.is_subst
    
    def is_adjoining_node(self) -> bool:
        """접합 노드 여부 확인"""
        return self.is_adj
    
    def is_substitution_site_for(self, other) -> bool:
        """다른 노드의 대체 위치인지 확인"""
        if not self.is_subst:
            return False
        
        # 대체 규칙 검사 (예: 두 노드의 레이블이 일치해야 함)
        return self.label == other.label
    
    def is_adjoining_site_for(self, other) -> bool:
        """다른 노드의 접합 위치인지 확인"""
        if not other.is_adj:
            return False
        
        # 접합 규칙 검사 (예: 레이블이 일치해야 함)
        return self.label == other.label
    
    def __str__(self) -> str:
        """문자열 표현"""
        if self.is_terminal():
            return f"{self.word}/{self.pos}"
        else:
            return f"{self.label}{'↓' if self.is_subst else ''}{'*' if self.is_adj else ''}"


# TAG 트리를 표현하는 클래스
class TAGTree:
    """TAG 구문 트리"""
    
    def __init__(self, id: Optional[str] = None, label: Optional[str] = None):
        """
        TAG 트리 초기화
        
        Args:
            id: 트리 ID
            label: 트리 레이블 (예: 감정 분류)
        """
        self.id = id
        self.label = label
        self.root = None
        self.nodes = {}  # ID -> 노드 매핑
        self.terminals = []  # 단말 노드들
    
    def add_node(self, node: TAGNode):
        """노드 추가"""
        self.nodes[node.id] = node
        if node.is_terminal():
            self.terminals.append(node)
    
    def set_root(self, root: TAGNode):
        """루트 노드 설정"""
        self.root = root
    
    def get_node(self, id: int) -> Optional[TAGNode]:
        """ID로 노드 조회"""
        return self.nodes.get(id)
    
    def all_nodes(self) -> List[TAGNode]:
        """모든 노드 목록 반환"""
        return list(self.nodes.values())
    
    def calculate_depths(self):
        """모든 노드의 깊이 계산"""
        def _set_depth(node, depth):
            node.depth = depth
            for child in node.children:
                _set_depth(child, depth + 1)
        
        if self.root:
            _set_depth(self.root, 0)
    
    def get_sentence(self) -> str:
        """트리에서 문장 추출"""
        return ' '.join([node.word for node in self.terminals])
    
    def get_pos_tags(self) -> List[str]:
        """품사 태그 목록 추출"""
        return [node.pos for node in self.terminals]
    
    def get_elementary_types(self) -> List[int]:
        """요소 타입 목록 추출 (alpha=1, beta=2)"""
        # 각 단말 노드의 상위 초기 트리 타입을 결정
        types = []
        for node in self.terminals:
            # 상위 노드 탐색하여 초기 트리 타입 결정
            curr = node
            while curr.parent:
                curr = curr.parent
                
                if curr.is_adj:
                    types.append(2)  # beta
                    break
            else:
                types.append(1)  # alpha
        
        return types
    
    def get_heads(self) -> List[int]:
        """의존 관계 헤드 목록 추출"""
        # 단순 구현: 각 단말 노드의 부모-자식 관계를 헤드로 간주
        heads = []
        terminal_indices = {node.id: i for i, node in enumerate(self.terminals)}
        
        for node in self.terminals:
            if node.parent:
                # 부모 노드의 첫 번째 단말 노드를 헤드로 간주
                parent_terminals = [n for n in node.parent.children if n.is_terminal()]
                if parent_terminals and parent_terminals[0].id in terminal_indices:
                    heads.append(terminal_indices[parent_terminals[0].id])
                else:
                    heads.append(0)  # 루트에 연결
            else:
                heads.append(0)  # 루트에 연결
        
        return heads
    
    def get_syntax_features(self) -> List[List[int]]:
        """구문 특성 추출 (깊이, NP 여부, VP 여부)"""
        features = []
        
        for node in self.terminals:
            # 특성 1: 깊이
            depth = node.depth
            
            # 특성 2: NP(명사구) 여부
            is_np = 1 if node.parent and node.parent.label.startswith('NP') else 0
            
            # 특성 3: VP(동사구) 여부
            is_vp = 1 if node.parent and node.parent.label.startswith('VP') else 0
            
            features.append([depth, is_np, is_vp])
        
        return features
    
    def create_relation_matrix(self) -> List[List[List[float]]]:
        """관계 행렬 생성 (대체, 접합, 부모-자식, 형제 관계)"""
        n_terminals = len(self.terminals)
        relation_matrix = [[[0.0, 0.0, 0.0, 0.0] for _ in range(n_terminals)] for _ in range(n_terminals)]
        
        # 노드 간 관계 설정
        for i, node_i in enumerate(self.terminals):
            for j, node_j in enumerate(self.terminals):
                # 부모-자식 관계
                if node_i.parent == node_j.parent:
                    if i != j:
                        # 형제 관계
                        relation_matrix[i][j][3] = 1.0
                
                # 문법적 관계 (단순화된 예시)
                if j > 0 and i > 0 and abs(i - j) == 1:
                    # 인접 토큰 간 관계 가중치
                    relation_matrix[i][j][2] = 0.5
        
        return relation_matrix


def parse_tag_file(file_path: str) -> List[TAGTree]:
    """
    TAG 파일 파싱
    
    Args:
        file_path: TAG 파일 경로
        
    Returns:
        TAG 트리 목록
    """
    trees = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # 파일 내용을 문장 단위로 분리
        entries = re.split(r'\n\s*\n', content)
        
        for entry in entries:
            if not entry.strip():
                continue
            
            tree = parse_tag_entry(entry)
            if tree:
                trees.append(tree)
    
    return trees


def parse_tag_entry(entry: str) -> Optional[TAGTree]:
    """
    TAG 엔트리 파싱
    
    Args:
        entry: TAG 엔트리 문자열
        
    Returns:
        TAG 트리 또는 None
    """
    lines = entry.strip().split('\n')
    
    # 트리 생성
    tree = TAGTree()
    
    # 메타데이터 파싱 (주석 라인)
    meta_lines = [line for line in lines if line.startswith('#')]
    for meta in meta_lines:
        if 'LABEL:' in meta:
            tree.label = meta.split('LABEL:')[1].strip()
        elif 'ID:' in meta:
            tree.id = meta.split('ID:')[1].strip()
    
    # 노드 파싱
    data_lines = [line for line in lines if not line.startswith('#')]
    
    nodes = {}
    parent_child = {}
    
    for line in data_lines:
        parts = line.split('\t')
        if len(parts) < 4:
            continue
        
        node_id = int(parts[0])
        label = parts[1]
        parent_id = int(parts[2]) if parts[2] and parts[2] != '-1' else None
        
        # 추가 정보
        extra = {}
        if len(parts) > 3:
            for i in range(3, len(parts)):
                if ':' in parts[i]:
                    key, value = parts[i].split(':', 1)
                    extra[key.strip()] = value.strip()
        
        # 노드 생성
        word = extra.get('word')
        pos = extra.get('pos')
        is_subst = 'subst' in extra.get('type', '')
        is_adj = 'adj' in extra.get('type', '')
        
        node = TAGNode(node_id, label, word, pos, is_subst, is_adj)
        nodes[node_id] = node
        tree.add_node(node)
        
        if parent_id is not None:
            parent_child[node_id] = parent_id
    
    # 부모-자식 관계 설정
    for child_id, parent_id in parent_child.items():
        if parent_id in nodes and child_id in nodes:
            nodes[parent_id].add_child(nodes[child_id])
    
    # 루트 노드 설정
    for node in nodes.values():
        if node.parent is None:
            tree.set_root(node)
            break
    
    # 깊이 계산
    tree.calculate_depths()
    
    return tree


def extract_features_from_tag_tree(tree: TAGTree) -> Dict:
    """
    TAG 트리에서 특성 추출
    
    Args:
        tree: TAG 트리
        
    Returns:
        특성 딕셔너리
    """
    features = {}
    
    # 기본 특성
    features['tokens'] = [node.word for node in tree.terminals]
    features['pos_tags'] = tree.get_pos_tags()
    features['elem_types'] = tree.get_elementary_types()
    
    # 구문 특성
    features['syntax_features'] = tree.get_syntax_features()
    features['heads'] = tree.get_heads()
    
    # 관계 행렬
    features['tag_relations'] = tree.create_relation_matrix()
    
    return features


# 간단한 TAG 파일 변환 함수
def convert_penn_to_tag(penn_file: str, output_file: str):
    """
    Penn Treebank 형식을 TAG 형식으로 변환
    
    Args:
        penn_file: Penn Treebank 파일 경로
        output_file: 출력 TAG 파일 경로
    """
    # 구현할 로직: Penn Treebank 파싱 및 TAG 변환
    # 이 부분은 실제 Penn Treebank 형식에 맞게 구현 필요
    pass


# 예시 사용법
if __name__ == "__main__":
    # TAG 파일 파싱 예시
    trees = parse_tag_file("sample.tag")
    for tree in trees:
        print(f"문장: {tree.get_sentence()}")
        print(f"품사: {tree.get_pos_tags()}")
        
        # 특성 추출
        features = extract_features_from_tag_tree(tree)
        print(f"구문 특성: {features['syntax_features']}")