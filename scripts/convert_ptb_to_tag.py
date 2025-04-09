#!/usr/bin/env python
"""
Penn Treebank 형식에서 TAGbank 형식으로 변환하는 스크립트

이 스크립트는 Penn Treebank 스타일의 구구조 트리를 Tree-Adjoining Grammar(TAG) 유도 구조로 변환합니다.
변환 과정은 논문 "Proposing TAGbank as a Corpus of Tree-Adjoining Grammar Derivations"에서 설명한
방법론을 따릅니다.
"""

import os
import sys
import argparse
import re
import nltk
from nltk.tree import Tree
from typing import List, Dict, Tuple, Optional, Set
import logging


class PennToTAGConverter:
    """Penn Treebank에서 TAGbank 형식으로 변환하는 클래스"""
    
    def __init__(self):
        """초기화"""
        # 헤드 투영 테이블 (Collins 1999)
        self.head_rules = {
            'ADJP': ['JJ', 'JJR', 'JJS', 'ADJP', 'VBN', 'VBG', 'ADVP', 'IN', 'VP', 'NP'],
            'ADVP': ['RB', 'RBR', 'RBS', 'ADVP', 'IN', 'JJ'],
            'NP': ['NN', 'NNP', 'NNPS', 'NNS', 'NX', 'POS', 'JJR', 'NP'],
            'PP': ['IN', 'TO', 'PP'],
            'S': ['VP', 'S', 'SBAR', 'ADJP', 'UCP', 'NP'],
            'SBAR': ['IN', 'WHNP', 'WHPP', 'WHADVP', 'WHADJP', 'WDT', 'S', 'SQ', 'SINV', 'SBAR'],
            'VP': ['VB', 'VBD', 'VBN', 'VBP', 'VBZ', 'VP', 'ADJP', 'S', 'SBAR', 'NP']
        }
        
        # 인자/부가어 식별 규칙
        self.argument_labels = {'SBJ', 'OBJ', 'CLR', 'DTV', 'PRD', 'TPC', 'LGS'}
        
        # POS 태그 매핑
        self.pos_map = {}  # 필요에 따라 POS 태그 변환 규칙 추가
    
    def find_head(self, tree: Tree) -> Tuple[int, str]:
        """
        구문 트리에서 헤드 노드의 인덱스와 헤드 노드를 찾습니다.
        
        Args:
            tree: NLTK Tree 객체
            
        Returns:
            (헤드 인덱스, 헤드 노드)의 튜플
        """
        if not isinstance(tree, Tree) or len(tree) == 0:
            return 0, None
        
        label = tree.label()
        if isinstance(label, str):
            # 함수 태그 제거 (예: NP-SBJ -> NP)
            base_label = label.split('-')[0]
            
            # 헤드 규칙 찾기
            head_candidates = self.head_rules.get(base_label, [])
            
            # 오른쪽에서 왼쪽으로 탐색 (영어의 경우 주로 오른쪽 헤드)
            for pos in head_candidates:
                for i, child in enumerate(tree):
                    if isinstance(child, Tree):
                        child_label = child.label()
                        if isinstance(child_label, str) and child_label.split('-')[0] == pos:
                            return i, child
            
            # 왼쪽에서 오른쪽으로 탐색 (기본 전략)
            for i, child in enumerate(tree):
                if isinstance(child, Tree):
                    return i, child
        
        # 기본값: 첫 번째 자식
        return 0, tree[0] if len(tree) > 0 else None
    
    def is_argument(self, label: str) -> bool:
        """
        노드 라벨이 인자(argument)인지 판단합니다.
        
        Args:
            label: 노드 라벨
            
        Returns:
            인자이면 True, 부가어(adjunct)이면 False
        """
        # 함수 태그가 있는 경우 (예: NP-SBJ)
        if '-' in label:
            func_tag = label.split('-')[1]
            return func_tag in self.argument_labels
        
        return False
    
    def extract_elementary_trees(self, tree: Tree) -> List[Dict]:
        """
        구문 트리에서 TAG 기본 트리(elementary trees)를 추출합니다.
        
        Args:
            tree: NLTK Tree 객체
            
        Returns:
            기본 트리 목록 (딕셔너리 형태)
        """
        if not isinstance(tree, Tree):
            return []
        
        elementary_trees = []
        processed_indices = set()
        
        # 트리 구조 분석 및 헤드 찾기
        head_idx, head_node = self.find_head(tree)
        
        if head_idx is not None:
            processed_indices.add(head_idx)
            
            # 헤드가 단말 노드인 경우 (어휘 앵커)
            if not isinstance(head_node, Tree):
                # 초기 트리(alpha tree) 생성
                alpha_tree = {
                    'type': 'alpha',  # 초기 트리
                    'anchor': head_node,  # 어휘 앵커
                    'root': tree.label(),  # 루트 노드 라벨
                    'substitution_sites': []  # 대체 사이트 (인자 노드)
                }
                
                # 인자 노드 처리
                for i, child in enumerate(tree):
                    if i != head_idx and isinstance(child, Tree):
                        child_label = child.label()
                        if self.is_argument(child_label):
                            alpha_tree['substitution_sites'].append({
                                'idx': i,
                                'label': child_label
                            })
                            # 재귀적으로 인자 노드의 기본 트리 추출
                            elementary_trees.extend(self.extract_elementary_trees(child))
                            processed_indices.add(i)
                
                elementary_trees.append(alpha_tree)
            
            # 헤드가 비단말 노드인 경우
            else:
                # 헤드 노드의 기본 트리 재귀적 추출
                elementary_trees.extend(self.extract_elementary_trees(head_node))
        
        # 부가어(adjunct) 노드 처리
        for i, child in enumerate(tree):
            if i not in processed_indices and isinstance(child, Tree):
                child_label = child.label()
                if not self.is_argument(child_label):
                    # 보조 트리(beta tree) 생성
                    beta_tree = {
                        'type': 'beta',  # 보조 트리
                        'root': tree.label(),  # 루트 노드 라벨
                        'foot': tree.label(),  # 발 노드 라벨 (동일)
                        'adjunction_site': i  # 부가 위치
                    }
                    elementary_trees.append(beta_tree)
                    
                    # 재귀적으로 부가어 노드의 기본 트리 추출
                    elementary_trees.extend(self.extract_elementary_trees(child))
                    processed_indices.add(i)
        
        return elementary_trees
    
    def build_derivation_tree(self, elementary_trees: List[Dict]) -> Dict:
        """
        기본 트리 목록에서 TAG 유도 트리(derivation tree)를 구축합니다.
        
        Args:
            elementary_trees: 기본 트리 목록
            
        Returns:
            TAG 유도 트리 (딕셔너리 형태)
        """
        # 초기 트리(alpha tree)를 찾아 루트로 설정
        root_tree = None
        for tree in elementary_trees:
            if tree['type'] == 'alpha' and 'anchor' in tree:
                root_tree = tree
                break
        
        if not root_tree:
            return {}
        
        # 유도 트리 구축
        derivation_tree = {
            'type': root_tree['type'],
            'anchor': root_tree.get('anchor', ''),
            'root': root_tree['root'],
            'children': []
        }
        
        # 인자 노드(substitution sites) 처리
        for subst_site in root_tree.get('substitution_sites', []):
            site_idx = subst_site['idx']
            site_label = subst_site['label']
            
            # 해당 라벨을 가진 초기 트리 찾기
            for tree in elementary_trees:
                if tree['type'] == 'alpha' and tree['root'] == site_label:
                    child_deriv = self.build_derivation_tree([tree] + [
                        t for t in elementary_trees if t != tree and t != root_tree
                    ])
                    if child_deriv:
                        child_deriv['operation'] = 'substitution'
                        child_deriv['address'] = site_idx
                        derivation_tree['children'].append(child_deriv)
                    break
        
        # 부가어 노드(adjunction sites) 처리
        for tree in elementary_trees:
            if tree['type'] == 'beta' and tree['root'] == root_tree['root']:
                adjunct_site = tree.get('adjunction_site')
                if adjunct_site is not None:
                    child_deriv = {
                        'type': 'beta',
                        'root': tree['root'],
                        'foot': tree['foot'],
                        'operation': 'adjunction',
                        'address': adjunct_site,
                        'children': []
                    }
                    derivation_tree['children'].append(child_deriv)
        
        return derivation_tree
    
    def convert_to_tagbank_format(self, tree: Tree) -> List[Dict]:
        """
        NLTK Tree를 TAGbank 형식으로 변환합니다.
        
        Args:
            tree: NLTK Tree 객체
            
        Returns:
            TAGbank 형식의 토큰 목록
        """
        # 1. 기본 트리 추출
        elementary_trees = self.extract_elementary_trees(tree)
        
        # 2. 유도 트리 구축
        derivation_tree = self.build_derivation_tree(elementary_trees)
        
        # 3. TAGbank 형식으로 변환
        tagbank_tokens = []
        token_idx = 1
        
        # 트리의 단말 노드 추출
        leaves = tree.leaves()
        pos_tags = [pos for _, pos in tree.pos()]
        
        for i, (leaf, pos) in enumerate(zip(leaves, pos_tags)):
            token_entry = {
                'IDX': token_idx,
                'LEX': leaf.lower(),
                'POS': pos.lower(),
                'HD': 0,  # 기본값, 추후 조정
                'ELEM': 'alpha',  # 기본값, 추후 조정
                'RHS': '',
                'LHS': ''
            }
            
            # 헤드 정보와 요소 타입 설정 (단순 휴리스틱)
            for etree in elementary_trees:
                if etree.get('anchor') == leaf:
                    token_entry['ELEM'] = etree['type']
                    # 헤드 정보는 유도 트리에서 추출 (여기서는 단순화)
                    break
            
            # 구구조 정보 (RHS/LHS)
            # 실제 구현에서는 더 복잡한 로직 필요
            token_entry['RHS'] = '(S ' if i == 0 else ''
            token_entry['LHS'] = ')S' if i == len(leaves) - 1 else ''
            
            tagbank_tokens.append(token_entry)
            token_idx += 1
        
        # 헤드 정보 보완 (간단한 휴리스틱)
        for i, token in enumerate(tagbank_tokens):
            if token['ELEM'] == 'alpha':
                # 초기 트리의 경우 루트 노드로 설정
                token['HD'] = 0
            elif i > 0:
                # 보조 트리의 경우 이전 토큰 인덱스로 설정
                token['HD'] = i
        
        return tagbank_tokens
    
    def convert_file(self, input_path: str, output_path: str):
        """
        Penn Treebank 파일을 TAGbank 형식으로 변환합니다.
        
        Args:
            input_path: 입력 파일 경로 (Penn Treebank 형식)
            output_path: 출력 파일 경로 (TAGbank 형식)
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        trees = []
        try:
            # 여러 트리가 있을 수 있음
            trees = nltk.Tree.fromstring(content)
            if not isinstance(trees, list):
                trees = [trees]
        except:
            # 한 줄에 하나의 트리가 있는 형식
            trees = []
            for line in content.strip().split('\n'):
                if line.strip():
                    try:
                        tree = nltk.Tree.fromstring(line)
                        trees.append(tree)
                    except:
                        logging.warning(f"Failed to parse line: {line}")
        
        # TAGbank 형식으로 변환
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# TAGbank format converted from Penn Treebank\n")
            f.write("IDX LEX POS HD ELEM RHS LHS\n")
            
            for tree in trees:
                # 문장 구분
                f.write("\n# Sentence: " + ' '.join(tree.leaves()) + "\n")
                
                # 변환 및 출력
                tagbank_tokens = self.convert_to_tagbank_format(tree)
                for token in tagbank_tokens:
                    f.write(f"{token['IDX']} {token['LEX']} {token['POS']} {token['HD']} {token['ELEM']} {token['RHS']} {token['LHS']}\n")
                
                f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Convert Penn Treebank format to TAGbank format")
    parser.add_argument('--input', type=str, required=True, help="Input file (Penn Treebank format)")
    parser.add_argument('--output', type=str, required=True, help="Output file (TAGbank format)")
    args = parser.parse_args()
    
    converter = PennToTAGConverter()
    converter.convert_file(args.input, args.output)
    print(f"Conversion completed. Output saved to {args.output}")


if __name__ == "__main__":
    main()
