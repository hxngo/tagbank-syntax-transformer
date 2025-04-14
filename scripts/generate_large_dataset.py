import os
import nltk
import spacy
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.corpus import treebank
from collections import defaultdict

def download_and_prepare_data():
    """필요한 데이터와 모델을 다운로드"""
    nltk.download('treebank')
    return spacy.load('en_core_web_sm')

def convert_tree_to_tag(tree, nlp):
    """구문 트리를 TAGbank 형식으로 변환"""
    tokens = []
    depths = []
    np_vp_tags = []
    
    def process_tree(node, depth=0):
        if isinstance(node, str):
            doc = nlp(node)
            tokens.append(node)
            depths.append(depth)
            np_vp_tags.append('O')
            return
            
        label = node.label()
        is_np = label == 'NP'
        is_vp = label == 'VP'
        
        for child in node:
            if isinstance(child, str):
                tokens.append(child)
                depths.append(depth)
                np_vp_tags.append('NP' if is_np else 'VP' if is_vp else 'O')
            else:
                process_tree(child, depth + 1)
    
    process_tree(tree)
    return tokens, depths, np_vp_tags

def generate_tagbank_data(output_dir, num_samples=1000000):
    """대규모 TAGbank 데이터 생성"""
    nlp = download_and_prepare_data()
    os.makedirs(output_dir, exist_ok=True)
    
    # Penn Treebank 데이터 처리
    trees = list(treebank.parsed_sents())
    print(f"Processing {len(trees)} trees from Penn Treebank...")
    
    train_file = open(os.path.join(output_dir, 'train.tag'), 'w', encoding='utf-8')
    val_file = open(os.path.join(output_dir, 'validation.tag'), 'w', encoding='utf-8')
    test_file = open(os.path.join(output_dir, 'test.tag'), 'w', encoding='utf-8')
    
    for i, tree in enumerate(tqdm(trees)):
        tokens, depths, np_vp_tags = convert_tree_to_tag(tree, nlp)
        
        # TAGbank 형식으로 출력
        output = ""
        for token, depth, tag in zip(tokens, depths, np_vp_tags):
            output += f"{token}\t{depth}\t{tag}\n"
        output += "\n"
        
        # 데이터 분할 (80/10/10)
        if i < len(trees) * 0.8:
            train_file.write(output)
        elif i < len(trees) * 0.9:
            val_file.write(output)
        else:
            test_file.write(output)
    
    # 파일 닫기
    train_file.close()
    val_file.close()
    test_file.close()
    
    print(f"Generated TAGbank data in {output_dir}")
    print("Train file:", os.path.join(output_dir, 'train.tag'))
    print("Validation file:", os.path.join(output_dir, 'validation.tag'))
    print("Test file:", os.path.join(output_dir, 'test.tag'))

if __name__ == '__main__':
    # 현재 스크립트의 디렉토리를 기준으로 상대 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(os.path.dirname(current_dir), 'data', 'large_tagbank')
    generate_tagbank_data(output_dir) 