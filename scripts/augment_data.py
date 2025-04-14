import os
import random
import nltk
import torch
from nltk.corpus import wordnet
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
import numpy as np

def load_tagbank_file(file_path):
    """TAGbank 파일 로드"""
    sentences = []
    current_sentence = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                token, depth, tag = line.split('\t')
                current_sentence.append((token, int(depth), tag))
            elif current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
    
    if current_sentence:
        sentences.append(current_sentence)
    
    return sentences

def save_tagbank_file(sentences, file_path):
    """TAGbank 형식으로 저장"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            for token, depth, tag in sentence:
                f.write(f"{token}\t{depth}\t{tag}\n")
            f.write("\n")

def synonym_replacement(sentence, n=1):
    """동의어 치환을 통한 증강"""
    new_sentence = sentence.copy()
    replaceable_indices = [i for i, (token, _, tag) in enumerate(sentence)
                         if tag in ['NP', 'VP']]
    
    if not replaceable_indices:
        return new_sentence
        
    n = min(n, len(replaceable_indices))
    indices_to_replace = random.sample(replaceable_indices, n)
    
    for idx in indices_to_replace:
        token = sentence[idx][0]
        synsets = wordnet.synsets(token)
        if synsets:
            synonyms = []
            for syn in synsets:
                for lemma in syn.lemmas():
                    if lemma.name() != token:
                        synonyms.append(lemma.name())
            if synonyms:
                new_token = random.choice(synonyms)
                new_sentence[idx] = (new_token, sentence[idx][1], sentence[idx][2])
    
    return new_sentence

def swap_subtrees(sentence):
    """구문 구존하면서 부분 트리 교환"""
    if len(sentence) < 4:  # 너무 짧은 문장은 건너뜀
        return sentence
        
    new_sentence = sentence.copy()
    depths = [d for _, d, _ in sentence]
    
    # 같은 깊이를 가진 연속된 토큰들을 찾음
    depth_ranges = []
    start = 0
    for i in range(1, len(depths)):
        if depths[i] != depths[i-1]:
            if i - start > 1:  # 최소 2개 이상의 토큰
                depth_ranges.append((start, i))
            start = i
    
    if not depth_ranges:
        return new_sentence
        
    # 랜덤하게 범위를 선택하고 해당 범위 내의 토큰들을 섞음
    range_idx = random.choice(range(len(depth_ranges)))
    start, end = depth_ranges[range_idx]
    
    # 해당 범위의 토큰들을 섞음
    tokens = new_sentence[start:end]
    random.shuffle(tokens)
    new_sentence[start:end] = tokens
    
    return new_sentence

def random_mask(sentence, mask_prob=0.15):
    """랜덤 마스킹을 통한 증강"""
    new_sentence = sentence.copy()
    for i in range(len(sentence)):
        if random.random() < mask_prob:
            new_sentence[i] = ('[MASK]', sentence[i][1], sentence[i][2])
    return new_sentence

def augment_dataset(input_file, output_file, augmentation_factor=5):
    """데이터셋 증강"""
    print(f"Loading data from {input_file}...")
    sentences = load_tagbank_file(input_file)
    augmented_sentences = []
    
    # 1. 동의어 치환
    print("Performing synonym replacement...")
    for _ in range(augmentation_factor // 3):
        for sent in tqdm(sentences):
            aug_sent = synonym_replacement(sent, n=2)
            if aug_sent:
                augmented_sentences.append(aug_sent)
    
    # 2. 부분 트리 교환
    print("Performing subtree swapping...")
    for _ in range(augmentation_factor // 3):
        for sent in tqdm(sentences):
            aug_sent = swap_subtrees(sent)
            if aug_sent:
                augmented_sentences.append(aug_sent)
    
    # 3. 랜덤 마스킹
    print("Performing random masking...")
    for _ in range(augmentation_factor // 3):
        for sent in tqdm(sentences):
            aug_sent = random_mask(sent)
            augmented_sentences.append(aug_sent)
    
    # 원본 데이터도 포함
    all_sentences = sentences + augmented_sentences
    random.shuffle(all_sentences)
    
    print(f"Saving augmented data to {output_file}...")
    save_tagbank_file(all_sentences, output_file)
    print(f"Original sentences: {len(sentences)}")
    print(f"Augmented sentences: {len(augmented_sentences)}")
    print(f"Total sentences: {len(all_sentences)}")

if __name__ == '__main__':
    nltk.download('wordnet')
    # 현재 스크립트의 디렉토리를 기준으로 상대 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    input_file = os.path.join(base_dir, 'data', 'large_tagbank', 'train.tag')
    output_file = os.path.join(base_dir, 'data', 'large_tagbank', 'train_augmented.tag')
    augment_dataset(input_file, output_file, augmentation_factor=15)  # 15배로 증강 