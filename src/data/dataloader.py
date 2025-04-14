import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from ..utils.augmentation import augment_batch
import json
import logging

logger = logging.getLogger(__name__)

class TagDataset(Dataset):
    def __init__(self, data_path):
        """
        TAG 데이터셋 초기화
        
        Args:
            data_path: TAG 파일 경로
        """
        self.data = []
        self.vocab = {'PAD': 0, 'UNK': 1}  # 기본 토큰
        self.label2idx = {}  # 레이블을 인덱스로 매핑
        self.idx2label = {}  # 인덱스를 레이블로 매핑
        
        # 데이터 로드
        self._load_data(data_path)
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
        logger.info(f"Vocabulary size: {len(self.vocab)}")
        logger.info(f"Number of labels: {len(self.label2idx)}")
    
    def _load_data(self, data_path):
        """TAG 파일에서 데이터 로드"""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                raw_data = [line.strip() for line in f if line.strip()]
            
            # 데이터 파싱
            for line in raw_data:
                try:
                    item = json.loads(line)
                    
                    # 토큰 처리
                    tokens = item.get('tokens', [])
                    for token in tokens:
                        if token not in self.vocab:
                            self.vocab[token] = len(self.vocab)
                    
                    # 레이블 처리
                    label = item.get('label', 0)  # 기본값 0
                    if isinstance(label, str):
                        if label not in self.label2idx:
                            idx = len(self.label2idx)
                            self.label2idx[label] = idx
                            self.idx2label[idx] = label
                        label = self.label2idx[label]
                    
                    # 토큰을 인덱스로 변환
                    token_ids = [self.vocab.get(token, self.vocab['UNK']) for token in tokens]
                    
                    # 구문 관련 특성 처리
                    elem_types = item.get('elem_types', None)
                    syntax_features = item.get('syntax_features', None)
                    tag_relations = item.get('tag_relations', None)
                    
                    # 데이터 아이템 생성
                    processed_item = {
                        'token_ids': torch.tensor(token_ids),
                        'attention_mask': torch.ones(len(token_ids)),
                        'labels': torch.tensor(label),
                        'elem_types': torch.tensor(elem_types) if elem_types is not None else None,
                        'syntax_features': torch.tensor(syntax_features) if syntax_features is not None else None,
                        'tag_relations': torch.tensor(tag_relations) if tag_relations is not None else None
                    }
                    
                    self.data.append(processed_item)
                    
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse line: {line}")
                except Exception as e:
                    logger.warning(f"Error processing line: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {str(e)}")
            raise
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, config=None):
    """배치 데이터를 처리하고 증강하는 함수"""
    # 시퀀스 길이 계산
    max_len = max(item['token_ids'].size(0) for item in batch)
    
    # 패딩된 텐서 초기화
    token_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    labels = torch.zeros(len(batch), dtype=torch.long)
    
    # 구문 관련 특성을 위한 텐서 초기화
    has_elem_types = batch[0]['elem_types'] is not None
    has_syntax_features = batch[0]['syntax_features'] is not None
    has_tag_relations = batch[0]['tag_relations'] is not None
    
    if has_elem_types:
        elem_types = torch.zeros((len(batch), max_len), dtype=torch.long)
    if has_syntax_features:
        syntax_features = torch.zeros((len(batch), max_len, batch[0]['syntax_features'].size(-1)), dtype=torch.float)
    if has_tag_relations:
        tag_relations = torch.zeros((len(batch), max_len, max_len), dtype=torch.long)
    
    # 배치 데이터 채우기
    for i, item in enumerate(batch):
        seq_len = item['token_ids'].size(0)
        token_ids[i, :seq_len] = item['token_ids']
        attention_mask[i, :seq_len] = item['attention_mask']
        labels[i] = item['labels']
        
        if has_elem_types:
            elem_types[i, :seq_len] = item['elem_types']
        if has_syntax_features:
            syntax_features[i, :seq_len] = item['syntax_features']
        if has_tag_relations:
            tag_relations[i, :seq_len, :seq_len] = item['tag_relations']
    
    # 데이터 증강 적용
    if config and config.get('data', {}).get('augmentation', {}).get('enabled', False):
        token_ids, attention_mask, labels = augment_batch(
            input_ids=token_ids,
            attention_mask=attention_mask,
            labels=labels,
            mask_prob=config['data']['augmentation'].get('mask_prob', 0.15),
            shuffle_prob=config['data']['augmentation'].get('shuffle_prob', 0.1),
            max_shuffle_distance=config['data']['augmentation'].get('max_shuffle_distance', 3)
        )
    
    # 결과 딕셔너리 생성
    result = {
        'token_ids': token_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
    
    # 구문 관련 특성 추가
    if has_elem_types:
        result['elem_types'] = elem_types
    if has_syntax_features:
        result['syntax_features'] = syntax_features
    if has_tag_relations:
        result['tag_relations'] = tag_relations
    
    return result

class TAGDataLoader:
    def __init__(self, dataset, config, shuffle=True):
        """
        TAG 데이터 로더 초기화
        
        Args:
            dataset: TAG 데이터셋
            config: 설정 딕셔너리
            shuffle: 데이터 셔플 여부
        """
        self.dataset = dataset
        self.config = config
        self.shuffle = shuffle
        self.batch_size = config['training']['batch_size']
    
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            yield collate_fn(batch, self.config)
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
// ... existing code ... 