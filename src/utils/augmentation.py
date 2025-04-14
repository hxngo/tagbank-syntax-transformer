import random
import torch
import numpy as np

def augment_batch(input_ids, attention_mask, labels=None, mask_prob=0.15, shuffle_prob=0.1, max_shuffle_distance=3):
    """
    배치 단위로 데이터 증강을 수행합니다.
    
    Args:
        input_ids (torch.Tensor): 입력 토큰 ID 텐서 [batch_size, seq_len]
        attention_mask (torch.Tensor): 어텐션 마스크 텐서 [batch_size, seq_len]
        labels (torch.Tensor, optional): 레이블 텐서 [batch_size, seq_len]
        mask_prob (float): 토큰을 마스킹할 확률
        shuffle_prob (float): 토큰 순서를 섞을 확률
        max_shuffle_distance (int): 토큰을 섞을 때 최대 이동 거리
        
    Returns:
        tuple: (증강된 input_ids, 증강된 attention_mask, 증강된 labels)
    """
    augmented_input_ids = input_ids.clone()
    augmented_attention_mask = attention_mask.clone()
    if labels is not None:
        augmented_labels = labels.clone()
    
    batch_size, seq_len = input_ids.shape
    
    # 특수 토큰 위치 찾기 (CLS, SEP 등)
    special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    special_tokens_mask[:, 0] = True  # CLS token
    special_tokens_mask[torch.arange(batch_size), attention_mask.sum(dim=1) - 1] = True  # SEP token
    
    for b in range(batch_size):
        valid_indices = torch.where((attention_mask[b] == 1) & ~special_tokens_mask[b])[0]
        num_tokens = len(valid_indices)
        
        # 증강할 토큰 위치 선택 (마스킹과 셔플링이 겹치지 않도록)
        num_total_aug = int(num_tokens * (mask_prob + shuffle_prob))
        if num_total_aug > 0:
            aug_indices = random.sample(valid_indices.tolist(), min(num_total_aug, num_tokens))
            
            # 마스킹 적용
            num_to_mask = int(num_tokens * mask_prob)
            if num_to_mask > 0:
                mask_indices = aug_indices[:num_to_mask]
                augmented_input_ids[b, mask_indices] = 103  # [MASK] 토큰의 ID
            
            # 토큰 순서 섞기 (남은 인덱스 사용)
            shuffle_indices = aug_indices[num_to_mask:]
            if len(shuffle_indices) >= 2:  # 최소 2개의 토큰이 필요
                # 연속된 토큰들을 그룹으로 묶어서 섞기
                groups = []
                current_group = [shuffle_indices[0]]
                
                for i in range(1, len(shuffle_indices)):
                    if shuffle_indices[i] - shuffle_indices[i-1] <= max_shuffle_distance:
                        current_group.append(shuffle_indices[i])
                    else:
                        if len(current_group) >= 2:
                            groups.append(current_group)
                        current_group = [shuffle_indices[i]]
                
                if len(current_group) >= 2:
                    groups.append(current_group)
                
                # 각 그룹 내에서 토큰 순서 섞기
                for group in groups:
                    if len(group) >= 2:
                        tokens = augmented_input_ids[b, group].clone()
                        perm = torch.randperm(len(group))
                        augmented_input_ids[b, group] = tokens[perm]
                        if labels is not None:
                            label_tokens = augmented_labels[b, group].clone()
                            augmented_labels[b, group] = label_tokens[perm]
    
    if labels is not None:
        return augmented_input_ids, augmented_attention_mask, augmented_labels
    return augmented_input_ids, augmented_attention_mask

def mask_tokens(batch, config, mask_prob=0.15):
    """일부 토큰을 마스킹 처리합니다."""
    token_ids = batch['token_ids'].clone()
    attention_mask = batch['attention_mask']
    
    # 마스킹할 위치 선택 (특수 토큰 제외)
    special_tokens = config['data']['special_tokens']
    prob_matrix = torch.full(token_ids.shape, mask_prob)
    for special_token in special_tokens:
        prob_matrix[token_ids == special_token] = 0
    prob_matrix = prob_matrix * attention_mask
    
    masked_indices = torch.bernoulli(prob_matrix).bool()
    batch['token_ids'][masked_indices] = config['data']['mask_token_id']
    
    return batch

def shuffle_tokens(batch, config, max_spans=3):
    """문장 내 일부 토큰 스팬의 순서를 변경합니다."""
    token_ids = batch['token_ids'].clone()
    attention_mask = batch['attention_mask']
    batch_size, seq_len = token_ids.shape
    
    for i in range(batch_size):
        # 유효한 토큰 위치 찾기
        valid_positions = torch.where(attention_mask[i] == 1)[0]
        if len(valid_positions) <= 3:  # 너무 짧은 문장은 건너뛰기
            continue
            
        # 랜덤하게 스팬 선택 및 섞기
        num_spans = random.randint(1, min(max_spans, len(valid_positions) // 3))
        span_length = random.randint(2, 4)
        
        for _ in range(num_spans):
            start_idx = random.randint(0, len(valid_positions) - span_length)
            span = token_ids[i, start_idx:start_idx + span_length].clone()
            # 스팬 내 토큰 순서 변경
            perm = torch.randperm(span_length)
            token_ids[i, start_idx:start_idx + span_length] = span[perm]
    
    batch['token_ids'] = token_ids
    return batch 