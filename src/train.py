"""
TAG 구문 정보를 활용하는 Transformer 모델 훈련 스크립트
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import math
from torch.utils.data import DataLoader, random_split
import argparse
import os
import logging
import time
import copy
import numpy as np
import yaml
from typing import Dict, Tuple
from tqdm import tqdm
import torch.nn.functional as F
from models.syntax_transformer import create_syntax_aware_transformer
from data.tag_dataset import TAGDataset, create_tag_dataloaders
import wandb
from torch import amp
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.cuda.amp import autocast
import random
from torch.optim import AdamW
from utils.augmentation import augment_batch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def get_cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1
):
    """
    코사인 스케줄링과 웜업을 결합한 학습률 스케줄러
    
    Args:
        optimizer: 옵티마이저
        num_warmup_steps: 웜업 스텝 수
        num_training_steps: 전체 학습 스텝 수
        num_cycles: 코사인 사이클 수
        last_epoch: 마지막 에포크
        
    Returns:
        LambdaLR 스케줄러
    """
    def lr_lambda(current_step):
        # 웜업 단계
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # 웜업 이후 코사인 감소
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_linear_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1
):
    """
    선형 스케줄링과 웜업을 결합한 학습률 스케줄러
    
    Args:
        optimizer: 옵티마이저
        num_warmup_steps: 웜업 스텝 수
        num_training_steps: 전체 학습 스텝 수
        last_epoch: 마지막 에포크
        
    Returns:
        LambdaLR 스케줄러
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, (float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def apply_smote(features: np.ndarray, labels: np.ndarray, k_neighbors: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """SMOTE를 사용한 데이터 증강"""
    try:
        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
        features_resampled, labels_resampled = smote.fit_resample(features, labels)
        return features_resampled, labels_resampled
    except Exception as e:
        logger.warning(f"SMOTE 적용 실패: {str(e)}")
        return features, labels


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: Dict,
    save_path: str
):
    """체크포인트 저장"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics
    }, save_path)


def calculate_metrics(y_true, y_pred):
    """클래스별 성능 메트릭 계산"""
    metrics = {}
    
    # 전체 정확도
    metrics['accuracy'] = (y_true == y_pred).float().mean().item()
    
    # 클래스별 메트릭
    for cls in range(3):
        cls_pred = y_pred == cls
        cls_true = y_true == cls
        
        # True Positives, False Positives, False Negatives
        tp = (cls_pred & cls_true).sum().float()
        fp = (cls_pred & ~cls_true).sum().float()
        fn = (~cls_pred & cls_true).sum().float()
        
        # Precision
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        metrics[f'precision_cls{cls}'] = float(precision)
        
        # Recall
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        metrics[f'recall_cls{cls}'] = float(recall)
        
        # F1 Score
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        metrics[f'f1_cls{cls}'] = float(f1)
    
    return metrics


def evaluate_model(model, val_loader, criterion, device, vocab=None):
    """
    모델 평가 수행
    
    Args:
        model: 평가할 모델
        val_loader: 검증 데이터 로더
        criterion: 손실 함수
        device: 연산 장치
        vocab: 어휘 사전 (옵션)
    
    Returns:
        평가 결과 (손실, 정확도, 상세 메트릭)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []  # 확률값 저장
    num_classes = None
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            try:
                # 기본 입력 텐서들을 device로 이동
                token_ids = batch['token_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # 구문 관련 특성들을 device로 이동 (없을 수 있음)
                pos_tags = batch.get('pos_tags', None)
                if pos_tags is not None:
                    pos_tags = pos_tags.to(device)
                    
                elem_types = batch.get('elem_types', None)
                if elem_types is not None:
                    elem_types = elem_types.to(device)
                    
                syntax_features = batch.get('syntax_features', None)
                if syntax_features is not None:
                    syntax_features = syntax_features.to(device)
                    
                tag_relations = batch.get('tag_relations', None)
                if tag_relations is not None:
                    tag_relations = tag_relations.to(device)
                    
                subst_nodes = batch.get('subst_nodes', None)
                if subst_nodes is not None:
                    subst_nodes = subst_nodes.to(device)
                    
                adj_nodes = batch.get('adj_nodes', None)
                if adj_nodes is not None:
                    adj_nodes = adj_nodes.to(device)
                    
                rel_positions = batch.get('rel_positions', None)
                if rel_positions is not None:
                    rel_positions = rel_positions.to(device)
                
                # 순전파
                outputs = model(
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
                
                if num_classes is None and len(outputs.shape) >= 2:
                    num_classes = outputs.size(-1)
                
                # 손실 계산 (outputs 형태에 따라 조정)
                if len(outputs.shape) == 2:  # [batch_size, num_classes]
                    loss = criterion(outputs, labels)
                    probs = torch.softmax(outputs, dim=-1)  # 확률값 계산
                    preds = outputs.argmax(dim=-1)  # [batch_size]
                    
                    # 예측값, 레이블, 확률값 저장
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    
                elif len(outputs.shape) == 3:  # [batch_size, seq_len, num_classes]
                    loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                    probs = torch.softmax(outputs, dim=-1)  # 확률값 계산
                    preds = outputs.argmax(dim=-1)  # [batch_size, seq_len]
                    
                    # 배치의 각 시퀀스에 대해 마스킹된 위치의 예측값과 실제값 저장
                    for i in range(outputs.size(0)):
                        mask = attention_mask[i].bool()
                        if mask.any():
                            masked_preds = preds[i][mask]
                            masked_labels = labels[i][mask]
                            masked_probs = probs[i][mask]
                            all_preds.extend(masked_preds.cpu().numpy())
                            all_labels.extend(masked_labels.cpu().numpy())
                            all_probs.extend(masked_probs.cpu().numpy())
                else:
                    raise ValueError(f"Unexpected output shape: {outputs.shape}")
                
                total_loss += loss.item()
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                logger.error(f"Batch shapes - token_ids: {token_ids.shape}, "
                           f"attention_mask: {attention_mask.shape}, "
                           f"labels: {labels.shape}, "
                           f"outputs: {outputs.shape}")
                raise
    
    # 메트릭 계산
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    if len(all_preds) == 0:
        logger.warning("No valid predictions found!")
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': 0.0,
            'classification_report': "No valid predictions",
            'confusion_matrix': np.array([[0]])
        }
    
    # 기본 메트릭
    accuracy = (all_preds == all_labels).mean()
    avg_loss = total_loss / len(val_loader)
    
    # 클래스별 메트릭
    unique_classes = np.unique(np.concatenate([all_preds, all_labels]))
    class_names = [str(i) for i in range(len(unique_classes))]
    if vocab is not None and hasattr(vocab, 'idx2label'):
        class_names = [vocab.idx2label.get(i, str(i)) for i in unique_classes]
    
    # 분류 보고서
    report = classification_report(
        all_labels,
        all_preds,
        labels=unique_classes,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    
    # 혼동 행렬
    cm = confusion_matrix(all_labels, all_preds, labels=unique_classes)
    
    # ROC-AUC와 PR-AUC 계산 (이진 분류인 경우)
    roc_auc = None
    pr_auc = None
    if len(unique_classes) == 2:
        from sklearn.metrics import roc_auc_score, average_precision_score
        try:
            roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
            pr_auc = average_precision_score(all_labels, all_probs[:, 1])
        except Exception as e:
            logger.warning(f"Failed to calculate ROC-AUC or PR-AUC: {str(e)}")
    
    # 혼동 행렬 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # 결과 반환
    results = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'classes': unique_classes.tolist()
    }
    
    # 이진 분류 메트릭 추가
    if roc_auc is not None:
        results['roc_auc'] = roc_auc
    if pr_auc is not None:
        results['pr_auc'] = pr_auc
    
    return results


class EarlyStopping:
    """Early stopping 구현"""
    def __init__(self, patience=10, min_delta=0.001, min_epochs=20):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss, epoch):
        if epoch < self.min_epochs:
            return False
            
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
        return self.early_stop


def train_epoch(model, train_loader, optimizer, criterion, config, device):
    """
    한 에포크 동안의 학습 수행
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in train_loader:
        # 기본 입력 텐서들을 device로 이동
        token_ids = batch['token_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 구문 관련 특성들을 device로 이동 (없을 수 있음)
        pos_tags = batch.get('pos_tags', None)
        if pos_tags is not None:
            pos_tags = pos_tags.to(device)
            
        elem_types = batch.get('elem_types', None)
        if elem_types is not None:
            elem_types = elem_types.to(device)
            
        syntax_features = batch.get('syntax_features', None)
        if syntax_features is not None:
            syntax_features = syntax_features.to(device)
            
        tag_relations = batch.get('tag_relations', None)
        if tag_relations is not None:
            tag_relations = tag_relations.to(device)
            
        subst_nodes = batch.get('subst_nodes', None)
        if subst_nodes is not None:
            subst_nodes = subst_nodes.to(device)
            
        adj_nodes = batch.get('adj_nodes', None)
        if adj_nodes is not None:
            adj_nodes = adj_nodes.to(device)
            
        rel_positions = batch.get('rel_positions', None)
        if rel_positions is not None:
            rel_positions = rel_positions.to(device)
        
        optimizer.zero_grad()
        
        # 순전파
        outputs = model(
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
        
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        loss.backward()
        
        # 그래디언트 클리핑
        if config['optimization'].get('clip_grad_norm', False):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['optimization']['max_grad_norm']
            )
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # 정확도 계산
        pred = outputs.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
    
    return total_loss / len(train_loader), correct / total


def train_with_validation(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    scaler,
    config,
    device,
    vocab_size,
    logger
):
    """
    모델 훈련 및 검증 수행
    
    Args:
        model: 훈련할 모델
        train_loader: 훈련 데이터 로더
        val_loader: 검증 데이터 로더
        optimizer: 옵티마이저
        scheduler: 학습률 스케줄러
        criterion: 손실 함수
        scaler: 그래디언트 스케일러
        config: 학습 설정
        device: 연산 장치
        vocab_size: 어휘 크기
        logger: 로거
    """
    best_model = None
    best_val_loss = float('inf')
    
    # Early stopping 초기화
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping']['patience'],
        min_delta=config['training']['early_stopping']['min_delta'],
        min_epochs=config['training']['min_epochs']
    )
    
    for epoch in range(config['training']['epochs']):
        try:
            # 훈련
            train_loss, train_accuracy = train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                config=config,
                device=device
            )
            
            # 검증
            val_results = evaluate_model(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device
            )
            
            val_loss = val_results['loss']
            val_accuracy = val_results['accuracy']
            
            # 학습률 스케줄러 업데이트
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            # 로깅
            logger.info(
                f'Epoch {epoch+1}/{config["training"]["epochs"]} - '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, '
                f'LR: {optimizer.param_groups[0]["lr"]:.2e}'
            )
            
            # 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)
                
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics={'loss': val_loss, 'accuracy': val_accuracy},
                    save_path=os.path.join(config['logging']['save_dir'], 'best_model.pt')
                )
                logger.info(f'New best model saved! (Val Loss: {val_loss:.4f})')
            
            # Early stopping 체크
            if early_stopping(val_loss, epoch):
                logger.info(f'Early stopping triggered after {epoch+1} epochs')
                break
            
        except Exception as e:
            logger.error(f"훈련 중 오류 발생: {str(e)}")
            raise
    
    return best_model


def main():
    parser = argparse.ArgumentParser(description='TAG Transformer 모델 훈련')
    parser.add_argument('--config', type=str, required=True, help='설정 파일 경로')
    parser.add_argument('--train_data', type=str, required=True, help='훈련 데이터 파일 경로')
    parser.add_argument('--val_data', type=str, required=True, help='검증 데이터 파일 경로')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'], help='실행 모드')
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    config['data']['train_data'] = args.train_data
    config['data']['val_data'] = args.val_data
    
    # 학습 설정 조정
    config['training']['batch_size'] = 32
    config['training']['learning_rate'] = 5e-5
    config['training']['weight_decay'] = 0.01
    config['optimization']['clip_grad_norm'] = True
    config['optimization']['max_grad_norm'] = 1.0
    
    # 부동소수점 정밀도 설정
    torch.set_float32_matmul_precision('high')
    
    # wandb 초기화
    if args.mode == 'train':
        wandb.init(
            project=config['logging']['project_name'],
            config=config
        )
    
    # 출력 디렉토리 생성
    os.makedirs(config['logging']['save_dir'], exist_ok=True)
    
    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    try:
        # 데이터셋 로드
        logger.info("Loading datasets...")
        train_dataset = TAGDataset(config['data']['train_data'])
        val_dataset = TAGDataset(config['data']['val_data'])
        
        # 데이터 로더 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['data']['num_workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=True
        )
        
        # 모델 생성
        logger.info("Creating model...")
        model = create_syntax_aware_transformer(
            vocab_size=len(train_dataset.vocab),
            max_length=config['model']['max_length'],
            d_model=config['model']['d_model'],
            n_heads=config['model']['n_heads'],
            n_encoder_layers=config['model']['n_encoder_layers'],
            d_ff=config['model']['d_ff'],
            dropout=config['model']['dropout'],
            num_classes=config['model']['num_classes'],
            use_checkpoint=config['optimization']['gradient_checkpointing']
        ).to(device)
        
        # 손실 함수
        criterion = nn.CrossEntropyLoss(
            label_smoothing=config['training']['label_smoothing']
        )
        
        if args.mode == 'train':
            # 옵티마이저
            optimizer = AdamW(
                model.parameters(),
                lr=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay'],
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
            # 학습률 스케줄러
            num_training_steps = len(train_loader) * config['training']['epochs']
            num_warmup_steps = num_training_steps // 10
            
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
            
            # 모델 훈련
            logger.info('Starting training...')
            best_model = train_with_validation(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                scaler=amp.GradScaler(),
                config=config,
                device=device,
                vocab_size=len(train_dataset.vocab),
                logger=logger
            )
            logger.info('Training completed!')
            
        else:  # 평가 모드
            # 저장된 모델 로드
            checkpoint = torch.load(
                os.path.join(config['logging']['save_dir'], 'best_model.pt'),
                map_location=device,
                weights_only=False  # PyTorch 2.6 이상 버전 호환성을 위해 추가
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 모델 평가
            logger.info('Starting evaluation...')
            eval_results = evaluate_model(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                vocab=train_dataset.vocab
            )
            
            # 평가 결과 출력
            logger.info('\n' + '='*50)
            logger.info('Evaluation Results:')
            logger.info(f"Loss: {eval_results['loss']:.4f}")
            logger.info(f"Accuracy: {eval_results['accuracy']:.4f}")
            if 'roc_auc' in eval_results:
                logger.info(f"ROC-AUC: {eval_results['roc_auc']:.4f}")
            if 'pr_auc' in eval_results:
                logger.info(f"PR-AUC: {eval_results['pr_auc']:.4f}")
            logger.info('\nClassification Report:')
            logger.info('\n' + eval_results['classification_report'])
            logger.info('\nConfusion matrix saved as confusion_matrix.png')
            logger.info('='*50)
    
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
        raise
    
    finally:
        if args.mode == 'train':
            wandb.finish()


if __name__ == "__main__":
    main()