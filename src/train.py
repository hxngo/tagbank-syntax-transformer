"""
Syntax-aware Transformer 모델 훈련 스크립트
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import logging
from tqdm import tqdm
from datetime import datetime

# 로컬 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.tag_dataset import create_tagbank_dataloaders
from src.models.syntax_transformer import create_syntax_aware_transformer, create_pad_mask


def set_seed(seed):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """한 에폭 훈련"""
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # 데이터를 디바이스로 이동
        token_ids = batch['token_ids'].to(device)
        pos_tags = batch['pos_tags'].to(device)
        elem_types = batch['elem_types'].to(device)
        syntax_features = batch['syntax_features'].to(device)
        heads = batch['heads'].to(device)
        labels = batch['labels'].to(device) if 'labels' in batch else None
        
        # 마스크 생성
        mask = create_pad_mask(token_ids).to(device)
        
        # 모델 전방 패스
        optimizer.zero_grad()
        logits = model(token_ids, pos_tags, elem_types, syntax_features, heads, mask)
        
        # 손실 계산
        loss = criterion(logits, labels)
        
        # 역전파 및 옵티마이저 스텝
        loss.backward()
        optimizer.step()
        
        # 통계
        epoch_loss += loss.item()
        
        # 정확도 계산 (분류 태스크)
        if labels is not None:
            pred = torch.argmax(logits, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        
        # 진행 바 업데이트
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{correct/total:.4f}" if total > 0 else "N/A"
        })
    
    # 에폭 평균 손실 및 정확도
    avg_loss = epoch_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """모델 평가"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # 데이터를 디바이스로 이동
            token_ids = batch['token_ids'].to(device)
            pos_tags = batch['pos_tags'].to(device)
            elem_types = batch['elem_types'].to(device)
            syntax_features = batch['syntax_features'].to(device)
            heads = batch['heads'].to(device)
            labels = batch['labels'].to(device) if 'labels' in batch else None
            
            # 마스크 생성
            mask = create_pad_mask(token_ids).to(device)
            
            # 모델 전방 패스
            logits = model(token_ids, pos_tags, elem_types, syntax_features, heads, mask)
            
            # 손실 계산
            if labels is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item()
                
                # 정확도 계산
                pred = torch.argmax(logits, dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
    
    # 평균 손실 및 정확도
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def main(args):
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # 시드 설정
    set_seed(args.seed)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 데이터 로더 생성
    logger.info("Loading datasets...")
    dataloaders = create_tagbank_dataloaders(
        train_path=args.train_data,
        valid_path=args.valid_data,
        test_path=args.test_data,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_mwe=args.use_mwe
    )
    
    # 어휘 크기 및 클래스 수 추출
    train_dataset = dataloaders['train'].dataset
    vocab_size = len(train_dataset.vocab)
    num_classes = args.num_classes
    
    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"Number of classes: {num_classes}")
    
    # 모델 생성
    logger.info("Creating model...")
    model = create_syntax_aware_transformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_encoder_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_length=args.max_length,
        dropout=args.dropout,
        num_classes=num_classes
    )
    model.to(device)
    
    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 학습률 스케줄러
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2,
        verbose=True
    )
    
    # 체크포인트 디렉토리
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 훈련 루프
    logger.info("Starting training...")
    best_valid_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}")
        
        # 훈련
        train_loss, train_acc = train_epoch(
            model=model,
            dataloader=dataloaders['train'],
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # 검증
        if 'valid' in dataloaders:
            valid_loss, valid_acc = evaluate(
                model=model,
                dataloader=dataloaders['valid'],
                criterion=criterion,
                device=device
            )
            
            logger.info(f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")
            
            # 학습률 스케줄러 업데이트
            scheduler.step(valid_loss)
            
            # 최고 성능 모델 저장
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                
                model_path = os.path.join(args.output_dir, f"best_model.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'valid_acc': valid_acc,
                    'vocab': train_dataset.vocab
                }, model_path)
                
                logger.info(f"Saved best model to {model_path}")
        
        # 체크포인트 저장
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc
            }, checkpoint_path)
            
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # 테스트 평가
    if 'test' in dataloaders:
        logger.info("Evaluating on test set...")
        
        # 최고 모델 로드
        best_model_path = os.path.join(args.output_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        test_loss, test_acc = evaluate(
            model=model,
            dataloader=dataloaders['test'],
            criterion=criterion,
            device=device
        )
        
        logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Syntax-aware Transformer model")
    
    # 데이터 인자
    parser.add_argument("--train_data", type=str, required=True, help="Training data path")
    parser.add_argument("--valid_data", type=str, default=None, help="Validation data path")
    parser.add_argument("--test_data", type=str, default=None, help="Test data path")
    parser.add_argument("--use_mwe", action="store_true", help="Use multi-word expressions")
    
    # 모델 인자
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of encoder layers")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes for classification")
    
    # 훈련 인자
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every n epochs")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    
    args = parser.parse_args()
    
    main(args)
