"""
TAG 구문 정보를 활용하는 Transformer 모델 평가 스크립트
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import logging
import numpy as np
from typing import Dict, List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from tqdm import tqdm

from models.syntax_transformer import create_syntax_aware_transformer, SyntaxAwareTransformer
from data.tag_dataset import TAGDataset, create_tag_dataloaders
from utils.tag_parser import parse_tag_file, extract_features_from_tag_tree


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(model_path, config):
    model = SyntaxAwareTransformer(
        vocab_size=config['model']['vocab_size'],
        d_model=config['model']['d_model'],
        n_encoder_layers=config['model']['n_encoder_layers'],
        n_heads=config['model']['n_heads'],
        d_ff=config['model']['d_ff'],
        max_length=config['model']['max_length'],
        dropout=config['model']['dropout'],
        num_classes=config['model']['num_classes']
    )
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    logger.info("Starting evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="평가 중")):
            logger.info(f"\nProcessing batch {batch_idx + 1}")
            
            # 입력 데이터 확인
            token_ids = batch['token_ids'].to(device)
            pos_tags = batch['pos_tags'].to(device)
            elem_types = batch['elem_types'].to(device)
            syntax_features = batch['syntax_features'].to(device)
            tag_relations = batch['tag_relations'].to(device)
            labels = batch['labels'].to(device)
            mask = batch['attention_mask'].to(device)
            
            logger.info(f"Batch shapes:")
            logger.info(f"- token_ids: {token_ids.shape}")
            logger.info(f"- pos_tags: {pos_tags.shape}")
            logger.info(f"- elem_types: {elem_types.shape}")
            logger.info(f"- syntax_features: {syntax_features.shape}")
            logger.info(f"- tag_relations: {tag_relations.shape}")
            logger.info(f"- labels: {labels.shape}")
            logger.info(f"- mask: {mask.shape}")
            
            outputs = model(
                token_ids=token_ids,
                pos_tags=pos_tags,
                elem_types=elem_types,
                syntax_features=syntax_features,
                tag_relations=tag_relations,
                mask=mask
            )
            
            preds = torch.argmax(outputs, dim=1)
            
            logger.info(f"Predictions for this batch: {preds.cpu().numpy()}")
            logger.info(f"True labels for this batch: {labels.cpu().numpy()}")
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    logger.info("\nEvaluation completed")
    logger.info(f"Total predictions: {len(all_preds)}")
    logger.info(f"Total labels: {len(all_labels)}")
    logger.info(f"Unique predictions: {np.unique(all_preds)}")
    logger.info(f"Unique labels: {np.unique(all_labels)}")
    
    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(cm, classes, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('혼동 행렬')
    plt.ylabel('실제 레이블')
    plt.xlabel('예측 레이블')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_syntax_complexity(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    구문 복잡성에 따른 모델 성능 평가
    
    Args:
        model: 모델
        test_loader: 테스트 데이터로더
        device: 장치
        
    Returns:
        평가 지표 딕셔너리
    """
    model.eval()
    
    # 예측 및 실제 라벨 저장
    y_true = []
    y_pred = []
    
    # 문장 복잡성별 예측/실제 라벨 저장
    complexity_results = {
        'simple': {'true': [], 'pred': []},
        'medium': {'true': [], 'pred': []},
        'complex': {'true': [], 'pred': []}
    }
    
    with torch.no_grad():
        for batch in test_loader:
            # 배치 데이터 추출 및 장치 이동
            token_ids = batch['token_ids'].to(device)
            pos_tags = batch['pos_tags'].to(device)
            elem_types = batch['elem_types'].to(device)
            syntax_features = batch['syntax_features'].to(device)
            heads = batch['heads'].to(device)
            tag_relations = batch['tag_relations'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 구문 복잡성 결정 (실제 구현 필요)
            complexities = determine_complexity(syntax_features)
            
            # 순전파
            logits = model(
                token_ids=token_ids,
                pos_tags=pos_tags,
                elem_types=elem_types,
                syntax_features=syntax_features,
                tag_relations=tag_relations,
                mask=attention_mask
            )
            
            # 예측
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            # 결과 저장
            y_true.extend(labels_np)
            y_pred.extend(preds)
            
            # 복잡성별 결과 저장
            for i, comp in enumerate(complexities):
                complexity_results[comp]['true'].append(labels_np[i])
                complexity_results[comp]['pred'].append(preds[i])
    
    # 전체 성능 평가
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    # 복잡성별 성능 평가
    for comp in complexity_results:
        if complexity_results[comp]['true']:
            comp_true = complexity_results[comp]['true']
            comp_pred = complexity_results[comp]['pred']
            
            comp_acc = accuracy_score(comp_true, comp_pred)
            comp_prec, comp_rec, comp_f1, _ = precision_recall_fscore_support(
                comp_true, comp_pred, average='weighted', zero_division=0
            )
            
            results[f'{comp}_accuracy'] = comp_acc
            results[f'{comp}_precision'] = comp_prec
            results[f'{comp}_recall'] = comp_rec
            results[f'{comp}_f1'] = comp_f1
    
    return results


def determine_complexity(syntax_features: torch.Tensor) -> List[str]:
    """
    구문 특성을 바탕으로 문장 복잡성 결정
    
    Args:
        syntax_features: 구문 특성 텐서 [batch_size, seq_len, 3]
        
    Returns:
        복잡성 레이블 목록 ['simple', 'medium', 'complex']
    """
    # 간단한 복잡성 결정 기준 (실제 구현 시 더 정교한 로직 필요)
    # 구문 특성: [깊이, NP여부, VP여부]
    complexities = []
    
    # 배치의 각 문장에 대해
    for i in range(syntax_features.shape[0]):
        # 최대 깊이 계산
        max_depth = syntax_features[i, :, 0].max().item()
        
        # NP, VP 비율 계산
        np_ratio = syntax_features[i, :, 1].sum().item() / syntax_features[i, :, 1].numel()
        vp_ratio = syntax_features[i, :, 2].sum().item() / syntax_features[i, :, 2].numel()
        
        # 복잡성 결정
        if max_depth <= 3:
            complexities.append('simple')
        elif max_depth <= 6:
            complexities.append('medium')
        else:
            complexities.append('complex')
    
    return complexities


def compare_with_without_syntax(
    test_loader: DataLoader,
    model_with_syntax_path: str,
    model_without_syntax_path: str,
    device: torch.device
) -> Dict[str, Dict[str, float]]:
    """
    구문 정보 사용 여부에 따른 모델 성능 비교
    
    Args:
        test_loader: 테스트 데이터로더
        model_with_syntax_path: 구문 정보 사용 모델 경로
        model_without_syntax_path: 구문 정보 미사용 모델 경로
        device: 장치
        
    Returns:
        비교 결과 딕셔너리
    """
    # 모델 불러오기
    checkpoint_with = torch.load(model_with_syntax_path, map_location=device)
    checkpoint_without = torch.load(model_without_syntax_path, map_location=device)
    
    # 모델 생성
    vocab_size = len(test_loader.dataset.vocab)
    num_classes = max(test_loader.dataset.labels) + 1
    
    model_with = create_syntax_aware_transformer(
        vocab_size=vocab_size,
        num_classes=num_classes
    )
    model_with.load_state_dict(checkpoint_with['model_state_dict'])
    model_with.to(device)
    
    model_without = create_syntax_aware_transformer(
        vocab_size=vocab_size,
        num_classes=num_classes
    )
    model_without.load_state_dict(checkpoint_without['model_state_dict'])
    model_without.to(device)
    
    # 평가
    results_with = evaluate_syntax_complexity(model_with, test_loader, device)
    results_without = evaluate_syntax_complexity(model_without, test_loader, device)
    
    return {
        'with_syntax': results_with,
        'without_syntax': results_without
    }


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Evaluate a syntax-aware transformer model")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--test_data", type=str, required=True, help="Path to the TAG test data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--compare_model", type=str, default=None, help="Path to another model for comparison")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 설정 로드
    config = load_config(args.config)
    logger.info("Configuration loaded:")
    logger.info(f"Model config: {config['model']}")
    
    # 테스트 데이터셋 로드
    logger.info("\nLoading test dataset...")
    test_dataset = TAGDataset(args.test_data, max_seq_len=config['model']['max_length'])
    logger.info(f"Test dataset size: {len(test_dataset)}")
    logger.info(f"Sample from test dataset: {test_dataset[0]}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    # 모델 로드
    logger.info("\nLoading model...")
    model = load_model(args.model_path, config)
    model = model.to(device)
    logger.info("Model loaded successfully")
    
    # 결과 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 평가 수행
    logger.info("Evaluating model...")
    predictions, labels = evaluate(model, test_loader, device)
    
    # 결과 분석
    unique_labels = np.unique(labels)
    class_names = [str(i) for i in range(len(unique_labels))]
    logger.info(f"Found {len(class_names)} classes in the test data")
    
    # 분류 보고서 생성
    report = classification_report(labels, predictions, target_names=class_names)
    with open(output_dir / 'classification_report.txt', 'w') as f:
        f.write(report)
    print("\n분류 보고서:")
    print(report)
    
    # 혼동 행렬 생성 및 저장
    cm = confusion_matrix(labels, predictions)
    plot_confusion_matrix(cm, class_names, output_dir / 'confusion_matrix.png')
    
    # 정확도 계산
    accuracy = (predictions == labels).mean()
    print(f"\n전체 정확도: {accuracy:.4f}")
    
    # 구문 정보 사용 여부 비교 (선택적)
    if args.compare_model:
        logger.info("Comparing models with and without syntax information...")
        comparison = compare_with_without_syntax(
            test_loader, 
            args.model_path, 
            args.compare_model, 
            device
        )
        
        # 비교 결과 출력
        logger.info("Comparison results:")
        logger.info(f"With syntax - accuracy: {comparison['with_syntax']['accuracy']:.4f}, F1: {comparison['with_syntax']['f1']:.4f}")
        logger.info(f"Without syntax - accuracy: {comparison['without_syntax']['accuracy']:.4f}, F1: {comparison['without_syntax']['f1']:.4f}")
        
        # 복잡성별 비교
        for comp in ['simple', 'medium', 'complex']:
            if f'{comp}_accuracy' in comparison['with_syntax'] and f'{comp}_accuracy' in comparison['without_syntax']:
                logger.info(f"{comp.capitalize()} accuracy - with syntax: {comparison['with_syntax'][f'{comp}_accuracy']:.4f}, without syntax: {comparison['without_syntax'][f'{comp}_accuracy']:.4f}")
                logger.info(f"{comp.capitalize()} F1 - with syntax: {comparison['with_syntax'][f'{comp}_f1']:.4f}, without syntax: {comparison['without_syntax'][f'{comp}_f1']:.4f}")


if __name__ == "__main__":
    main()