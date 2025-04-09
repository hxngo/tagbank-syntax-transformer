"""
Syntax-aware Transformer 모델 평가 스크립트
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import json
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

# 로컬 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.tag_dataset import create_tagbank_dataloaders
from src.models.syntax_transformer import create_syntax_aware_transformer, create_pad_mask


def evaluate_model(model, dataloader, device, criterion=None):
    """모델 평가"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
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
            
            # 예측 및 라벨 수집
            if labels is not None:
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # 손실 계산
                if criterion is not None:
                    loss = criterion(logits, labels)
                    total_loss += loss.item()
    
    # 결과 반환
    results = {
        'predictions': all_predictions,
        'labels': all_labels
    }
    
    if criterion is not None:
        results['loss'] = total_loss / len(dataloader)
    
    return results


def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    """혼동 행렬 시각화"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    display.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.close()


def analyze_attention(model, dataloader, device, output_dir, num_samples=5):
    """어텐션 패턴 분석"""
    model.eval()
    
    # 어텐션 추출 설정 (TAGAttention 레이어의 전방 패스 반환값 수정 필요)
    attention_weights = []
    processed_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if processed_samples >= num_samples:
                break
                
            # 데이터를 디바이스로 이동
            token_ids = batch['token_ids'].to(device)
            pos_tags = batch['pos_tags'].to(device)
            elem_types = batch['elem_types'].to(device)
            syntax_features = batch['syntax_features'].to(device)
            heads = batch['heads'].to(device)
            
            # 마스크 생성
            mask = create_pad_mask(token_ids).to(device)
            
            # 모델 전방 패스 - 어텐션 가중치 수집 코드 추가 필요
            # 다음은 참고용 의사 코드입니다:
            # 실제 구현에서는 모델 내부에서 어텐션 가중치를 추출하는 방식으로 수정 필요
            """
            attn_weights = []
            
            def attention_hook(module, input, output):
                # 출력의 두 번째 값이 어텐션 가중치
                attn_weights.append(output[1].detach())
            
            # 모델의 각 어텐션 레이어에 후크 등록
            hooks = []
            for layer in model.encoder.layers:
                hook = layer.tag_attn.register_forward_hook(attention_hook)
                hooks.append(hook)
            
            # 모델 전방 패스
            _ = model(token_ids, pos_tags, elem_types, syntax_features, heads, mask)
            
            # 후크 제거
            for hook in hooks:
                hook.remove()
            """
            
            # 여기서는 더미 데이터 사용 (실제 구현 시 수정 필요)
            dummy_attn = torch.randn(token_ids.size(0), 8, token_ids.size(1), token_ids.size(1))
            attention_weights.append(dummy_attn.cpu().numpy())
            
            processed_samples += token_ids.size(0)
    
    # 어텐션 시각화 (첫 번째 샘플의 마지막 레이어 시각화)
    if attention_weights:
        os.makedirs(os.path.join(output_dir, 'attention_maps'), exist_ok=True)
        
        # 각 샘플에 대해 어텐션 맵 생성
        for sample_idx in range(min(num_samples, len(attention_weights[0]))):
            plt.figure(figsize=(12, 10))
            
            # 마지막 레이어의 각 헤드에 대한 어텐션 맵
            sample_attn = attention_weights[0][sample_idx]
            num_heads = sample_attn.shape[0]
            
            for head_idx in range(num_heads):
                plt.subplot(2, 4, head_idx + 1)
                plt.imshow(sample_attn[head_idx], cmap='viridis')
                plt.title(f'Head {head_idx+1}')
                plt.colorbar()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'attention_maps', f'sample_{sample_idx+1}_attention.png'))
            plt.close()


def main(args):
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'evaluation.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 데이터 로드
    logger.info("Loading test data...")
    dataloaders = create_tagbank_dataloaders(
        train_path=args.train_data if args.train_data else args.test_data,  # 어휘 구축용
        test_path=args.test_data,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_mwe=args.use_mwe
    )
    
    test_dataloader = dataloaders['test']
    dataset = test_dataloader.dataset
    vocab_size = len(dataset.vocab)
    
    # 모델 로드
    logger.info(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    model = create_syntax_aware_transformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_encoder_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_length=args.max_length,
        dropout=args.dropout,
        num_classes=args.num_classes
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # 손실 함수
    criterion = nn.CrossEntropyLoss()
    
    # 모델 평가
    logger.info("Evaluating model...")
    results = evaluate_model(model, test_dataloader, device, criterion)
    
    # 평가 지표 계산
    if 'labels' in results and results['labels']:
        logger.info(f"Test Loss: {results['loss']:.4f}")
        
        # 분류 보고서
        y_true = results['labels']
        y_pred = results['predictions']
        
        class_names = args.class_names.split(',') if args.class_names else [str(i) for i in range(args.num_classes)]
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
        logger.info("\nClassification Report:\n" + report)
        
        # 혼동 행렬
        if args.plot_cm:
            logger.info("Plotting confusion matrix...")
            cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
            plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
            logger.info(f"Confusion matrix saved to {cm_path}")
        
        # 예측 결과 저장
        predictions_path = os.path.join(args.output_dir, 'predictions.json')
        with open(predictions_path, 'w') as f:
            json.dump({
                'true_labels': results['labels'],
                'predictions': results['predictions']
            }, f)
        logger.info(f"Predictions saved to {predictions_path}")
    
    # 어텐션 분석
    if args.analyze_attention:
        logger.info("Analyzing attention patterns...")
        analyze_attention(model, test_dataloader, device, args.output_dir, args.num_attn_samples)
        logger.info(f"Attention maps saved to {os.path.join(args.output_dir, 'attention_maps')}")
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Syntax-aware Transformer model")
    
    # 데이터 인자
    parser.add_argument("--test_data", type=str, required=True, help="Test data path")
    parser.add_argument("--train_data", type=str, default=None, help="Train data path (for vocabulary)")
    parser.add_argument("--use_mwe", action="store_true", help="Use multi-word expressions")
    
    # 모델 인자
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of encoder layers")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--class_names", type=str, default=None, help="Comma-separated class names")
    
    # 평가 인자
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--plot_cm", action="store_true", help="Plot confusion matrix")
    parser.add_argument("--analyze_attention", action="store_true", help="Analyze attention patterns")
    parser.add_argument("--num_attn_samples", type=int, default=5, help="Number of samples for attention analysis")
    parser.add_argument("--output_dir", type=str, default="./evaluation", help="Output directory")
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
