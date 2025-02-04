import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
from pathlib import Path
import os
# 导入模型和数据集定义
from model_v2 import SwinClassifier, load_datasets

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(args):
    """测试模型的函数"""
    # 检查模型文件是否存在
    if not os.path.exists(args.test_model_path):
        raise FileNotFoundError(f"Model file not found: {args.test_model_path}")
        
    print(f"Loading model from {args.test_model_path}")
    
    # 检查数据集路径
    for path, name in [(args.imagewoof_root, 'ImageWoof'),
                      (args.fgvc_root, 'FGVC-Aircraft'),
                      (args.flowers_root, 'Flowers-102')]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} dataset not found at: {path}")
    
 
    print("Loading test dataset...")
    test_dataset = load_datasets(
        imagewoof_root=args.imagewoof_root,
        fgvc_aircraft_root=args.fgvc_root,
        flowers102_root=args.flowers_root,
        is_training=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 加载模型
    model = SwinClassifier(args.num_classes).to(device)
    
    # 加载检查点
    checkpoint = torch.load(args.test_model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("Loading from checkpoint with full state")
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'best_acc' in checkpoint:
            print(f"Best accuracy from training: {checkpoint['best_acc']:.2f}%")
    else:
        print("Loading state dict directly")
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # 初始化评估指标
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    test_loss = 0
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}
    
    print("Starting evaluation...")
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Testing'):
            # 测试时增强：水平翻转
            batch_size = inputs.size(0)
            inputs = torch.cat([inputs, torch.flip(inputs, dims=[3])], dim=0)
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            # 合并TTA预测
            outputs = outputs[:batch_size] + outputs[batch_size:]
            outputs = outputs / 2
            
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 统计每个类别的准确率
            for target, pred in zip(targets, predicted):
                t = target.item()
                if t not in class_correct:
                    class_correct[t] = 0
                    class_total[t] = 0
                class_total[t] += 1
                if pred == target:
                    class_correct[t] += 1
    
    # 计算总体准确率和损失
    overall_acc = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    print(f'\nTest Results:')
    print(f'Average Loss: {avg_loss:.4f}')
    print(f'Overall Accuracy: {overall_acc:.2f}%')
    
    # 计算并保存每个类别的准确率
    class_accuracies = {}
    print('\nPer-class Accuracy:')
    for class_idx in sorted(class_correct.keys()):
        acc = 100. * class_correct[class_idx] / class_total[class_idx]
        class_accuracies[str(class_idx)] = {
            'accuracy': acc,
            'correct': class_correct[class_idx],
            'total': class_total[class_idx]
        }
        print(f'Class {class_idx}: {acc:.2f}% ({class_correct[class_idx]}/{class_total[class_idx]})')
    
    # 保存测试结果
    if args.save_results:
        results = {
            'test_model_path': args.test_model_path,
            'overall_accuracy': overall_acc,
            'average_loss': avg_loss,
            'total_samples': total,
            'total_correct': correct,
            'class_accuracies': class_accuracies
        }
        
        # 创建保存目录
        save_dir = Path('test_results')
        save_dir.mkdir(exist_ok=True)
        
        # 使用模型文件名和时间戳作为结果文件名
        model_name = Path(args.model_path).stem
        save_path = save_dir / f'{model_name}_results.json'
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f'\nTest results saved to {save_path}')
    
    return overall_acc, avg_loss, class_accuracies

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the model')
    # 数据集路径
    parser.add_argument('--imagewoof_root', type=str, default='./data/imagewoof2-160',
                      help='path to imagewoof dataset')
    parser.add_argument('--fgvc_root', type=str, default='./data',
                      help='path to FGVC-Aircraft dataset')
    parser.add_argument('--flowers_root', type=str, default='./data/flowers-102',
                      help='path to Flowers102 dataset')
    
    # 模型参数
    parser.add_argument('--test_model_path', type=str,default='experiments/best_model.pth',
                      help='path to the trained model checkpoint for testing')
    parser.add_argument('--num_classes', type=int, default=212,
                      help='number of classes')
    
    # 测试参数
    parser.add_argument('--batch_size', type=int, default=128,
                      help='batch size for testing')
    parser.add_argument('--num_workers', type=int, default=8,
                      help='number of workers for data loading')
    parser.add_argument('--save_results', action='store_true',
                      help='save test results to a JSON file')
    
    args = parser.parse_args()
    test(args)
