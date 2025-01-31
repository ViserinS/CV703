import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime
from tqdm import tqdm
import argparse
import time
import wandb
from torch.cuda.amp import autocast, GradScaler
import numpy as np


from model_v2 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, train_loader, criterion, optimizer, epoch, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    epoch_start_time = time.time()
    batch_times = []  
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        batch_start_time = time.time()
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # 使用混合精度训练
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # 使用scaler进行反向传播
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 计算当前batch的处理时间
        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)
        
        # 更新进度条信息
        pbar.set_postfix({
            'loss': running_loss/(batch_idx+1),
            'acc': 100.*correct/total,
            'batch_time': f'{batch_time:.3f}s'
        })
        
        # 记录每个batch的指标到wandb
        if batch_idx % 10 == 0:  # 每10个batch记录一次
            wandb.log({
                'batch_loss': loss.item(),
                'batch_accuracy': 100.*predicted.eq(targets).sum().item()/targets.size(0),
                'batch_time': batch_time,
                'learning_rate': optimizer.param_groups[0]['lr']
            }, step=epoch * len(train_loader) + batch_idx)
    
    epoch_time = time.time() - epoch_start_time
    # 计算和记录epoch级别的指标
    avg_loss = running_loss/len(train_loader)
    accuracy = 100.*correct/total
    avg_batch_time = np.mean(batch_times)
    
    wandb.log({
        'epoch': epoch,
        'train_loss': avg_loss,
        'train_accuracy': accuracy,
        'epoch_time': epoch_time,
        'average_batch_time': avg_batch_time,
        'batches_per_second': 1.0/avg_batch_time
    }, step=epoch * len(train_loader))
    
    return avg_loss, accuracy, epoch_time

def validate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc='Validating'):
            batch_size = inputs.size(0)
            # 测试时增强：水平翻转
            inputs = torch.cat([inputs, torch.flip(inputs, dims=[3])], dim=0)
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            # 合并TTA预测
            outputs = outputs[:batch_size] + outputs[batch_size:]
            outputs = outputs / 2
            
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss/len(val_loader), 100.*correct/total

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth'):
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(save_dir, 'best_model.pth')
        torch.save(state, best_filepath)

def train(args):
    
    wandb.init(
        project="CV703-Assignment1",
        config={
            "architecture": "Swin-Large",
            "dataset": "ImageWoof+FGVC+Flowers102",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_lr": args.max_lr,
            "weight_decay": args.weight_decay,
            "num_classes": args.num_classes,
            "optimizer": "AdamW",
            "scheduler": "OneCycleLR",
        }
    )
    
    # 创建保存目录
    save_dir = os.path.join('experiments', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据集
    print("Loading datasets...")
    train_dataset = load_datasets(
        imagewoof_root=args.imagewoof_root,
        fgvc_aircraft_root=args.fgvc_root,
        flowers102_root=args.flowers_root,
        is_training=True
    )
    val_dataset = load_datasets(
        imagewoof_root=args.imagewoof_root,
        fgvc_aircraft_root=args.fgvc_root,
        flowers102_root=args.flowers_root,
        is_training=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size * 2,
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
 
    print(f"Creating model with {args.num_classes} classes...")

    # 
    # model = SwinClassifier(args.num_classes, model_path=args.train_model_path).to(device)
    
    # use this code to get baseline classifier
    model = SwinBaselineClassifier(args.num_classes, model_path=args.train_model_path).to(device)

 
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
 
    wandb.log({
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
    })
    
  
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # 使用 OneCycleLR 调度器
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        total_steps=args.epochs * len(train_loader),
        pct_start=0.05,
        anneal_strategy='cos'
    )
    print('total_steps: ', args.epochs * len(train_loader))


    # 创建混合精度训练的scaler
    scaler = GradScaler()
    
    best_acc = 0
    patience_counter = 0
    
    try:
        for epoch in range(args.epochs):
            train_loss, train_acc, epoch_time = train_epoch(
                model, train_loader, criterion, optimizer, epoch, scaler
            )
            
            val_loss, val_acc = validate(model, val_loader, criterion)
            
            
            scheduler.step()
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                  f'Time: {epoch_time:.1f}s')
            
           
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }, is_best, save_dir)
            
            # 早停
            if patience_counter >= args.patience:
                print(f'Early stopping triggered after {epoch} epochs')
                break
                
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving final checkpoint...")
    
    print(f"Training completed. Best accuracy: {best_acc:.2f}%")
    
    # 结束wandb记录
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model')
    
    # wandb相关参数
    parser.add_argument('--wandb_project', type=str, default='CV703-Assignment1',
                      help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                      help='WandB entity/username')
    parser.add_argument('--wandb_name', type=str, default=None,
                      help='WandB run name')
    # 数据集路径
    parser.add_argument('--imagewoof_root', type=str, default='./data/imagewoof2-160/imagewoof2-160',
                      help='path to imagewoof dataset')
    parser.add_argument('--fgvc_root', type=str, default='./data',
                      help='path to FGVC-Aircraft dataset')
    parser.add_argument('--flowers_root', type=str, default='./data/flowers-102',
                      help='path to Flowers102 dataset')
    
    # 模型参数
    parser.add_argument('--train_model_path', type=str, 
                      default='./swin-large-patch4-window7-224-in22k',
                      help='path to pretrained model for training')
    parser.add_argument('--num_classes', type=int, default=212,
                      help='number of classes')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64,
                      help='batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                      help='number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                      help='initial learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-4,
                      help='maximum learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                      help='weight decay')
    parser.add_argument('--patience', type=int, default=10,
                      help='patience for early stopping')
    parser.add_argument('--num_workers', type=int, default=8,
                      help='number of workers for data loading')
    
    args = parser.parse_args()
    train(args)