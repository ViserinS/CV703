import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from collections import Counter
import time
from datetime import datetime
import json
from thop import profile, clever_format
from torch.cuda.amp import autocast, GradScaler
from transformers import SwinModel

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据集根目录
fgvc_aircraft_root = './data'
imagewoof_root = './data/imagewoof2-160'
flowers102_root = './data/flowers-102'

# 定义数据变换流程
class SwinTransform:
    def __init__(self, is_training=True):
        self.size = 224
        if is_training:
            self.transforms = transforms.Compose([
                transforms.RandomResizedCrop(
                    self.size,
                    scale=(0.8, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandAugment(num_ops=2, magnitude=9),  # 添加RandAugment
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                transforms.RandomErasing(p=0.3)  # 添加随机擦除
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize(
                    int(self.size * 1.143),  # 256/224 ≈ 1.143
                    interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __call__(self, image):
        return self.transforms(image)

# 定义完整的数据转换流程
full_transform = SwinTransform()

# 修改后的 FGVC Aircraft 数据集类
class ModifiedFGVCAircraft(torchvision.datasets.FGVCAircraft):
    def __init__(self, root, split='trainval', transform=None, startlabel=0):
        super(ModifiedFGVCAircraft, self).__init__(root=root, split=split, download=False, transform=transform)
        self.startlabel = startlabel
        
    def __getitem__(self, index):
        image, label = super(ModifiedFGVCAircraft, self).__getitem__(index)
        label += self.startlabel
        return image, label

# 修改后的 Flowers102 数据集类
class ModifiedFlowers102(torchvision.datasets.Flowers102):
    def __init__(self, root, split='train', transform=None, startlabel=0):
        super(ModifiedFlowers102, self).__init__(root=root, split=split, download=False, transform=transform)
        self.startlabel = startlabel
        
    def __getitem__(self, index):
        image, label = super(ModifiedFlowers102, self).__getitem__(index)
        label += self.startlabel
        return image, label

def update_targets(dataset, start_label):
    dataset.targets = [label for _, label in dataset]
    unique_labels = set(dataset.targets)
    return dataset, start_label + len(unique_labels)

def load_datasets():
    # 添加缓存机制
    cache_file = 'dataset_cache.pth'
    if os.path.exists(cache_file):
        print("Loading cached dataset...")
        cache_data = torch.load(cache_file)
        return cache_data['train_dataset'], cache_data['test_dataset']
        
    print("Building dataset from scratch...")
    train_dir = os.path.join(imagewoof_root, 'imagewoof2-160/train')
    valid_dir = os.path.join(imagewoof_root, 'imagewoof2-160/val')
    imagewoof_train = ImageFolder(root=train_dir, transform=full_transform)
    imagewoof_val = ImageFolder(root=valid_dir, transform=full_transform)

    start_label = 0
    imagewoof_train, start_label_fgcv = update_targets(imagewoof_train, start_label)
    imagewoof_val, _ = update_targets(imagewoof_val, start_label)

    fgvc_trainval = ModifiedFGVCAircraft(root=fgvc_aircraft_root, split='trainval', 
                                        transform=full_transform, startlabel=start_label_fgcv)
    fgvc_test = ModifiedFGVCAircraft(root=fgvc_aircraft_root, split='test', 
                                    transform=full_transform, startlabel=start_label_fgcv)
    
    fgvc_trainval, start_label_flowers = update_targets(fgvc_trainval, start_label_fgcv)
    fgvc_test, _ = update_targets(fgvc_test, start_label_fgcv)

    flowers_train = ModifiedFlowers102(root=flowers102_root, split='test', 
                                     transform=full_transform, startlabel=start_label_flowers)
    flowers_test = ModifiedFlowers102(root=flowers102_root, split='train', 
                                    transform=full_transform, startlabel=start_label_flowers)
    
    flowers_train, _ = update_targets(flowers_train, start_label_flowers)
    flowers_test, _ = update_targets(flowers_test, start_label_flowers)

    train_dataset = ConcatDataset([imagewoof_train, fgvc_trainval, flowers_train])
    test_dataset = ConcatDataset([imagewoof_val, fgvc_test, flowers_test])
    
    # 保存缓存
    torch.save({
        'train_dataset': train_dataset,
        'test_dataset': test_dataset
    }, 'dataset_cache.pth')
    
    return train_dataset, test_dataset

class SwinClassifier(nn.Module):
    def __init__(self, num_classes, model_path='./swin-large-patch4-window7-224-in22k'):
        super().__init__()
        self.swin = SwinModel.from_pretrained(model_path)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.swin.config.hidden_size),
            nn.Dropout(0.2),
            nn.Linear(self.swin.config.hidden_size, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, num_classes)
        )
        
        # 解冻更多层以提高性能
        for name, param in self.swin.named_parameters():
            if 'layers.2' in name or 'layers.3' in name:  # 训练最后两个stage
                param.requires_grad = True
            else:
                param.requires_grad = True

    def forward(self, x):
        outputs = self.swin(x)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)

def create_model(num_classes):
    model = SwinClassifier(num_classes)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return model.to(device), total_params, trainable_params

def calculate_flops(model, input_size=(1, 3, 224, 224)):
    input_tensor = torch.randn(input_size).to(device)
    flops, params = profile(model, inputs=(input_tensor,))
    return flops, params

def train_epoch(model, train_loader, criterion, optimizer, epoch, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    epoch_start_time = time.time()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (inputs, targets) in enumerate(pbar):
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
        
        pbar.set_postfix({
            'loss': running_loss/(batch_idx+1),
            'acc': 100.*correct/total
        })
    
    epoch_time = time.time() - epoch_start_time
    return running_loss/len(train_loader), 100.*correct/total, epoch_time

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

def train():
    # 创建保存目录
    save_dir = os.path.join('experiments', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据集
    print("Loading datasets...")
    train_dataset, test_dataset = load_datasets()
    num_epochs = 100
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=64,  # RTX 4090可以支持更大的batch size
        shuffle=True, 
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True  # 保持worker进程存活
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=128,  # 测试时用更大的batch size
        shuffle=False, 
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    # 创建模型和优化器
    num_classes = 212  # 总类别数
    print(f"Creating model with {num_classes} classes...")
    model, total_params, trainable_params = create_model(num_classes)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 添加标签平滑
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-5,  # 降低初始学习率
        weight_decay=0.05,  # 增加权重衰减
        betas=(0.9, 0.999)
    )
    
    # 使用带有预热的余弦退火学习率调度
    warmup_steps = 5  # 预热轮数
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,  # 最大学习率
        total_steps=num_epochs * len(train_dataloader),
        pct_start=0.05,  # 预热比例
        anneal_strategy='cos'
    )
    
    # 创建混合精度训练的scaler
    scaler = GradScaler()
    
    # 训练参数
    
    patience = 10
    best_acc = 0
    patience_counter = 0
    
    try:
        for epoch in range(num_epochs):
            train_loss, train_acc, epoch_time = train_epoch(model, train_dataloader, criterion, optimizer, epoch, scaler)
            val_loss, val_acc = validate(model, test_dataloader, criterion)
            scheduler.step()
            
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # 保存检查点
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
            
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch} epochs')
                break
                
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving final checkpoint...")
    
    print(f"Training completed. Best accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    train()