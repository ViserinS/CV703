import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.models.convnext import convnext_tiny
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import time
from datetime import datetime
import json
from thop import profile, clever_format

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义数据集根目录
fgvc_aircraft_root = './data'
imagewoof_root = './data/imagewoof2-160'
flowers102_root = './data/flowers-102'

# 定义数据变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

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

# 更新标签函数
def update_targets(dataset, start_label):
    dataset.targets = [label for _, label in dataset]
    unique_labels = set(dataset.targets)
    return dataset, start_label + len(unique_labels)

def load_datasets():
    # 加载数据集的代码保持不变...
    train_dir = os.path.join(imagewoof_root, 'imagewoof2-160/train')
    valid_dir = os.path.join(imagewoof_root, 'imagewoof2-160/val')
    imagewoof_train = ImageFolder(root=train_dir, transform=transform)
    imagewoof_val = ImageFolder(root=valid_dir, transform=transform)

    start_label = 0
    imagewoof_train, start_label_fgcv = update_targets(imagewoof_train, start_label)
    imagewoof_val, _ = update_targets(imagewoof_val, start_label)

    fgvc_trainval = ModifiedFGVCAircraft(root=fgvc_aircraft_root, split='trainval', transform=transform, startlabel=start_label_fgcv)
    fgvc_test = ModifiedFGVCAircraft(root=fgvc_aircraft_root, split='test', transform=transform, startlabel=start_label_fgcv)
    
    fgvc_trainval, start_label_flowers = update_targets(fgvc_trainval, start_label_fgcv)
    fgvc_test, _ = update_targets(fgvc_test, start_label_fgcv)

    flowers_train = ModifiedFlowers102(root=flowers102_root, split='test', transform=transform, startlabel=start_label_flowers)
    flowers_test = ModifiedFlowers102(root=flowers102_root, split='train', transform=transform, startlabel=start_label_flowers)
    
    flowers_train, _ = update_targets(flowers_train, start_label_flowers)
    flowers_test, _ = update_targets(flowers_test, start_label_flowers)

    train_dataset = ConcatDataset([imagewoof_train, fgvc_trainval, flowers_train])
    test_dataset = ConcatDataset([imagewoof_val, fgvc_test, flowers_test])
    
    return train_dataset, test_dataset


def create_model(num_classes):
    model = convnext_tiny(pretrained=True)
    model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
    
    for name, param in model.named_parameters():
        if 'stages.0' in name or 'stages.1' in name:
            param.requires_grad = False
    
    # Calculate total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return model.to(device), total_params, trainable_params

def calculate_flops(model, input_size=(1, 3, 224, 224)):
    """计算模型的FLOPs"""
    input_tensor = torch.randn(input_size).to(device)
    flops, params = profile(model, inputs=(input_tensor,))
    return flops, params

def train_epoch(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    epoch_start_time = time.time()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
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
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss/len(val_loader), 100.*correct/total

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth'):
    """保存检查点"""
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(save_dir, 'best_model.pth')
        torch.save(state, best_filepath)

def load_checkpoint(model, optimizer, scheduler, save_dir):
    """加载检查点"""
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        train_history = checkpoint.get('train_history', [])
        return start_epoch, best_acc, train_history
    else:
        return 0, 0, []
def save_log(log_file, message):
    """保存训练日志到文件"""
    with open(log_file, 'a') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")


def save_training_history(history, save_dir):
    """保存训练历史到JSON文件"""
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)


def main():
    # 创建保存目录
    save_dir = os.path.join('experiments', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, 'training.log')
    
    # 加载数据集
    print("Loading datasets...")
    save_log(log_file, "Loading datasets...")
    train_dataset, test_dataset = load_datasets()
    
    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 创建模型和优化器
    num_classes = 212
    print(f"Creating model with {num_classes} classes...")
    model, total_params, trainable_params = create_model(num_classes)
    
    # 计算FLOPs
    flops, _ = calculate_flops(model)
    flops_readable, _ = clever_format([flops, 0], "%.3f")
    
    # 记录模型统计信息
    model_stats = f"""Model Statistics:
    Total Parameters: {total_params:,}
    Trainable Parameters: {trainable_params:,}
    FLOPs: {flops_readable}
    """
    print(model_stats)
    save_log(log_file, model_stats)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-4, 
        weight_decay=1e-2
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    
    # 尝试加载检查点
    start_epoch, best_acc, train_history = load_checkpoint(model, optimizer, scheduler, save_dir)
    
    # 训练参数
    num_epochs = 100
    patience = 10
    patience_counter = 0
    total_training_time = 0
    
    print(f"Starting training from epoch {start_epoch}...")
    save_log(log_file, f"Starting training from epoch {start_epoch}...")
    start_time = time.time()
    
    try:
        for epoch in range(start_epoch, num_epochs):
            # 训练和验证
            train_loss, train_acc, epoch_time = train_epoch(model, train_dataloader, criterion, optimizer, epoch)
            val_loss, val_acc = validate(model, test_dataloader, criterion)
            
            total_training_time += epoch_time
            
            # 更新学习率
            scheduler.step()
            
            # 记录训练信息
            epoch_info = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time,
                'total_training_time': total_training_time
            }
            train_history.append(epoch_info)
            
            # 保存日志
            log_message = f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}, ' \
                         f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}, ' \
                         f'LR: {optimizer.param_groups[0]["lr"]:.6f}, ' \
                         f'Epoch Time: {epoch_time:.2f}s, ' \
                         f'Total Training Time: {total_training_time/3600:.2f}h'
            print(log_message)
            save_log(log_file, log_message)
            
            # 保存检查点
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'train_history': train_history,
                'total_training_time': total_training_time
            }
            save_checkpoint(checkpoint_state, is_best, save_dir)
            save_training_history(train_history, save_dir)
            
            # 早停
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch} epochs')
                save_log(log_file, f'Early stopping triggered after {epoch} epochs')
                break
                
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving checkpoint...")
        save_log(log_file, "Training interrupted by user")
        save_final_stats(save_dir, log_file, total_training_time, best_acc, flops, total_params, trainable_params)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        save_log(log_file, f"Error occurred: {str(e)}")
        save_final_stats(save_dir, log_file, total_training_time, best_acc, flops, total_params, trainable_params)
        raise e
    
    # 保存最终统计信息
    save_final_stats(save_dir, log_file, total_training_time, best_acc, flops, total_params, trainable_params)

def save_final_stats(save_dir, log_file, total_training_time, best_acc, flops, total_params, trainable_params):
    """保存最终的统计信息"""
    flops_readable, _ = clever_format([flops, 0], "%.3f")
    final_stats = f"""
Final Training Statistics:
    Total Training Time: {total_training_time/3600:.2f} hours
    Best Validation Accuracy: {best_acc:.2f}%
    Model FLOPs: {flops_readable}
    Total Parameters: {total_params:,}
    Trainable Parameters: {trainable_params:,}
    """
    print(final_stats)
    save_log(log_file, final_stats)
    
    # 保存为JSON格式
    stats_dict = {
        'total_training_time_hours': total_training_time/3600,
        'best_validation_accuracy': best_acc,
        'flops': flops,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params
    }
    
    stats_path = os.path.join(save_dir, 'final_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats_dict, f, indent=4)

if __name__ == '__main__':
    main()