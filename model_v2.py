import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset
from torchvision.datasets import ImageFolder
from transformers import SwinModel

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
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                transforms.RandomErasing(p=0.3)
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize(
                    int(self.size * 1.143),
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
    
class baseSwinTransform:
    def __init__(self):
        self.size = 224
        self.transforms = transforms.Compose([
            transforms.Resize(
                int(self.size * 1.143),
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

class ModifiedFGVCAircraft(torchvision.datasets.FGVCAircraft):
    def __init__(self, root, split='trainval', transform=None, startlabel=0):
        super().__init__(root=root, split=split, download=False, transform=transform)
        self.startlabel = startlabel
        
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        label += self.startlabel
        return image, label

class ModifiedFlowers102(torchvision.datasets.Flowers102):
    def __init__(self, root, split='train', transform=None, startlabel=0):
        super().__init__(root=root, split=split, download=False, transform=transform)
        self.startlabel = startlabel
        
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        label += self.startlabel
        return image, label

class SwinClassifier(nn.Module):
    def __init__(self, num_classes, model_path='./swin-large-patch4-window7-224-in22k'):
        super().__init__()
        self.swin = SwinModel.from_pretrained(model_path)
        
        # 修改模型配置以进行下游任务
        self.swin.config.hidden_size = 1536  # Swin-Large的隐藏层大小
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.swin.config.hidden_size),
            nn.Dropout(0.2),
            nn.Linear(self.swin.config.hidden_size, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        outputs = self.swin(x)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)
    
class SwinBaselineClassifier(nn.Module):
    def __init__(self, num_classes, model_path='./swin-large-patch4-window7-224-in22k'):
        super().__init__()
        self.swin = SwinModel.from_pretrained(model_path)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.swin.config.hidden_size),
            nn.Linear(self.swin.config.hidden_size, num_classes)
        )

    def forward(self, x):
        outputs = self.swin(x)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)


def load_datasets(imagewoof_root='./data/imagewoof2-160', 
                 fgvc_aircraft_root='./data',
                 flowers102_root='./data/flowers-102',
                 is_training=True):
    
    # to baseline model, you need to modify
    # transform = baseSwinTransform(is_training=is_training)
    transform = SwinTransform(is_training=is_training)
    
    
    train_dir = os.path.join(imagewoof_root, 'train' if is_training else 'val')
    
   
    imagewoof = ImageFolder(root=train_dir, transform=transform)
    start_label = len(imagewoof.classes)  
    
    fgvc = ModifiedFGVCAircraft(
        root=fgvc_aircraft_root,
        split='trainval' if is_training else 'test',
        transform=transform,
        startlabel=start_label
    )
    start_label += 100  # FGVC Aircraft有100个类别
    
    flowers = ModifiedFlowers102(
        root=flowers102_root,
        split='train' if is_training else 'test',
        transform=transform,
        startlabel=start_label
    )
    
    for dataset in [imagewoof, fgvc, flowers]:
        if hasattr(dataset, 'targets'):
            print(f"Dataset {type(dataset).__name__} label range: {min(dataset.targets)} - {max(dataset.targets)}")
    
    dataset = ConcatDataset([imagewoof, fgvc, flowers])
    return dataset