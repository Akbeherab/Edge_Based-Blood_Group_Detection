
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# Settings
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 8
LR = 0.0001
DATA_PATH = r"C:\Users\nielitpatna\Desktop\blood dataset\split_dataset"

print(f"Device: {device}\n")

# -------------------------------
# Data Augmentation
# -------------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load data
train_ds = datasets.ImageFolder(DATA_PATH + '/train', train_transform)
val_ds = datasets.ImageFolder(DATA_PATH + '/val', test_transform)
test_ds = datasets.ImageFolder(DATA_PATH + '/test', test_transform)

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=0)

classes = train_ds.classes
print(f"Classes: {classes}")
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}\n")

# -------------------------------
# VGG19 - FULL TRAINING (No Freezing)
# -------------------------------
print("Loading VGG19 with pretrained weights...")
model = models.vgg19(pretrained=True)

# Replace final classifier layer
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, NUM_CLASSES)

# ALL LAYERS WILL BE TRAINED
for param in model.parameters():
    param.requires_grad = True

model = model.to(device)

print("VGG19 - FULL TRAINING MODE")
print(f"Original VGG19 classifier replaced: 4096 → {NUM_CLASSES}")
print(f"All layers trainable")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([
    {'params': model.features.parameters(), 'lr': LR/10},  # Lower LR for backbone
    {'params': model.classifier.parameters(), 'lr': LR}    # Higher LR for classifier
], lr=LR)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

# -------------------------------
# Training Functions
# -------------------------------
def train(model, loader):
    model.train()
    loss_sum = correct = total = 0
    for inputs, labels in tqdm(loader, desc='Train', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        loss_sum += loss.item() * inputs.size(0)
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
    return loss_sum / total, correct / total

def validate(model, loader):
    model.eval()
    loss_sum = correct = total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Val ', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss_sum += loss.item() * inputs.size(0)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    return loss_sum / total, correct / total

# -------------------------------
# Training Loop
# -------------------------------
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_acc = 0.0

print(f"{'='*80}")
print("TRAINING VGG19 - FULL BASELINE (NO FREEZING)")
print(f"{'='*80}\n")

for epoch in range(EPOCHS):
    tr_loss, tr_acc = train(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)
    
    scheduler.step(val_loss)
    
    history['train_loss'].append(tr_loss)
    history['train_acc'].append(tr_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}", end="")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_vgg19_full_baseline.pth')
        print(" >>> SAVED (Best Model)")
    else:
        print()

print(f"\n{'='*80}")
print("TRAINING COMPLETE!")
print(f"{'='*80}\n")

# -------------------------------
# Save Training Curves
# -------------------------------
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history['train_acc'], 'b-o', label='Train Accuracy', linewidth=2)
plt.plot(history['val_acc'], 'r-s', label='Validation Accuracy', linewidth=2)
plt.title('VGG19 Full Baseline - Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])

plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
plt.plot(history['val_loss'], 'r-s', label='Validation Loss', linewidth=2)
plt.title('VGG19 Full Baseline - Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('vgg19_full_baseline_training_curves.png', dpi=300, bbox_inches='tight')
print("Saved: vgg19_full_baseline_training_curves.png")
plt.show()

# -------------------------------
# Test & Confusion Matrix
# -------------------------------
print("Loading best model for final testing...")
model.load_state_dict(torch.load('best_vgg19_full_baseline.pth'))

test_loss, test_acc = validate(model, test_loader)
print(f"\nFinal Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)\n")

# Predictions
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Final Testing"):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, pred = torch.max(outputs, 1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.numpy())

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(11, 9))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes,
            cbar_kws={'label': 'Count'}, annot_kws={'size': 11})
plt.title('Confusion Matrix - VGG19 Full Baseline', fontsize=15, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('vgg19_full_baseline_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Saved: vgg19_full_baseline_confusion_matrix.png")
plt.show()

# -------------------------------
# Final Results
# -------------------------------
print(f"\n{'='*80}")
print("FINAL RESULTS - VGG19 FULL BASELINE")
print(f"{'='*80}")
print(f"Architecture       : VGG19 (Full Training - No Freezing)")
print(f"Classifier         : Original VGG19 last layer replaced (4096 → {NUM_CLASSES})")
print(f"Learning Rate      : Backbone = {LR/10}, Classifier = {LR}")
print(f"Best Val Accuracy  : {best_acc:.4f}")
print(f"Test Accuracy      : {test_acc:.4f}")
print(f"Total Epochs       : {len(history['train_loss'])}")
print(f"{'='*80}\n")

print("All files saved:")
print("   • best_vgg19_full_baseline.pth")
print("   • vgg19_full_baseline_training_curves.png")
print("   • vgg19_full_baseline_confusion_matrix.png")
