
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

# -------------------------------
# Data
# -------------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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
# Model: DenseNet-121 + 1 FC
# -------------------------------
model = models.densenet121(pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
model = model.to(device)

print(f"Model: DenseNet-121 + 1 FC Layer")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
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
print("TRAINING DENSENET-121 (1 FC)")
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
        torch.save(model.state_dict(), 'best_densenet121_model.pth')
        print(" >>> SAVED (Best Model)")
    else:
        print()

print(f"\n{'='*80}")
print("TRAINING DONE!")
print(f"{'='*80}\n")

# -------------------------------
# Save Training & Validation Curves
# -------------------------------
print("Saving training and validation curves...")
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history['train_acc'], 'b-o', label='Train Accuracy', linewidth=2)
plt.plot(history['val_acc'], 'r-s', label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])

plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
plt.plot(history['val_loss'], 'r-s', label='Validation Loss', linewidth=2)
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_validation_curves.png', dpi=300, bbox_inches='tight')
print("Saved: training_validation_curves.png")
plt.show()

# -------------------------------
# Test & Confusion Matrix
# -------------------------------
print("Loading best model for testing...")
model.load_state_dict(torch.load('best_densenet121_model.pth'))

test_loss, test_acc = validate(model, test_loader)
print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)\n")

# Predictions
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Predicting"):
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
plt.title('Confusion Matrix - DenseNet-121', fontsize=15, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Saved: confusion_matrix.png")
plt.show()

# -------------------------------
# Final Results
# -------------------------------
print(f"\n{'='*80}")
print("FINAL RESULTS")
print(f"{'='*80}")
print(f"Best Validation Accuracy : {best_acc:.4f}")
print(f"Test Accuracy            : {test_acc:.4f}")
print(f"Total Epochs             : {len(history['train_loss'])}")
print(f"{'='*80}\n")

print("All files saved:")
print("   • best_densenet121_model.pth")
print("   • training_validation_curves.png")
print("   • confusion_matrix.png")
print("\nTraining complete!")
