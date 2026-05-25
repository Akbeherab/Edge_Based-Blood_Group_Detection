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

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Parameters
IMG_HEIGHT = 224  # ResNet-18 standard input size
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 8
LEARNING_RATE = 0.0001

# Dataset path
dataset_path = r"C:\Users\nielitpatna\Desktop\blood dataset\split_dataset"

# Verify dataset exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

# Data transforms (using ImageNet statistics for ResNet)
train_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
print("\nLoading datasets...")
train_dataset = datasets.ImageFolder(
    os.path.join(dataset_path, 'train'),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    os.path.join(dataset_path, 'val'),
    transform=val_test_transform
)

test_dataset = datasets.ImageFolder(
    os.path.join(dataset_path, 'test'),
    transform=val_test_transform
)

# Data loaders - Adjusted num_workers for Windows
num_workers = 0 if os.name == 'nt' else 4  # 0 for Windows, 4 for Linux/Mac

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=num_workers, 
    pin_memory=True if torch.cuda.is_available() else False
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=num_workers, 
    pin_memory=True if torch.cuda.is_available() else False
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=num_workers, 
    pin_memory=True if torch.cuda.is_available() else False
)

# Get class names
class_names = train_dataset.classes
print(f"\nClasses: {class_names}")
print(f"Number of training images: {len(train_dataset)}")
print(f"Number of validation images: {len(val_dataset)}")
print(f"Number of test images: {len(test_dataset)}\n")

# Modified ResNet-18 Model with 2 FC Layers
class BloodCellResNet18(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super(BloodCellResNet18, self).__init__()
        
        # Load pretrained ResNet-18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Get the number of features from the last layer
        num_features = self.resnet.fc.in_features  # 512 for ResNet-18
        
        # Replace the final fully connected layer with 2 FC layers
        self.resnet.fc = nn.Sequential(
            # FC Layer 1
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.BatchNorm1d(256),
            
            # FC Layer 2 (Output Layer)
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.resnet(x)

# Initialize model
print("Initializing ResNet-18 model with 2 FC layers...")
model = BloodCellResNet18(num_classes=NUM_CLASSES, pretrained=True).to(device)

# Print model architecture
print("\nModel Architecture:")
print("="*100)
print(model)
print("="*100)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}\n")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

# Training function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

# Validation function
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation', leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

# Training loop
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

best_val_acc = 0.0
patience_counter = 0
early_stopping_patience = 10

print("="*100)
print("Starting Training with ResNet-18 (2 FC Layers)...")
print("="*100 + "\n")

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    # Update scheduler
    scheduler.step(val_loss)
    
    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # Print metrics in one line
    print(f"Epoch {epoch+1:02d}/{EPOCHS} - "
          f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - "
          f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}", end='')
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
        }, 'best_resnet18_2fc_model.pth')
        print(f" >>> SAVED (Val Acc: {val_acc:.4f})")
        patience_counter = 0
    else:
        print()
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= early_stopping_patience:
        print(f"\nEarly stopping triggered after {epoch+1} epochs")
        break

print("\n" + "="*100)
print("Training Completed!")
print("="*100 + "\n")

# Plot training history
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    # Accuracy plot
    axes[0].plot(epochs_range, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2, marker='o', markersize=4)
    axes[0].plot(epochs_range, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
    axes[0].set_title('ResNet-18 (2 FC) Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Loss plot
    axes[1].plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
    axes[1].plot(epochs_range, history['val_loss'], 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
    axes[1].set_title('ResNet-18 (2 FC) Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('resnet18_2fc_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Training history plot saved as 'resnet18_2fc_training_history.png'\n")

plot_history(history)

# Load best model for testing
print("Loading best model...")
checkpoint = torch.load('best_resnet18_2fc_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Best model from epoch {checkpoint['epoch']+1} loaded.\n")

# Test evaluation
print("="*100)
print("Evaluating on Test Set...")
print("="*100 + "\n")

test_loss, test_acc = validate(model, test_loader, criterion, device)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}\n")

# Generate predictions for confusion matrix
def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Generating predictions"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)

predicted_classes, true_classes = get_predictions(model, test_loader, device)

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'},
            annot_kws={'size': 10})
plt.title('Confusion Matrix - Blood Cell Classification (ResNet-18 with 2 FC)', 
          fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('resnet18_2fc_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("Confusion matrix saved as 'resnet18_2fc_confusion_matrix.png'\n")

# Classification Report
print("="*100)
print("Classification Report")
print("="*100 + "\n")
print(classification_report(true_classes, predicted_classes, target_names=class_names, digits=4))

# Per-class accuracy
print("\n" + "="*100)
print("Per-Class Accuracy")
print("="*100 + "\n")

for i, class_name in enumerate(class_names):
    class_correct = cm[i, i]
    class_total = np.sum(cm[i, :])
    class_accuracy = class_correct / class_total if class_total > 0 else 0
    print(f"{class_name:25s}: {class_accuracy*100:6.2f}% ({class_correct}/{class_total})")

# Save final model
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': class_names,
    'test_acc': test_acc,
    'test_loss': test_loss,
}, 'final_resnet18_2fc_model.pth')

# Final summary
print("\n" + "="*100)
print("FINAL SUMMARY - ResNet-18 (2 FC Layers)")
print("="*100)
print(f"Architecture:             ResNet-18 + 2 FC Layers (512 -> 256 -> {NUM_CLASSES})")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")
print(f"Test Accuracy:            {test_acc:.4f}")
print(f"Test Loss:                {test_loss:.4f}")
print(f"Total Epochs:             {len(history['train_loss'])}")
print(f"Model saved as:           'best_resnet18_2fc_model.pth' and 'final_resnet18_2fc_model.pth'")
print("="*100 + "\n")

print("✓ Training complete!")
print("✓ Files saved: resnet18_2fc_training_history.png, resnet18_2fc_confusion_matrix.png")
print("✓ Models saved: best_resnet18_2fc_model.pth, final_resnet18_2fc_model.pth")

# Print model summary
print("\n" + "="*100)
print("MODEL SUMMARY")
print("="*100)
print("\nFully Connected Layers:")
print(model.resnet.fc)
print("="*100 + "\n")