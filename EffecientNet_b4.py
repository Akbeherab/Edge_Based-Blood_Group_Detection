import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import json
import urllib.request

# ================= CONFIGURATION =================
CONFIG = {
    'model_name': 'efficientnet_b4',
    'batch_size': 8,
    'epochs': 50,
    'lr': 0.0001,
    'patience': 10,
    'num_classes': 4,
    'img_size': 380
}

# ================= PATHS =================
BASE_DIR = r"C:\Users\nielitpatna\Desktop\blood dataset"
train_dir = os.path.join(BASE_DIR, "dataset2-master", "dataset2-master", "images", "TRAIN")
test_dir = os.path.join(BASE_DIR, "dataset2-master", "dataset2-master", "images", "TEST")
output_dir = os.path.join(BASE_DIR, "OUTPUT_EFFICIENTNET_B4")
weights_path = os.path.join(BASE_DIR, "efficientnet_b4_weights.pth")

# ================= FUNCTIONS (DEFINE BEFORE MAIN) =================
def download_weights():
    """Download weights bypassing hash check"""
    cache_file = os.path.join(
        os.path.expanduser("~/.cache/torch/hub/checkpoints"),
        "efficientnet_b4_rwightman-7eb33cd5.pth"
    )
    if os.path.exists(cache_file):
        print(f"Removing old cache: {cache_file}")
        os.remove(cache_file)

    if not os.path.exists(weights_path):
        print("Downloading EfficientNet-B4 weights (no hash check)...")
        url = "https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth"
        urllib.request.urlretrieve(url, weights_path)
        size_mb = os.path.getsize(weights_path) / (1024 * 1024)
        print(f"Downloaded: {weights_path} ({size_mb:.1f} MB)")
        if size_mb < 70:
            print("WARNING: File size too small - may be corrupted!")
            print("Try disabling antivirus and re-run.")
    else:
        size_mb = os.path.getsize(weights_path) / (1024 * 1024)
        print(f"Weights found: {weights_path} ({size_mb:.1f} MB)")

def create_model(num_classes, device):
    from torchvision.models import efficientnet_b4

    print("\nBuilding EfficientNet-B4 model...")
    model = efficientnet_b4(weights=None)

    print("Loading pretrained weights...")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Pretrained weights loaded successfully!")

    # Freeze first 100 layers
    for idx, (name, param) in enumerate(model.named_parameters()):
        if idx < 100:
            param.requires_grad = False

    # Replace classifier
    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 64),
    nn.ReLU(),

    nn.Linear(64, 128),
    nn.ReLU(),

    nn.Linear(128, 256),
    nn.ReLU(),

    nn.Linear(256, 512),
    nn.ReLU(),

    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    return model

# ================= MAIN EXECUTION =================
if __name__ == '__main__':
    # Fix for Windows multiprocessing
    from multiprocessing import freeze_support
    freeze_support()
    
    os.makedirs(output_dir, exist_ok=True)

    # ================= DEVICE =================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ================= DOWNLOAD WEIGHTS =================
    download_weights()

    # ================= TRANSFORMS =================
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ================= DATA =================
    print("\nLoading datasets...")
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    print(f"Training samples: {len(train_data)}")
    print(f"Test samples    : {len(test_data)}")
    print(f"Classes         : {train_data.classes}")
    print(f"Class distribution (train): {Counter(train_data.targets)}")

    # Weighted sampler for class imbalance
    class_counts = np.bincount(train_data.targets)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[train_data.targets]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_data, batch_size=CONFIG['batch_size'], sampler=sampler, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

    class_names = train_data.classes
    num_classes = len(class_names)

    # ================= MODEL =================
    model = create_model(num_classes, device)

    # ================= LOSS & OPTIMIZER =================
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    # ================= TRAINING =================
    best_test_acc = 0.0
    epochs_no_improve = 0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'lr': []}

    print("\n" + "=" * 60)
    print("  STARTING TRAINING - EFFICIENTNET-B4")
    print("=" * 60)

    for epoch in range(CONFIG['epochs']):
        # ----- TRAIN -----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            if (batch_idx + 1) % 100 == 0:
                batch_acc = (preds == labels).float().mean().item()
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f} - Acc: {batch_acc:.4f}")

        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # ----- TEST -----
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)

        test_loss = test_loss / len(test_loader)
        test_acc = test_correct / test_total
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['lr'].append(current_lr)

        scheduler.step(test_acc)

        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'train_acc': train_acc,
                'config': CONFIG,
                'class_names': class_names
            }, os.path.join(output_dir, "best_model.pth"))
            print(f"  [Epoch {epoch+1}/{CONFIG['epochs']}] Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | LR: {current_lr:.6f}  ** BEST **")
        else:
            epochs_no_improve += 1
            print(f"  [Epoch {epoch+1}/{CONFIG['epochs']}] Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | LR: {current_lr:.6f}  ({epochs_no_improve}/{CONFIG['patience']})")

        # Early stopping
        if epochs_no_improve >= CONFIG['patience']:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {CONFIG['patience']} epochs)")
            break

    print(f"\nBest test accuracy: {best_test_acc:.4f}")

    # ================= PLOT TRAINING HISTORY =================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history['train_loss'], label='Train Loss', marker='.')
    axes[0].plot(history['test_loss'], label='Test Loss', marker='.')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_acc'], label='Train Acc', marker='.')
    axes[1].plot(history['test_acc'], label='Test Acc', marker='.')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history['lr'], label='Learning Rate', marker='.', color='red')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('LR')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=300)
    plt.close()
    print("Saved: training_curves.png")

    # ================= FINAL EVALUATION =================
    print("\nLoading best model for final evaluation...")
    best_checkpoint = torch.load(os.path.join(output_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # ================= METRICS =================
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision_w = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall_w = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_w = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print("\n" + "=" * 60)
    print("  FINAL RESULTS - EFFICIENTNET-B4")
    print("=" * 60)
    print(f"Best Epoch       : {best_checkpoint['epoch']+1}")
    print(f"Accuracy         : {acc:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall    (macro): {recall:.4f}")
    print(f"F1 Score  (macro): {f1:.4f}")
    print(f"Precision (wt)   : {precision_w:.4f}")
    print(f"Recall    (wt)   : {recall_w:.4f}")
    print(f"F1 Score  (wt)   : {f1_w:.4f}")

    # ================= CLASSIFICATION REPORT =================
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print("\nClassification Report:")
    print(report)

    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write("BLOOD CELL CLASSIFICATION - EFFICIENTNET-B4\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Best Epoch         : {best_checkpoint['epoch']+1}\n")
        f.write(f"Best Test Accuracy : {best_test_acc:.4f}\n\n")
        f.write(f"Final Accuracy         : {acc:.4f}\n")
        f.write(f"Final Precision (macro): {precision:.4f}\n")
        f.write(f"Final Recall    (macro): {recall:.4f}\n")
        f.write(f"Final F1 Score  (macro): {f1:.4f}\n")
        f.write(f"Final Precision (wt)   : {precision_w:.4f}\n")
        f.write(f"Final Recall    (wt)   : {recall_w:.4f}\n")
        f.write(f"Final F1 Score  (wt)   : {f1_w:.4f}\n\n")
        f.write(report)

    # ================= CONFUSION MATRIX =================
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                square=True, linewidths=0.5)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title(f"Confusion Matrix - EfficientNet-B4 (Acc: {acc:.4f})", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: confusion_matrix.png")

    # ================= NORMALIZED CONFUSION MATRIX =================
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap='YlOrRd',
                xticklabels=class_names,
                yticklabels=class_names,
                square=True, linewidths=0.5, vmin=0, vmax=1)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title(f"Normalized Confusion Matrix (Acc: {acc:.4f})", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix_normalized.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: confusion_matrix_normalized.png")

    # ================= PER-CLASS ACCURACY =================
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(class_names, per_class_acc, color=['#2196F3', '#4CAF50', '#FF9800', '#F44336'], edgecolor='black', linewidth=0.8)
    plt.ylim([0, 1.05])
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Class', fontsize=12)
    plt.title('Per-Class Accuracy - EfficientNet-B4', fontsize=14)
    plt.xticks(rotation=45, ha='right')

    for bar, val in zip(bars, per_class_acc):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_accuracy.png"), dpi=300)
    plt.close()
    print("Saved: per_class_accuracy.png")

    # ================= ROC CURVES (ONE-VS-REST) =================
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc

    y_bin = label_binarize(all_labels, classes=list(range(num_classes)))

    plt.figure(figsize=(8, 6))
    colors_roc = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors_roc[i], lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - EfficientNet-B4', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=300)
    plt.close()
    print("Saved: roc_curves.png")

    # ================= SAVE HISTORY JSON =================
    history['best_test_acc'] = float(best_test_acc)
    history['best_epoch'] = int(best_checkpoint['epoch'] + 1)
    history['final_acc'] = float(acc)
    history['final_precision_macro'] = float(precision)
    history['final_recall_macro'] = float(recall)
    history['final_f1_macro'] = float(f1)
    history['final_precision_weighted'] = float(precision_w)
    history['final_recall_weighted'] = float(recall_w)
    history['final_f1_weighted'] = float(f1_w)
    history['class_names'] = class_names
    history['per_class_accuracy'] = {name: float(val) for name, val in zip(class_names, per_class_acc)}

    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)
    print("Saved: training_history.json")

    # ================= SAVE PREDICTIONS =================
    np.savez(os.path.join(output_dir, "predictions.npz"),
             preds=all_preds,
             labels=all_labels,
             probs=all_probs,
             class_names=class_names)
    print("Saved: predictions.npz")

    # ================= SUMMARY =================
    print("\n" + "=" * 60)
    print("  ALL DONE! Files saved to:")
    print(f"  {output_dir}")
    print("=" * 60)
    print(f"  best_model.pth")
    print(f"  classification_report.txt")
    print(f"  confusion_matrix.png")
    print(f"  confusion_matrix_normalized.png")
    print(f"  training_curves.png")
    print(f"  per_class_accuracy.png")
    print(f"  roc_curves.png")
    print(f"  training_history.json")
    print(f"  predictions.npz")
    print("=" * 60)
    print(f"\n  FINAL ACCURACY: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  BEST  ACCURACY: {best_test_acc:.4f} ({best_test_acc*100:.2f}%)")
    print("=" * 60)