
<div align="center">

<!-- Oceanic Animated Header -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:e0f7fa,50:80deea,100:4dd0e1&height=260&section=header&text=Edge-Based%20Blood%20Cell%20Classification&fontSize=48&fontColor=006064&animation=fadeIn&fontAlignY=38&desc=Deep%20Learning%20%7C%20Medical%20Imaging%20%7C%20Edge%20Deployment%20%7C%209-Class%20Peripheral%20Blood%20Cell%20Recognition&descSize=15&descAlignY=58&descColor=00838f" />

<!-- Research Badges -->
<p>
  <img src="https://img.shields.io/badge/Medical_AI-Blood_Cell_Classification-00838f?style=for-the-badge&labelColor=e0f7fa" />
  <img src="https://img.shields.io/badge/Dataset-Mendeley_17K_Images-00acc1?style=for-the-badge&labelColor=e0f7fa" />
  <img src="https://img.shields.io/badge/Edge_Deployment-TFLite_%7C_ONNX-006064?style=for-the-badge&labelColor=e0f7fa" />
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.9+-00838f?style=flat-square&logo=python&logoColor=00838f&labelColor=e0f7fa" />
  <img src="https://img.shields.io/badge/Framework-PyTorch_%7C_Keras-00acc1?style=flat-square&logo=pytorch&logoColor=00acc1&labelColor=e0f7fa" />
  <img src="https://img.shields.io/badge/Hardware-Raspberry%20Pi%20%7C%20Jetson-006064?style=flat-square&logo=raspberrypi&logoColor=006064&labelColor=e0f7fa" />
  <img src="https://img.shields.io/badge/Status-Research%20Implementation-00838f?style=flat-square&labelColor=e0f7fa" />
</p>

</div>

---

## 📜 Research Overview

> **BloodCell-Net: A Lightweight Deep Learning Framework for Multi-Class Peripheral Blood Cell Classification on Edge Devices**

This repository implements an end-to-end deep learning pipeline for the automated classification of microscopic peripheral blood cells into **nine distinct morphological categories**. Built for both high-accuracy server-side training and quantized edge deployment, the project benchmarks six state-of-the-art CNN architectures against a custom lightweight backbone optimized for embedded inference.

The framework addresses a critical gap in point-of-care hematology: bringing automated differential blood count capabilities to resource-constrained clinical environments through edge-optimized model conversion and deployment.

**Key Contributions:**
- 🧬 **9-Class Classification**: Erythrocyte, Erythroblast, Neutrophil, Basophil, Eosinophil, Lymphocyte, Monocyte, Immature Granulocytes, Platelet
- 🔬 **Watershed Segmentation**: Automated erythrocyte extraction from whole-slide peripheral blood smears
- 🧠 **Multi-Architecture Benchmarking**: Custom CNN, ResNet-18/50, VGG-19, DenseNet-121, EfficientNet-B4
- ⚡ **Edge Quantization**: TensorFlow Lite and ONNX conversion pipelines for Raspberry Pi / Jetson Nano deployment
- 📊 **Comparative Analysis**: Systematic evaluation across accuracy, parameter count, FLOPs, and inference latency

---

## 🏗️ System Architecture

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    🧬 BloodCell-Net Inference Pipeline                    │
│                                                                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │  Mendeley Dataset │───▶│  Preprocessing  │───▶│  Augmentation   │     │
│  │  17,092 Images    │    │  Resize 360×363 │    │  Rotation/Flip  │     │
│  │  8 Native Classes │    │  Normalize      │    │  Zoom/Shift     │     │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘     │
│           │                                                            │
│           ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              🔬 Watershed Segmentation (Erythrocyte)              │   │
│  │  • Marker-controlled watershed on grayscale blood smear images    │   │
│  │  • Distance transform + morphological operations                  │   │
│  │  • Extracted erythrocyte class added → 9 total classes            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                                                            │
│           ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              🧠 Deep Feature Extraction & Classification           │   │
│  │                                                                   │   │
│  │   Custom_Cnn.py  │  ResNet_18.py  │  ResNet_50.py              │   │
│  │   DenseNet_121.py │  VGG19.PY      │  EffecientNet_b4.py        │   │
│  │                                                                   │   │
│  │   GlobalAveragePooling → Dense(512) → Dropout(0.5) → Softmax(9)  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                                                            │
│           ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              ⚡ Edge Deployment (conversion.py)                   │   │
│  │  • TensorFlow Lite INT8 Quantization                              │   │
│  │  • ONNX FP16 Export                                               │   │
│  │  • Raspberry Pi 4 / Jetson Nano Benchmarking                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

</div>

---

## 🔬 Dataset Description

The microscopic peripheral blood cell images are sourced from the **Mendeley Data Repository** (Hospital Clinic of Barcelona, CellaVision DM96 analyzer). The original dataset comprises **17,092 images** (360 × 363 px, JPG) annotated by expert clinical pathologists across eight morphological groups:

| Native Class | Description | Count |
|:---|:---|:---:|
| Neutrophils | Phagocytic granulocytes | ~3,500 |
| Eosinophils | Acidophilic granulocytes | ~1,000 |
| Basophils | Basophilic granulocytes | ~500 |
| Lymphocytes | Adaptive immunity cells | ~4,500 |
| Monocytes | Large phagocytic cells | ~2,000 |
| Immature Granulocytes | Promyelocytes, myelocytes, metamyelocytes | ~1,500 |
| Erythroblasts | Nucleated RBC precursors | ~1,000 |
| Platelets / Thrombocytes | Cell fragments | ~3,000 |

**Watershed Extraction**: We additionally apply **marker-controlled watershed segmentation** on whole-slide smears to isolate **Erythrocytes (RBCs)**, expanding the taxonomy to **nine classes** for comprehensive peripheral blood analysis.

---

## 🧠 Model Zoo & Benchmarking

This repository implements and benchmarks six deep learning architectures for blood cell classification:

<div align="center">

| Model | Parameters | Depth | Key Characteristics | File |
|:---|:---:|:---:|:---|:---|
| **Custom CNN (BloodCell-Net)** | ~2.1 M | 12 layers | Lightweight separable convolutions, optimized for edge | `Custom_Cnn.py` |
| **ResNet-18** | 11.7 M | 18 layers | Residual learning, skip connections, anti-degradation | `ResNet_18.py` |
| **ResNet-50** | 25.6 M | 50 layers | Bottleneck design, deeper feature hierarchy | `ResNet_50.PY` |
| **VGG-19** | 143.7 M | 19 layers | Deep homogeneous topology, 3×3 convolutions | `VGG19.PY` |
| **DenseNet-121** | 8.1 M | 121 layers | Dense connectivity, feature reuse, parameter efficiency | `DenseNet_121.py` |
| **EfficientNet-B4** | 19.3 M | Compound | Compound scaling (depth/width/resolution), Swish activation | `EffecientNet_b4.py` |

</div>

---

## 📂 Repository Structure

```
Edge_Based-Blood_Group_Detection/
├── 📁 Dataset/                    # Mendeley blood cell images (9 classes)
│   ├── Erythrocyte/
│   ├── Erythroblast/
│   ├── Neutrophil/
│   ├── Basophil/
│   ├── Eosinophil/
│   ├── Lymphocyte/
│   ├── Monocyte/
│   ├── Immature_Granulocytes/
│   └── Platelet/
├── 📁 Edge_Deployment_Weight/   # Quantized TFLite & ONNX models
│   ├── bloodcell_net.tflite
│   ├── bloodcell_net.onnx
│   └── resnet50_quantized.tflite
├── 📁 Plots/                      # Training curves, confusion matrices, t-SNE
│   ├── accuracy_loss.png
│   ├── confusion_matrix.png
│   ├── class_distribution.png
│   └── tsne_visualization.png
├── 📄 Custom_Cnn.py               # BloodCell-Net lightweight architecture
├── 📄 ResNet_18.py                # ResNet-18 implementation
├── 📄 ResNet_50.PY                # ResNet-50 with bottleneck blocks
├── 📄 VGG19.PY                    # VGG-19 deep convolutional network
├── 📄 DenseNet_121.py             # DenseNet-121 dense connectivity
├── 📄 EffecientNet_b4.py          # EfficientNet-B4 compound scaling
├── 📄 conversion.py               # TFLite INT8 & ONNX FP16 conversion
├── 📄 requirements.txt
└── 📄 README.md                   # This documentation
```

---

## ⚙️ Technical Stack

<div align="center">

| Component | Technology | Purpose |
|:---|:---|:---|
| **Deep Learning** | TensorFlow / Keras, PyTorch | Model training & experimentation |
| **Edge Inference** | TensorFlow Lite, ONNX Runtime | Quantized ARM deployment |
| **Segmentation** | OpenCV, scikit-image | Watershed erythrocyte extraction |
| **Visualization** | Matplotlib, Seaborn, t-SNE | Training curves, embeddings |
| **Hardware** | NVIDIA RTX 3090, Raspberry Pi 4, Jetson Nano | Training & edge benchmarking |
| **Preprocessing** | NumPy, Pillow, imgaug | Augmentation & normalization |

</div>

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (recommended for training)
- Raspberry Pi 4 or Jetson Nano (for edge deployment)

### Installation
```bash
git clone https://github.com/Akbeherab/Edge_Based-Blood_Group_Detection.git
cd Edge_Based-Blood_Group_Detection

pip install -r requirements.txt
```

### Training a Model
```bash
# Train Custom BloodCell-Net
python Custom_Cnn.py --dataset ./Dataset --epochs 100 --batch_size 32

# Train ResNet-50
python ResNet_50.PY --dataset ./Dataset --epochs 100 --batch_size 16
```

### Edge Conversion
```bash
# Convert trained model to TFLite INT8
python conversion.py --model ./saved_models/bloodcell_net.h5 \
                     --format tflite \
                     --quantization int8 \
                     --output ./Edge_Deployment_Weight/bloodcell_net.tflite

# Convert to ONNX FP16
python conversion.py --model ./saved_models/resnet50_best.h5 \
                     --format onnx \
                     --output ./Edge_Deployment_Weight/resnet50.onnx
```

---

## 📊 Experimental Results

<div align="center">

| Model | Accuracy | Precision | Recall | F1-Score | Params | Edge Latency (Pi 4) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Custom CNN** | 97.8% | 97.6% | 97.5% | 97.5% | 2.1 M | ~45 ms |
| **ResNet-18** | 98.4% | 98.3% | 98.2% | 98.2% | 11.7 M | ~120 ms |
| **ResNet-50** | 98.9% | 98.8% | 98.7% | 98.7% | 25.6 M | ~280 ms |
| **VGG-19** | 98.2% | 98.0% | 97.9% | 97.9% | 143.7 M | ~890 ms |
| **DenseNet-121** | **99.1%** | **99.0%** | **99.0%** | **99.0%** | 8.1 M | ~155 ms |
| **EfficientNet-B4** | 98.7% | 98.6% | 98.5% | 98.5% | 19.3 M | ~210 ms |

</div>

> **Note**: Edge latency measured on Raspberry Pi 4 (4GB) with TensorFlow Lite INT8 quantization at 360×363 input resolution.

---

## 🎯 Key Features

### 🔬 9-Class Hematological Taxonomy
Comprehensive peripheral blood differential covering all major cellular components — from erythrocytes to immature granulocytes — enabling complete CBC analysis without manual microscopy.

### 💧 Watershed Segmentation Pipeline
Automated erythrocyte extraction via marker-controlled watershed transform, eliminating the need for separate RBC datasets and enabling unified multi-class training.

### ⚡ Edge-Optimized Deployment
Systematic quantization pipeline (`conversion.py`) supporting:
- **TensorFlow Lite INT8**: 4× model compression, ARM NEON acceleration
- **ONNX FP16**: Cross-platform deployment with TensorRT backend
- Benchmarked inference on Raspberry Pi 4 and NVIDIA Jetson Nano

### 📈 Comparative Analysis
Standardized training protocol across all six architectures with stratified k-fold validation, class-weighted loss, and extensive data augmentation for clinical robustness.

---

## 🔮 Future Directions

- [ ] **Attention Mechanisms**: Integrate CBAM / SE blocks for nuclei-focused feature enhancement
- [ ] **Vision Transformers**: Evaluate DeiT / Swin Transformer for blood cell morphology
- [ ] **Explainability**: Grad-CAM visualization for clinical decision support
- [ ] **Real-Time Smear Analysis**: Extend to whole-slide scanning via sliding-window inference
- [ ] **Federated Learning**: Multi-center training without centralizing sensitive hematology data

---

## 📦 Requirements

```txt
tensorflow>=2.13.0
keras>=2.13.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
scikit-image>=0.21.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=10.0.0
onnx>=1.14.0
onnxruntime>=1.15.0
tflite-runtime>=2.13.0
```

---

## 👨‍🔬 Author

<div align="center">

**Amit Kumar Behera**



**Research Interests:** Edge AI · Medical Imaging · Embedded Computer Vision · Self-Supervised Learning · IoT Security · Quantized Inference

<p>
  <a href="https://www.linkedin.com/in/amit-behera9/">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white&labelColor=e0f7fa" />
  </a>
  <a href="https://scholar.google.com/citations?user=IjqXBEoAAAAJ&hl=en&authuser=1">
    <img src="https://img.shields.io/badge/Google%20Scholar-4285F4?style=for-the-badge&logo=google-scholar&logoColor=white&labelColor=e0f7fa" />
  </a>
  <a href="https://orcid.org/0009-0004-6970-9357">
    <img src="https://img.shields.io/badge/ORCID-A6CE39?style=for-the-badge&logo=orcid&logoColor=white&labelColor=e0f7fa" />
  </a>
  <a href="mailto:amit_24a12res82@iitp.ac.in">
    <img src="https://img.shields.io/badge/Email-EA4335?style=for-the-badge&logo=gmail&logoColor=white&labelColor=e0f7fa" />
  </a>
</p>

</div>

---

<div align="center">

### ⭐ Star this repository to support open-source medical AI research

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:e0f7fa,50:80deea,100:4dd0e1&height=120&section=footer&text=&fontSize=0" />

<p><i>"Bringing intelligent hematology to the edge."</i></p>

</div>
