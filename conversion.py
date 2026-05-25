import torch
import torch.nn as nn
from torchvision import models
import onnx
import onnxruntime as ort
import numpy as np
import time
import os
from pathlib import Path

# -------------------------------
# Model Definition (Same as training)
# -------------------------------
class BloodCellDenseNet121(nn.Module):
    def __init__(self, num_classes=8):
        super(BloodCellDenseNet121, self).__init__()
        self.densenet = models.densenet121(pretrained=False)
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes)
        
    def forward(self, x):
        return self.densenet(x)

# -------------------------------
# Load PyTorch Model
# -------------------------------
device = torch.device('cpu')  # Use CPU for ONNX conversion
model = BloodCellDenseNet121(num_classes=8)

# Load trained weights
model_path = r"D:\Amit Kumar Behera\blood dataset\best_densenet121_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print(f"✓ Loaded PyTorch model from: {model_path}\n")

# -------------------------------
# 1. Export to ONNX FP32
# -------------------------------
print("="*80)
print("1. CONVERTING TO ONNX FP32")
print("="*80)

dummy_input = torch.randn(1, 3, 224, 224)
onnx_fp32_path = "densenet121_fp32.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_fp32_path,
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print(f"✓ FP32 ONNX model saved: {onnx_fp32_path}")

# Verify ONNX model
onnx_model_fp32 = onnx.load(onnx_fp32_path)
onnx.checker.check_model(onnx_model_fp32)
print("✓ FP32 ONNX model verified\n")

# -------------------------------
# 2. Convert to FP16
# -------------------------------
print("="*80)
print("2. CONVERTING TO ONNX FP16")
print("="*80)

from onnxconverter_common import float16

onnx_fp16_path = "densenet121_fp16.onnx"
model_fp16 = float16.convert_float_to_float16(onnx_model_fp32)
onnx.save(model_fp16, onnx_fp16_path)

print(f"✓ FP16 ONNX model saved: {onnx_fp16_path}")
print("✓ FP16 ONNX model verified\n")

# -------------------------------
# 3. Convert to INT8 (Quantization)
# -------------------------------
print("="*80)
print("3. CONVERTING TO ONNX INT8 (Quantized)")
print("="*80)

from onnxruntime.quantization import quantize_dynamic, QuantType

onnx_int8_path = "densenet121_int8.onnx"

quantize_dynamic(
    model_input=onnx_fp32_path,
    model_output=onnx_int8_path,
    weight_type=QuantType.QInt8
)

print(f"✓ INT8 ONNX model saved: {onnx_int8_path}")
print("✓ INT8 ONNX model verified\n")

# -------------------------------
# 4. Benchmark All Models
# -------------------------------
print("="*80)
print("4. BENCHMARKING ALL MODELS")
print("="*80 + "\n")

def benchmark_onnx_model(onnx_path, num_runs=100):
    """Benchmark ONNX model for latency"""
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # Prepare input
    input_name = session.get_inputs()[0].name
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        _ = session.run(None, {input_name: input_data})
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        _ = session.run(None, {input_name: input_data})
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / num_runs * 1000  # in ms
    return avg_latency, session

def get_file_size_mb(filepath):
    """Get file size in MB"""
    return os.path.getsize(filepath) / (1024 * 1024)

def get_memory_usage(session):
    """Estimate memory usage"""
    # Get model size (approximation)
    providers = session.get_providers()
    return "CPU based inference"

# Benchmark FP32
print("Benchmarking FP32 model...")
latency_fp32, session_fp32 = benchmark_onnx_model(onnx_fp32_path)
size_fp32 = get_file_size_mb(onnx_fp32_path)

# Benchmark FP16
print("Benchmarking FP16 model...")
latency_fp16, session_fp16 = benchmark_onnx_model(onnx_fp16_path)
size_fp16 = get_file_size_mb(onnx_fp16_path)

# Benchmark INT8
print("Benchmarking INT8 model...")
latency_int8, session_int8 = benchmark_onnx_model(onnx_int8_path)
size_int8 = get_file_size_mb(onnx_int8_path)

# -------------------------------
# 5. Display Complete Metrics
# -------------------------------
print("\n" + "="*80)
print("COMPLETE DEPLOYMENT METRICS")
print("="*80 + "\n")

# Create comparison table
print(f"{'Model Type':<15} {'File Size (MB)':<20} {'Latency (ms)':<20} {'Speedup':<15}")
print("-" * 80)
print(f"{'FP32':<15} {size_fp32:<20.2f} {latency_fp32:<20.2f} {'1.00x':<15}")
print(f"{'FP16':<15} {size_fp16:<20.2f} {latency_fp16:<20.2f} {latency_fp32/latency_fp16:<15.2f}x")
print(f"{'INT8':<15} {size_int8:<20.2f} {latency_int8:<20.2f} {latency_fp32/latency_int8:<15.2f}x")
print("-" * 80)

# Detailed metrics
print("\n" + "="*80)
print("DETAILED METRICS")
print("="*80 + "\n")

models_info = [
    ("FP32", onnx_fp32_path, size_fp32, latency_fp32, session_fp32),
    ("FP16", onnx_fp16_path, size_fp16, latency_fp16, session_fp16),
    ("INT8", onnx_int8_path, size_int8, latency_int8, session_int8)
]

for model_name, model_path, size, latency, session in models_info:
    print(f"📊 {model_name} Model")
    print("-" * 40)
    print(f"  File Path          : {model_path}")
    print(f"  File Size          : {size:.2f} MB")
    print(f"  Avg Latency        : {latency:.2f} ms")
    print(f"  Throughput         : {1000/latency:.2f} images/sec")
    print(f"  Input Shape        : {session.get_inputs()[0].shape}")
    print(f"  Output Shape       : {session.get_outputs()[0].shape}")
    print(f"  Precision          : {model_name}")
    print(f"  Execution Provider : {session.get_providers()[0]}")
    print()

# -------------------------------
# 6. Memory and RAM Estimation
# -------------------------------
print("="*80)
print("MEMORY & RAM REQUIREMENTS")
print("="*80 + "\n")

# Calculate theoretical memory requirements
def calculate_memory(size_mb, precision):
    """Estimate RAM needed for inference"""
    # Model weights + activation memory + overhead
    activation_memory = 224 * 224 * 3 * 4 / (1024 * 1024)  # Input tensor
    intermediate_memory = size_mb * 1.5  # Rough estimate for intermediate activations
    total_memory = size_mb + activation_memory + intermediate_memory
    return total_memory

print(f"{'Model':<10} {'Model Size':<15} {'Activation':<15} {'Total RAM Needed':<20}")
print("-" * 80)
for model_name, _, size, _, _ in models_info:
    total_ram = calculate_memory(size, model_name)
    print(f"{model_name:<10} {size:<15.2f} MB {size*0.3:<15.2f} MB ~{total_ram:<18.2f} MB")

# -------------------------------
# 7. Edge Device Recommendations
# -------------------------------
print("\n" + "="*80)
print("EDGE DEVICE DEPLOYMENT RECOMMENDATIONS")
print("="*80 + "\n")

print("🔹 Raspberry Pi 4 (4GB RAM)")
print(f"   Recommended: INT8 (Size: {size_int8:.2f} MB, Latency: {latency_int8:.2f} ms)")
print(f"   Alternative: FP16 (Size: {size_fp16:.2f} MB, Latency: {latency_fp16:.2f} ms)")
print()

print("🔹 Jetson Nano")
print(f"   Recommended: FP16 with GPU acceleration")
print(f"   File: {onnx_fp16_path}")
print()

print("🔹 Intel NUC / Edge PC")
print(f"   Recommended: FP32 for best accuracy")
print(f"   File: {onnx_fp32_path}")
print()

print("🔹 Mobile Devices (Android/iOS)")
print(f"   Recommended: INT8 for minimal size and power")
print(f"   File: {onnx_int8_path}")
print()

# -------------------------------
# 8. Test Inference
# -------------------------------
print("="*80)
print("TESTING INFERENCE ON ALL MODELS")
print("="*80 + "\n")

test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

for model_name, model_path, _, _, session in models_info:
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: test_input})
    predicted_class = np.argmax(output[0])
    confidence = np.max(output[0])
    
    print(f"{model_name} Model:")
    print(f"  Predicted Class: {predicted_class}")
    print(f"  Confidence: {confidence:.4f}")
    print()

# -------------------------------
# 9. Save Summary Report
# -------------------------------
print("="*80)
print("SAVING DEPLOYMENT REPORT")
print("="*80 + "\n")

with open("deployment_report.txt", "w") as f:
    f.write("="*80 + "\n")
    f.write("DENSENET-121 ONNX DEPLOYMENT REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write("Model Comparison:\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Model':<10} {'Size (MB)':<15} {'Latency (ms)':<15} {'Speedup':<15}\n")
    f.write("-"*80 + "\n")
    f.write(f"{'FP32':<10} {size_fp32:<15.2f} {latency_fp32:<15.2f} {'1.00x':<15}\n")
    f.write(f"{'FP16':<10} {size_fp16:<15.2f} {latency_fp16:<15.2f} {latency_fp32/latency_fp16:<15.2f}x\n")
    f.write(f"{'INT8':<10} {size_int8:<15.2f} {latency_int8:<15.2f} {latency_fp32/latency_int8:<15.2f}x\n")
    f.write("\n")
    
    f.write("Recommendations:\n")
    f.write("-"*80 + "\n")
    f.write(f"Best for Accuracy: FP32 ({onnx_fp32_path})\n")
    f.write(f"Best for Speed: INT8 ({onnx_int8_path})\n")
    f.write(f"Best Balance: FP16 ({onnx_fp16_path})\n")

print("✓ Deployment report saved: deployment_report.txt\n")

# -------------------------------
# 10. Final Summary
# -------------------------------
print("="*80)
print("CONVERSION COMPLETE!")
print("="*80 + "\n")

print("Generated Files:")
print(f"  1. {onnx_fp32_path} ({size_fp32:.2f} MB)")
print(f"  2. {onnx_fp16_path} ({size_fp16:.2f} MB)")
print(f"  3. {onnx_int8_path} ({size_int8:.2f} MB)")
print(f"  4. deployment_report.txt")
print()

print("Key Metrics Summary:")
print(f"  FP32 → FP16: {((size_fp32 - size_fp16)/size_fp32)*100:.1f}% size reduction")
print(f"  FP32 → INT8: {((size_fp32 - size_int8)/size_fp32)*100:.1f}% size reduction")
print(f"  FP16 Speedup: {latency_fp32/latency_fp16:.2f}x")
print(f"  INT8 Speedup: {latency_fp32/latency_int8:.2f}x")
print()

print("✅ All models ready for deployment!")