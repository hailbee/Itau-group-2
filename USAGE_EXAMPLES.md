# Complete Usage Examples

## 1. OCR Threshold Testing Examples

### Example 1.1: Single Threshold (Default Behavior)
```bash
python main.py --mode ocr \
  --test_filepath data/processed/validate_pairs_ref_10k.parquet \
  --fuzzy_threshold 80
```

**Output:**
```
Testing OCR encoder on glyphs...
Testing Pytesseract OCR with fuzzy threshold=80...
Extracting text via OCR...
Calculating text similarity via fuzzy matching...
Evaluating OCR accuracy...

OCR Results Summary (fuzzy_threshold=80):
{
  'accuracy': 0.8234,
  'precision': 0.8456,
  'recall': 0.7987,
  'f1_score': 0.8218,
  'roc_auc': 0.8945,
  'threshold': 0.7645
}
```

---

### Example 1.2: Predefined Multiple Thresholds
```bash
python main.py --mode ocr \
  --test_filepath data/processed/validate_pairs_ref_10k.parquet \
  --ocr_thresholds all
```

**Output:**
```
Testing OCR encoder on glyphs...
Testing OCR with predefined fuzzy thresholds...

============================================================
Testing Pytesseract OCR with fuzzy threshold=10...
============================================================
Generating glyphs for 10000 text pairs...
Extracting text via OCR...
...

============================================================
Testing Pytesseract OCR with fuzzy threshold=30...
============================================================
...

OCR Results Summary (Predefined Thresholds):
pytesseract_10: {'accuracy': 0.4123, 'precision': 0.2345, ...}
pytesseract_30: {'accuracy': 0.6234, 'precision': 0.5123, ...}
pytesseract_50: {'accuracy': 0.7456, 'precision': 0.7234, ...}
pytesseract_60: {'accuracy': 0.7834, 'precision': 0.7645, ...}
pytesseract_70: {'accuracy': 0.8123, 'precision': 0.8012, ...}
pytesseract_80: {'accuracy': 0.8234, 'precision': 0.8456, ...}
pytesseract_90: {'accuracy': 0.7956, 'precision': 0.8234, ...}
```

---

### Example 1.3: Custom Thresholds
```bash
python main.py --mode ocr \
  --test_filepath data/processed/validate_pairs_ref_10k.parquet \
  --ocr_thresholds custom \
  --ocr_custom_thresholds "45,55,65,75,85,95"
```

**Output:**
```
Testing OCR encoder on glyphs...
Testing OCR with custom fuzzy thresholds: [45, 55, 65, 75, 85, 95]...

============================================================
Testing Pytesseract OCR with fuzzy threshold=45...
============================================================
...

OCR Results Summary (Custom Thresholds):
pytesseract_45: {'accuracy': 0.5678, 'precision': 0.5234, ...}
pytesseract_55: {'accuracy': 0.6890, 'precision': 0.6456, ...}
pytesseract_65: {'accuracy': 0.7645, 'precision': 0.7234, ...}
pytesseract_75: {'accuracy': 0.8012, 'precision': 0.7945, ...}
pytesseract_85: {'accuracy': 0.8089, 'precision': 0.8123, ...}
pytesseract_95: {'accuracy': 0.7234, 'precision': 0.7456, ...}
```

---

### Example 1.4: OCR with ROC Curve Plotting
```bash
python main.py --mode ocr \
  --test_filepath data/processed/validate_pairs_ref_10k.parquet \
  --ocr_thresholds all \
  --plot_roc True
```

**Output:**
```
Testing OCR encoder on glyphs...
Testing OCR with predefined fuzzy thresholds...

... (testing each threshold)

Evaluation plots saved to 'ocr_evaluation_results.png'
```

---

## 2. Latency Measurement Examples

### Example 2.1: OCR Latency (Including Glyph Generation)
```bash
python main.py --mode ocr \
  --test_filepath data/processed/validate_pairs_ref_10k.parquet \
  --measure_latency True \
  --latency_num_runs 20 \
  --latency_warmup 5
```

**Output:**
```
Testing OCR encoder on glyphs...
Testing Pytesseract OCR with fuzzy threshold=80...
...

============================================================
Measuring OCR latency...
============================================================
Model: pytesseract (including glyph generation) (OCR)
Batch Size: 100

Runtime Metrics (ms):
  Mean: 2345.67
  Min: 2234.45
  Max: 2456.78
  Std: 78.92

Memory Metrics (MB):
  Allocated: 512.34
  Reserved: 768.00
  Peak: 512.34
  CPU: 234.56

Utilization:
  CPU: 45.3%

Throughput: 42.63 samples/sec
```

---

### Example 2.2: OCR Latency (Inference Only)
```bash
python main.py --mode ocr \
  --test_filepath data/processed/validate_pairs_ref_10k.parquet \
  --measure_latency True \
  --include_preprocessing_time False
```

**Output:**
```
Testing OCR encoder on glyphs...
Testing Pytesseract OCR with fuzzy threshold=80...
...

============================================================
Measuring OCR latency...
============================================================
Pre-generating glyphs for 100 texts...
Model: pytesseract (inference only) (OCR)
Batch Size: 100

Runtime Metrics (ms):
  Mean: 1234.56
  Min: 1123.45
  Max: 1345.67
  Std: 56.78

Throughput: 81.00 samples/sec
```

---

### Example 2.3: Image Encoder Latency with All Models
```bash
python main.py --mode image_encoder \
  --image_encoder all \
  --test_filepath data/processed/validate_pairs_ref_10k.parquet \
  --measure_latency True \
  --include_preprocessing_time False
```

**Output:**
```
Testing all available image encoders on glyphs...

============================================================
Testing vit - google/vit-base-patch16-224
============================================================
...

Image Encoder Results Summary:
vit: {'accuracy': 0.8945, ...}
resnet: {'accuracy': 0.8723, ...}
convnext: {'accuracy': 0.8867, ...}
vitmae: {'accuracy': 0.8654, ...}
siglip: {'accuracy': 0.9123, ...}

============================================================
Measuring image encoder latency...
============================================================
Model: google/vit-base-patch16-224 (inference only) (ImageEncoderFactory)
Batch Size: 100

Runtime Metrics (ms):
  Mean: 567.89
  Min: 534.56
  Max: 612.34
  Std: 28.45

Throughput: 176.12 samples/sec
```

---

### Example 2.4: Siamese Model Latency
```bash
python main.py --mode evaluate_saved \
  --test_filepath data/processed/validate_pairs_ref_10k.parquet \
  --backbone siglip \
  --model_weights weights/best_model_siglip_pair.pt \
  --measure_latency True
```

**Output:**
```
Evaluating saved Siamese model with SIGLIP backbone...

Siamese Model Results Summary:
accuracy: 0.9234
precision: 0.9456
recall: 0.9012
f1_score: 0.9232
roc_auc: 0.9567
threshold: 0.6789

============================================================
Measuring Siamese model latency...
============================================================
Model: siamese (including glyph generation) (SiameseModel)
Batch Size: 100

Runtime Metrics (ms):
  Mean: 892.34
  Min: 845.67
  Max: 934.56
  Std: 35.67

Memory Metrics (MB):
  Allocated: 1024.56
  Reserved: 1280.00
  Peak: 1024.56
  CPU: 512.34

Utilization:
  CPU: 52.1%
  GPU Memory: 1024.56
  GPU Utilization: 85.3%

Throughput: 112.07 samples/sec
```

---

### Example 2.5: VLM Baseline Latency (Dedicated Mode)
```bash
python main.py --mode latency \
  --baseline_model all \
  --test_filepath data/processed/validate_pairs_ref_10k.parquet \
  --latency_num_runs 15 \
  --latency_warmup 3
```

**Output:**
```
Starting latency measurement for VLM baselines...
Number of runs: 15
Warmup runs: 3
Sample texts: 100

Profiling CLIP...
Model: OpenAI CLIP ViT-B/32 (clip)
Batch Size: 32
...
Runtime Metrics (ms):
  Mean: 123.45
  Min: 112.34
  Max: 145.67
  Std: 9.23

Profiling COCA...
...

Profiling SIGLIP...
...

================================================================================
LATENCY COMPARISON REPORT
================================================================================

Summary (sorted by mean latency):
--------------------------------------------------------------------------------
Model                     Latency (ms)        Throughput         GPU Mem (MB)    
--------------------------------------------------------------------------------
OpenAI CLIP ViT-B/32      123.45±9.23         259.1 s/s           2048.0          
SigLiP ViT-B/32           145.67±11.23        219.3 s/s           2304.0          
COCA                      234.56±18.45        136.3 s/s           3072.0          
FLAVA                     267.89±22.34        119.4 s/s           3456.0          
CogVLM                    456.78±34.56        70.1 s/s             4096.0          
Qwen-VL                   523.45±45.67        61.2 s/s             4608.0          
Gemma-2-Vision            678.90±56.78        47.1 s/s             5120.0          
--------------------------------------------------------------------------------

Detailed Results:
--------------------------------------------------------------------------------
Model: OpenAI CLIP ViT-B/32 (clip)
Batch Size: 32
...
```

---

## 3. Combined Examples (Thresholds + Latency)

### Example 3.1: OCR All Thresholds with Latency
```bash
python main.py --mode ocr \
  --test_filepath data/processed/validate_pairs_ref_10k.parquet \
  --ocr_thresholds all \
  --measure_latency True \
  --latency_num_runs 10
```

**Output:**
```
Testing OCR encoder on glyphs...
Testing OCR with predefined fuzzy thresholds...

... (testing each threshold) ...

OCR Results Summary (Predefined Thresholds):
pytesseract_10: {'accuracy': 0.4123, ...}
pytesseract_30: {'accuracy': 0.6234, ...}
pytesseract_50: {'accuracy': 0.7456, ...}
...

============================================================
Measuring OCR latency...
============================================================
Model: pytesseract (including glyph generation) (OCR)
Batch Size: 100

Runtime Metrics (ms):
  Mean: 2345.67
  Min: 2234.45
  Max: 2456.78
  Std: 78.92

Throughput: 42.63 samples/sec
```

---

### Example 3.2: Full Workflow: Test Multiple Thresholds, Measure Latency, Plot Results
```bash
python main.py --mode ocr \
  --test_filepath data/processed/validate_pairs_ref_10k.parquet \
  --ocr_thresholds custom \
  --ocr_custom_thresholds "50,60,70,80" \
  --plot_roc True \
  --measure_latency True \
  --include_preprocessing_time False \
  --latency_num_runs 20
```

**Output:**
```
Testing OCR encoder on glyphs...
Testing OCR with custom fuzzy thresholds: [50, 60, 70, 80]...

============================================================
Testing Pytesseract OCR with fuzzy threshold=50...
============================================================
Generating glyphs for 10000 text pairs...
Extracting text via OCR...
Calculating text similarity via fuzzy matching...
Evaluating OCR accuracy...
Evaluation plots saved to 'ocr_evaluation_results.png'

OCR Results Summary (Custom Thresholds):
pytesseract_50: {'accuracy': 0.7456, 'precision': 0.7234, ...}
pytesseract_60: {'accuracy': 0.7834, 'precision': 0.7645, ...}
pytesseract_70: {'accuracy': 0.8123, 'precision': 0.8012, ...}
pytesseract_80: {'accuracy': 0.8234, 'precision': 0.8456, ...}

============================================================
Measuring OCR latency...
============================================================
Pre-generating glyphs for 100 texts...
Model: pytesseract (inference only) (OCR)
Batch Size: 100

Runtime Metrics (ms):
  Mean: 1234.56
  Min: 1123.45
  Max: 1345.67
  Std: 56.78

Throughput: 81.00 samples/sec
```

---

## 4. Error Handling Examples

### Example 4.1: Missing Custom Thresholds
```bash
python main.py --mode ocr \
  --test_filepath data/processed/validate_pairs_ref_10k.parquet \
  --ocr_thresholds custom
```

**Error Output:**
```
ValueError: For 'custom' OCR thresholds, --ocr_custom_thresholds must be provided 
(e.g., '50,60,70,80')
```

---

### Example 4.2: Invalid Custom Threshold Format
```bash
python main.py --mode ocr \
  --test_filepath data/processed/validate_pairs_ref_10k.parquet \
  --ocr_thresholds custom \
  --ocr_custom_thresholds "50,abc,70"
```

**Error Output:**
```
ValueError: Invalid custom thresholds format: 50,abc,70. 
Expected comma-separated integers.
```

---

## 5. Integration with Scripts

### Example 5.1: Python Script for Comparative Analysis
```python
import subprocess
import json

# Test OCR with different threshold strategies
strategies = [
    {
        'name': 'Single Threshold (Default)',
        'args': ['--mode', 'ocr', '--fuzzy_threshold', '80']
    },
    {
        'name': 'Predefined Thresholds',
        'args': ['--mode', 'ocr', '--ocr_thresholds', 'all']
    },
    {
        'name': 'Custom Thresholds',
        'args': ['--mode', 'ocr', '--ocr_thresholds', 'custom', 
                 '--ocr_custom_thresholds', '50,60,70,80,90']
    }
]

for strategy in strategies:
    print(f"\n{'='*60}")
    print(f"Running: {strategy['name']}")
    print(f"{'='*60}\n")
    
    cmd = ['python', 'main.py', '--test_filepath', 'data/processed/validate_pairs_ref_10k.parquet']
    cmd.extend(strategy['args'])
    
    subprocess.run(cmd)
```

---

### Example 5.2: Latency Comparison Script
```python
import subprocess

# Test latency for different modes
modes = [
    ('ocr', ['--measure_latency', 'True']),
    ('image_encoder', ['--image_encoder', 'vit', '--measure_latency', 'True']),
    ('image_encoder', ['--image_encoder', 'siglip', '--measure_latency', 'True']),
]

test_file = 'data/processed/validate_pairs_ref_10k.parquet'

for mode, extra_args in modes:
    print(f"\nTesting {mode} latency...")
    cmd = ['python', 'main.py', '--mode', mode, '--test_filepath', test_file] + extra_args
    subprocess.run(cmd)
```

---

## 6. Command Line Tips

### Useful Flag Combinations:

```bash
# Quick test with all default settings
python main.py --mode ocr --test_filepath <path>

# Comprehensive OCR evaluation
python main.py --mode ocr --test_filepath <path> --ocr_thresholds all --plot_roc True

# Performance profiling
python main.py --mode ocr --test_filepath <path> --measure_latency True --latency_num_runs 50

# Multi-threaded image encoder comparison
python main.py --mode image_encoder --image_encoder all --test_filepath <path> \
  --measure_latency True --include_preprocessing_time False

# Production-ready Siamese model evaluation
python main.py --mode evaluate_saved --test_filepath <path> \
  --backbone siglip --model_weights weights/best_model_siglip_pair.pt \
  --plot True --measure_latency True --save_misclassified True
```

---

## 7. Expected File Sizes & Runtime

### Estimated Execution Times:

| Configuration | Data Size | Est. Runtime | Memory |
|---------------|-----------|--------------|--------|
| OCR single threshold | 10k pairs | 2-3 min | 500 MB |
| OCR all thresholds | 10k pairs | 12-15 min | 500 MB |
| OCR latency (20 runs) | 100 texts | 3-5 min | 500 MB |
| Image encoder (ViT) | 10k pairs | 5-8 min | 3 GB |
| Image encoder all | 10k pairs | 25-35 min | 3 GB per model |
| Siamese evaluation | 10k pairs | 3-5 min | 2 GB |
| Siamese + latency | 100 pairs | 2-4 min | 2 GB |
| VLM latency (7 models) | 100 texts | 15-25 min | 4 GB |

**Note:** Times vary based on GPU, batch size, and system load.

