# Swin-UNet  
## 🧠 Swin UNet – The Simplest & Most Powerful Image Segmentation Workflow(Advanced Satellite Imagery Segmentation for GIS and Urban Planning)

[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.17%2B-orange)](https://www.tensorflow.org/)
[![Python 3.7+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)


Welcome to **Swin UNet**, a cutting-edge architecture that combines the power of Swin Transformers and the robustness of UNet for **pixel-perfect image segmentation**.

Whether you're a **researcher**, **engineer**, or **practitioner**, this library is the **cleanest, shortest, and most modular way** to:

- 🏋️‍♀️ **Train** powerful segmentation models in a single call. 
- 🧪 **Test** models with visual outputs and metrics.   
- 🖼️ **Infer** on any image with professional overlay masks. 

> Backed by **Keras + TensorFlow**, and fully customizable without rewriting boilerplate code.

## 🚀 Key Features for Remote Sensing Professionals
**State-of-the-art deep learning solution** for road extraction from satellite imagery, combining Swin Transformers and U-Net architecture. Ideal for:

- **GIS Specialists**: Accurate geospatial analysis for urban planning
- **AI Researchers**: Cutting-edge transformer-based segmentation models
- **Civil Engineers**: Infrastructure planning with precise road network detection
- **Environmental Scientists**: Land use monitoring and change detection

### Technical Highlights 🔬
- **Transformer-Powered Segmentation**: Swin Transformer backbone for superior spatial dependency handling
- **Multi-Class Capability**: Easily extendable from binary road extraction to complex land cover classification
- **AUC Focal Loss**: Optimized for imbalanced satellite datasets (98.38% accuracy on DeepGlobe)
- **GPU-Ready Implementation**: TensorFlow 2.x optimized for rapid training on large geospatial datasets



## 🛠️ Quick Start for Developers

### 1. Installation
```bash
pip install keras-swin-unet
```

## 🗂️ Dataset Preparation
Structure your satellite imagery data:
```bash
data/
├── images/  # High-res satellite images (RGB)
└── masks/   # Pixel-level road annotations
```
⚙️ **Training Configuration**

## ✅ Step 1: Train a Swin UNet Model
The library includes a dynamic data loader that handles both data ingestion and model training, with the dataset split into 80% for training, 10% for validation, and 10% for testing. During each training run, the training and validation sets are used automatically, and model checkpoints are saved whenever the validation loss decreases. You can also tune the focal loss parameters (alpha and gamma) and any other hyperparameters as needed.
```
from keras_swin_unet import swin_train

swin_train(
    data="demo_data",              # Dataset path (must contain images/ and masks/)
    model_dir="./checkpoint",      # Folder to save model + logs
    num_classes=2,                 # 2 for binary, >2 for multi-class segmentation
    epochs=50,                     # Training epochs
    bs=4,                          # Batch size
    patience=5,                    # Early stopping patience
    filter=64,                     # Initial number of filters
    depth=4,                       # Depth of encoder/decoder
    stack_down=2, stack_up=2,      # Swin transformer blocks per level
    patch_size=[4, 4],             # Patch size for Swin layers
    num_heads=[4, 8, 8, 8],        # Attention heads at each level
    window_size=[4, 2, 2, 2],      # Attention window per level
    num_mlp=512,                   # MLP size in Swin blocks
    gamma=2.0, alpha=0.25,         # Focal loss hyperparameters
    input_shape=[512, 512, 3],     # Input image shape
    input_scale=255,               # Divide input by this value (e.g. 255 for 8-bit)
    mask_scale=255,                # Same for masks
    visualize=10                   # Save visual results on N test samples
)
```
## ✅ Step 2: Test a Swin UNet Model 
When you invoke the testing step for the trained Swin UNet model, the held‑out test dataset is automatically used for inference. Results computed from the specified checkpoint are saved alongside the checkpoint files: numerical metrics are written to a JSON file, and evaluation plots are exported as PNGs for easy review and use.
```
from keras_swin_unet import swin_infer

swin_infer(
    data="demo_data",             # Folder with test images/masks
    model_dir="./checkpoint",     # Where the model was saved
    num_classes=2,                # Must match training
    gamma=2.0, alpha=0.25,        # Same focal loss settings
    input_scale=255,
    mask_scale=255,               # Match input normalization
    visualize=10                  # Save N test results (use -1 for full test set)
)
```
## ✅ Step 3: Run Inference on a Single Image
If you just have single image and want to get inference result on pretrained weights use the following call.
```python
swin_infer(
    image="demo_data/images/104.jpg",  # Path to a single image
    output="output_overlay.png",       # Save overlay to this file
    model_dir="./checkpoint",          # Where the trained model is
    num_classes=2,                     # Match with training setup
    input_scale=255,
    gamma=2.0, alpha=0.25,
    visualize=1                        # 1 for overlay, >1 for grid comparison
)
```
🌍 **Real-World Applications**

- 🏙️ **Urban Planning**: Automated road network mapping for smart cities
- 🚨 **Disaster Response**: Rapid infrastructure assessment post-natural disasters
- 🚗 **Autonomous Navigation**: High-precision road data for self-driving systems
- 🌾 **Agricultural Logistics**: Rural road network analysis for crop distribution

📊 **Performance Insights**
The Swin Unet model  is trained with deep glob dataset and following results are achieved. 

**Confusion Matrix**

|                     | Predicted Road | Predicted Non-Road |
|---------------------|----------------|--------------------|
| **Actual Road**      | 2,752,25       | 72,120             |
| **Actual Non-Road**  | 64,087         | 7,977,176          |

**Metric Breakdown**

| Metric     | Formula                              | Value   |
|------------|--------------------------------------|---------|
| Precision  | TP / (TP + FP)                       | 90.11%  |
| Recall     | TP / (TP + FN)                       | 89.22%  |
| F1 Score   | 2*(Precision*Recall)/(Precision+Recall) | 0.8966  |


![Road Extraction Visualization](Results/3.png)
*Visual comparison showing precise road network detection in challenging terrain*

**Maintainer**:  
Laeeq Aslam(laeeq.aslam.100@gmail.com)

📚 **Citations & Acknowledgements**

```bibtex
@inproceedings{liu2021swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and others},
  booktitle={ICCV},
  year={2021}
}
