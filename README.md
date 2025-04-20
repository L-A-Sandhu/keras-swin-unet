# Swin-UNet Road Extraction Model

## Overview  
This repository provides a **Swin-UNet** implementation for **road extraction** from satellite imagery. Combining **Swin Transformers** for feature extraction and **U-Net** for segmentation, the model achieves state-of-the-art performance on the **DeepGlobe Road Extraction Dataset**. Key metrics include accuracy, F1 score, precision, recall, and AUC.

---

## Features  
- **Hybrid Architecture**: Swin Transformer + U-Net for efficient segmentation  
- **Flexible Input Shapes**: Supports 512x512, 256x256, and custom resolutions  
- **Customizable Training**: Adjust batch size, learning rate, and hyperparameters  
- **Performance Metrics**: Accuracy, F1 Score, Precision, Recall, AUC, Confusion Matrix  
- **Visualization Tools**: Compare predictions vs. ground truth with TP/FP/FN/TN overlays  
- **Lightweight Inference**: Average prediction time of 35ms per image  

---

## Dependencies  
- **Python 3.7+**  
- **TensorFlow 2.10.0+**  
- **Keras**  
- **Matplotlib**  
- **scikit-learn**  
- **Pillow (PIL)**  
- **NumPy**  

Install requirements:  
```bash
pip install -r requirements.txt
# Getting Started

## 1. Clone Repository

```bash
git clone https://github.com/your-repo/swin-unet-road-extraction.git
cd swin-unet-road-extraction
```
## 2. Prepare Dataset

Download and organize the DeepGlobe Road Extraction Dataset:

```plaintext
data/
├── images/     # Satellite images
└── masks/      # Ground truth masks
```
## 3. Train Model

```bash
python main.py \
  --model_dir './checkpoint/' \
  --data './data/' \
  --num_classes 2 \
  --inps 'train' \
  --b_s 64 \
  --e 100 \
  --input_shape 512 512 3
  
### Key Arguments:
- `--model_dir`: Model checkpoint directory
- `--b_s`: Batch size (default: 64)
- `--e`: Epochs (default: 100)
- `--input_shape`: Image dimensions (height, width, channels)

## 4. Run Inference

```bash
python main.py \
  --model_dir './checkpoint/' \
  --data './data/' \
  --inps 'infer' \
  --input_shape 512 512 3
## Performance Metrics

Trained on DeepGlobe dataset (512x512 images):

| Metric                | Value     |
|-----------------------|-----------|
| Accuracy             | 98.38%    |
| F1 Score             | 0.8966    |
| Precision            | 0.9011    |
| Recall               | 0.8922    |
| AUC                  | 0.8922    |
| Model Size           | 322.01 MB |
| Inference Time/Image | 35.20 ms  |

### Confusion Matrix:

```json
[[7977176, 64087],  // TN, FP
 [72120, 275225]]   // FN, TP
## Visualization

Sample outputs include:
- Input satellite image
- Predicted road mask
- Ground truth mask
- TP/FP/FN overlay

## Prediction Visualization

## License

MIT License - see LICENSE for details.

## Contact

For questions: your.email@example.com

## Acknowledgements

- DeepGlobe for dataset
- Swin Transformer authors
- TensorFlow/Keras teams
