# Sports Image Classification using CNN and Transfer Learning

## Project Overview
This project applies Convolutional Neural Networks (CNNs) to classify images into 100 different sports categories using the Kaggle 100 Sports Image Classification dataset.

## Dataset
- Dataset name: 100 Sports Image Classification
- Dataset link: https://www.kaggle.com/datasets/gpiosenka/sports-classification
- Image format: JPG
- Image size: 224 x 224 x 3
- Number of classes: 100
- Training images: 13,493
- Validation images: 500
- Test images: 500

## Models Implemented
1. Custom CNN model
2. MobileNetV2 transfer learning model

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Training and validation accuracy/loss curves

## Repository Structure
```text
sports_phase2_project/
├── README.md
├── requirements.txt
├── proposal.docx
├── sports_classification_phase2.ipynb
├── dataset/
│   └── dataset_link.txt
├── src/
│   └── train.py
├── outputs/
└── models/
```

## How to Run

### Option 1: Kaggle Notebook
1. Open Kaggle Notebook.
2. Add the dataset: `gpiosenka/sports-classification`.
3. Upload or paste the notebook/code.
4. Run all cells.

### Option 2: Local Python Environment
```bash
pip install -r requirements.txt
python src/train.py
```

If running locally, update `DATASET_DIR` in `src/train.py` to the folder where the dataset is extracted.

## Expected Outputs
The code saves the following outputs:
- `outputs/cnn_accuracy_loss_curves.png`
- `outputs/mobilenet_accuracy_loss_curves.png`
- `outputs/cnn_confusion_matrix.png`
- `outputs/mobilenet_confusion_matrix.png`
- `outputs/model_comparison.csv`
- `models/custom_cnn_model.keras`
- `models/mobilenetv2_model.keras`
