# Multi-Stage Hybrid CNN-ViT for Chest X-Ray Classification

This repository implements a deep learning pipeline for classifying Chest X-Ray images into three categories: **Effusion**, **Infiltration**, and **No Finding**.

The project utilizes a **Hybrid Architecture** combining **DenseNet121** (for feature extraction) and a **Vision Transformer (ViT)** (for global attention), employed within a **Hierarchical Inference Strategy** to maximize accuracy on difficult-to-distinguish classes.

## 📂 Repository Content

- **`Main.ipynb`**: The complete Jupyter Notebook containing data preprocessing, model architecture definition, training loops (Stage 1 & 2), and evaluation.
- **`stage1_best_state.pth`**: Model weights for the 3-class hybrid classifier.
- **`stage2_best_state.pth`**: Model weights for the specialized binary classifier (Infiltration vs. No Finding).

## 🧠 Model Architecture

The core model is a **CNN-ViT Hybrid**:
1.  **CNN Backbone:** `DenseNet121` (pretrained on ImageNet) is used to extract local spatial features.
2.  **Projection:** CNN features are flattened and projected to an embedding dimension.
3.  **Transformer:** A standard `TransformerEncoder` applies self-attention mechanisms to capture global dependencies in the X-ray.
4.  **Classifier:** A final Multi-Layer Perceptron (MLP) head for prediction.

## ⚙️ Hierarchical Inference Strategy

To improve performance, the model does not rely on a single pass. Instead, it uses a two-stage decision process:

1.  **Stage 1 (General Classification):** The image is passed through the 3-class Hybrid Model.
2.  **Stage 2 (Refinement):** * If Stage 1 predicts **Effusion**, the prediction is accepted.
    * If Stage 1 predicts **Infiltration** or **No Finding** (which are often visually similar), the image is passed to a specialized **Binary Hybrid Model** trained exclusively to distinguish between these two.

```mermaid
graph TD;
    A[Input X-Ray] --> B[Stage 1: 3-Class Hybrid Model];
    B -- Predicts Effusion --> C[Final: Effusion];
    B -- Predicts Infiltration/No Finding --> D[Stage 2: Binary Model];
    D --> E[Final: Infiltration OR No Finding];
