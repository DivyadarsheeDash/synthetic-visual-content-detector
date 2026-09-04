# 🔍 Real vs. AI-Generated Image Classification

An end-to-end Deep Learning framework developed in PyTorch for classifying synthetic/AI-generated images vs. authentic photographs using custom CNNs and Transfer Learning.

---

## 📌 Project Highlights

- **Dataset Scale**: 120,000 images (100k Train / 20k Test) perfectly balanced between `FAKE` and `REAL` classes.
- **Interpolation Benchmarking**: Comparison between `NEAREST`, `BILINEAR`, and `BICUBIC` upscaling.
- **Custom Noise Engineering**: 10% Salt & Pepper noise injection applied dynamically on-the-fly to 25.8% of samples for real-world robustness.
- **State-of-the-Art Results**: Achieved **99.11% Accuracy**, **99.10% F1-Score**, and **99.58% Precision** using fine-tuned **EfficientNet-B0**.

---

## 📊 Benchmark Results

| Model | Input Size | Preprocessing / Noise | Epochs | Accuracy | F1-Score | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TinyVGG (Baseline)** | 32 × 32 | TrivialAugmentWide | 1 | 80.58% | 78.74% | 87.00% | 71.91% |
| **TinyVGG (Upscaled)** | 256 × 256 | Bicubic + Salt & Pepper (25.8%) | 1 | 85.33% | 85.04% | 86.72% | 83.42% |
| **EfficientNet-B0** | 256 × 256 | Pretrained ImageNet + S&P | 1 | 98.74% | 98.74% | 98.63% | 98.85% |
| **EfficientNet-B0 (+Sched)**| 256 × 256 | Pretrained + ReduceLROnPlateau | 2 | **99.11%** | **99.10%** | **99.58%** | **99.83%** |

---

## 📁 Repository Structure

- [`Data_getting.ipynb`](file:///c:/Dash_Strikes/REAL_VS_AI/Data_getting.ipynb): Data ingestion, directory scanning, and CSV generation (`Train_Data.csv`, `Test_Data.csv`).
- [`dataset_maker_and_transformer.ipynb`](file:///c:/Dash_Strikes/REAL_VS_AI/dataset_maker_and_transformer.ipynb): Interpolation exploration, channel normalization, and PyTorch dataset serialization.
- [`Dataloader_&_Models.ipynb`](file:///c:/Dash_Strikes/REAL_VS_AI/Dataloader_&_Models.ipynb): Dynamic noise augmentation, TinyVGG & EfficientNet-B0 model training and evaluation.
- [`REAL_VS_AI_PROJECT_DOCUMENTATION.md`](file:///c:/Dash_Strikes/REAL_VS_AI/REAL_VS_AI_PROJECT_DOCUMENTATION.md): Complete in-depth technical report.
