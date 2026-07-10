# Brain Tumor Classification

Classification of brain tumors on MRI images into 4 classes: **Glioma, Meningioma, No Tumor, Pituitary**, comparing the following 4 deep learning models:

| Model | Type | Library |
|---|---|---|
| CNN (`GeneralCNN`) | Custom-built classifier | PyTorch |
| Vision Transformer (`vit_base_patch16_224`) | Fine-tuned classifier | timm |
| YOLOv8 (`yolov8m-cls`) | Fine-tuned classifier | Ultralytics |
| YOLOv11 (`yolo11m-cls`) | Fine-tuned classifier | Ultralytics |

All 4 models share the same ImageFolder dataset format (`data/processed/{train,valid,test}/<class>`).

This is a restructured and rebuilt version of the original scientific research (NCKH) project (previously located at `D:\Data\NCKH\BrainTumor`, which consisted of 3 Google Drive exports with code, config, and weight/dataset zip files mixed together). Weights (~20MB-1.2GB per file) and the original dataset (~1.2GB) **are not copied here** — only the folder structure and code have been rebuilt; see [Data and Model Weights](#data-and-model-weights).

## Research Results

The results below are extracted from the student scientific research report **"Image Classification using Deep Learning Models"** (Project ID: SV2025-176, Faculty of International Education — Ho Chi Minh City University of Technology and Education, 10/2025), which serves as the academic foundation of this project.

**Experimental Data:** 7,023 brain MRI images from Kaggle (authored by Masoud Nickparvar) — Glioma 1,621 / Meningioma 1,645 / Pituitary 1,757 / No Tumor 2,000 images. Split: Train 90% – Test 10%, Validation = 10% of the Train set.

**Performance comparison of the 4 models on the Validation set:**

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| CNN (baseline) | 89.58% | 89.18% | 89.15% | 89.11% |
| Vision Transformer | 98.14% | 98.10% | 98.09% | 98.09% |
| **YOLOv8** | **99.57%** | **99.60%** | **99.55%** | **99.57%** |
| YOLOv11 | 96.71% | 97.36% | 96.63% | 96.87% |

**Inference speed and model size:**

| Model | Training Time | Evaluation Time / Image | Size |
|---|---|---|---|
| CNN | ~222 minutes | ~0.16s | ~800 MB |
| ViT | ~220 minutes | ~0.02s (fastest) | ~327 MB |
| YOLOv8 | ~174 minutes | ~0.27s | ~30.2 MB |
| YOLOv11 | ~247 minutes | ~0.74s | ~19.9 MB (smallest) |

**Observations:**

- **YOLOv8** is the overall best model — all metrics are around 99.5-99.6%, nearly perfect across all 4 tumor classes (Confusion Matrix only misclassified 1-2 samples per class), fast inference speed, and compact size. This model is recommended for practical deployment in diagnostic support.
- **ViT** ranks second and remains stable thanks to the global self-attention mechanism, while also achieving the fastest inference speed among the 4 models (~0.02s/image).
- **YOLOv11** is the most compact (~19.9 MB) but still has significant confusion in the No Tumor class (Precision of only 90.09%), requiring further improvement.
- **CNN** serves as the baseline model for comparison, performing weakest on the Meningioma class (F1 score of only 80.25%) due to similar visual features among different tumor types.

For full details (class-by-class evaluation, confusion matrix, comparison of SGD/Adam/AdamW optimizers, future work, etc.), please refer to the original report file `BaoCaoNCKH_PhanLoaiHinhAnh_KhoaDTQT_Lan2.docx`.

## Project Structure

```
brain-tumor/
├── configs/                 # Hyperparameters & paths (YAML), not hardcoded in the code
│   ├── paths.yaml            # Paths to data/weights, overridable by environment variables
│   └── cnn.yaml / vit.yaml / yolov8.yaml / yolov11.yaml
├── data/
│   ├── raw/{images,labels}/  # Original images + YOLO labels (1 class id per file)
│   └── processed/{train,valid,test}/<class>/  # Shared ImageFolder for all 4 models
├── weights/{cnn,vit,yolov8,yolov11}/  # Trained checkpoints (.pth / .pt)
├── src/brain_tumor/          # Main package, independent of Streamlit
│   ├── config.py             # Load configs/paths.yaml -> Paths (dataclass)
│   ├── constants.py          # Class names, image extensions
│   ├── data/                 # splitting.py, transforms.py
│   ├── models/               # cnn.py, vit.py, yolo.py
│   ├── training/             # train_cnn.py, train_vit.py, train_yolo.py
│   ├── evaluation/           # evaluate_cnn.py, evaluate_vit.py, evaluate_yolo.py,
│   │                         # compare.py (model comparison), reporting.py (charts)
│   └── inference/            # predict.py (predict single image, used for upload GUI)
├── app/streamlit_app.py       # Streamlit UI, only calls functions in brain_tumor
├── scripts/                   # CLI: check_dataset.py, split_dataset.py, train.py,
│                               # evaluate.py, download_weights.py, download_dataset.py
└── tests/                     # pytest, does not require GPU/heavy models to run
```

### Why restructuring it this way?

The original code (`args.py`, `Library.py`, `GUI_version2.py`, `cnn.py`) mixed the dataset splitting, training, evaluation, and Streamlit UI logic into one or two files. It hardcoded Windows paths (`D:\Dataset\...`) in one version and Colab paths (`/content/...`) in another, while duplicating almost the exact dataset splitting code for CNN and ViT. This restructured version:

- Separates **config** (YAML) from **code** — changing machines or moving data only requires modifying `configs/paths.yaml` or setting the `BRAIN_TUMOR_DATA_ROOT` / `BRAIN_TUMOR_WEIGHTS_ROOT` environment variables.
- Separates **logic** (`src/brain_tumor/`) from the **user interface** (`app/`) — training/evaluation functions accept a `progress_callback` instead of calling `st.progress` directly, making them reusable in the CLI (`scripts/`), tests, or other user interfaces.
- Merges two identical dataset splitting functions (`split_dataset_CNN` / `split_dataset_ViT`) into a single `split_classification_dataset` shared by all 4 models (YOLOv8/v11 also use ImageFolder classification instead of detection).
- Adds **actual training** for all 4 models — the original version only had a "Train" button displaying static text, and the image upload panel showed a fixed dummy result; both now invoke the actual model (`src/brain_tumor/training/`, `src/brain_tumor/inference/predict.py`).

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
pip install -e .                  # install the brain_tumor package (src layout)
```

`ultralytics` requires an internet connection to download pretrained checkpoints (`yolov8m-cls.pt`, `yolo11m-cls.pt`) on the first training run if they are not already present in the working directory.

## Data and Model Weights

1. Copy the original MRI images into `data/raw/images/` and the corresponding YOLO labels (one line of `class_id x_center y_center w h`, with class_id following the table in `src/brain_tumor/constants.py`) into `data/raw/labels/`.
2. Verify integrity:
   ```bash
   python scripts/check_dataset.py
   ```
3. Split the dataset into a single ImageFolder structure shared by all 4 models:
   ```bash
   python scripts/split_dataset.py --train 70 --val 15 --test 15
   ```
4. Download pre-trained checkpoints to `weights/<model>/` using the script (see section below), or train them from scratch using `scripts/train.py`.

### Download Dataset from Hugging Face Hub

The pre-split image dataset (`data/processed/{train,valid,test}`, ~12.7k images, ~1.2GB) is stored in the dataset repository [Lomuto/brain-tumor-mri-dataset](https://huggingface.co/datasets/Lomuto/brain-tumor-mri-dataset):

```bash
python scripts/download_dataset.py
```

### Download Pre-trained Models from Hugging Face Hub

The 4 trained checkpoints are hosted on the Hugging Face Hub (not included in Git due to size limitations, see `.gitignore`):

| Model | Repo |
|---|---|
| CNN | [Lomuto/cnn-brain-tumor-clasification](https://huggingface.co/Lomuto/cnn-brain-tumor-clasification) |
| ViT | [Lomuto/vit-brain-tumor-classification](https://huggingface.co/Lomuto/vit-brain-tumor-classification) |
| YOLOv8 | [Lomuto/yolov8-brain-tumor-classification](https://huggingface.co/Lomuto/yolov8-brain-tumor-classification) |
| YOLOv11 | [Lomuto/yolov11-brain-tumor-classification](https://huggingface.co/Lomuto/yolov11-brain-tumor-classification) |

Download all 4 models to the locations specified in `configs/paths.yaml`:

```bash
python scripts/download_weights.py
```

Or download only a specific model using `--model cnn|vit|yolov8|yolov11`, or use `--force` to re-download if the files already exist.

If the data/weights are located on another drive or machine, there is no need to modify the code — simply set:

```bash
set BRAIN_TUMOR_DATA_ROOT=D:\Dataset       # Windows cmd
set BRAIN_TUMOR_WEIGHTS_ROOT=D:\Dataset\Result
```

## Training

```bash
python scripts/train.py --model cnn
python scripts/train.py --model vit
python scripts/train.py --model yolov8
python scripts/train.py --model yolov11
```

Hyperparameters are read from `configs/<model>.yaml`. The best checkpoint (based on validation accuracy) is saved to the path specified in `configs/paths.yaml`.

## Evaluation

```bash
python scripts/evaluate.py --model cnn --split val
python scripts/evaluate.py --model yolov11 --split val
```

## Streamlit Interface

```bash
streamlit run app/streamlit_app.py
```

Allows: selecting models, viewing parameters, splitting the dataset, training, validating, testing, comparing all 4 models (tables + charts), and classifying/predicting an uploaded image.

## Testing

```bash
pytest
```

`tests/` only tests the dataset splitting logic and config loading using mock data in a temporary directory — no PyTorch, GPU, or actual models are required.
