# Brain Tumor Classification

Phan loai u nao tren anh MRI thanh 4 lop: **Glioma, Meningioma, No Tumor,
Pituitary**, so sanh 4 mo hinh hoc sau:

| Mo hinh | Loai | Thu vien |
|---|---|---|
| CNN (`GeneralCNN`) | Classifier tu xay dung | PyTorch |
| Vision Transformer (`vit_base_patch16_224`) | Classifier fine-tune | timm |
| YOLOv8 (`yolov8m-cls`) | Classifier fine-tune | Ultralytics |
| YOLOv11 (`yolo11m-cls`) | Classifier fine-tune | Ultralytics |

Ca 4 mo hinh dung chung 1 bo dataset ImageFolder (`data/processed/{train,valid,test}/<lop>`).

Day la ban xay dung lai co cau truc cua du an NCKH goc (truoc day nam o
`D:\Data\NCKH\BrainTumor`, gom 3 ban Google Drive export voi code, config va
weight/dataset zip nam lan lon voi nhau). Weight (~20MB-1.2GB moi file) va
dataset goc (~1.2GB) **khong duoc sao chep vao day** — chi cau truc thu muc va
code duoc xay dung lai; xem [Du lieu va trong so mo hinh](#du-lieu-va-trong-so-mo-hinh).

## Ket qua nghien cuu

Ket qua duoi day trich tu bao cao nghien cuu khoa hoc sinh vien **"Phan loai
hinh anh bang mo hinh hoc sau"** (ma so SV2025-176, Khoa Dao Tao Quoc Te —
Truong Dai hoc Su pham Ky thuat TP.HCM, 10/2025), la co so hoc thuat cua
project nay.

**Du lieu thuc nghiem:** 7,023 anh MRI nao tu Kaggle (tac gia Masoud
Nickparvar) — Glioma 1,621 / Meningioma 1,645 / Pituitary 1,757 / No Tumor
2,000 anh. Chia Train 90% – Test 10%, Validation = 10% cua tap Train.

**So sanh hieu suat 4 mo hinh tren tap Validation:**

| Mo hinh | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| CNN (baseline) | 89.58% | 89.18% | 89.15% | 89.11% |
| Vision Transformer | 98.14% | 98.10% | 98.09% | 98.09% |
| **YOLOv8** | **99.57%** | **99.60%** | **99.55%** | **99.57%** |
| YOLOv11 | 96.71% | 97.36% | 96.63% | 96.87% |

**Toc do suy luan va kich thuoc mo hinh:**

| Mo hinh | Thoi gian huan luyen | Thoi gian danh gia / anh | Kich thuoc |
|---|---|---|---|
| CNN | ~222 phut | ~0.16s | ~800 MB |
| ViT | ~220 phut | ~0.02s (nhanh nhat) | ~327 MB |
| YOLOv8 | ~174 phut | ~0.27s | ~30.2 MB |
| YOLOv11 | ~247 phut | ~0.74s | ~19.9 MB (nho nhat) |

**Nhan xet:**

- **YOLOv8** la mo hinh tot nhat tong the — tat ca chi so deu ~99.5-99.6%,
  gan nhu hoan hao tren ca 4 lop u (Confusion Matrix chi lech 1-2 mau moi
  lop), toc do suy luan nhanh va kich thuoc nho gon. Day la mo hinh duoc de
  xuat de trien khai thuc te ho tro chan doan.
- **ViT** xep thu 2 va on dinh nho co che self-attention toan cuc, dong thoi
  co toc do suy luan nhanh nhat trong 4 mo hinh (~0.02s/anh).
- **YOLOv11** nho gon nhat (~19.9 MB) nhung van con nham lan dang ke o lop
  Notumor (Precision chi 90.09%), can cai thien them.
- **CNN** dong vai tro mo hinh co so (baseline) de doi chieu, yeu nhat o lop
  Meningioma (F1 chi 80.25%) do cac loai u co dac trung hinh anh tuong dong.

Chi tiet day du (danh gia tung lop, ma tran nham lan, so sanh optimizer
SGD/Adam/AdamW, huong phat trien...) xem file bao cao goc
`BaoCaoNCKH_PhanLoaiHinhAnh_KhoaDTQT_Lan2.docx`.

## Cau truc du an

```
brain-tumor/
├── configs/                 # Hyperparameter & duong dan (YAML), khong hardcode trong code
│   ├── paths.yaml            # Duong dan data/weights, override bang bien moi truong
│   └── cnn.yaml / vit.yaml / yolov8.yaml / yolov11.yaml
├── data/
│   ├── raw/{images,labels}/  # Anh + nhan YOLO goc (1 class id / file)
│   └── processed/{train,valid,test}/<lop>/  # ImageFolder dung chung ca 4 model
├── weights/{cnn,vit,yolov8,yolov11}/  # Checkpoint da train (.pth / .pt)
├── src/brain_tumor/          # Package chinh, khong phu thuoc Streamlit
│   ├── config.py             # Load configs/paths.yaml -> Paths (dataclass)
│   ├── constants.py          # Ten lop, phan mo rong anh
│   ├── data/                 # splitting.py, transforms.py
│   ├── models/               # cnn.py, vit.py, yolo.py
│   ├── training/             # train_cnn.py, train_vit.py, train_yolo.py
│   ├── evaluation/           # evaluate_cnn.py, evaluate_vit.py, evaluate_yolo.py,
│   │                         # compare.py (so sanh 4 mo hinh), reporting.py (bieu do)
│   └── inference/            # predict.py (du doan 1 anh, dung cho GUI upload)
├── app/streamlit_app.py       # Giao dien Streamlit, chi goi ham trong brain_tumor
├── scripts/                   # CLI: check_dataset.py, split_dataset.py, train.py,
│                               # evaluate.py, download_weights.py, download_dataset.py
└── tests/                     # pytest, khong can GPU/model nang de chay
```

### Vi sao xay dung lai theo cach nay

Code goc (`args.py`, `Library.py`, `GUI_version2.py`, `cnn.py`) tron logic
tach dataset, train, evaluate va UI Streamlit vao chung 1-2 file, hardcode
duong dan Windows (`D:\Dataset\...`) o mot ban va duong dan Colab
(`/content/...`) o ban khac, dong thoi lap lai gan nguyen code chia
dataset cho CNN va ViT. Ban xay dung lai nay:

- Tach **config** (YAML) khoi **code** — doi may/may di chuyen du lieu chi can
  sua `configs/paths.yaml` hoac dat bien moi truong `BRAIN_TUMOR_DATA_ROOT` /
  `BRAIN_TUMOR_WEIGHTS_ROOT`.
- Tach **logic** (`src/brain_tumor/`) khoi **giao dien** (`app/`) — cac ham
  training/evaluation nhan `progress_callback` thay vi goi thang
  `st.progress`, nen co the dung lai trong CLI (`scripts/`), test, hoac giao
  dien khac.
- Gop 2 ham chia dataset giong het nhau (`split_dataset_CNN` /
  `split_dataset_ViT`) thanh mot `split_classification_dataset` dung chung
  cho ca 4 model (YOLOv8/v11 cung dung ImageFolder classify thay vi detector).
- Them **training that su** cho ca 4 mo hinh — ban goc chi co nut "Train"
  hien text tinh, va panel upload-anh chi hien ket qua vi du co dinh; ca hai
  gio goi dung model that (`src/brain_tumor/training/`,
  `src/brain_tumor/inference/predict.py`).

## Cai dat

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
pip install -e .                  # cai package brain_tumor (src layout)
```

`ultralytics` yeu cau internet de tai checkpoint pretrained (`yolov8m-cls.pt`,
`yolo11m-cls.pt`) o lan train dau tien neu chua co san trong thu muc lam viec.

## Du lieu va trong so mo hinh

1. Copy anh MRI goc vao `data/raw/images/` va nhan YOLO tuong ung (1 dong
   `class_id x_center y_center w h`, class_id theo bang trong
   `src/brain_tumor/constants.py`) vao `data/raw/labels/`.
2. Kiem tra tinh toan ven:
   ```bash
   python scripts/check_dataset.py
   ```
3. Chia dataset thanh 1 bo ImageFolder dung chung ca 4 model:
   ```bash
   python scripts/split_dataset.py --train 70 --val 15 --test 15
   ```
4. Tai checkpoint da train san ve `weights/<model>/` bang script (xem muc
   duoi), hoac tu train lai bang `scripts/train.py`.

### Tai dataset tu Hugging Face Hub

Bo anh da chia san (`data/processed/{train,valid,test}`, ~12.7k anh, ~1.2GB)
duoc luu tren repo dataset [Lomuto/brain-tumor-mri-dataset](https://huggingface.co/datasets/Lomuto/brain-tumor-mri-dataset):

```bash
python scripts/download_dataset.py
```

### Tai san model tu Hugging Face Hub

4 checkpoint da train duoc luu tren Hugging Face Hub (khong dua vao git vi
qua nang, xem `.gitignore`):

| Mo hinh | Repo |
|---|---|
| CNN | [Lomuto/cnn-brain-tumor-clasification](https://huggingface.co/Lomuto/cnn-brain-tumor-clasification) |
| ViT | [Lomuto/vit-brain-tumor-classification](https://huggingface.co/Lomuto/vit-brain-tumor-classification) |
| YOLOv8 | [Lomuto/yolov8-brain-tumor-classification](https://huggingface.co/Lomuto/yolov8-brain-tumor-classification) |
| YOLOv11 | [Lomuto/yolov11-brain-tumor-classification](https://huggingface.co/Lomuto/yolov11-brain-tumor-classification) |

Tai ca 4 model ve dung vi tri khai bao trong `configs/paths.yaml`:

```bash
python scripts/download_weights.py
```

Hoac chi tai mot model (`--model cnn|vit|yolov8|yolov11`), hay `--force` de
tai lai neu file da co san.

Neu du lieu/weight nam o o dia hoac may khac, khong can sua code — chi dat:

```bash
set BRAIN_TUMOR_DATA_ROOT=D:\Dataset       # Windows cmd
set BRAIN_TUMOR_WEIGHTS_ROOT=D:\Dataset\Result
```

## Huan luyen

```bash
python scripts/train.py --model cnn
python scripts/train.py --model vit
python scripts/train.py --model yolov8
python scripts/train.py --model yolov11
```

Hyperparameter doc tu `configs/<model>.yaml`. Checkpoint tot nhat (theo val
accuracy) duoc luu vao duong dan khai bao trong `configs/paths.yaml`.

## Danh gia

```bash
python scripts/evaluate.py --model cnn --split val
python scripts/evaluate.py --model yolov11 --split val
```

## Giao dien Streamlit

```bash
streamlit run app/streamlit_app.py
```

Cho phep: chon mo hinh, xem tham so, chia dataset, train, validate, test,
so sanh ca 4 mo hinh (bang + bieu do), va phan loai/du doan mot anh tai len.

## Kiem thu

```bash
pytest
```

`tests/` chi kiem tra logic chia dataset va load config bang du lieu gia
lap trong thu muc tam — khong yeu cau torch/GPU/model that.
