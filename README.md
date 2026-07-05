# Brain Tumor Classification

Phan loai u nao tren anh MRI thanh 4 lop: **Glioma, Meningioma, No Tumor,
Pituitary**, so sanh 4 mo hinh hoc sau:

| Mo hinh | Loai | Thu vien |
|---|---|---|
| CNN (`GeneralCNN`) | Classifier tu xay dung | PyTorch |
| Vision Transformer (`vit_base_patch16_224`) | Classifier fine-tune | timm |
| YOLOv8 | Detector (lop = box class) | Ultralytics |
| YOLOv10 | Detector (lop = box class) | Ultralytics |

Day la ban xay dung lai co cau truc cua du an NCKH goc (truoc day nam o
`D:\Data\NCKH\BrainTumor`, gom 3 ban Google Drive export voi code, config va
weight/dataset zip nam lan lon voi nhau). Weight (~300MB-1.2GB moi file) va
dataset goc **khong duoc sao chep vao day** — chi cau truc thu muc va code
duoc xay dung lai; xem [Du lieu va trong so mo hinh](#du-lieu-va-trong-so-mo-hinh).

## Cau truc du an

```
brain-tumor/
├── configs/                 # Hyperparameter & duong dan (YAML), khong hardcode trong code
│   ├── paths.yaml            # Duong dan data/weights, override bang bien moi truong
│   ├── cnn.yaml / vit.yaml / yolov8.yaml / yolov10.yaml
│   └── yolo_dataset.yaml     # Dataset descriptor cho Ultralytics (dung chung v8/v10)
├── data/
│   ├── raw/{images,labels}/  # Anh + nhan YOLO goc (1 class id / file)
│   └── processed/{cnn,vit,yolo}/  # Sinh ra tu scripts/split_dataset.py
├── weights/{cnn,vit,yolov8,yolov10}/  # Checkpoint da train (.pth / .pt)
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
├── scripts/                   # CLI: check_dataset.py, split_dataset.py, train.py, evaluate.py
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
  `split_dataset_ViT`) thanh mot `split_classification_dataset` dung chung.
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

`ultralytics` yeu cau internet de tai checkpoint pretrained (`yolov8m.pt`,
`yolov10m.pt`) o lan train dau tien neu chua co san trong thu muc lam viec.

## Du lieu va trong so mo hinh

1. Copy anh MRI goc vao `data/raw/images/` va nhan YOLO tuong ung (1 dong
   `class_id x_center y_center w h`, class_id theo bang trong
   `src/brain_tumor/constants.py`) vao `data/raw/labels/`.
2. Kiem tra tinh toan ven:
   ```bash
   python scripts/check_dataset.py
   ```
3. Chia dataset theo dinh dang tung mo hinh can (ImageFolder cho CNN/ViT,
   images/+labels/ cho YOLO):
   ```bash
   python scripts/split_dataset.py --target cnn   --train 70 --val 15 --test 15
   python scripts/split_dataset.py --target vit   --train 70 --val 15 --test 15
   python scripts/split_dataset.py --target yolo  --train 70 --val 15 --test 15
   ```
4. Dat checkpoint da train san (neu co) vao `weights/<model>/` theo ten file
   trong `configs/paths.yaml` (vd `weights/cnn/cnn_checkpoint.pth`).

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
python scripts/train.py --model yolov10
```

Hyperparameter doc tu `configs/<model>.yaml`. Checkpoint tot nhat (theo val
accuracy voi CNN/ViT, `best.pt` voi YOLO) duoc luu vao duong dan khai bao
trong `configs/paths.yaml`.

## Danh gia

```bash
python scripts/evaluate.py --model cnn --split val
python scripts/evaluate.py --model yolov10 --split val
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
