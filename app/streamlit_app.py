"""Streamlit demo app: dataset splitting, training, validation, testing and
cross-model comparison for the CNN / ViT / YOLOv8 / YOLOv10 tumor classifiers.

This is a rewrite of the original ``GUI_version2.py`` that delegates all
business logic to the ``brain_tumor`` package instead of embedding it
directly in the UI file, and turns the previously-decorative "Train" button
and "upload image" panel into real actions.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st
import torch
from PIL import Image

from brain_tumor.config import Paths, load_yaml, yolo_dataset_yaml
from brain_tumor.constants import CLASS_NAMES, NUM_CLASSES
from brain_tumor.data.splitting import split_classification_dataset, split_detection_dataset
from brain_tumor.data.transforms import compute_mean_std
from brain_tumor.evaluation.compare import compare_models
from brain_tumor.evaluation.evaluate_cnn import evaluate as evaluate_cnn
from brain_tumor.evaluation.evaluate_cnn import test as test_cnn
from brain_tumor.evaluation.evaluate_vit import evaluate as evaluate_vit
from brain_tumor.evaluation.evaluate_vit import test as test_vit
from brain_tumor.evaluation.evaluate_yolo import evaluate_yolo, predict_on_dir
from brain_tumor.evaluation.reporting import plot_all_metrics
from brain_tumor.inference.predict import predict_image_classifier, predict_image_yolo
from brain_tumor.models.cnn import GeneralCNN
from brain_tumor.models.vit import load_vit_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATHS = Paths.load()

st.set_page_config(page_title="Brain Tumor Classification", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { background-color: #e0e0e0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def st_progress_callback(bar, label: str):
    def _cb(percent: int, message: str) -> None:
        bar.progress(min(percent, 100), text=f"{label}: {message}")

    return _cb


# ---------------------------------------------------------------------------
# Khu vuc 1: Chon mo hinh + Train / Validation / Test / Comparison
# ---------------------------------------------------------------------------
col1, col2 = st.columns([3, 5])
with col1:
    st.header("Chon mo hinh")
    main_model = st.selectbox(
        "Chon loai mo hinh", ["CNN", "Yolo v8", "Yolo v10", "Vision Transformer"], key="main_model_select"
    )

MODEL_CONFIG_FILE = {
    "CNN": "cnn.yaml",
    "Vision Transformer": "vit.yaml",
    "Yolo v8": "yolov8.yaml",
    "Yolo v10": "yolov10.yaml",
}

train_col1, train_col2 = st.columns([3, 5])
with train_col1:
    st.subheader("Train")
    train_btn = st.button("Bat dau huan luyen", key="train_btn")
    train_status = st.empty()

with train_col2:
    st.subheader("Tham so huan luyen")
    cfg = load_yaml(MODEL_CONFIG_FILE[main_model])
    st.dataframe(pd.DataFrame(list(cfg.items()), columns=["Tham so", "Gia tri"]))

    if train_btn:
        progress_bar = st.progress(0, text="Chuan bi huan luyen...")
        callback = st_progress_callback(progress_bar, main_model)
        if main_model == "CNN":
            from brain_tumor.training.train_cnn import train as train_cnn_fn

            history = train_cnn_fn(
                PATHS.cnn_train, PATHS.cnn_val, PATHS.cnn_weights,
                num_classes=NUM_CLASSES, image_size=cfg["image_size"], epochs=cfg["epochs"],
                batch_size=cfg["batch_size"], lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                device=DEVICE, progress_callback=callback,
            )
            train_status.success(f"Val accuracy cuoi cung: {history['val_accuracy'][-1]:.2%}")
        elif main_model == "Vision Transformer":
            from brain_tumor.training.train_vit import train as train_vit_fn

            history = train_vit_fn(
                PATHS.vit_train, PATHS.vit_val, PATHS.vit_weights,
                num_classes=NUM_CLASSES, model_name=cfg["model"], image_size=cfg["image_size"],
                epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"],
                weight_decay=cfg["weight_decay"], warmup_epochs=cfg["warmup_epochs"],
                gradient_clipping=cfg["gradient_clipping"], device=DEVICE, progress_callback=callback,
            )
            train_status.success(f"Val accuracy cuoi cung: {history['val_accuracy'][-1]:.2%}")
        else:
            from brain_tumor.training.train_yolo import train as train_yolo_fn

            is_v8 = main_model == "Yolo v8"
            runs = PATHS.yolov8_runs if is_v8 else PATHS.yolov10_runs
            weights_out = PATHS.yolov8_weights if is_v8 else PATHS.yolov10_weights
            extra = {k: v for k, v in cfg.items() if k not in {"task", "model", "epochs", "imgsz", "batch", "device"}}
            train_status.info("Dang huan luyen YOLO (xem tien do trong console)...")
            train_yolo_fn(
                base_model=cfg["model"], data_yaml=yolo_dataset_yaml(), project=runs,
                weights_out=weights_out, epochs=cfg["epochs"], imgsz=cfg["imgsz"],
                batch=cfg["batch"], device=cfg.get("device", "cpu"), **extra,
            )
            train_status.success("Huan luyen YOLO hoan tat.")

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
val_col1, val_col2 = st.columns([3, 5])
with val_col1:
    st.subheader("Validation")
    val_model = st.selectbox(
        "Chon loai mo hinh", ["CNN", "Yolo v8", "Yolo v10", "Vision Transformer"], key="val_model_select"
    )
    btn_val = st.button("Bat dau danh gia", key="val_btn")

with val_col2:
    st.subheader("Ket qua Validation")
    if btn_val:
        progress_bar = st.progress(0, text="Dang danh gia...")
        callback = st_progress_callback(progress_bar, val_model)
        if val_model == "CNN":
            model = GeneralCNN(num_classes=NUM_CLASSES).to(DEVICE)
            model.load_state_dict(torch.load(PATHS.cnn_weights, map_location=DEVICE))
            _, precision = evaluate_cnn(model, PATHS.cnn_val, PATHS.cnn_train, CLASS_NAMES, device=DEVICE, progress_callback=callback)
            st.success(f"Precision: {precision:.2f}%")
        elif val_model == "Vision Transformer":
            model = load_vit_checkpoint(PATHS.vit_weights, num_classes=NUM_CLASSES, device=DEVICE)
            precision = evaluate_vit(model, PATHS.vit_val, PATHS.vit_train, device=DEVICE, progress_callback=callback)
            st.success(f"Precision: {precision:.2f}%")
        else:
            weights = PATHS.yolov8_weights if val_model == "Yolo v8" else PATHS.yolov10_weights
            runs = PATHS.yolov8_runs if val_model == "Yolo v8" else PATHS.yolov10_runs
            result = evaluate_yolo(weights, yolo_dataset_yaml(), runs, device=0 if DEVICE.type == "cuda" else "cpu")
            st.write(result)

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
test_col1, test_col2 = st.columns([3, 5])
with test_col1:
    st.subheader("Testing")
    test_model = st.selectbox(
        "Chon loai mo hinh", ["CNN", "Yolo v8", "Yolo v10", "Vision Transformer"], key="test_model_select"
    )
    btn_test = st.button("Bat dau test", key="test_btn")

with test_col2:
    st.subheader("Ket qua Test")
    if btn_test:
        progress_bar = st.progress(0, text="Dang test...")
        callback = st_progress_callback(progress_bar, test_model)
        if test_model == "CNN":
            model = GeneralCNN(num_classes=NUM_CLASSES).to(DEVICE)
            model.load_state_dict(torch.load(PATHS.cnn_weights, map_location=DEVICE))
            results, accuracy = test_cnn(model, PATHS.cnn_test, PATHS.cnn_train, CLASS_NAMES, device=DEVICE, progress_callback=callback)
            st.success(f"Test accuracy: {accuracy:.2f}%")
            for r in results[:10]:
                mark = "OK" if r["is_correct"] else "SAI"
                st.write(f"{os.path.basename(r['filename'])}: That={r['true_label']} Doan={r['predicted_label']} [{mark}]")
        elif test_model == "Vision Transformer":
            model = load_vit_checkpoint(PATHS.vit_weights, num_classes=NUM_CLASSES, device=DEVICE)
            classes = sorted(os.listdir(PATHS.vit_train))
            results, accuracy = test_vit(model, PATHS.vit_test, PATHS.vit_train, classes, device=DEVICE, progress_callback=callback)
            st.success(f"Test accuracy: {accuracy:.2f}%")
            cols = st.columns(4)
            for idx, r in enumerate(results):
                with cols[idx % 4]:
                    mark = "OK" if r["is_correct"] else "SAI"
                    st.image(r["filename"], caption=f"That={r['true_label']} Doan={r['predicted_label']} [{mark}]", use_container_width=True)
        else:
            weights = PATHS.yolov8_weights if test_model == "Yolo v8" else PATHS.yolov10_weights
            runs = PATHS.yolov8_runs if test_model == "Yolo v8" else PATHS.yolov10_runs
            images_dir = PATHS.yolo_test / "images"
            tested = predict_on_dir(weights, images_dir, runs, progress_callback=callback)
            st.success(f"Da du doan {len(tested)} anh.")
            cols = st.columns(4)
            for idx, img_path in enumerate(tested):
                with cols[idx % 4]:
                    st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
com_col1, com_col2 = st.columns([3, 5])
with com_col1:
    st.subheader("Comparison")
    btn_com = st.button("So sanh cac mo hinh", key="com_btn")

with com_col2:
    st.subheader("Ket qua so sanh")
    if btn_com:
        progress_bar = st.progress(0, text="Dang so sanh...")
        callback = st_progress_callback(progress_bar, "Comparison")
        df_results = compare_models(PATHS, device=DEVICE, progress_callback=callback)
        st.dataframe(df_results)
        for fig in plot_all_metrics(df_results):
            st.pyplot(fig)

# ---------------------------------------------------------------------------
# Upload & classify a single image
# ---------------------------------------------------------------------------
result_col1, result_col2 = st.columns([3, 5])
with result_col1:
    st.subheader("Ket qua")
with result_col2:
    st.subheader("Phan loai anh tai len")
    uploaded_file = st.file_uploader("Tai len hinh anh", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Anh da tai len")

        if main_model in ("CNN", "Vision Transformer"):
            if main_model == "CNN":
                model = GeneralCNN(num_classes=NUM_CLASSES).to(DEVICE)
                model.load_state_dict(torch.load(PATHS.cnn_weights, map_location=DEVICE))
                image_size = cfg["image_size"]
                mean, std = compute_mean_std(PATHS.cnn_train, image_size)
            else:
                model = load_vit_checkpoint(PATHS.vit_weights, num_classes=NUM_CLASSES, device=DEVICE)
                image_size = cfg["image_size"]
                mean, std = compute_mean_std(PATHS.vit_train, image_size)

            label, confidence = predict_image_classifier(model, image, mean, std, image_size, device=DEVICE)
            st.write(f"Nhan du doan: {label}")
            st.write(f"Xac suat: {confidence:.2%}")
        else:
            weights = PATHS.yolov8_weights if main_model == "Yolo v8" else PATHS.yolov10_weights
            results = predict_image_yolo(weights, image)
            st.write(f"So doi tuong phat hien: {len(results[0].boxes)}")

# ---------------------------------------------------------------------------
# Khu vuc 2: Phan chia dataset
# ---------------------------------------------------------------------------
with col2:
    st.header("Phan chia Dataset")
    train_val, val_test = st.slider(
        "Phan chia ty le Train/Val/Test", 0.0, 100.0, (70.0, 85.0), step=1.0, format="%.0f%%"
    )
    train_ratio = train_val
    val_ratio = val_test - train_val
    test_ratio = 100.0 - val_test
    st.write(f"Train: {train_ratio:.0f}%   Val: {val_ratio:.0f}%   Test: {test_ratio:.0f}%")

    split_btn = st.button("Phan chia dataset", key="split_btn")
    if split_btn:
        progress_bar = st.progress(0, text="Dang chia dataset...")
        callback = st_progress_callback(progress_bar, "Split")
        if main_model == "Vision Transformer":
            output_dir = PATHS.vit_train.parent
            split_classification_dataset(PATHS.raw_images, PATHS.raw_labels, output_dir, train_ratio, val_ratio, test_ratio, progress_callback=callback)
        elif main_model == "CNN":
            output_dir = PATHS.cnn_train.parent
            split_classification_dataset(PATHS.raw_images, PATHS.raw_labels, output_dir, train_ratio, val_ratio, test_ratio, progress_callback=callback)
        else:
            output_dir = PATHS.yolo_train.parent
            split_detection_dataset(PATHS.raw_images, PATHS.raw_labels, output_dir, train_ratio, val_ratio, test_ratio, progress_callback=callback)
        st.success("Phan chia dataset thanh cong!")
