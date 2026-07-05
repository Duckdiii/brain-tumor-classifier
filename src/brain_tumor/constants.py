"""Shared constants for the brain tumor classification project."""

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CLASS_ID_TO_NAME = dict(enumerate(CLASS_NAMES))
NUM_CLASSES = len(CLASS_NAMES)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
