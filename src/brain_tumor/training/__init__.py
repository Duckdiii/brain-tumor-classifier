"""Training entry points for CNN, ViT, YOLOv8 and YOLOv10.

The original project only ever trained models interactively in Colab
notebooks; the Streamlit "Train" button was a static placeholder. These
modules implement real training loops driven by the ``configs/*.yaml``
hyperparameters that were already checked into the project but never wired
up to code.
"""
