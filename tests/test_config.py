from brain_tumor.config import Paths, load_yaml


def test_load_yaml_reads_configs_by_bare_name():
    cfg = load_yaml("cnn.yaml")
    assert cfg["num_classes"] == 4
    assert cfg["model"] == "general_cnn"


def test_paths_resolve_under_project_root():
    paths = Paths.load()
    assert paths.classification_train.name == "train"
    assert paths.cnn_weights.name == "best_model.pth"
    assert paths.yolov8_weights.name == "best.pt"
