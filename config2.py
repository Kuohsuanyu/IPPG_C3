import os
import yaml
from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Create root config node
# -----------------------------------------------------------------------------
_C = CN()

# Toolbox mode: 'train_and_test', 'only_test', 'unsupervised_method'
_C.TOOLBOX_MODE = "train_and_test"

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 50
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.LR = 1e-4
_C.TRAIN.MODEL_FILE_NAME = ""

# Optimizer settings
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = "Adam"
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Dataset settings for train/valid/test
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.DATA_PATH = ""
_C.DATA.CACHED_PATH = "PreprocessedData"
_C.DATA.FILE_LIST_PATH = os.path.join(_C.DATA.CACHED_PATH, "DataFileLists")  # can be dir or csv
_C.DATA.DATASET = ""
_C.DATA.FS = 30  # video fps or sample frequency
_C.DATA.DATA_FORMAT = "NDCHW"  # data format (e.g. NCHW)

# Data pre-processing settings
_C.DATA.PREPROCESS = CN()
_C.DATA.PREPROCESS.DO_CROP_FACE = True
_C.DATA.PREPROCESS.CROP_FACE_BACKEND = "HC"  # face detector backend (HC=HaarCascade)
_C.DATA.PREPROCESS.CROP_FACE_LARGE_BOX = True
_C.DATA.PREPROCESS.CROP_FACE_LARGE_BOX_COEF = 1.5
_C.DATA.PREPROCESS.DO_CHUNK = True
_C.DATA.PREPROCESS.CHUNK_LENGTH = 180
_C.DATA.PREPROCESS.DATA_TYPE = ["RGB"]
_C.DATA.PREPROCESS.DATA_AUG = ["None"]
_C.DATA.PREPROCESS.LABEL_TYPE = "Raw"
_C.DATA.PREPROCESS.USE_PSEUDO_PPG_LABEL = False
_C.DATA.PREPROCESS.RESIZE = CN()
_C.DATA.PREPROCESS.RESIZE.W = 128
_C.DATA.PREPROCESS.RESIZE.H = 128

# -----------------------------------------------------------------------------
# Validation settings (can copy from train and override)
# -----------------------------------------------------------------------------
_C.VALID = CN()
_C.VALID.ENABLED = False
_C.VALID.DATA_PATH = ""
_C.VALID.DATASET = ""
_C.VALID.FS = 30
_C.VALID.BATCH_SIZE = 8
# You can add PREPROCESS etc. similar to TRAIN

# -----------------------------------------------------------------------------
# Testing / Inference settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.DATA_PATH = ""
_C.TEST.DATASET = ""
_C.TEST.FS = 30
_C.TEST.BATCH_SIZE = 4
_C.TEST.METRICS = ["MAE", "RMSE", "SUCI", "MER"]
_C.TEST.USE_LAST_EPOCH = True
_C.TEST.OUTPUT_SAVE_DIR = ""

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = "DeepPhys"  # model name
_C.MODEL.PRETRAINED = False
_C.MODEL.RESUME_PATH = ""  # checkpoint path

# You can add model specific parameters here
_C.MODEL.DROP_RATE = 0.0

# Example submodules for specific models
_C.MODEL.DEEPPHYS = CN()
_C.MODEL.DEEPPHYS.LAYERS = 18

_C.MODEL.PHYSNET = CN()
_C.MODEL.PHYSNET.FRAME_NUM = 64

# -----------------------------------------------------------------------------
# Inference / Evaluation settings
# -----------------------------------------------------------------------------
_C.INFERENCE = CN()
_C.INFERENCE.BATCH_SIZE = 4
_C.INFERENCE.EVALUATION_METHOD = "FFT"  # or "peak_detection"
_C.INFERENCE.EVALUATION_WINDOW = CN()
_C.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW = False
_C.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE = 10
_C.INFERENCE.MODEL_PATH = ""

# -----------------------------------------------------------------------------
# Logging and device settings
# -----------------------------------------------------------------------------
_C.LOG = CN()
_C.LOG.PATH = "runs/exp"
_C.DEVICE = "cuda:0"
_C.NUM_GPUS = 1

# -----------------------------------------------------------------------------
# Utilities for loading and merging config from yaml
# -----------------------------------------------------------------------------
def update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.safe_load(f)
    config.merge_from_other_cfg(CN(yaml_cfg))
    config.freeze()

def get_config(cfg_file=None):
    config = _C.clone()
    if cfg_file:
        update_config_from_file(config, cfg_file)
    return config
