import os

from easydict import EasyDict as edict

config = edict()

# === Core Configuration ===
config.pretrained = "ArcFace"  # Options: ArcFace, SphereFace, ElasticFace
config.data = "LFW"
config.classif = "Gender"

# === Training Parameters ===
config.alpha = 1.0  # Utility loss weight
config.beta = 5.0  # Privacy loss weight
config.lr_filter = 0.001
config.lr_clf = 0.0001
config.num_epoch = 30
config.batch_size = 64
config.n_classes = 2
config.index = 4
config.embedding_size = 512
config.init_zero = True
config.network = "iresnet50"
config.use_se = False

# === Logging and Checkpointing ===
config.log_freq = 120
config.global_step = 0
config.freq_clf = 100
config.iter_clf = 5
config.warmup_epoch = -1

# === Targets for Validation/Testing ===
config.val_targets = ["agedb_30"]
config.test_targets = ["lfw", "agedb_30", "colorferet"]

# === Paths (edit these for your local setup or server environment) ===

# Base directories
base_log_dir = "path/to/logs"  # e.g., 'logs/' or '/mnt/logs/'
base_data_dir = "path/to/datasets"  # e.g., 'datasets/' or '/mnt/datasets/'
base_model_dir = "path/to/saved_models"  # e.g., 'models/' or '/output/models/'
base_gender_model_dir = "path/to/gender_models"  # e.g., 'models_gender/'
base_estim_dir = "path/to/estimations"  # e.g., 'estim_results/'
base_reference_dir = "path/to/pretrained_models"  # e.g., 'pretrained/'

# Experiment-specific
config.experiment = f"Filter_At_Layer{config.index}_{config.classif}"
config.checkpoint_dir_name = f"{config.alpha}privacy_{config.beta}utility"

# Final constructed paths
config.log_dir = base_log_dir
config.data_dir = base_data_dir

config.output = os.path.join(
    base_model_dir,
    config.pretrained,
    config.experiment,
    config.data,
    config.checkpoint_dir_name,
)

config.estim_dir = os.path.join(base_estim_dir, config.pretrained, config.data)

config.output_g = os.path.join(
    base_gender_model_dir,
    config.pretrained, 'genderclassifier.pth'
)

config.reference_root = base_reference_dir
config.output_ori = os.path.join(
    config.reference_root, config.pretrained, "reference", "backbone.pth"
)
