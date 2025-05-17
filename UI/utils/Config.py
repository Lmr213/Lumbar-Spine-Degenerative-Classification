from pathlib import Path
from typing import List

from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)


DEBUG = False

@dataclass
class DatasetConfig:
    path: Path = Path('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/')
    label_columns = [
       'left_neural_foraminal_narrowing_l1_l2',
       'left_neural_foraminal_narrowing_l2_l3',
       'left_neural_foraminal_narrowing_l3_l4',
       'left_neural_foraminal_narrowing_l4_l5',
       'left_neural_foraminal_narrowing_l5_s1',
       'right_neural_foraminal_narrowing_l1_l2',
       'right_neural_foraminal_narrowing_l2_l3',
       'right_neural_foraminal_narrowing_l3_l4',
       'right_neural_foraminal_narrowing_l4_l5',
       'right_neural_foraminal_narrowing_l5_s1']
    label_map = {
        'Normal/Mild': 0,
        'Moderate': 1,
        'Severe': 2,
        -100: -100 # for missing values
    }
    # use_n_images: int = 18  # select images

@dataclass
class ModelConfig:
    name: str = 'efficientnet_b3.ra2_in1k' # resnet or efficientnet
    train_backbone: bool = False # fine-tuned?
    use_weights: bool = True  # use weighted loss?

@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    seed: int = 42
    n_folds: int = 5
        

config = Config()
