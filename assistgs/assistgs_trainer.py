# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code to train model.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import List, Literal, Type, DefaultDict
from collections import defaultdict
from nerfstudio.engine.trainer import TrainerConfig, Trainer
from nerfstudio.engine.callbacks import TrainingCallback
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from nerfstudio.utils.rich_utils import CONSOLE
from torch.cuda.amp.grad_scaler import GradScaler

TORCH_DEVICE = str


@dataclass
class AssistTrainerConfig(TrainerConfig):
    """Configuration for training regimen"""

    _target: Type = field(default_factory=lambda: AssistTrainer)
    """target class to instantiate"""


class AssistTrainer(Trainer):
    """Trainer class

    Args:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.

    Attributes:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.
        device: The device to run the training on.
        pipeline: The pipeline object.
        optimizers: The optimizers object.
        callbacks: The callbacks object.
        training_state: Current model training state.
    """

    pipeline: VanillaPipeline
    optimizers: Optimizers
    callbacks: List[TrainingCallback]

    def __init__(self, config: AssistTrainerConfig, local_rank: int = 0, world_size: int = 1) -> None:        
        super().__init__(config, local_rank, world_size)
        object_infos_path = self.config.data / "bboxs_aabb.npy"
        object_infos = np.load(object_infos_path.as_posix())
        num_objects = object_infos.shape[0]
        

        for object_id in range(1, num_objects + 1):
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                self.config.optimizers[f"object_{object_id}_{name}"] = self.config.optimizers[f"object_template_{name}"].copy()

