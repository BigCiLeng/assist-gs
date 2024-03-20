"""
Nerual scene graph kitti datamanager.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import random
from typing import Dict, Tuple, Type

import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig, FullImageDatamanager

from assistgs.data.assistgs_dataset import AssistDataset

@dataclass
class AssistDataManagerConfig(FullImageDatamanagerConfig):
    """A semantic datamanager - required to use with .setup()"""

    _target: Type = field(default_factory=lambda: AssistDataManager)


class AssistDataManager(FullImageDatamanager):  # pylint: disable=abstract-method
    """Data manager implementation for data that also requires processing semantic data.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    def create_train_dataset(self) -> AssistDataset:
        return AssistDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> AssistDataset:
        return AssistDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
        )

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next training batch

        Returns a Camera instead of raybundle"""
        image_idx = self.train_unseen_cameras.pop(random.randint(0, len(self.train_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.train_unseen_cameras) == 0:
            self.train_unseen_cameras = [i for i in range(len(self.train_dataset))]

        data = deepcopy(self.cached_train[image_idx])

        for data_name in data.keys():
            if not isinstance(data[data_name], torch.Tensor):
                continue
            data[data_name] = data[data_name].to(self.device)

        assert len(self.train_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.train_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx
        return camera, data

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle"""
        image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        data = deepcopy(self.cached_eval[image_idx])
        
        for data_name in data.keys():
            if not isinstance(data[data_name], torch.Tensor):
                continue
            data[data_name] = data[data_name].to(self.device)
        
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        return camera, data

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle

        TODO: Make sure this logic is consistent with the vanilladatamanager"""
        image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        data = deepcopy(self.cached_eval[image_idx])

        for data_name in data.keys():
            if not isinstance(data[data_name], torch.Tensor):
                continue
            data[data_name] = data[data_name].to(self.device)

        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        return camera, data
