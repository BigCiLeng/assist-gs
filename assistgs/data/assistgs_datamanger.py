"""
Nerual scene graph kitti datamanager.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import random

import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig, FullImageDatamanager, _undistort_image

import random
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type

from rich.progress import track
from typing_extensions import assert_never

from nerfstudio.utils.rich_utils import CONSOLE

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
    
    def _load_images(
        self, split: Literal["train", "eval"], cache_images_device: Literal["cpu", "gpu"]
    ) -> List[Dict[str, torch.Tensor]]:
        undistorted_images: List[Dict[str, torch.Tensor]] = []

        # Which dataset?
        if split == "train":
            dataset = self.train_dataset
        elif split == "eval":
            dataset = self.eval_dataset
        else:
            assert_never(split)

        def undistort_idx(idx: int) -> Dict[str, torch.Tensor]:
            data = dataset.get_data(idx, image_type=self.config.cache_images_type)
            camera = dataset.cameras[idx].reshape(())
            assert data["image"].shape[1] == camera.width.item() and data["image"].shape[0] == camera.height.item(), (
                f'The size of image ({data["image"].shape[1]}, {data["image"].shape[0]}) loaded '
                f'does not match the camera parameters ({camera.width.item(), camera.height.item()})'
            )
            if camera.distortion_params is None or torch.all(camera.distortion_params == 0):
                return data
            K = camera.get_intrinsics_matrices().numpy()
            distortion_params = camera.distortion_params.numpy()
            image = data["image"].numpy()

            K, image, mask = _undistort_image(camera, distortion_params, data, image, K)
            data["image"] = torch.from_numpy(image)
            if mask is not None:
                data["mask"] = mask

            if "instance_image" in data:
                instance_image = data["instance_image"].numpy()

                _, instance_image, _ = _undistort_image(camera, distortion_params, data, instance_image, K)
                data["instance_image"] = torch.from_numpy(instance_image)

            dataset.cameras.fx[idx] = float(K[0, 0])
            dataset.cameras.fy[idx] = float(K[1, 1])
            dataset.cameras.cx[idx] = float(K[0, 2])
            dataset.cameras.cy[idx] = float(K[1, 2])
            dataset.cameras.width[idx] = image.shape[1]
            dataset.cameras.height[idx] = image.shape[0]
            return data

        CONSOLE.log(f"Caching / undistorting {split} images")
        with ThreadPoolExecutor(max_workers=2) as executor:
            undistorted_images = list(
                track(
                    executor.map(
                        undistort_idx,
                        range(len(dataset)),
                    ),
                    description=f"Caching / undistorting {split} images",
                    transient=True,
                    total=len(dataset),
                )
            )

        # Move to device.
        if cache_images_device == "gpu":
            for cache in undistorted_images:
                cache["image"] = cache["image"].to(self.device)
                if "mask" in cache:
                    cache["mask"] = cache["mask"].to(self.device)
                if "depth" in cache:
                    cache["depth"] = cache["depth"].to(self.device)
                self.train_cameras = self.train_dataset.cameras.to(self.device)
        elif cache_images_device == "cpu":
            for cache in undistorted_images:
                cache["image"] = cache["image"].pin_memory()
                if "mask" in cache:
                    cache["mask"] = cache["mask"].pin_memory()
                self.train_cameras = self.train_dataset.cameras
        else:
            assert_never(cache_images_device)

        return undistorted_images
    