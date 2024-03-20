from typing import Dict, Literal
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import cv2
import numpy.typing as npt

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset

class AssistDataset(InputDataset):
    """Dataset that returns nerual scene graph needed: images, pose, render_pose,
    visible_objects, render_objects, objects_meta, hwf, bbox/object_mask.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.

    Returns:
        imgs: [n_frames, h, w, 3]
        instance_segm: [n_frames, h, w]
        poses: [n_frames, 4, 4]
        frame_id: [n_frames]: [frame, cam, 0]
        render_poses: [n_test_frames, 4, 4]
        hwf: [H, W, focal]
        i_split: [[train_split], [validation_split], [test_split]]
        visible_objects: [n_frames, n_obj, 23]
        object_meta: dictionary with metadata for each object with track_id as key
        render_objects: [n_test_frames, n_obj, 23]
        bboxes: 2D bounding boxes in the images stored for each of n_frames
    """

    def __init__(
        self,
        dataparser_outputs: DataparserOutputs,
        scale_factor: float = 1.0,
    ):
        super().__init__(dataparser_outputs, scale_factor)
        assert (
            ("instance_filenames" in dataparser_outputs.metadata.keys()
            and dataparser_outputs.metadata["instance_filenames"] is not None)
        )

        self.instance_filenames = self.metadata["instance_filenames"]
    
    def get_data(self, image_idx: int, image_type: Literal["uint8", "float32"] = "float32") -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        if image_type == "float32":
            image = self.get_image_float32(image_idx)
        elif image_type == "uint8":
            image = self.get_image_uint8(image_idx)
        else:
            raise NotImplementedError(f"image_type (={image_type}) getter was not implemented, use uint8 or float32")

        data = {"image_idx": image_idx}
        data["image"] = image
        metadata = self.get_metadata(data)
        data.update(metadata)

        return data

    def get_metadata(self, data: Dict) -> Dict:
        return_dict = {}
        instance_filepath = self.instance_filenames[data["image_idx"]]
        instance_height = int(self._dataparser_outputs.cameras.height[data["image_idx"]])
        instance_width = int(self._dataparser_outputs.cameras.width[data["image_idx"]])
        instance_image = self.get_instance_image_from_path(
            filepath=instance_filepath, height=instance_height, width=instance_width, scale_factor=self.scale_factor
        )
        return_dict["instance_image"] = instance_image
        return return_dict
    
    def get_instance_image_from_path(
        self,
        filepath: Path,
        height: int,
        width: int,
        scale_factor: float = 1.0,
        interpolation: int = cv2.INTER_NEAREST,
    ) -> torch.Tensor:
        """Loads, rescales and resizes depth images.
        Filepath points to a 16-bit or 32-bit depth image, or a numpy array `*.npy`.

        Args:
            filepath: Path to depth image.
            height: Target depth image height.
            width: Target depth image width.
            scale_factor: Factor by which to scale depth image.
            interpolation: Depth value interpolation for resizing.

        Returns:
            Depth image torch tensor with shape [height, width, 1].
        """
        if filepath.suffix == ".npy":
            image = np.load(filepath) * scale_factor
            image = cv2.resize(image, (width, height), interpolation=interpolation)
        else:
            image = np.array(Image.open(str(filepath.absolute())))
            image = image.astype(np.float32) * scale_factor
            image = cv2.resize(image, (width, height), interpolation=interpolation)
        return torch.from_numpy(image[:, :, np.newaxis])
    