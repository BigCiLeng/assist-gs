# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union
from jaxtyping import Float
from torch import Tensor
from nerfstudio.data.scene_box import OrientedBox

import torch
from torch.nn import Parameter
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.cameras import Cameras
from gsplat._torch_impl import quat_to_rotmat
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from gsplat.sh import num_sh_bases, spherical_harmonics
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.base_model import Model, ModelConfig
import math
import numpy as np

from pytorch_msssim import SSIM
from typing_extensions import Literal
# need following import for background color override
from nerfstudio.model_components import renderers
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.rich_utils import CONSOLE
# from nerfstudio.models.splatfacto import (
#     SplatfactoModel,
#     SplatfactoModelConfig,
#     projection_matrix,
# )
from assistgs.models.splatfacto import (
    SplatfactoModel,
    SplatfactoModelConfig,
    projection_matrix,
)

# need to change to a better implemention
SEMANTIC_COLOR_LIST = [
    [0, 0, 0],  # 黑色
    [255, 0, 0],  # 红色
    [0, 255, 0],  # 绿色
    [0, 0, 255],  # 蓝色
    [255, 255, 0],  # 黄色
    [255, 0, 255],  # 品红
    [0, 255, 255],  # 青色
    [192, 192, 192],  # 银色
]
SEMANTIC_COLOR_LIST = np.array(SEMANTIC_COLOR_LIST, dtype=np.uint8)
SEMANTIC_COLOR_LIST = torch.tensor(SEMANTIC_COLOR_LIST).float().to("cuda:0")


@dataclass
class AssistGSModelConfig(ModelConfig):
    """Neural Scene Graph Model Config"""

    _target: Type = field(default_factory=lambda: AssistGSModel)
    background_model: ModelConfig = SplatfactoModelConfig()
    object_model_template: ModelConfig = SplatfactoModelConfig()

    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 3000
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "random"
    """Whether to randomize the background color."""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians"""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling gaussians"""
    continue_cull_post_densification: bool = True
    """If True, continue to cull gaussians post refinement"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0002
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.0
    "Size of the cube to initialize random gaussians within"
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_split_at: int = 15000
    """stop splitting at this step"""
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    use_scale_regularization: bool = False
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 10.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    output_depth_during_training: bool = False
    """If True, output depth during training. Otherwise, only output depth during evaluation."""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """
    decomposed_rendering: bool = False
    """whether to render background and objects seperately"""
    render_semantics: bool = False
    """whether to render semantics image"""
    semantics_warmup_length: int = 500
    """period of steps where semantic loss is turned off"""


class AssistGSModel(Model):
    """Scene graph model

    Args:
        config: Scene graph configuration to instantiate model
    """

    config: AssistGSModelConfig
    object_meta: torch.Tensor
    object_list: Dict

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        self.seed_points = seed_points
        super().__init__(*args, **kwargs)

    def populate_modules(self):

        self.object_list = self.kwargs["object_list"]
        self.object_meta = self.kwargs["object_meta"]

        object_models = {}
        background_seed_mask = None
        if self.seed_points is not None:
            background_seed_mask = torch.ones_like(self.seed_points[0][:, 0], dtype=torch.bool)
        self.num_objects = self.object_meta.shape[0]
        
        for object_id in range(1, self.num_objects + 1):
            R = torch.eye(3)
            T = self.object_meta[object_id-1, 0:3]
            S = self.object_meta[object_id-1, 3:6]
            object_bbox = OrientedBox(R, T, S)
            # process seed_pts
            object_seed_pts = None
            if self.seed_points is not None:
                crop_ids = object_bbox.within(self.seed_points[0]).squeeze()
                object_pts = self.seed_points[0][crop_ids]
                object_pts_rgb = self.seed_points[1][crop_ids]
                object_seed_pts = (object_pts, object_pts_rgb)
                background_seed_mask[crop_ids] = False                
            
            object_model_name = f"object_{int(object_id)}"
            object_models[object_model_name] = self.config.object_model_template.setup(
                scene_box=object_bbox,
                num_train_data=self.num_train_data,
                metadata=self.kwargs["metadata"],
                device=self.kwargs["device"],
                grad_scaler=self.kwargs["grad_scaler"],
                seed_points=object_seed_pts,
            )
        self.object_models = torch.nn.ModuleDict(object_models)

        background_seed_pts = None
        if self.seed_points is not None:
            background_pts = self.seed_points[0][background_seed_mask]
            background_pts_rgb = self.seed_points[1][background_seed_mask]
            background_seed_pts = (background_pts, background_pts_rgb)
        self.background_model = self.config.background_model.setup(
            scene_box=self.scene_box,
            num_train_data=self.num_train_data,
            metadata=self.kwargs["metadata"],
            device=self.kwargs["device"],
            grad_scaler=self.kwargs["grad_scaler"],
            seed_points=background_seed_pts,
        )

        self.set_all_gaussians()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        # losses
        self.semantic_loss = torch.nn.MSELoss()

        # scene box
        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)

    @property
    def colors(self):
        colors_all = [self.background_model.colors]
        for object_id in range(1, self.num_objects + 1):
            object_model_name = f"object_{int(object_id)}"
            object_model = self.object_models[object_model_name]
            colors_all.append(object_model.colors)
        colors_all = torch.cat(colors_all, dim=0)
        return colors_all

    @property
    def shs_0(self):
        shs_0_all = [self.background_model.shs_0]
        for object_id in range(1, self.num_objects + 1):
            object_model_name = f"object_{int(object_id)}"
            object_model = self.object_models[object_model_name]
            shs_0_all.append(object_model.shs_0)
        shs_0_all = torch.cat(shs_0_all, dim=0)
        return shs_0_all
    @property
    def shs_rest(self):
        shs_rest_all = [self.background_model.shs_rest]
        for object_id in range(1, self.num_objects + 1):
            object_model_name = f"object_{int(object_id)}"
            object_model = self.object_models[object_model_name]
            shs_rest_all.append(object_model.shs_rest)
        shs_rest_all = torch.cat(shs_rest_all, dim=0)
        return shs_rest_all

    @property
    def num_points(self):
        num_points_all = self.background_model.num_points
        for object_id in range(1, self.num_objects + 1):
            object_model_name = f"object_{int(object_id)}"
            object_model = self.object_models[object_model_name]
            num_points_all += object_model.num_points
        return num_points_all

    @property
    def means(self):
        means_all = [self.background_model.means]
        for object_id in range(1, self.num_objects + 1):
            object_model_name = f"object_{int(object_id)}"
            object_model = self.object_models[object_model_name]
            means_all.append(object_model.means)
        means_all = torch.cat(means_all, dim=0)
        return means_all

    @property
    def scales(self):
        scales_all = [self.background_model.scales]
        for object_id in range(1, self.num_objects + 1):
            object_model_name = f"object_{int(object_id)}"
            object_model = self.object_models[object_model_name]
            scales_all.append(object_model.scales)
        scales_all = torch.cat(scales_all, dim=0)
        return scales_all

    @property
    def quats(self):
        quats_all = [self.background_model.quats]
        for object_id in range(1, self.num_objects + 1):
            object_model_name = f"object_{int(object_id)}"
            object_model = self.object_models[object_model_name]
            quats_all.append(object_model.quats)
        quats_all = torch.cat(quats_all, dim=0)
        return quats_all

    @property
    def features_dc(self):
        features_dc_all = [self.background_model.features_dc]
        for object_id in range(1, self.num_objects + 1):
            object_model_name = f"object_{int(object_id)}"
            object_model = self.object_models[object_model_name]
            features_dc_all.append(object_model.features_dc)
        features_dc_all = torch.cat(features_dc_all, dim=0)
        return features_dc_all

    @property
    def features_rest(self):
        features_rest_all = [self.background_model.features_rest]
        for object_id in range(1, self.num_objects + 1):
            object_model_name = f"object_{int(object_id)}"
            object_model = self.object_models[object_model_name]
            features_rest_all.append(object_model.features_rest)
        features_rest_all = torch.cat(features_rest_all, dim=0)
        return features_rest_all

    @property
    def opacities(self):
        opacities_all = [self.background_model.opacities]
        for object_id in range(1, self.num_objects + 1):
            object_model_name = f"object_{int(object_id)}"
            object_model = self.object_models[object_model_name]
            opacities_all.append(object_model.opacities)
        opacities_all = torch.cat(opacities_all, dim=0)
        return opacities_all

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        # TODO here load different component into model
        self.step = 30000
        
        gauss_params = {}
        for p in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
            gauss_params[p] = dict[f"background_model.{p}"]
        self.background_model.load_state_dict(gauss_params, **kwargs)

        for object_id in range(1, self.num_objects + 1):
            object_model_name = f"object_{int(object_id)}"
            object_model = self.object_models[object_model_name]
            gauss_params = {}
            for p in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                gauss_params[p] = dict[f"{object_model_name}.{p}"]
            object_model.load_state_dict(gauss_params, **kwargs)


    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        # assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        if "exp_avg" in param_state:
            param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
            param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state
        
    def remove_from_all_optim(self, optimizers, deleted_mask):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)
        torch.cuda.empty_cache()
        
    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        if "exp_avg" in param_state:
            repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))
            param_state["exp_avg"] = torch.cat(
                [
                    param_state["exp_avg"],
                    torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
            param_state["exp_avg_sq"] = torch.cat(
                [
                    param_state["exp_avg_sq"],
                    torch.zeros_like(param_state["exp_avg_sq"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param
        
    def dup_in_all_optim(self, optimizers, dup_mask, n):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.dup_in_optim(optimizers.optimizers[group], dup_mask, param, n)

    def after_train(self, step: int):
        assert step == self.step
        # to save some training time, we no longer need to update those stats post refinement
        if self.step >= self.config.stop_split_at:
            return
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            assert self.xys.grad is not None
            grads = self.xys.grad.detach().norm(dim=-1)  # TODO fill in
            # TODO (caojk) we only check the background model, maybe we should also check the object model?
            if self.background_model.xys_grad_norm is None:
                num_background_points = self.background_model.num_points
                self.background_model.xys_grad_norm = grads[0:num_background_points]
                self.background_model.vis_counts = torch.ones_like(self.background_model.xys_grad_norm)
                object_points_total = 0
                for object_id in range(1, self.num_objects + 1):
                    object_model_name = f"object_{int(object_id)}"
                    object_model = self.object_models[object_model_name]
                    num_object_points = object_model.num_points
                    object_model.xys_grad_norm = grads[
                        num_background_points + object_points_total: num_background_points + object_points_total + num_object_points
                    ]
                    object_model.vis_counts = torch.ones_like(object_model.xys_grad_norm)
                    object_points_total = object_points_total + num_object_points
            else:
                assert self.background_model.vis_counts is not None
                num_background_points = self.background_model.num_points
                background_visible_mask = visible_mask[0:num_background_points]
                self.background_model.vis_counts[background_visible_mask] = self.background_model.vis_counts[background_visible_mask] + 1
                self.background_model.xys_grad_norm[background_visible_mask] = grads[
                    0 : num_background_points
                ][background_visible_mask] + self.background_model.xys_grad_norm[background_visible_mask]

                object_points_total = 0
                for object_id in range(1, self.num_objects + 1):
                    object_model_name = f"object_{int(object_id)}"
                    object_model = self.object_models[object_model_name]
                    num_object_points = object_model.num_points
                    object_visible_mask = visible_mask[
                        num_background_points + object_points_total: num_background_points + object_points_total + num_object_points
                    ]
                    object_grads = grads[
                        num_background_points + object_points_total: num_background_points + object_points_total + num_object_points
                    ]
                    object_model.vis_counts[object_visible_mask] = object_model.vis_counts[object_visible_mask] + 1
                    object_model.xys_grad_norm[object_visible_mask] = object_grads[object_visible_mask] + object_model.xys_grad_norm[object_visible_mask]
                    object_points_total = object_points_total + num_object_points

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32, device=self.device)
                newradii = self.radii.detach()[visible_mask]
                self.max_2Dsize[visible_mask] = torch.maximum(
                    self.max_2Dsize[visible_mask], newradii / float(max(self.last_size[0], self.last_size[1]))
                )
            # here we need to update the sub-models as well
            if self.background_model.max_2Dsize is None:
                self.background_model.max_2Dsize = torch.zeros(self.background_model.num_points, dtype=torch.float32, device=self.device)
                for object_id in range(1, self.num_objects + 1):
                    object_model_name = f"object_{int(object_id)}"
                    object_model = self.object_models[object_model_name]
                    object_model.max_2Dsize = torch.zeros(object_model.num_points, dtype=torch.float32, device=self.device)
            
            # update the scene graph model
            newradii = self.radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask], newradii / float(max(self.last_size[0], self.last_size[1]))
            )

            # updat the submodels
            num_background_points = self.background_model.num_points
            background_visible_mask = visible_mask[0:num_background_points]
            background_newradii = self.radii.detach()[0:num_background_points][background_visible_mask]
            self.background_model.max_2Dsize[background_visible_mask] = torch.maximum(
                self.background_model.max_2Dsize[background_visible_mask], background_newradii / float(max(self.last_size[0], self.last_size[1]))
            )

            object_points_total = 0
            for object_id in range(1, self.num_objects + 1):
                object_model_name = f"object_{int(object_id)}"
                object_model = self.object_models[object_model_name]
                num_object_points = object_model.num_points
                object_visible_mask = visible_mask[
                    num_background_points + object_points_total: num_background_points + object_points_total + num_object_points
                ]
                object_newradii = self.radii.detach()[
                    num_background_points + object_points_total: num_background_points + object_points_total + num_object_points
                ][object_visible_mask]
                object_model.max_2Dsize[object_visible_mask] = torch.maximum(
                    object_model.max_2Dsize[object_visible_mask], object_newradii / float(max(self.last_size[0], self.last_size[1]))
                )
                object_points_total = object_points_total + num_object_points


    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, back_color: torch.Tensor):
        assert back_color.shape == (3,)
        self.back_color = back_color

    def refinement_after(self, optimizers: Optimizers, step: int):
        assert step == self.step
        if self.step <= self.config.warmup_length:
            return
        with torch.no_grad():
            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.config.reset_alpha_every * self.config.refine_every
            do_densification = (
                self.step < self.config.stop_split_at
                and self.step % reset_interval > self.num_train_data + self.config.refine_every
            )
            if do_densification:
                # then we densify
                assert (
                    self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                )
                avg_grad_norm = (
                    (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])
                )
                high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()
                splits = (self.scales.exp().max(dim=-1).values > self.config.densify_size_thresh).squeeze()
                if self.step < self.config.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.config.split_screen_size).squeeze()
                splits &= high_grads
                nsamps = self.config.n_split_samples
                split_params = self.split_gaussians(splits, nsamps)

                dups = (self.scales.exp().max(dim=-1).values <= self.config.densify_size_thresh).squeeze()
                dups &= high_grads
                dup_params = self.dup_gaussians(dups)
                for name, param in self.gauss_params.items():
                    self.gauss_params[name] = torch.nn.Parameter(
                        torch.cat([param.detach(), split_params[name], dup_params[name]], dim=0)
                    )

                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros_like(split_params["scales"][:, 0]),
                        torch.zeros_like(dup_params["scales"][:, 0]),
                    ],
                    dim=0,
                )
                
                split_idcs = torch.where(splits)[0]
                self.dup_in_all_optim(optimizers, split_idcs, nsamps)
                
                dup_idcs = torch.where(dups)[0]
                self.dup_in_all_optim(optimizers, dup_idcs, 1)  

                splits_mask = torch.cat(
                    (
                        splits,
                        torch.zeros(
                            nsamps * splits.sum() + dups.sum(),
                            device=self.device,
                            dtype=torch.bool,
                        ),
                    )
                )

                deleted_mask = self.cull_gaussians(splits_mask)
            elif self.step >= self.config.stop_split_at and self.config.continue_cull_post_densification:
                deleted_mask = self.cull_gaussians()
            else:
                # if we donot allow culling post refinement, no more gaussians will be pruned.
                deleted_mask = None
                
            if deleted_mask is not None:
                self.remove_from_all_optim(optimizers, deleted_mask)

            if self.step < self.config.stop_split_at and self.step % reset_interval == self.config.refine_every:
                reset_value = self.config.cull_alpha_thresh * 2.0
                self.background_model.opacities.data = torch.clamp(
                    self.background_model.opacities.data,
                    max=torch.logit(torch.tensor(reset_value, device=self.device)).item(),
                )
                
                for object_id in range(1, self.num_objects + 1):
                    object_model_name = f"object_{int(object_id)}"
                    object_model = self.object_models[object_model_name]
                    object_model.opacities.data = torch.clamp(
                        object_model.opacities.data,
                        max=torch.logit(torch.tensor(reset_value, device=self.device)).item(),
                    )
                    
                # reset the exp of optimizer
                background_optim = optimizers.optimizers["background_opacity"]
                background_param = background_optim.param_groups[0]["params"][0]
                background_param_state = background_optim.state[background_param]
                background_param_state["exp_avg"] = torch.zeros_like(background_param_state["exp_avg"])
                background_param_state["exp_avg_sq"] = torch.zeros_like(background_param_state["exp_avg_sq"])
                for object_id in range(1, self.num_objects + 1):
                    object_optim = optimizers.optimizers[f"object_{int(object_id)}_opacity"]
                    object_param = object_optim.param_groups[0]["params"][0]
                    object_param_state = object_optim.state[object_param]
                    object_param_state["exp_avg"] = torch.zeros_like(object_param_state["exp_avg"])
                    object_param_state["exp_avg_sq"] = torch.zeros_like(object_param_state["exp_avg_sq"])
            # set scene graph model
            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None
            # set background model and object model
            self.background_model.xys_grad_norm = None
            self.background_model.vis_counts = None
            self.background_model.max_2Dsize = None
            for object_id in range(1, self.num_objects + 1):
                object_model_name = f"object_{int(object_id)}"
                object_model = self.object_models[object_model_name]
                object_model.xys_grad_norm = None
                object_model.vis_counts = None
                object_model.max_2Dsize = None

    def cull_gaussians(self, extra_cull_mask: Optional[torch.Tensor] = None):
        """
        This function deletes gaussians with under a certain opacity threshold
        extra_cull_mask: a mask indicates extra gaussians to cull besides existing culling criterion
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (torch.sigmoid(self.opacities) < self.config.cull_alpha_thresh).squeeze()
        below_alpha_count = torch.sum(culls).item()
        toobigs_count = 0
        if extra_cull_mask is not None:
            culls = culls | extra_cull_mask
        if self.step > self.config.refine_every * self.config.reset_alpha_every:
            # cull huge ones
            toobigs = (torch.exp(self.scales).max(dim=-1).values > self.config.cull_scale_thresh).squeeze()
            if self.step < self.config.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                toobigs = toobigs | (self.max_2Dsize > self.config.cull_screen_size).squeeze()
            culls = culls | toobigs
            toobigs_count = torch.sum(toobigs).item()
        for name, param in self.gauss_params.items():
            self.gauss_params[name] = torch.nn.Parameter(param[~culls])

        CONSOLE.log(
            f"Culled {n_bef - self.num_points} gaussians "
            f"({below_alpha_count} below alpha thresh, {toobigs_count} too bigs, {self.num_points} remaining)"
        )

        return culls

    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """
        n_splits = split_mask.sum().item()
        CONSOLE.log(f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quats[split_mask] / self.quats[split_mask].norm(dim=-1, keepdim=True)  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        new_features_dc = self.features_dc[split_mask].repeat(samps, 1)
        new_features_rest = self.features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(samps, 1)
        self.scales[split_mask] = torch.log(torch.exp(self.scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self.quats[split_mask].repeat(samps, 1)
        out = {
            "means": new_means,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "opacities": new_opacities,
            "scales": new_scales,
            "quats": new_quats,
        }
        for name, param in self.gauss_params.items():
            if name not in out:
                out[name] = param[split_mask].repeat(samps, 1)
        return out

    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        CONSOLE.log(f"Duplicating {dup_mask.sum().item()/self.num_points} gaussians: {n_dups}/{self.num_points}")
        new_dups = {}
        for name, param in self.gauss_params.items():
            new_dups[name] = param[dup_mask]
        return new_dups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []

        # TODO (caojk) maybe it is faster to change the callback method into parallel?
        # in a single method instead of a "for" iteration
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                self.step_cb
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                self.background_model.step_cb
            )
        )
        for object_id in range(1, self.num_objects + 1):
            object_model_name = f"object_{int(object_id)}"
            cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                self.object_models[object_model_name].step_cb
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.background_model.after_train,
            )
        )
        for object_id in range(1, self.num_objects + 1):
            object_model_name = f"object_{int(object_id)}"
            cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.object_models[object_model_name].after_train,
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.background_model.refinement_after,
                update_every_num_iters=self.config.refine_every,
                args=[training_callback_attributes.optimizers,'background_'],
            )
        )
        for object_id in range(1, self.num_objects + 1):
            object_model_name = f"object_{int(object_id)}"
            cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.object_models[object_model_name].refinement_after,
                update_every_num_iters=self.config.refine_every,
                args=[training_callback_attributes.optimizers,object_model_name+'_'],
            )
        )
        return cbs

    def step_cb(self, step):
        self.step = step

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        
        param_groups = {}
        for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
            param_groups[f"background_{name}"] = [self.background_model.gauss_params[name]]
        for object_id in range(1, self.num_objects + 1):
            
            object_model_name = f"object_{int(object_id)}"
            object_model = self.object_models[object_model_name]
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                param_groups[f"object_{int(object_id)}_{name}"] = [object_model.gauss_params[name]]
        return param_groups

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        return gps

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max((self.config.num_downscales - self.step // self.config.resolution_schedule), 0)
        else:
            return 1
    
    def set_all_gaussians(self):
        param_groups = {}
        for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
            param_groups[name] = [self.background_model.gauss_params[name]]
        

        for object_id in range(1, self.num_objects + 1):
            
            object_model_name = f"object_{int(object_id)}"
            object_model = self.object_models[object_model_name]
            
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                param_groups[name].append(object_model.gauss_params[name])
        
        for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
            param_groups[name] = torch.cat(param_groups[name], dim=0)
        
        self.gauss_params = param_groups
        
    def _downscale_if_required(self, image):    
        d = self._get_downscale_factor()
        if d > 1:
            newsize = [image.shape[0] // d, image.shape[1] // d]

            # torchvision can be slow to import, so we do it lazily.
            import torchvision.transforms.functional as TF

            return TF.resize(image.permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        return image
    
    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"
        # if self.training:
            # currently relies on the branch vickie/camera-grads
            # self.camera_optimizer.apply_to_camera(camera)
        if self.training:
            if self.config.background_color == "random":
                background = torch.rand(3, device=self.device)
            elif self.config.background_color == "white":
                background = torch.ones(3, device=self.device)
            elif self.config.background_color == "black":
                background = torch.zeros(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        else:
            # logic for setting the background of the scene
            if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
                background = renderers.BACKGROUND_COLOR_OVERRIDE.to(self.device)
            else:
                background = self.background_color.to(self.device)
                
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                rgb = background.repeat(int(camera.height.item()), int(camera.width.item()), 1)
                depth = background.new_ones(*rgb.shape[:2], 1) * 10
                accumulation = background.new_zeros(*rgb.shape[:2], 1)
                return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}
        else:
            crop_ids = None
        camera_downscale = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_downscale)
        # shift the camera to center of scene looking at center
        R = camera.camera_to_worlds[0, :3, :3]  # 3 x 3
        T = camera.camera_to_worlds[0, :3, 3:4]  # 3 x 1
        # flip the z and y axes to align with gsplat conventions
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
        R = R @ R_edit
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        # calculate the FOV of the camera given fx and fy, width and height
        cx = camera.cx.item()
        cy = camera.cy.item()
        fovx = 2 * math.atan(camera.width / (2 * camera.fx))
        fovy = 2 * math.atan(camera.height / (2 * camera.fy))
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        projmat = projection_matrix(0.001, 1000, fovx, fovy, device=self.device)

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)
        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default

        # compose all gaussians here
        self.set_all_gaussians()

        # we consider that there is no need for crop since it is all done in submodels
        self.xys, depths, self.radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
            means_crop,
            torch.exp(scales_crop),
            1,
            quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            viewmat.squeeze()[:3, :],
            projmat.squeeze() @ viewmat.squeeze(),
            camera.fx.item(),
            camera.fy.item(),
            cx,
            cy,
            H,
            W,
            BLOCK_WIDTH,
        )  # type: ignore
        
        # rescale the camera back to original dimensions before returning
        camera.rescale_output_resolution(camera_downscale)
        
        if (self.radii).sum() == 0:
            rgb = background.repeat(H, W, 1)
            depth = background.new_ones(*rgb.shape[:2], 1) * 10
            accumulation = background.new_zeros(*rgb.shape[:2], 1)

            return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}

        # Important to allow xys grads to populate properly
        if self.training:
            self.xys.retain_grad()
        if self.config.sh_degree > 0:
            viewdirs = means_crop.detach() - camera.camera_to_worlds.detach()[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors_crop)
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
        else:
            rgbs = torch.sigmoid(colors_crop[:, 0, :])

        assert (num_tiles_hit > 0).any()  # type: ignore
        
        # apply the compensation of screen space blurring to gaussians
        opacities = None
        if self.config.rasterize_mode == "antialiased":
            opacities = torch.sigmoid(opacities_crop) * comp[:, None]
        elif self.config.rasterize_mode == "classic":
            opacities = torch.sigmoid(opacities_crop)
        else:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)
        
        
        # we need to propagate the results to the sub models
        num_background_points = self.background_model.num_points
        self.background_model.xys = self.xys[0:num_background_points]
        if self.training:
            self.background_model.xys.retain_grad()
        self.background_model.radii = self.radii[0:num_background_points]
        self.background_model.last_size = self.last_size
        object_points_total = 0
        for object_id in range(1, self.num_objects + 1):
            object_model_name = f"object_{int(object_id)}"
            object_model = self.object_models[object_model_name]
            num_object_points = object_model.num_points
            object_model.xys = self.xys[
                num_background_points + object_points_total: num_background_points + object_points_total + num_object_points
            ]
            if self.training:
                object_model.xys.retain_grad()
            object_model.radii = self.radii[
                num_background_points + object_points_total: num_background_points + object_points_total + num_object_points
            ]
            object_model.last_size = self.last_size
            object_points_total = object_points_total + num_object_points
        # aggeragate the xys and radii for populating the grads
        xys = [self.background_model.xys]
        radii = [self.background_model.radii]
        for object_id in range(1, self.num_objects + 1):    
            object_model_name = f"object_{int(object_id)}"
            object_model = self.object_models[object_model_name]
            xys.append(object_model.xys)
            radii.append(object_model.radii)
        self.xys = torch.cat(xys, dim=0)
        self.radii = torch.cat(radii, dim=0)

        outputs = {}

        # render outputs here
        rgb, alpha = rasterize_gaussians(  # type: ignore
            self.xys,
            depths,
            self.radii,
            conics,
            num_tiles_hit,  # type: ignore
            rgbs,
            opacities,
            H,
            W,
            BLOCK_WIDTH,
            background=background,
            return_alpha=True,
        )  # type: ignore
        alpha = alpha[..., None]
        rgb = torch.clamp(rgb, max=1.0)  # type: ignore
        depth_im = None
        if self.config.output_depth_during_training or not self.training:
            depth_im = rasterize_gaussians(  # type: ignore
                self.xys,
                depths,
                self.radii,
                conics,
                num_tiles_hit,  # type: ignore
                depths[:, None].repeat(1, 3),
                opacities,
                H,
                W,
                BLOCK_WIDTH,
                background=torch.zeros(3, device=self.device),
            )[..., 0:1]  # type: ignore
            depth_im = torch.where(alpha > 0, depth_im / alpha, depth_im.detach().max())
        outputs["rgb"] = rgb
        outputs["depth"] = depth_im
        outputs["accumulation"] = alpha
        outputs["background"] = background
        
        # use crop ids to render semantics
        if self.config.render_semantics:
            semantics_label = torch.zeros_like(rgbs[..., 0]).to(self.device)
            num_points_count = self.background_model.num_points
            for object_id in range(1, self.num_objects + 1):
                object_model_name = f"object_{int(object_id)}"
                object_model = self.object_models[object_model_name]
                # TODO need fixed
                semantics_label[num_points_count : num_points_count + object_model.num_points] = object_id
                num_points_count = num_points_count + object_model.num_points
            semantics = rasterize_gaussians(  # type: ignore
                self.xys,
                depths,
                self.radii,
                conics,
                num_tiles_hit,  # type: ignore
                semantics_label[:, None].repeat(1, 3),
                opacities,
                H,
                W,
                BLOCK_WIDTH,
                background=torch.zeros(3, device=self.device),
            )[..., 0:1]  # type: ignore
            semantics = torch.where(alpha > 0, semantics / alpha, semantics.detach().max())
            outputs["semantics"] = semantics

        # render background and objects seperately
        if self.config.decomposed_rendering:
            background_rgb = self.decomposed_rendering(self.background_model, viewmat, projmat, camera, BLOCK_WIDTH)
            outputs["background_rgb"] = background_rgb
            for object_id in range(1, self.num_objects + 1):    
                object_model_name = f"object_{int(object_id)}"
                object_model = self.object_models[object_model_name]
                object_rgb = self.decomposed_rendering(object_model, viewmat, projmat, camera, BLOCK_WIDTH)
                outputs[f"{object_model_name}_rgb"] = object_rgb

        # rescale the camera back to original dimensions
        camera.rescale_output_resolution(camera_downscale)

        return outputs  # type: ignore

    @torch.no_grad()
    def decomposed_rendering(
        self,
        specific_model: SplatfactoModel,
        viewmat: Float[Tensor, "4 4"],
        projmat: Float[Tensor, "4 4"],
        camera: Cameras,
        BLOCK_WIDTH: int,
    ) -> Tensor:
        # get the background color
        if self.training:
            if self.config.background_color == "random":
                background = torch.rand(3, device=self.device)
            elif self.config.background_color == "white":
                background = torch.ones(3, device=self.device)
            elif self.config.background_color == "black":
                background = torch.zeros(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        else:
            if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
                background = renderers.BACKGROUND_COLOR_OVERRIDE.to(self.device)
            else:
                background = self.background_color.to(self.device)
        if specific_model.crop_box is not None and not self.training:
            crop_ids = specific_model.crop_box.within(specific_model.means).squeeze()
            if crop_ids.sum() == 0:
                rgb = background.repeat(int(camera.height.item()), int(camera.width.item()), 1)
                depth = background.new_ones(*rgb.shape[:2], 1) * 10
                accumulation = background.new_zeros(*rgb.shape[:2], 1)
                return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}
        else:
            crop_ids = None
        cx = camera.cx.item()
        cy = camera.cy.item()
        W, H = int(camera.width.item()), int(camera.height.item())
        if crop_ids is not None:
            opacities_crop = specific_model.opacities[crop_ids]
            means_crop = specific_model.means[crop_ids]
            features_dc_crop = specific_model.features_dc[crop_ids]
            features_rest_crop = specific_model.features_rest[crop_ids]
            scales_crop = specific_model.scales[crop_ids]
            quats_crop = specific_model.quats[crop_ids]
        else:
            opacities_crop = specific_model.opacities
            means_crop = specific_model.means
            features_dc_crop = specific_model.features_dc
            features_rest_crop = specific_model.features_rest
            scales_crop = specific_model.scales
            quats_crop = specific_model.quats
        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)
        xys, depths, radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
            means_crop,
            torch.exp(scales_crop),
            1,
            quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            viewmat.squeeze()[:3, :],
            projmat.squeeze() @ viewmat.squeeze(),
            camera.fx.item(),
            camera.fy.item(),
            cx,
            cy,
            H,
            W,
            BLOCK_WIDTH,
        )  # type: ignore

        if (radii).sum() == 0:
            rgb = background.repeat(H, W, 1)
            depth = background.new_ones(*rgb.shape[:2], 1) * 10
            accumulation = background.new_zeros(*rgb.shape[:2], 1)

            return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}

        if self.config.sh_degree > 0:
            viewdirs = means_crop.detach() - camera.camera_to_worlds.detach()[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors_crop)
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
        else:
            rgbs = torch.sigmoid(colors_crop[:, 0, :])

        assert (num_tiles_hit > 0).any()  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        opacities = None
        if self.config.rasterize_mode == "antialiased":
            opacities = torch.sigmoid(opacities_crop) * comp[:, None]
        elif self.config.rasterize_mode == "classic":
            opacities = torch.sigmoid(opacities_crop)
        else:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        rgb, alpha = rasterize_gaussians(  # type: ignore
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,  # type: ignore
            rgbs,
            opacities,
            H,
            W,
            BLOCK_WIDTH,
            background=background,
            return_alpha=True,
        )  # type: ignore
        alpha = alpha[..., None]
        rgb = torch.clamp(rgb, max=1.0)  # type: ignore

        return rgb

    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)
    def get_gt_semantic(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return alpha * image[..., :3] + (1 - alpha) * background
        else:
            return image
        
    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points
        return metrics_dict
        
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]


        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)

        loss_dict = {}
        
        loss_dict["main_loss"] = (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss
        loss_dict["scale_reg"] = scale_reg
        

        # semantic loss
        if self.config.render_semantics and self.step > self.config.semantics_warmup_length:
            assert "instance_image" in batch, "There is no instance image in batch"
            # gt_semantics = batch["instance_image"]
            gt_semantics = self.get_gt_semantic(batch["instance_image"])
            predicted_semantics = outputs["semantics"]
            semantic_loss = self.semantic_loss(gt_semantics, predicted_semantics)
            loss_dict["semantic_loss"] = semantic_loss

        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera.to(self.device))
        return outs  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        d = self._get_downscale_factor()
        if d > 1:
            # torchvision can be slow to import, so we do it lazily.
            import torchvision.transforms.functional as TF

            newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]
            predicted_rgb = TF.resize(outputs["rgb"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        else:
            predicted_rgb = outputs["rgb"]

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb}

        if not self.training:
            images_dict["depth"] = outputs["depth"]
        
        if self.config.decomposed_rendering:
            images_dict["background_rgb"] = outputs["background_rgb"]
            for object_id in range(1, self.num_objects + 1):    
                object_model_name = f"object_{int(object_id)}"
                images_dict[f"{object_model_name}_rgb"] = outputs[f"{object_model_name}_rgb"]

        images_dict["semantics"] = outputs["semantics"]
        if self.config.render_semantics:
            assert "instance_image" in batch, "There is no instance image in batch"
            gt_semantics = batch["instance_image"]
            predicted_semantics = outputs["semantics"]
            gt_semantics_img = torch.zeros([gt_semantics.shape[0], gt_semantics.shape[1], 3]).to(self.device)
            predicted_semantics_img = torch.zeros([predicted_semantics.shape[0], predicted_semantics.shape[1], 3]).to(self.device)
            for object_id in range(1, self.num_objects + 1):
                semantic_color = SEMANTIC_COLOR_LIST[object_id]
                predicted_semantics = torch.round(predicted_semantics)
                gt_semantics_img[(gt_semantics == object_id).squeeze()] = semantic_color
                predicted_semantics_img[(predicted_semantics == object_id).squeeze()] = semantic_color
            combined_semantics = torch.cat([gt_semantics_img, predicted_semantics_img], dim=1)
            images_dict["semantics"] = combined_semantics

        return metrics_dict, images_dict
