from __future__ import annotations

from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.configs.base_config import ViewerConfig

from assistgs.data.assistgs_dataparser import AssistGSDataParserConfig
from assistgs.data.assistgs_datamanger import AssistDataManagerConfig
from assistgs.models.assistgs_model import AssistGSModelConfig
from assistgs.assistgs_trainer import AssistTrainerConfig
from assistgs.assistgs_pipeline import AssistPipelineConfig

MAX_NUM_ITERATIONS = 30000
STEPS_PER_SAVE = 2500
STEPS_PER_EVAL_IMAGE = 200
STEPS_PER_EVAL_ALL_IMAGES = 100
LR_START = 1e-2
LR_FINAL = 1e-5


assistgs = MethodSpecification(
    config=AssistTrainerConfig(
        method_name="assistgs",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=AssistPipelineConfig(
            datamanager=AssistDataManagerConfig(
                dataparser=AssistGSDataParserConfig(
                    load_3D_points=True,
                    points_to_load="dense",
                ),
            ),
            model=AssistGSModelConfig(
                decomposed_rendering=True,
                render_semantics=True
            ),
        ),
        optimizers={
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
            },
            "background_means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "background_features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "background_features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "background_opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "background_scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "background_quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            
            
            "object_template_means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "object_template_features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "object_template_features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "object_template_opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "object_template_scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "object_template_quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
        ),
    description="Neural Scene Graph implementation with NeRFacto model for backgruond and object models.",
)
