"""
Nerfstudio My Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations


from my_nerf.models.vanilla_nerf import VNeRF, VModelConfig
from my_nerf.my_pipeline import (
    MyPipelineConfig,
    DPipelineConfig,
)
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.configs.method_configs import VanillaDataManagerConfig
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

# my_config.py

from my_nerf.models.vanilla_nerf import VNeRF
from my_nerf.models.d_nerf import  MyDModelConfig, MyDNeRF
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from pathlib import Path

MyMethod = MethodSpecification(
    config=TrainerConfig(
        method_name="v-nerf", 
        pipeline=MyPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
            ),
            model=VModelConfig(_target=VNeRF),
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
                "scheduler": None,
            },
        },
    ),
    description="Vanilla NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis",
)


DMethod = MethodSpecification(
    config=TrainerConfig(
        method_name="d-nerf",  # TODO: rename to your own model
        pipeline=DPipelineConfig(
                datamanager=VanillaDataManagerConfig(
                    dataparser=DNeRFDataParserConfig()),
                model=MyDModelConfig(
                    _target=MyDNeRF,
                    enable_temporal_distortion=True,
                    temporal_distortion_params={"kind": TemporalDistortionKind.DNERF},
                ),
            ),
            optimizers={
                "fields": {
                    "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
                    "scheduler": None,
                },
            },
        ),
    description="D-NeRF: A Temporally Coherent Neural Radiance Field for Dynamic Scenes",
)