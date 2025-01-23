"""
Copyright (c) 2022 Inria & NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


# Standard Library
from typing import Union

# MegaPose
from megapose.lib3d.rigid_mesh_database import BatchedMeshes

# Pose models
from megapose.models.pose_rigid import PosePredictor
from megapose.panda3d_renderer.panda3d_batch_renderer import Panda3dBatchRenderer
from megapose.training.training_config import TrainingConfig
from megapose.utils.logging import get_logger

logger = get_logger(__name__)


def check_update_config(cfg: TrainingConfig) -> TrainingConfig:
    """Useful for loading models previously trained with different configurations."""
    return cfg


def create_model_pose(
    cfg: TrainingConfig,
    renderer: Panda3dBatchRenderer,
    mesh_db: BatchedMeshes,
) -> PosePredictor:

    model = PosePredictor(
        cfg = cfg,
        renderer=renderer,
        mesh_db=mesh_db,
        n_rendered_views=cfg.n_rendered_views,
        multiview_type=cfg.multiview_type,
        render_normals=cfg.render_normals,
        render_depth=cfg.render_depth,
        input_depth=cfg.input_depth,
        predict_rendered_views_logits=cfg.predict_rendered_views_logits,
        remove_TCO_rendering=cfg.remove_TCO_rendering,
        predict_pose_update=cfg.predict_pose_update,
        depth_normalization_type=cfg.depth_normalization_type,
    )
    return model
