from pathlib import Path
import numpy as np
import cv2

from megapose.config import WDS_DS_DIR

from megapose.datasets.datasets_cfg import make_object_dataset
from megapose.datasets.web_scene_dataset import WebSceneDataset, IterableWebSceneDataset
from megapose.visualization.bokeh_plotter import BokehPlotter

from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.panda3d_renderer.types import Panda3dLightData
from megapose.utils.conversion import convert_wds_scene_observation_to_panda3d


def render_scene(obs):
    camera_data, object_datas = convert_wds_scene_observation_to_panda3d(
        obs.camera_data, obs.object_datas, scene_ds.id2label
    )
    light_datas = [
        Panda3dLightData(
            light_type='ambient',
            color=((1., 1., 1., 1)),
        ),
    ]
    renderings = renderer.render_scene(
        object_datas,
        [camera_data],
        light_datas,
        render_depth=True,
        render_normals=True,
        render_binary_mask=True,
        copy_arrays=True,
    )
    return renderings[0]


scene_ds = WebSceneDataset(WDS_DS_DIR / "bop_gso", label_format="gso_")
iterator = iter(IterableWebSceneDataset(scene_ds))

sample = next(iterator)
# Convert the RGB image to a format suitable for OpenCV
rgb_image = np.array(sample.rgb)

obj_ds_name = 'gso.normalized'
object_ds = make_object_dataset(obj_ds_name)

RENDER_SCENE = False
if RENDER_SCENE:
    # sample.rgb array (540, 720, 3) uint8
    # renderings.rgb array (540, 720, 3) unit8
    # renderings.normals (540, 720, 3) uint8
    # renderings.depth (540, 720, 1) float32 depth in meters
    # sample.depth (540, 720) float32 depth in meters
    # Convert the rendered RGB image to a format suitable for OpenCV
    renderer = Panda3dSceneRenderer(object_ds)
    renderings = render_scene(sample)

    rendered_rgb_image = np.array(renderings.rgb)

    # Display the original and rendered images using OpenCV
    cv2.imshow("Original RGB Image", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    cv2.imshow("Rendered RGB Image", cv2.cvtColor(rendered_rgb_image, cv2.COLOR_RGB2BGR))

    # Display the rendered depth image
    depth_image = renderings.depth
    depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    depth_image_normalized = depth_image_normalized.astype(np.uint8)
    cv2.imshow("Rendered Depth Image", depth_image_normalized)

    # Display the rendered normals image
    normals_image = renderings.normals
    cv2.imshow("Rendered Normals Image", cv2.cvtColor(normals_image, cv2.COLOR_RGB2BGR))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

from megapose.datasets.pose_dataset import PoseDataset
ds_train = PoseDataset(
        IterableWebSceneDataset(scene_ds),
        apply_rgb_augmentation=False,
        apply_background_augmentation=False,
        apply_depth_augmentation=False,
        min_area=1000.0,
        keep_labels_set=None,
    )

from torch.utils.data import DataLoader

ds_iter_train = DataLoader(
        ds_train,
        batch_size=4,
        num_workers=1,
        collate_fn=ds_train.collate_fn,
        persistent_workers=True,
        pin_memory=False,
    )
iter_train = iter(ds_iter_train)
data = next(iter_train)