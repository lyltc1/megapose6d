from pathlib import Path
import numpy as np
import cv2

from megapose.config import GSO_DIR

from megapose.datasets.datasets_cfg import make_object_dataset
from megapose.datasets.web_scene_dataset import WebSceneDataset, IterableWebSceneDataset
from megapose.visualization.bokeh_plotter import BokehPlotter

from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.panda3d_renderer.types import Panda3dLightData
from megapose.utils.conversion import convert_wds_scene_observation_to_panda3d

scene_ds = WebSceneDataset(GSO_DIR, label_format="gso_")
iterator = iter(IterableWebSceneDataset(scene_ds))

sample = next(iterator)
# Convert the RGB image to a format suitable for OpenCV
rgb_image = np.array(sample.rgb)

obj_ds_name = 'gso.normalized'
object_ds = make_object_dataset(obj_ds_name)

renderer = Panda3dSceneRenderer(object_ds)

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


# sample.rgb array (540, 720, 3) uint8
# renderings.rgb array (540, 720, 3) unit8
# renderings.normals (540, 720, 3) uint8
# renderings.depth (540, 720, 1) float32 depth in meters
# sample.depth (540, 720) float32 depth in meters

renderings = render_scene(sample)

# Convert the rendered RGB image to a format suitable for OpenCV
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
