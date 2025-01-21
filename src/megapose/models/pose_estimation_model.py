import torch
import torch.nn as nn

from megapose.models.feature_extraction import ViTEncoder
from megapose.models.coarse_point_matching import CoarsePointMatching
from megapose.models.pose_diffusion_model import PoseDiffusionModel
from megapose.models.transformer import GeometricStructureEmbedding


class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg
        self.coarse_npoint = cfg.coarse_npoint
        self.fine_npoint = cfg.fine_npoint

        self.feature_extraction = ViTEncoder(cfg.feature_extraction, self.fine_npoint)
        self.geo_embedding = GeometricStructureEmbedding(cfg.geo_embedding)
        self.coarse_point_matching = CoarsePointMatching(cfg.coarse_point_matching)
        self.pose_diffusion_model = PoseDiffusionModel(cfg.pose_diffusion_model)

    def forward(self, end_points):
        dense_pm, dense_fm, dense_po, dense_fo, img_class_token, tem_class_token, radius = self.feature_extraction(end_points)
        end_points['img_class_token'] = img_class_token
        end_points['tem_class_token'] = tem_class_token

        # pre-compute geometric embeddings for geometric transformer
        bg_point = torch.ones(dense_pm.size(0),1,3).float().to(dense_pm.device) * 100

        sparse_pm, sparse_fm, fps_idx_m = sample_pts_feats(
            dense_pm, dense_fm, self.coarse_npoint, return_index=True
        )
        geo_embedding_m = self.geo_embedding(torch.cat([bg_point, sparse_pm], dim=1))  # [B, 1 + N, 3] -> [B, 1+N, 1+N, 256]

        sparse_po, sparse_fo, fps_idx_o = sample_pts_feats(
            dense_po, dense_fo, self.coarse_npoint, return_index=True
        )
        geo_embedding_o = self.geo_embedding(torch.cat([bg_point, sparse_po], dim=1))  # [B, N + 1, 3] -> [B, 1+N, 1+N, 256]

        # coarse_point_matching
        end_points = self.coarse_point_matching(
            sparse_pm, sparse_fm, geo_embedding_m,
            sparse_po, sparse_fo, geo_embedding_o,
            radius, end_points,
        )

        # [B, M, 3], [B, M, 256], [B, 1 + N, 1 + N, 256], [B, N]
        end_points = self.pose_diffusion_model(
            dense_pm, dense_fm, geo_embedding_m, fps_idx_m,
            dense_po, dense_fo, geo_embedding_o, fps_idx_o,
            radius, end_points
        )

        return end_points
