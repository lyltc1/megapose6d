import torch
import torch.nn as nn
import torch.nn.functional as F
from .gaussian_diffuser import GaussianDiffusion
from .denoiser import Denoiser
from .transformer import SparseToDenseTransformer
from .camera_transform import Rt_to_pose_encoding, pose_encoding_to_Rt
from .model_utils import compute_feature_similarity
from .loss_utils import compute_correspondence_loss
from .pointnet2.pointnet2_utils import QueryAndGroup
from .pointnet2.pytorch_utils import SharedMLP, Conv1d


class PoseDiffusionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.pose_encoding_type = cfg.pose_encoding_type
        self.in_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.out_dim)
        self.bg_token = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim) * .02)
        self.PE = PositionalEncoding(cfg.hidden_dim, r1=cfg.pe_radius1, r2=cfg.pe_radius2)
        self.diffuser = GaussianDiffusion(cfg.gaussian_diffuser)
        denoiser = Denoiser(cfg.denoiser)
        self.diffuser.model = denoiser

        self.target_dim = denoiser.target_dim

    def forward(
        self, p1, f1, geo1, fps_idx1, p2, f2, geo2, fps_idx2, radius, end_points, cond_start_step=0):
        B = p1.size(0)
        init_R = end_points['init_R']
        init_t = end_points['init_t']
        # p1_ = (p1 - init_t.unsqueeze(1)) @ init_R
        # p1_ = p1
        # p1_embeding = self.PE(p1_)
        # f1 = self.in_proj(f1) + p1_embeding
        # f1 = torch.cat([self.bg_token.repeat(B,1,1), f1], dim=1) # adding bg
        # p2_embeding = self.PE(p2)
        # f2 = self.in_proj(f2) + p2_embeding
        # f2 = torch.cat([self.bg_token.repeat(B,1,1), f2], dim=1) # adding bg
        # pose = Rt_to_pose_encoding(init_R, init_t,pose_encoding_type=self.pose_encoding_type)
        
        img_class_token = end_points['img_class_token'].reshape(B, 1, 768)
        tem_class_token = end_points['tem_class_token'].reshape(B, -1, 768)
        class_token = torch.cat([img_class_token, tem_class_token], dim=1)

        if self.training:
            gt_R = end_points['rotation_label']
            gt_t = end_points['translation_label'] / (radius.reshape(-1, 1)+1e-6) / 10.0
            # gt_delta_R = gt_R @ init_R.transpose(1, 2)
            # gt_delta_t = - init_t + gt_t

            # end_points = compute_correspondence_loss(
            #     end_points, atten_list, p1, p2, gt_R, gt_t,
            #     dis_thres=self.cfg.loss_dis_thres,
            #     loss_str='fine'
            # )
            # corr = atten_list[-1][:, 1:, :]  # [B, N + 1, N + 1]
            # max_corr_idx = torch.argmax(corr, dim=2)  # [B, N]

            # bg_point = torch.ones(B,1,3).float().to(p1.device) * 100
            # p2 = torch.cat([bg_point, p2], dim=1)
            # p2_max_corr = torch.gather(p2, 1, max_corr_idx.unsqueeze(2).expand(-1, -1, p2.size(2)))  # [B, N, 3]

            # bg_embeding = torch.zeros(B, 1, 256).float().to(p1.device)
            # p2_embeding = torch.cat([bg_embeding, p2_embeding], dim=1)
            # p2_embeding_max_corr = torch.gather(p2_embeding, 1, max_corr_idx.unsqueeze(2).expand(-1, -1, p2_embeding.size(2)))  # [B, N, 256]

            # f2_max_corr = torch.gather(f2, 1, max_corr_idx.unsqueeze(2).expand(-1, -1, f2.size(2)))  # [B, N, hidden_dim]

            # f1 = f1[:, 1:, :]
            # f2 = f2[:, 1:, :]

            gt_encoding = Rt_to_pose_encoding(gt_R, gt_t, pose_encoding_type=self.pose_encoding_type)
            model_out = self.diffuser(gt_encoding, class_token)
            #model_out = self.diffuser(gt_encoding, z=torch.concat([p1, p1_embeding, f1, p2, p2_embeding, f2], dim=2))
            end_points['diffusion_loss'] = model_out['loss']
            est_R, est_t = pose_encoding_to_Rt(model_out['x_0_pred'], pose_encoding_type=self.pose_encoding_type)
            end_points['pred_R'] = est_R
            end_points['pred_t'] = est_t * (radius.reshape(-1, 1)+1e-6) * 10.0
            end_points['init_t'] *= (radius.reshape(-1, 1)+1e-6)
            return end_points
        else:
            # corr = atten_list[-1][:, 1:, :]  # [B, N + 1, N + 1]
            # max_corr_idx = torch.argmax(corr, dim=2)  # [B, N]

            # bg_point = torch.ones(B,1,3).float().to(p1.device) * 100
            # p2 = torch.cat([bg_point, p2], dim=1)
            # p2_max_corr = torch.gather(p2, 1, max_corr_idx.unsqueeze(2).expand(-1, -1, p2.size(2)))  # [B, N, 3]

            # bg_embeding = torch.zeros(B, 1, 256).float().to(p1.device)
            # p2_embeding = torch.cat([bg_embeding, p2_embeding], dim=1)
            # p2_embeding_max_corr = torch.gather(p2_embeding, 1, max_corr_idx.unsqueeze(2).expand(-1, -1, p2_embeding.size(2)))  # [B, N, 256]

            # f2_max_corr = torch.gather(f2, 1, max_corr_idx.unsqueeze(2).expand(-1, -1, f2.size(2)))  # [B, N, hidden_dim]

            # f1 = f1[:, 1:, :]
            # f2 = f2[:, 1:, :]

            target_shape = [B, self.target_dim]

            # sampling
            (pred_pose, pose_encoding_diffusion_samples) = self.diffuser.sample(
                #target_shape, torch.concat([p1, p1_embeding, f1, p2, p2_embeding, f2], dim=2), cond_start_step=cond_start_step
                target_shape, class_token, cond_start_step=cond_start_step
            )

            pred_R, pred_t = pose_encoding_to_Rt(pred_pose, pose_encoding_type=self.pose_encoding_type)
            step_pred_R, step_pred_t = pose_encoding_to_Rt(pose_encoding_diffusion_samples, pose_encoding_type=self.pose_encoding_type)
            end_points['pred_R'] = pred_R
            end_points['pred_t'] = pred_t * (radius.reshape(-1, 1)+1e-6) * 10.0
            end_points['pred_pose_score'] = torch.ones(B).to(p1.device)
            end_points['step_pred_R'] = step_pred_R
            end_points['step_pred_t'] = step_pred_t * (radius.reshape(-1, 1, 1)+1e-6) * 10.0
            end_points['init_t'] *= (radius.reshape(-1, 1)+1e-6)
            return end_points


class PositionalEncoding(nn.Module):
    def __init__(self, out_dim, r1=0.1, r2=0.2, nsample1=32, nsample2=64, use_xyz=True, bn=True):
        super(PositionalEncoding, self).__init__()
        self.group1 = QueryAndGroup(r1, nsample1, use_xyz=use_xyz)
        self.group2 = QueryAndGroup(r2, nsample2, use_xyz=use_xyz)
        input_dim = 6 if use_xyz else 3

        self.mlp1 = SharedMLP([input_dim, 32, 64, 128], bn=bn)
        self.mlp2 = SharedMLP([input_dim, 32, 64, 128], bn=bn)
        self.mlp3 = Conv1d(256, out_dim, 1, activation=None, bn=None)

    def forward(self, pts1, pts2=None):
        if pts2 is None:
            pts2 = pts1

        # scale1
        feat1 = self.group1(
                pts1.contiguous(), pts2.contiguous(), pts1.transpose(1,2).contiguous()
            )
        feat1 = self.mlp1(feat1)
        feat1 = F.max_pool2d(
            feat1, kernel_size=[1, feat1.size(3)]
        )

        # scale2
        feat2 = self.group2(
                pts1.contiguous(), pts2.contiguous(), pts1.transpose(1,2).contiguous()
            )
        feat2 = self.mlp2(feat2)
        feat2 = F.max_pool2d(
            feat2, kernel_size=[1, feat2.size(3)]
        )

        feat = torch.cat([feat1, feat2], dim=1).squeeze(-1)
        feat = self.mlp3(feat).transpose(1,2)
        return feat

