import torch
from pytorch3d.transforms.rotation_conversions import matrix_to_quaternion, quaternion_to_matrix


def Rt_to_pose_encoding(R, t, pose_encoding_type):
    if pose_encoding_type == "absT_quaR":
        # Convert rotation matrix to quaternion
        quaternion_R = matrix_to_quaternion(R)
        pose_encoding = torch.cat([t, quaternion_R], dim=-1)
    else:
        raise NotImplementedError(f"pose_encoding_type {pose_encoding_type} not implemented")
    return pose_encoding


def pose_encoding_to_Rt(pose_encoding, pose_encoding_type):
    if pose_encoding_type == "absT_quaR":
        t = pose_encoding[..., :3]
        quaternion_R = pose_encoding[..., 3:7]
        R = quaternion_to_matrix(quaternion_R)
    else:
        raise NotImplementedError(f"pose_encoding_type {pose_encoding_type} not implemented")
    return R, t
