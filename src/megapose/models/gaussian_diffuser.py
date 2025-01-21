from math import pi
from collections import namedtuple
import torch
from torch import nn
import torch.nn.functional as F
import argparse
# constants
import gorilla
import os
import os.path as osp
ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])

# helpers functions
def get_parser():
    parser = argparse.ArgumentParser(
        description="Pose Estimation Debug Testing")
    # pem
    parser.add_argument("--gpus",
                        type=str,
                        default="0")
    parser.add_argument("--model",
                        type=str,
                        default="pose_estimation_model",
                        help="path to model file")
    parser.add_argument("--config",
                        type=str,
                        default="config/gsomini.yaml",
                        help="path to config file")
    parser.add_argument("--exp_id",
                        type=int,
                        default=0,
                        help="experiment id")
    parser.add_argument("--iter",
                        type=int,
                        default=80000,
                        help="epoch num. for testing")
    args_cfg = parser.parse_args()

    return args_cfg

def init():
    args = get_parser()
    exp_name = args.model + '_' + \
        osp.splitext(args.config.split("/")[-1])[0] + '_id' + str(args.exp_id)
    log_dir = osp.join("log", exp_name)

    cfg = gorilla.Config.fromfile(args.config)
    cfg.exp_name = exp_name
    cfg.gpus     = args.gpus
    cfg.model_name = args.model
    cfg.log_dir  = log_dir
    cfg.test_iter = args.iter
    cfg.output_dir = osp.join(cfg.log_dir, 'visualize')
    if not os.path.isdir(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    gorilla.utils.set_cuda_visible_devices(gpu_ids = cfg.gpus)

    return  cfg

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(self, cfg):
        super(GaussianDiffusion, self).__init__()
        self.cfg = cfg
        self.objective = cfg.objective
        assert self.objective in {"pred_noise", "pred_x0"}, "objective must be either pred_noise (predict noise) or pred_x0 (predict image start)"
        self.timesteps = cfg.timesteps
        self.sampling_timesteps = cfg.sampling_timesteps
        self.beta_1 = cfg.beta_1
        self.beta_T = cfg.beta_T
        self.loss_type = cfg.loss_type
        self.beta_schedule = cfg.beta_schedule
        self.p2_loss_weight_gamma = cfg.p2_loss_weight_gamma
        self.p2_loss_weight_k = cfg.p2_loss_weight_k

        self.init_diff_hyper(
            self.timesteps,
            self.sampling_timesteps,
            self.beta_1,
            self.beta_T,
            self.loss_type,
            self.beta_schedule,
            self.p2_loss_weight_gamma,
            self.p2_loss_weight_k,
        )

    def init_diff_hyper(
        self,
        timesteps,
        sampling_timesteps,
        beta_1,
        beta_T,
        loss_type,
        beta_schedule,
        p2_loss_weight_gamma,
        p2_loss_weight_k,
    ):
        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "custom":
            betas = torch.linspace(beta_1, beta_T, timesteps, dtype=torch.float64)
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters
        self.sampling_timesteps = timesteps if sampling_timesteps <= 0 else sampling_timesteps

        assert self.sampling_timesteps <= timesteps

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0
        # at the beginning of the diffusion chain
        register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer(
            "posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        # calculate p2 reweighting
        register_buffer(
            "p2_loss_weight", (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma
        )

    # helper functions
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return (posterior_mean, posterior_variance, posterior_log_variance_clipped)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def model_predictions(self, x, t, z, x_self_cond=None):
        model_output = self.model(x, t, z)

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, model_output)

        elif self.objective == "pred_x0":
            pred_noise = self.predict_noise_from_start(x, t, model_output)
            x_start = model_output

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(
        self, x: torch.Tensor, t: int, z: torch.Tensor, x_self_cond=None, clip_denoised=False  # B x N_x x dim
    ):
        preds = self.model_predictions(x, t, z)

        x_start = preds.pred_x_start

        if clip_denoised:
            raise NotImplementedError(
                "We don't clip the output because \
                    pose does not have a clear bound."
            )

        (model_mean, posterior_variance, posterior_log_variance) = self.q_posterior(x_start=x_start, x_t=x, t=t)

        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,  # B x N_x x dim
        t: int,
        z: torch.Tensor,
        x_self_cond=None,
        clip_denoised=False,
        cond_fn=None,
        cond_start_step=0,
    ):
        ################################################################################
        # from util.utils import seed_all_random_engines
        # seed_all_random_engines(0)
        ################################################################################

        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, z=z, x_self_cond=x_self_cond, clip_denoised=clip_denoised
        )

        if cond_fn is not None and t < cond_start_step:
            # print(model_mean[...,3:7].norm(dim=-1, keepdim=True))
            # tmp = model_mean.clone()
            model_mean = cond_fn(model_mean, t)
            # diff_norm = torch.norm(tmp-model_mean)
            # print(f"the diff norm is {diff_norm}")
            noise = 0.0
        else:
            noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0

        pred = model_mean + (0.5 * model_log_variance).exp() * noise

        return pred, x_start

    
    @torch.no_grad()
    def p_sample_loop(self, shape, z: torch.Tensor, cond_fn=None, cond_start_step=0):
        batch, device = shape[0], self.betas.device

        # Init here
        pose = torch.randn(shape, device=device)

        x_start = None

        pose_process = []
        pose_process.append(pose.unsqueeze(1))
        
        for t in reversed(range(0, self.num_timesteps)):
                pose, _ = self.p_sample(x=pose, t=t, z=z, cond_fn=cond_fn, cond_start_step=cond_start_step)
                pose_process.append(pose.unsqueeze(1))
        return pose, torch.concat(pose_process, dim=1)

    @torch.no_grad()
    def sample(self, shape, z, cond_fn=None, cond_start_step=0):
        # TODO: add more variants
        sample_fn = self.p_sample_loop
        return sample_fn(shape, z=z, cond_fn=cond_fn, cond_start_step=cond_start_step)

    def p_losses(self, x_start, t, z=None, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x, t, z)

        if self.objective == "pred_noise":
            target = noise
            x_0_pred = self.predict_start_from_noise(x, t, model_out)
        elif self.objective == "pred_x0":
            target = x_start
            x_0_pred = model_out
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = self.loss_fn(model_out, target, reduction="none")
        # loss = reduce(loss, "b ... -> b (...)", "mean")
        # loss = loss * extract(self.p2_loss_weight, t, loss.shape)

        return {"loss": loss, "noise": noise, "x_0_pred": x_0_pred, "x_t": x, "t": t}

    def forward(self, pose, z=None, *args, **kwargs):
        b = len(pose)
        t = torch.randint(0, self.num_timesteps, (b,), device=pose.device).long()
        return self.p_losses(pose, t, z=z, *args, **kwargs)

    @property
    def loss_fn(self):
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")
