import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from dataset import HazePairDataset
from models.cgnet import CGNet
from models.noise_estimation_network import NoiseEstimationNetwork
from utils.image_range import to_bipolar_range, to_image_range
from utils.logging import load_checkpoint, save_checkpoint, save_image
from utils.reverse_sampling import overlapping_reverse_sampling, reverse_sampling


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1.0 - self.mu) * param.data + self.mu * self.shadow[name].data.to(
                    "cuda")

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps,
                            dtype=np.float64) ** 2
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps,
                                  dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def optimizer_to(optimizer, device):
    for param in optimizer.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for sub_param in param.values():
                if isinstance(sub_param, torch.Tensor):
                    sub_param.data = sub_param.data.to(device)
                    if sub_param._grad is not None:
                        sub_param._grad.data = sub_param._grad.data.to(device)


def build_optimizer(parameters, optim_config):
    if optim_config.optimizer == "AdamW":
        optimizer_cls = torch.optim.AdamW
    elif optim_config.optimizer == "Adam":
        optimizer_cls = torch.optim.Adam
    else:
        raise NotImplementedError(f"optimizer {optim_config.optimizer} is not supported")

    return optimizer_cls([{
        "params": parameters,
        "lr": optim_config.lr,
        "betas": tuple(optim_config.betas),
        "weight_decay": optim_config.weight_decay,
        "eps": optim_config.eps,
    }], fused=True)


def fetch(checkpoint, *keys):
    for key in keys:
        if key in checkpoint:
            return checkpoint[key]
    raise KeyError(f"checkpoint does not contain any of {keys}")


class PRDiffDehaze(object):
    def __init__(self, config, rank, world_size):
        super().__init__()

        self.config = config
        self.cgnet_step = 0
        self.noise_step = 0

        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        self.rank = rank
        self.world_size = world_size

        self.cgnet = CGNet().cuda(rank)
        self.cgnet = DDP(self.cgnet, device_ids=[rank], find_unused_parameters=False)
        self.optimizer_cgnet = build_optimizer(self.cgnet.module.parameters(), config.optim.cgnet)
        self.ema_cgnet = EMAHelper()
        self.ema_cgnet.register(self.cgnet)

        self.noise_net = NoiseEstimationNetwork().cuda(rank)
        self.noise_net = DDP(self.noise_net, device_ids=[rank], find_unused_parameters=False)
        self.optimizer_noise = build_optimizer(self.noise_net.module.parameters(),
                                              config.optim.noise)

        self.ema_noise = EMAHelper()
        self.ema_noise.register(self.noise_net)

        self.device = self.noise_net.device

        optimizer_to(self.optimizer_cgnet, rank)
        optimizer_to(self.optimizer_noise, rank)

        for param_group in self.optimizer_cgnet.param_groups:
            for p in param_group["params"]:
                p.data = p.data.to(rank)
                if p.grad is not None:
                    p.grad.data = p.grad.data.to(rank)
        for param_group in self.optimizer_noise.param_groups:
            for p in param_group["params"]:
                p.data = p.data.to(rank)
                if p.grad is not None:
                    p.grad.data = p.grad.data.to(rank)

        start_time = time.time()
        self.train_dataset = HazePairDataset(
            os.path.join(self.config.data.train_data_dir),
            n=self.config.training.patch_n,
            patch_size=self.config.data.image_load_size,
            transforms=self.transforms,
            parse_patches=True,
        )
        self.val_dataset = HazePairDataset(
            os.path.join(self.config.data.test_data_dir),
            n=self.config.training.patch_n,
            patch_size=self.config.data.image_load_size,
            transforms=self.transforms,
            parse_patches=True,
        )
        print(f"dataset loaded in {time.time() - start_time:.6f}s")

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = self.betas.shape[0]

    def load_checkpoint(self, checkpoint_path, model_type="noise", training_mode=True):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

        checkpoint = load_checkpoint(checkpoint_path, device="cpu")

        if model_type == "noise":
            self.noise_step = fetch(checkpoint, "noise_step", "step_ddm")

            if training_mode:
                state_dict = fetch(checkpoint, "noise_net", "model_ddm")
            else:
                ema_state = fetch(checkpoint, "noise_ema", "ema_helper_ddm")
                state_dict = {k.replace("module.", "", 1): v for k, v in ema_state.items()}
            self.noise_net.module.load_state_dict(state_dict, strict=True)

            optimizer_state = fetch(checkpoint, "optimizer_noise", "optimizer_ddm")
            for state in optimizer_state["state"].values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            self.optimizer_noise.load_state_dict(optimizer_state)
            self.ema_noise.load_state_dict(fetch(checkpoint, "noise_ema", "ema_helper_ddm"))

            if self.rank == 0:
                print(f"loaded noise estimation checkpoint {checkpoint_path} "
                      f"(step {self.noise_step})")

        elif model_type == "cgnet":
            self.cgnet_step = fetch(checkpoint, "cgnet_step", "step_trs")

            if training_mode:
                state_dict = fetch(checkpoint, "cgnet", "model_trs")
            else:
                ema_state = fetch(checkpoint, "cgnet_ema", "ema_helper_trs")
                state_dict = {k.replace("module.", "", 1): v for k, v in ema_state.items()}
            self.cgnet.module.load_state_dict(state_dict, strict=True)

            optimizer_state = fetch(checkpoint, "optimizer_cgnet", "optimizer_trs")
            for state in optimizer_state["state"].values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            self.optimizer_cgnet.load_state_dict(optimizer_state)
            self.ema_cgnet.load_state_dict(fetch(checkpoint, "cgnet_ema", "ema_helper_trs"))

            if self.rank == 0:
                print(f"loaded CGNet checkpoint {checkpoint_path} "
                      f"(step {self.cgnet_step})")

        else:
            raise ValueError(f"unknown model type: {model_type}")

    @staticmethod
    def _snapshot_prefix(path):
        return os.path.join(os.path.dirname(path), os.path.basename(path).split("_")[0])

    def save_cgnet_checkpoint(self):
        base_name = self._snapshot_prefix(self.config.training.resume_cgnet)
        save_checkpoint({
            "cgnet_step": self.cgnet_step,
            "cgnet": self.cgnet.module.state_dict(),
            "optimizer_cgnet": self.optimizer_cgnet.state_dict(),
            "cgnet_ema": self.ema_cgnet.state_dict(),
            "config": self.config,
        }, filename=f"{base_name}_{self.cgnet_step}")
        print(f"saved CGNet checkpoint, step={self.cgnet_step}")
        torch.cuda.empty_cache()

    def save_noise_checkpoint(self):
        base_name = self._snapshot_prefix(self.config.training.resume_noise)
        save_checkpoint({
            "noise_step": self.noise_step,
            "noise_net": self.noise_net.module.state_dict(),
            "optimizer_noise": self.optimizer_noise.state_dict(),
            "noise_ema": self.ema_noise.state_dict(),
            "config": self.config,
        }, filename=f"{base_name}_{self.noise_step}")
        print(f"saved noise estimation checkpoint, step={self.noise_step}")
        torch.cuda.empty_cache()

    @staticmethod
    def _iterate(loader, step, total_steps):
        epoch = 0
        while step < total_steps:
            loader.sampler.set_epoch(epoch)
            for batch in loader:
                step += 1
                yield step, batch
                if step >= total_steps:
                    return
            epoch += 1

    def _build_loaders(self):
        sampler = DistributedSampler(self.train_dataset, shuffle=True)
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            sampler=sampler,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
            persistent_workers=False,
            drop_last=False,
        )
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config.sampling.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            persistent_workers=False,
            pin_memory=True,
        )
        return train_loader, val_loader

    def train(self):
        noise_checkpoint = self.config.training.resume_noise + ".pth.tar"
        cgnet_checkpoint = self.config.training.resume_cgnet + ".pth.tar"

        if os.path.isfile(noise_checkpoint):
            self.load_checkpoint(noise_checkpoint, model_type="noise")
        if os.path.isfile(cgnet_checkpoint):
            self.load_checkpoint(cgnet_checkpoint, model_type="cgnet")

        train_loader, val_loader = self._build_loaders()

        print("start training CGNet")
        self.train_cgnet(train_loader, val_loader)
        print("start training noise estimation network")
        self.train_noise(train_loader, val_loader)

    def train_cgnet(self, loader, val_loader):
        scaler = torch.cuda.amp.GradScaler()
        total_steps = self.config.training.cgnet_iters

        for step, (x, _) in self._iterate(loader, self.cgnet_step, total_steps):
            self.cgnet_step = step
            self.cgnet.train()
            x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
            n = x.size(0)
            x = to_bipolar_range(x.to(self.device))
            x.requires_grad_(True)

            with torch.cuda.amp.autocast():
                guidance, dehazed = self.cgnet(x[:, :3, :, :])
                guidance.requires_grad_(True)
                dehazed.requires_grad_(True)
                loss = (F.l1_loss(guidance, x[:, 3:, :, :], reduction="sum")
                        + 0.1 * F.l1_loss(dehazed, x[:, 3:, :, :], reduction="sum")) / n
                if step % 100 == 0 and self.rank == 0:
                    print("cgnet loss == {}, step == {}".format(loss.item(), step))

            if torch.isnan(loss).any():
                print(f"NaN detected at step {step}")
                continue

            self.optimizer_cgnet.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer_cgnet)
            scaler.update()
            self.ema_cgnet.update(self.cgnet)

            if step % self.config.training.validation_freq == 0 and self.rank == 0:
                self.cgnet.eval()
                self.save_validation_grid(val_loader, step, False)

            if (step % self.config.training.snapshot_freq == 0 or step == 1) and self.rank == 0:
                self.save_cgnet_checkpoint()

    def train_noise(self, loader, val_loader):
        scaler = torch.cuda.amp.GradScaler()
        total_steps = self.config.training.noise_iters
        transition_steps = total_steps * 2 // 3

        for step, (x, _) in self._iterate(loader, self.noise_step, total_steps):
            self.noise_step = step
            self.noise_net.train()
            self.cgnet.eval()
            x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
            n = x.size(0)
            x = to_bipolar_range(x.to(self.device))
            x.requires_grad_(True)

            with torch.cuda.amp.autocast():
                guidance, _ = self.cgnet(x[:, :3, :, :])
                t = torch.randint(0, self.num_timesteps, (n,), device=self.device)
                alpha_bar = (1 - self.betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
                noise = torch.randn_like(x[:, 3:, :, :], device=self.device)
                x_t = x[:, 3:, :, :] * alpha_bar.sqrt() + noise * (1.0 - alpha_bar).sqrt()
                predicted_noise = self.noise_net(x_t, t, guidance)

                if step <= transition_steps:
                    lam2 = step / transition_steps
                    lam1 = 1.0 - lam2
                    loss = (lam1 * F.l1_loss(noise, predicted_noise, reduction="sum")
                            + lam2 * F.mse_loss(noise, predicted_noise, reduction="sum")) / n
                else:
                    loss = F.l1_loss(noise, predicted_noise, reduction="sum") / n
                if step % 100 == 0 and self.rank == 0:
                    print("noise loss == {}, step == {}".format(loss.item(), step))

            if torch.isnan(loss).any():
                print(f"NaN detected at step {step}")
                continue

            self.optimizer_noise.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer_noise)
            scaler.update()
            self.ema_noise.update(self.noise_net)

            if step % self.config.training.validation_freq == 0 and self.rank == 0:
                self.noise_net.eval()
                self.cgnet.eval()
                self.save_validation_grid(val_loader, step, True)

            if (step % self.config.training.snapshot_freq == 0 or step == 1) and self.rank == 0:
                self.save_noise_checkpoint()

    def reverse_sample(self, x, x_cond, last=True, patch_locs=None, patch_size=None):
        total_steps = self.config.diffusion.num_diffusion_timesteps
        sample_steps = self.config.sampling.sampling_timesteps

        skip = max(total_steps // sample_steps, 1)
        seq = list(range(0, total_steps, skip))

        sampling_fn = overlapping_reverse_sampling if patch_locs else reverse_sampling

        sampling_args = {
            "x": x,
            "x_cond": x_cond,
            "seq": seq,
            "noise_net": self.noise_net.module,
            "cgnet": self.cgnet.module,
            "betas": self.betas,
            "eta": 0.0,
        }
        if patch_locs is not None:
            sampling_args.update({"corners": patch_locs, "p_size": patch_size})

        with torch.cuda.amp.autocast(), torch.inference_mode():
            xs, _ = sampling_fn(**sampling_args)

        if last:
            result = xs[-1].clone().to(x_cond.device)
            del xs
            return result
        return [t.to(x_cond.device) for t in xs]

    def save_validation_grid(self, val_loader, step, is_noise_net):
        image_folder = os.path.join(self.config.data.val_save_dir,
                                    "noise" if is_noise_net else "cgnet")

        with torch.no_grad(), torch.cuda.amp.autocast():
            print(f"processing a single batch of validation images at step: {step}")

            x = next(iter(val_loader))[0]
            x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
            x = to_bipolar_range(x).to(self.device)
            hazy = x[:, :3, :, :].to(self.device, non_blocking=True)
            guidance, dehazed = self.cgnet(hazy)
            n = hazy.size(0)
            if is_noise_net:
                noise = torch.randn(n, 3, hazy.size(2), hazy.size(3), device=self.device)
                refined = self.reverse_sample(noise, guidance)

        hazy = to_image_range(hazy.detach().cpu())
        dehazed = to_image_range(dehazed.detach().cpu())
        guidance = to_image_range(guidance.detach().cpu())
        ground_truth = to_image_range(x[:, 3:, :, :].detach().cpu())

        if is_noise_net:
            refined = to_image_range(refined.detach().cpu())
            save_batch = torch.cat([hazy, dehazed, guidance, refined, ground_truth], dim=3)
        else:
            save_batch = torch.cat([hazy, dehazed, guidance, ground_truth], dim=3)

        save_image(
            save_batch,
            os.path.join(image_folder, f"batch_{step}.png"),
            nrow=int(n ** 0.5),
            padding=2,
            normalize=False,
        )

        if is_noise_net:
            del hazy, dehazed, guidance, refined
        else:
            del hazy, dehazed, guidance
        torch.cuda.empty_cache()
