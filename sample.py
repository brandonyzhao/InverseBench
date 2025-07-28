import os
from omegaconf import OmegaConf
import copy
import pickle
import hydra
from hydra.utils import instantiate

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator

from utils.helper import create_logger, count_parameters, update_ema, unwrap_model


@hydra.main(version_base="1.3", config_path="configs/pretrain", config_name="navier-stokes")
def main(config):
    if config.train.tf32:
        torch.set_float32_matmul_precision("high")
    wandb_log = "wandb" if config.log.wandb else None
    accelerator = Accelerator(log_with=wandb_log)
    if config.log.wandb:
        wandb_init_kwargs = {"project": config.log.project, "group": config.log.group}
        accelerator.init_trackers(
            config.log.project,
            config=OmegaConf.to_container(config),
            init_kwargs=wandb_init_kwargs,
        )

    exp_dir = os.path.join(config.log.exp_dir, config.log.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    ckpt_dir = os.path.join(exp_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = create_logger(exp_dir, main_process=accelerator.is_main_process)
    logger.info(f"Experiment dir created at {exp_dir}")

    # dataset

    dataset = instantiate(config.data)

    batch_size = config.train.batch_size // accelerator.num_processes
    assert (
        batch_size * accelerator.num_processes == config.train.batch_size
    ), "Batch size must be divisible by num processes"

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    logger.info(f"Dataset loaded with {len(dataset)} samples")
    # construct loss function
    loss_fn = instantiate(config.loss)

    # build model
    net = instantiate(config.model)

    logger.info(f"Number of parameters: {count_parameters(net)}")

    ema_net = copy.deepcopy(net).eval().requires_grad_(False).to(accelerator.device)

    # optimizer
    warmup_steps = config.train.warmup_steps
    optimizer = torch.optim.Adam(net.parameters(), lr=config.train.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(step, warmup_steps) / warmup_steps
    )

    # load checkpoints
    if config.train.resume != 'None':
        checkpoint = torch.load(config.train.resume)
        net.load_state_dict(checkpoint["net"])
        ema_net.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        logger.info(f"Resuming from checkpoint {config.train.resume}")
        start_steps = int(os.path.basename(config.train.resume).split(".")[0].split("_")[-1])