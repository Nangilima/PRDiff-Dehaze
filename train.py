import os

import numpy as np
import torch
import torch.distributed as dist

from models import PRDiffDehaze
from utils.config import load_config


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"Global Rank: {dist.get_rank()}, Local Rank: {rank}, World Size: {dist.get_world_size()}")


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size):
    setup(rank, world_size)

    config = load_config("config", "config.yml")

    device = torch.device(f"cuda:{rank}")
    print(f"=> using device: {device}")
    config.device = device

    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)
    torch.backends.cudnn.benchmark = True

    print("=> creating PRDiff-Dehaze")
    pipeline = PRDiffDehaze(config, rank, world_size)
    pipeline.train()
    cleanup()


if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"
    torch.multiprocessing.set_start_method("spawn")
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
