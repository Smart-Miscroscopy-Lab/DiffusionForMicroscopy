import os
import torch
import torch.distributed as dist

def setup_distributed():
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank = int(os.getenv("SLURM_PROCID", 0))
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = os.getenv("MASTER_PORT", "29500")
    dist.init_process_group(
        backend="nccl",  # Use NCCL for GPU training
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
    )


