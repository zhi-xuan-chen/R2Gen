import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch
import os

def ddp_setup(rank, world_size):
   """
   Args:
       rank: Unique identifier of each process
       world_size: Total number of processes
   """
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12355"
   init_process_group(backend="nccl", rank=rank, world_size=world_size)
   torch.cuda.set_device(rank)


