"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 4

SETUP_RETRY_COUNT = 3

def print_distributed_environment():
    """
    Print key environment variables related to distributed setup to confirm torchrun is working properly.
    """
    rank = os.environ.get('RANK', 'Not Set')
    world_size = os.environ.get('WORLD_SIZE', 'Not Set')
    local_rank = os.environ.get('LOCAL_RANK', 'Not Set')
    master_addr = os.environ.get('MASTER_ADDR', 'Not Set')
    master_port = os.environ.get('MASTER_PORT', 'Not Set')

    print(f"Distributed Environment Variables:")
    print(f"RANK: {rank}")
    print(f"WORLD_SIZE: {world_size}")
    print(f"LOCAL_RANK: {local_rank}")
    print(f"MASTER_ADDR: {master_addr}")
    print(f"MASTER_PORT: {master_port}")


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        print(f"Process {os.environ['RANK']} using MASTER_ADDR={os.environ['MASTER_ADDR']} and MASTER_PORT={os.environ['MASTER_PORT']}")
        return

    comm = MPI.COMM_WORLD
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda:{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
    data = MPI.COMM_WORLD.bcast(data)
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)

def _find_free_port():
    for _ in range(10):  # Try up to 10 different ports
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("", 0))  # Bind to a free port provided by the OS
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port = s.getsockname()[1]
            s.close()
            return port
        except Exception as e:
            print(f"Error finding port: {e}")
    raise RuntimeError("Unable to find a free port")
