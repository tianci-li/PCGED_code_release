


import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class DistributedConfig:

    backend: str = 'nccl'
    strategy: str = 'ddp'
    find_unused_parameters: bool = True
    broadcast_buffers: bool = True
    gradient_as_bucket_view: bool = True


def detect_device_type() -> str:

    if torch.cuda.is_available():

        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            return 'rocm'
        return 'cuda'
    return 'cpu'


def get_backend_for_device(device_type: str) -> str:

    if device_type == 'rocm':
        return 'rccl'
    elif device_type == 'cuda':
        return 'nccl'
    else:
        return 'gloo'


def setup_distributed(
    backend: Optional[str] = None,
    init_method: Optional[str] = None,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
) -> Tuple[int, int, torch.device]:


    if world_size is None:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
    if rank is None:
        rank = int(os.environ.get('RANK', 0))

    local_rank = int(os.environ.get('LOCAL_RANK', 0))


    device_type = detect_device_type()
    if backend is None:
        backend = get_backend_for_device(device_type)


    if device_type in ('cuda', 'rocm'):
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')


    if world_size > 1:
        if init_method is None:
            init_method = 'env://'

        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=world_size,
                rank=rank,
            )

    return rank, world_size, device


def cleanup_distributed():

    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:

    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:

    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_local_rank() -> int:

    return int(os.environ.get('LOCAL_RANK', 0))


def is_main_process() -> bool:

    return get_rank() == 0


def barrier():

    if dist.is_initialized():
        dist.barrier()


def broadcast_object(obj: Any, src: int = 0) -> Any:

    if not dist.is_initialized():
        return obj

    object_list = [obj] if get_rank() == src else [None]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:

    if not dist.is_initialized():
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor


def all_gather_tensors(tensor: torch.Tensor) -> torch.Tensor:

    if not dist.is_initialized():
        return tensor

    world_size = get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


class DistributedManager:


    def __init__(
        self,
        config: Optional[DistributedConfig] = None,
        device_type: str = 'auto',
    ):
        self.config = config or DistributedConfig()


        if device_type == 'auto':
            self.device_type = detect_device_type()
        else:
            self.device_type = device_type


        if self.config.backend == 'nccl' and self.device_type == 'rocm':
            self.config.backend = 'rccl'

        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.device = torch.device('cpu')
        self.is_distributed = False

    def setup(self) -> 'DistributedManager':

        self.rank, self.world_size, self.device = setup_distributed(
            backend=self.config.backend
        )
        self.local_rank = get_local_rank()
        self.is_distributed = self.world_size > 1

        if self.is_main_process:
            print(f"OK Distributed setup complete:")
            print(f"   Backend: {self.config.backend}")
            print(f"   Device type: {self.device_type}")
            print(f"   World size: {self.world_size}")
            print(f"   Device: {self.device}")

        return self

    def cleanup(self):

        cleanup_distributed()

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:

        model = model.to(self.device)

        if not self.is_distributed:
            return model

        if self.config.strategy == 'ddp':
            return DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=self.config.find_unused_parameters,
                broadcast_buffers=self.config.broadcast_buffers,
                gradient_as_bucket_view=self.config.gradient_as_bucket_view,
            )
        else:
            raise ValueError(f"Unsupported strategy: {self.config.strategy}")

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:

        if isinstance(model, DDP):
            return model.module
        return model

    def barrier(self):

        barrier()

    def broadcast(self, obj: Any, src: int = 0) -> Any:

        return broadcast_object(obj, src)

    def all_reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:

        return all_reduce_mean(tensor)

    @contextmanager
    def main_process_first(self):

        if not self.is_main_process:
            self.barrier()

        yield

        if self.is_main_process:
            self.barrier()

    def print(self, *args, **kwargs):

        if self.is_main_process:
            print(*args, **kwargs)

    def save_checkpoint(self, state: Dict, path: str):

        if self.is_main_process:
            torch.save(state, path)
        self.barrier()

    def load_checkpoint(self, path: str, map_location: Optional[str] = None) -> Dict:

        if map_location is None:
            map_location = f'cuda:{self.local_rank}'
        return torch.load(path, map_location=map_location)


def setup_for_distributed_training(config: Dict) -> DistributedManager:

    dist_config = config.get('distributed', {})
    hardware_config = config.get('hardware', {})


    ddp_config = dist_config.get('ddp', {})
    distributed_config = DistributedConfig(
        backend=dist_config.get('backend', 'nccl'),
        strategy=dist_config.get('strategy', 'ddp'),
        find_unused_parameters=ddp_config.get('find_unused_parameters', False),
        broadcast_buffers=ddp_config.get('broadcast_buffers', True),
        gradient_as_bucket_view=ddp_config.get('gradient_as_bucket_view', True),
    )


    device_type = hardware_config.get('device_type', 'auto')


    if device_type in ('auto', 'cuda'):
        cuda_config = hardware_config.get('cuda', {})
        if cuda_config.get('allow_tf32', True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if cuda_config.get('benchmark', True):
            torch.backends.cudnn.benchmark = True
        if cuda_config.get('deterministic', False):
            torch.backends.cudnn.deterministic = True


    manager = DistributedManager(
        config=distributed_config,
        device_type=device_type,
    )

    return manager.setup()


if __name__ == '__main__':

    print("Testing distributed utilities...")

    print(f"Device type: {detect_device_type()}")
    print(f"Backend: {get_backend_for_device(detect_device_type())}")


    manager = DistributedManager()
    manager.setup()

    print(f"Rank: {manager.rank}")
    print(f"World size: {manager.world_size}")
    print(f"Device: {manager.device}")
    print(f"Is main process: {manager.is_main_process}")

    print("\nOK Distributed utilities test passed!")
