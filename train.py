"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
# os.environ['NCCL_P2P_DISABLE'] = '1'
# os.environ['NCCL_IGNORE_DISABLED_P2P'] = '1'
import time
import math
import pickle
from contextlib import AbstractContextManager, ExitStack, nullcontext
from types import TracebackType
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence, TYPE_CHECKING, TypedDict, TypeVar, cast, runtime_checkable

import numpy as np
import numpy.typing as npt
import torch
import torch._dynamo
import torch.nn as nn
from torch import Generator
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler as TorchGradScaler
from torch.optim.optimizer import Optimizer

from model import GPTConfig, GPT


_M = TypeVar("_M", bound=nn.Module)


@runtime_checkable
class _SuppressErrorsConfig(Protocol):
    suppress_errors: bool


def _configure_dynamo() -> None:
    dynamo_config = getattr(torch._dynamo, "config", None)
    if isinstance(dynamo_config, _SuppressErrorsConfig):
        dynamo_config.suppress_errors = True


_configure_dynamo()

if TYPE_CHECKING:
    def manual_seed(seed: int) -> Generator: ...

    def _from_numpy(array: npt.NDArray[np.int64]) -> torch.Tensor: ...

    def _torch_load(path: str, map_location: str | torch.device) -> Mapping[str, object]: ...

    def _load_pretrained_model(model_type: str, override_args: Mapping[str, object] | None = None) -> GPT: ...

    def _typed_torch_save(obj: object, f: str) -> None: ...
else:
    manual_seed = torch.manual_seed
    _from_numpy = torch.from_numpy
    _torch_load = torch.load
    _typed_torch_save = torch.save

    def _load_pretrained_model(model_type: str, override_args: Mapping[str, object] | None = None) -> GPT:
        return GPT.from_pretrained(model_type, dict(override_args) if override_args is not None else None)


class _AutocastWrapper(AbstractContextManager[None]):
    def __init__(self, device_type: str, dtype: torch.dtype):
        self._device_type = device_type
        self._dtype = dtype
        self._stack = ExitStack()

    def __enter__(self) -> None:
        if self._device_type == 'cuda':
            self._stack.enter_context(autocast(device_type='cuda', dtype=self._dtype))
        elif self._device_type == 'cpu':
            self._stack.enter_context(autocast(device_type='cpu'))
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        return self._stack.__exit__(exc_type, exc, tb)


def _autocast_context(device_type: str, dtype: torch.dtype) -> AbstractContextManager[None]:
    if device_type in {'cuda', 'cpu'}:
        return _AutocastWrapper(device_type, dtype)
    return nullcontext()


def _forward_with_context(
    module: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    ctx: AbstractContextManager[None],
) -> tuple[torch.Tensor, torch.Tensor]:
    with ctx:
        outputs = module(inputs, targets)
    return cast(tuple[torch.Tensor, torch.Tensor], outputs)


def _get_model_config(model: GPT) -> GPTConfig:
    return model.config


def _crop_model_block_size(model: GPT, block_size: int) -> None:
    crop_block_size_method = getattr(model, "crop_block_size")
    crop = cast("Callable[[int], None]", crop_block_size_method)
    crop(block_size)


def _configure_model_optimizers(
    model: GPT,
    weight_decay: float,
    learning_rate: float,
    betas: tuple[float, float],
    device_type: str,
) -> Optimizer:
    configure_method = getattr(model, "configure_optimizers")
    configure = cast(
        "Callable[[float, float, tuple[float, float], str], Optimizer]",
        configure_method,
    )
    return configure(weight_decay, learning_rate, betas, device_type)


def _compile_model(module: _M) -> _M:
    compile_attr = getattr(torch, "compile", None)
    if not callable(compile_attr):
        return module
    compile_fn = cast("Callable[[nn.Module], nn.Module]", compile_attr)
    compiled = compile_fn(module)
    return cast(_M, compiled)


class _DummyGradScaler:
    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def step(self, optimizer: Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None

    def unscale_(self, optimizer: Optimizer) -> None:
        return None


def _scale_loss_tensor(scaler: TorchGradScaler | _DummyGradScaler, loss: torch.Tensor) -> torch.Tensor:
    scale_method = getattr(scaler, "scale")
    scale = cast("Callable[[torch.Tensor], torch.Tensor]", scale_method)
    return scale(loss)


def _tensor_backward(tensor: torch.Tensor) -> None:
    backward_method = getattr(tensor, "backward")
    backward = cast("Callable[[], None]", backward_method)
    backward()


def _save_checkpoint(obj: Mapping[str, object], path: str) -> None:
    obj_arg = cast(object, obj)
    _typed_torch_save(obj_arg, path)


BatchSplit = Literal['train', 'val']
BATCH_SPLITS: tuple[BatchSplit, BatchSplit] = ('train', 'val')
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


@dataclass
class ModelArgs:
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    bias: bool
    vocab_size: int | None
    dropout: float
    n_exp: int
    top_k: int
    use_aux_loss: bool
    use_router_z_loss: bool
    use_noisy_top_k: bool
    aux_loss_weight: float
    router_z_loss_weight: float
    train_capacity: float
    eval_capacity: float
    min_capacity: int
    stride: int
    use_switch_tfm_init: bool
    switch_tfm_init_scale: float
    router_use_full_prec: bool

    def to_config(self) -> GPTConfig:
        if self.vocab_size is None:
            raise ValueError("vocab_size must be set before creating GPTConfig")
        return GPTConfig(
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            block_size=self.block_size,
            bias=self.bias,
            vocab_size=self.vocab_size,
            dropout=self.dropout,
            n_exp=self.n_exp,
            top_k=self.top_k,
            use_aux_loss=self.use_aux_loss,
            use_router_z_loss=self.use_router_z_loss,
            use_noisy_top_k=self.use_noisy_top_k,
            aux_loss_weight=self.aux_loss_weight,
            router_z_loss_weight=self.router_z_loss_weight,
            train_capacity=self.train_capacity,
            eval_capacity=self.eval_capacity,
            min_capacity=self.min_capacity,
            stride=self.stride,
            use_switch_tfm_init=self.use_switch_tfm_init,
            switch_tfm_init_scale=self.switch_tfm_init_scale,
            router_use_full_prec=self.router_use_full_prec,
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "ModelArgs":
        return cls(
            n_layer=cast(int, data['n_layer']),
            n_head=cast(int, data['n_head']),
            n_embd=cast(int, data['n_embd']),
            block_size=cast(int, data['block_size']),
            bias=cast(bool, data['bias']),
            vocab_size=cast(int | None, data.get('vocab_size')),
            dropout=cast(float, data['dropout']),
            n_exp=cast(int, data['n_exp']),
            top_k=cast(int, data['top_k']),
            use_aux_loss=cast(bool, data['use_aux_loss']),
            use_router_z_loss=cast(bool, data['use_router_z_loss']),
            use_noisy_top_k=cast(bool, data['use_noisy_top_k']),
            aux_loss_weight=cast(float, data['aux_loss_weight']),
            router_z_loss_weight=cast(float, data['router_z_loss_weight']),
            train_capacity=cast(float, data['train_capacity']),
            eval_capacity=cast(float, data['eval_capacity']),
            min_capacity=cast(int, data['min_capacity']),
            stride=cast(int, data['stride']),
            use_switch_tfm_init=cast(bool, data['use_switch_tfm_init']),
            switch_tfm_init_scale=cast(float, data['switch_tfm_init_scale']),
            router_use_full_prec=cast(bool, data['router_use_full_prec']),
        )


class TrainingCheckpoint(TypedDict):
    model: dict[str, torch.Tensor]
    optimizer: dict[str, object]
    model_args: Mapping[str, object] | ModelArgs
    iter_num: int
    best_val_loss: float
    config: dict[str, object]


def _ensure_tensor_state(mapping: Mapping[str, object]) -> dict[str, torch.Tensor]:
    tensor_state: dict[str, torch.Tensor] = {}
    for key, value in mapping.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"State dict entry '{key}' is not a tensor: {type(value)!r}")
        tensor_state[key] = value
    return tensor_state


def _load_checkpoint(path: str, map_location: str | torch.device) -> TrainingCheckpoint:
    raw_mapping = _torch_load(path, map_location=map_location)
    required_keys = {"model", "optimizer", "model_args", "iter_num", "best_val_loss", "config"}
    missing = required_keys.difference(raw_mapping.keys())
    if missing:
        missing_list = ", ".join(sorted(str(key) for key in missing))
        raise KeyError(f"Checkpoint missing keys: {missing_list}")
    model_state_obj_raw = raw_mapping["model"]
    if not isinstance(model_state_obj_raw, Mapping):
        raise TypeError("Checkpoint model state must be a mapping")
    model_state_obj = cast(Mapping[str, object], model_state_obj_raw)
    optimizer_state_obj_raw = raw_mapping["optimizer"]
    if not isinstance(optimizer_state_obj_raw, Mapping):
        raise TypeError("Checkpoint optimizer state must be a mapping")
    optimizer_state_obj = cast(Mapping[str, object], optimizer_state_obj_raw)
    config_obj_raw = raw_mapping["config"]
    if not isinstance(config_obj_raw, Mapping):
        raise TypeError("Checkpoint config must be a mapping")
    config_obj = cast(Mapping[str, object], config_obj_raw)
    iter_value = raw_mapping["iter_num"]
    if not isinstance(iter_value, int):
        raise TypeError("Checkpoint iter_num must be an int")
    best_val_loss_value = raw_mapping["best_val_loss"]
    if not isinstance(best_val_loss_value, (float, int)):
        raise TypeError("Checkpoint best_val_loss must be a float")
    optimizer_state: dict[str, object] = dict(optimizer_state_obj.items())
    config_state: dict[str, object] = dict(config_obj.items())
    checkpoint: TrainingCheckpoint = {
    "model": _ensure_tensor_state(model_state_obj),
        "optimizer": optimizer_state,
        "model_args": cast(Mapping[str, object] | ModelArgs, raw_mapping["model_args"]),
        "iter_num": iter_value,
        "best_val_loss": float(best_val_loss_value),
        "config": config_state,
    }
    return checkpoint
from manager import MANAGER

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from: str = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

# wandb logging
wandb_log = True # False # disabled by default
wandb_project = 'nano-moe'
wandb_run_name = 'gpt2-124M-owt' + str(time.time())
wandb_mode = os.environ.get('WANDB_MODE', None)
wandb_group = None
wandb_tags: str | Sequence[str] | None = None
AllowedWandbMode = Literal['online', 'offline', 'disabled', 'shared']
_ALLOWED_WANDB_MODES: tuple[AllowedWandbMode, ...] = ('online', 'offline', 'disabled', 'shared')


class _WandbModule(Protocol):
    def init(
        self,
        *,
        project: str,
        name: str | None,
        config: Mapping[str, object],
        mode: AllowedWandbMode | None,
        group: str | None,
        tags: Sequence[str] | None,
    ) -> None: ...

    def log(self, data: Mapping[str, float], *, step: int | None = None) -> None: ...

# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?

# moe
n_exp = 1 # if n_exp = 1 we just use regular MLP layers
top_k = 2
use_aux_loss = False
use_router_z_loss = False
use_noisy_top_k = False
aux_loss_weight = 0.001
router_z_loss_weight = 0.01
train_capacity = 1.25
eval_capacity = 2.0
min_capacity = 4
stride = 2
use_switch_tfm_init = False
switch_tfm_init_scale = 1.0  # recommended 0.1 for stability (pg.10, https://arxiv.org/abs/2101.03961)
router_use_full_prec = False

# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.

# system
device = 'auto' # 'auto', 'cpu', 'cuda', 'cuda:0', etc., or try 'mps' on macbooks
def _default_dtype() -> str:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return 'bfloat16'
    return 'float16'


dtype = _default_dtype()
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
init_from = str(init_from)
config['init_from'] = init_from

auto_requested = (device == 'auto')
if device == 'auto':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device.startswith('cuda') and not torch.cuda.is_available():
    print("[train.py] Requested CUDA but no GPU detected. Falling back to CPU.")
    device = 'cpu'
if 'cuda' not in device and backend == 'nccl':
    print("[train.py] Backend 'nccl' requires CUDA. Switching to 'gloo' for CPU training.")
    backend = 'gloo'
if 'cuda' not in device and dtype != 'float32':
    print("[train.py] Forcing dtype='float32' on CPU.")
    dtype = 'float32'
if 'cuda' not in device and compile:
    print("[train.py] Disabling torch.compile on CPU. Pass --compile=True to override.")
    compile = False

if decay_lr and lr_decay_iters <= warmup_iters:
    suggested = warmup_iters + 1
    print(
        f"[train.py] Adjusting lr_decay_iters from {lr_decay_iters} to {suggested} to avoid division by zero (warmup_iters={warmup_iters})."
    )
    lr_decay_iters = suggested

config.update(
    dict(
        device=device,
        dtype=dtype,
        backend=backend,
        compile=compile,
        auto_requested=auto_requested,
        lr_decay_iters=lr_decay_iters,
    )
)
print(config)
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
seed_offset: int
ddp_local_rank: int = 0
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    if 'cuda' in device:
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
seed_value: int = 1337 + seed_offset
manual_seed(seed_value)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = _autocast_context(device_type, ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)


def _load_memmap(path: str) -> npt.NDArray[np.uint16]:
    return cast(npt.NDArray[np.uint16], np.memmap(path, dtype=np.uint16, mode='r'))


Int64Array = np.ndarray[Any, np.dtype[np.int64]]


def _slice_window(data: npt.NDArray[np.uint16], start: int, length: int) -> Int64Array:
    window: Int64Array = np.asarray(data[start : start + length], dtype=np.int64)
    return window


def _tensor_from_numpy(array: np.ndarray[Any, np.dtype[np.int64]]) -> torch.Tensor:
    return _from_numpy(array)


def get_batch(split: BatchSplit) -> tuple[torch.Tensor, torch.Tensor]:
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    data_path = os.path.join(data_dir, 'train.bin' if split == 'train' else 'val.bin')
    data = _load_memmap(data_path)
    max_start = max(1, int(len(data) - block_size))
    ix = torch.randint(max_start, (batch_size,))
    x_tensors: list[torch.Tensor] = []
    y_tensors: list[torch.Tensor] = []
    start = 0
    for offset in range(ix.size(0)):
        start = int(ix[offset].item())
    source_window = _slice_window(data, start, block_size)
    target_window = _slice_window(data, start + 1, block_size)
    x_tensors.append(_tensor_from_numpy(source_window))
    y_tensors.append(_tensor_from_numpy(target_window))
    x = torch.stack(x_tensors)
    y = torch.stack(y_tensors)
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = ModelArgs(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
    n_exp=n_exp,
    top_k=top_k,
    use_aux_loss=use_aux_loss,
    use_router_z_loss=use_router_z_loss,
    use_noisy_top_k=use_noisy_top_k,
    aux_loss_weight=aux_loss_weight,
    router_z_loss_weight=router_z_loss_weight,
    train_capacity=train_capacity,
    eval_capacity=eval_capacity,
    min_capacity=min_capacity,
    stride=stride,
    use_switch_tfm_init=use_switch_tfm_init,
    switch_tfm_init_scale=switch_tfm_init_scale,
    router_use_full_prec=router_use_full_prec,
)
print('\n\n')
print(model_args)
print('\n\n')
gptconf: GPTConfig | None = None
checkpoint_data: TrainingCheckpoint | None = None
init_mode: str = init_from
base_model: GPT | None = None
if init_mode == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args.vocab_size = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = model_args.to_config()
    base_model = GPT(gptconf)
elif init_mode == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint_data = _load_checkpoint(ckpt_path, map_location=device)
    raw_checkpoint_args = checkpoint_data['model_args']
    if isinstance(raw_checkpoint_args, ModelArgs):
        checkpoint_model_args = raw_checkpoint_args
    else:
        checkpoint_model_args = ModelArgs.from_mapping(raw_checkpoint_args)
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for field_name in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        setattr(model_args, field_name, getattr(checkpoint_model_args, field_name))
    # create the model
    gptconf = model_args.to_config()
    base_model = GPT(gptconf)
    state_dict = checkpoint_data['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    base_model.load_state_dict(state_dict)
    iter_num = checkpoint_data['iter_num']
    best_val_loss = checkpoint_data['best_val_loss']
    config.update(checkpoint_data['config'])
elif init_mode.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    base_model = _load_pretrained_model(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    pretrained_config = _get_model_config(base_model)
    for field_name in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        setattr(model_args, field_name, getattr(pretrained_config, field_name))
# crop down the model block size if desired, using model surgery
if base_model is None:
    raise RuntimeError("Model initialization failed")
model: nn.Module = base_model

model_config = _get_model_config(base_model)
if block_size < model_config.block_size:
    _crop_model_block_size(base_model, block_size)
    model_args.block_size = block_size # so that the checkpoint will have the right value
base_model.to(device)
model = base_model

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler: TorchGradScaler | _DummyGradScaler
if device_type == 'cuda':
    scaler = TorchGradScaler('cuda', enabled=(dtype == 'float16'))
else:
    scaler = _DummyGradScaler()

# optimizer
optimizer: Optimizer = _configure_model_optimizers(model, weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume' and checkpoint_data is not None:
    optimizer.load_state_dict(checkpoint_data['optimizer'])
    checkpoint_data = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    model = _compile_model(model)

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
def estimate_loss() -> dict[BatchSplit, torch.Tensor]:
    MANAGER.reset_routing_stats()
    results: dict[BatchSplit, torch.Tensor] = {}
    model.eval()
    with torch.no_grad():
        for split in BATCH_SPLITS:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                input_batch, target_batch = get_batch(split)
                with ctx:
                    _, loss_tensor = model(input_batch, target_batch)
                losses[k] = float(loss_tensor.item())
            results[split] = losses.mean()
    model.train()
    MANAGER.reset_routing_stats()
    return results

# learning rate decay scheduler (cosine with warmup)
def get_lr(it: int) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return float(learning_rate * (it + 1) / (warmup_iters + 1))
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return float(min_lr)
    # 3) in between, use cosine decay down to min learning rate
    denom = max(1, lr_decay_iters - warmup_iters)
    decay_ratio = float(it - warmup_iters) / float(denom)
    decay_ratio = min(1.0, max(0.0, decay_ratio))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return float(min_lr + coeff * (learning_rate - min_lr))

# logging
wandb_module: _WandbModule | None = None
if wandb_log and master_process:
    import wandb as _wandb
    mode_candidate: str | None = None
    if wandb_mode is None and not os.environ.get('WANDB_API_KEY'):
        os.environ.setdefault('WANDB_MODE', 'offline')
        mode_candidate = os.environ['WANDB_MODE']
        print("[train.py] WANDB_API_KEY not found; defaulting to offline logging mode.")
    elif wandb_mode is not None:
        mode_candidate = wandb_mode
    if mode_candidate is not None and mode_candidate not in _ALLOWED_WANDB_MODES:
        print(f"[train.py] Ignoring unsupported wandb mode '{mode_candidate}'.")
        mode_candidate = None
    mode_arg: AllowedWandbMode | None = mode_candidate
    group_arg: str | None = wandb_group
    tags_arg: list[str] | None = None
    if wandb_tags:
        if isinstance(wandb_tags, str):
            tags_arg = [tag.strip() for tag in wandb_tags.split(',') if tag.strip()]
        else:
            tags_arg = [tag for tag in wandb_tags]
    _wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config=config,
        mode=mode_arg,
        group=group_arg,
        tags=tags_arg,
    )
    wandb_module = cast(_WandbModule, _wandb)

# training loop
x_batch, y_batch = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
ddp_model: DDP | None = model if isinstance(model, DDP) else None
if ddp_model is not None:
    module_candidate = cast(nn.Module, getattr(ddp_model, "module"))
    if not isinstance(module_candidate, GPT):
        raise TypeError("DDP-wrapped module is expected to be GPT")
    raw_model: GPT = module_candidate
else:
    if not isinstance(model, GPT):
        raise TypeError("Model must be an instance of GPT")
    raw_model = model
running_mfu: float = -1.0
last_loss: torch.Tensor | None = None
while True:

    last_loss = None

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    MANAGER.reset_routing_stats()

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_module is not None:
            eval_payload: dict[str, float] = {
                "iter": float(iter_num),
                "train/loss": float(losses['train']),
                "val/loss": float(losses['val']),
                "lr": float(lr),
                "mfu": float(running_mfu*100), # convert to percentage
            }
            wandb_module.log(eval_payload, step=iter_num)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint: dict[str, object] = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args.as_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': float(best_val_loss),
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                checkpoint_path: str = os.path.join(out_dir, 'ckpt.pt')
                _save_checkpoint(checkpoint, checkpoint_path)
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp_model is not None:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            ddp_model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        _, raw_loss = _forward_with_context(model, x_batch, y_batch, ctx)
        normalized_loss = raw_loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        last_loss = normalized_loss
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        x_batch, y_batch = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaled_loss = _scale_loss_tensor(scaler, normalized_loss)
        _tensor_backward(scaled_loss)
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        if last_loss is None:
            raise RuntimeError("loss tensor was not produced during gradient accumulation")
        lossf = float(last_loss.item() * gradient_accumulation_steps)
        if local_iter_num >= 5: # let the training loop settle a bit
            estimate_mfu_callable = cast("Callable[[int, float], float]", getattr(raw_model, "estimate_mfu"))
            mfu = float(estimate_mfu_callable(batch_size * gradient_accumulation_steps, dt))
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        routing_metrics_source = MANAGER.routing_summary()
        routing_metrics: dict[str, float] = {key: float(value) for key, value in routing_metrics_source.items()}
        aggregated_metrics: dict[str, float] = {}
        if routing_metrics:
            fractions = [value for key, value in routing_metrics.items() if key.endswith('_fraction') and 'expert' in key]
            if fractions:
                max_frac = max(fractions)
                min_frac = min(fractions)
                print(f"  routing balance: min {min_frac:.3f}, max {max_frac:.3f}")
            layer_fractions: dict[str, list[float]] = {}
            for key, value in routing_metrics.items():
                if key.endswith('_fraction') and 'expert' in key:
                    parts = key.split('/')
                    if len(parts) >= 3:
                        layer = parts[1]
                        per_layer = layer_fractions.setdefault(layer, [])
                        per_layer.append(value)
            all_layer_values = [val for values in layer_fractions.values() for val in values]
            if all_layer_values:
                total_experts = len(all_layer_values)
                unused = sum(1 for val in all_layer_values if val <= 1e-6)
                aggregated_metrics.update(
                    {
                        "routing/global/fraction_min": min(all_layer_values),
                        "routing/global/fraction_max": max(all_layer_values),
                        "routing/global/fraction_mean": sum(all_layer_values) / total_experts,
                        "routing/global/fraction_unused_ratio": unused / total_experts,
                    }
                )
            for layer, values in layer_fractions.items():
                total = len(values)
                if total == 0:
                    continue
                unused = sum(1 for val in values if val <= 1e-6)
                aggregated_metrics.update(
                    {
                        f"routing/{layer}/fraction_min": min(values),
                        f"routing/{layer}/fraction_max": max(values),
                        f"routing/{layer}/fraction_unused_ratio": unused / total,
                    }
                )
        if wandb_module is not None:
            log_payload: dict[str, float] = {
                "iter": float(iter_num),
                "train/iter_loss": float(lossf),
                "train/iter_time_ms": float(dt * 1000),
                "train/mfu": float(running_mfu*100),
            }
            log_payload.update(routing_metrics)
            log_payload.update(aggregated_metrics)
            wandb_module.log(log_payload, step=iter_num)
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
