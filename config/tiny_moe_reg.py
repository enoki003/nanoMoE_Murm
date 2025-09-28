import time

# Lightweight MoE run on Tiny Shakespeare with router regularization terms enabled.
# Mirrors tiny_moe_no_reg.py but activates auxiliary and router z-loss penalties.

wandb_log = True
wandb_project = 'nano-moe'
wandb_group = 'tiny-moe-regularization'
wandb_tags = ['tinyshakespeare', 'regularized']
wandb_run_name = 'tiny-moe-reg-' + time.strftime('%Y-%m-%d_%H-%M-%S')

# data / runtime
out_dir = 'out/tiny_moe_reg'
dataset = 'tinyshakespeare'
device = 'auto'
compile = False
max_iters = 600
eval_interval = 50
log_interval = 1
eval_iters = 20

# model / moe
n_layer = 2
n_head = 2
n_embd = 64
block_size = 64
batch_size = 2
gradient_accumulation_steps = 1
n_exp = 2
top_k = 1
use_aux_loss = True
aux_loss_weight = 0.01
use_router_z_loss = True
router_z_loss_weight = 0.001
use_noisy_top_k = False
train_capacity = 1.25
eval_capacity = 2.0
min_capacity = 4
stride = 2
router_use_full_prec = False

# optimization
learning_rate = 6e-4
weight_decay = 1e-1
grad_clip = 1.0
warmup_iters = 200
lr_decay_iters = max_iters
min_lr = learning_rate / 10
