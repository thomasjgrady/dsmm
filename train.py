from dataclasses import dataclass, field
from model import *
from torch import Tensor
from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def default_perturbation(x: Tensor, t: float) -> Tensor:
    return (1-t)*1e-3*torch.randn_like(x)

@dataclass
class TrainConfig:

    # Training data path
    train_data_path: str = os.path.expanduser('~/data/wikitext-103/wiki.train.tokens_bpe.npy')

    # Validation data path
    valid_data_path: str = os.path.expanduser('~/data/wikitext-103/wiki.valid.tokens_bpe.npy')

    # Total number of training examples
    n_examples: int = 10_000_000

    # Batch size
    batch_size: int = 100

    # Total number of batches
    n_batches: int = field(init=False)

    # Minimum learning rate
    min_lr: float = 1e-4

    # Maximum learning rate
    max_lr: float = 1e-3

    # Learning rate warmup iterations (in examples)
    n_warmup: int = 1000
    n_warmup_batches: int = field(init=False)

    # Weight decay
    weight_decay: float = 1e-4

    # Checkpoint interval (in examples)
    checkpoint_interval: int = 20_000
    checkpoint_interval_batches: int = field(init=False)

    # Path to save checkpoints to
    checkpoint_dir = 'checkpoints'

    # Perturbation applied to x as a function of t to add noise resillience
    perturbation: Callable = default_perturbation

    # Scale of loss for stability
    loss_scaling: float = 10.0

    # Is this a interactive or submitted job?
    interactive: bool = True

    def __post_init__(self) -> None:

        self.n_batches = self.n_examples // self.batch_size
        self.n_warmup_batches = self.n_warmup // self.batch_size
        self.checkpoint_interval_batches = self.checkpoint_interval // self.batch_size

def setup_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """
    Sets up ADAM optimizer w/ weight decay. Taken from Andrej Karpathy
    https://github.com/karpathy/nanoGPT/blob/ae3a8d5fdd3ddb8b13fab182723476523961e3ab/model.py#L269
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear)
    blacklist_weight_modules = (nn.LayerNorm)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
    
    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": config.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=config.max_lr)

    return optimizer

def get_train_batch(x: np.ndarray, model_config: ModelConfig, train_config: TrainConfig) -> Tuple[Tensor, Tensor]:
    """
    Gets a training batch by randomly sampling from x.
    """

    js = np.random.randint(0, x.shape[0] - model_config.n_tokens - 1, size=train_config.batch_size)
    xs = [torch.tensor(x[j:j+model_config.n_tokens].astype(np.int64), dtype=torch.long) for j in js]
    ys = [torch.tensor(x[j+1:j+2].astype(np.int64), dtype=torch.long) for j in js]
    xs = torch.cat([v.unsqueeze(0) for v in xs], dim=0).to(model_config.device)
    ys = torch.cat([v.unsqueeze(0) for v in ys], dim=0).to(model_config.device)
    
    return xs, ys

def get_lr(batch_idx: int, config: TrainConfig) -> float:
    if batch_idx < config.n_warmup_batches:
        return np.sin(batch_idx/config.n_warmup_batches)
    else:
        frac = np.cos((batch_idx-config.n_warmup_batches)/config.n_batches)
        return frac*config.max_lr + (1-frac)*config.min_lr

if __name__ == '__main__':

    from simple_parsing import ArgumentParser

    import time

    timestamp = int(time.time())

    parser = ArgumentParser()
    parser.add_arguments(ModelConfig, dest='model')
    parser.add_arguments(TrainConfig, dest='train')

    args = parser.parse_args()

    # Setup from configs

    # Load data as memmapped file
    print('Loading training and validation data... ', end='', flush=True)
    train_data = np.memmap(args.train.train_data_path, dtype=np.uint16, mode='r')
    valid_data = np.memmap(args.train.valid_data_path, dtype=np.uint16, mode='r')
    print('done.', flush=True)

    n_train_tokens = train_data.shape[0]
    n_valid_tokens = valid_data.shape[0]
    print(f'# training tokens   = {n_train_tokens/1e9:.4f}B', flush=True)
    print(f'# validation tokens = {n_valid_tokens/1e6:.4f}M', flush=True)

    # Setup model
    print('Setting up model... ', end='', flush=True)
    model = Model(args.model)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([np.prod(p.shape) for p in trainable_params])
    print('done.', flush=True)
    print(f'# trainable params  = {n_params}', flush=True)

    # Setup optimizers
    print('Setting up optimizer... ', end='', flush=True)
    optim = setup_optimizer(model, args.train)
    print('done.', flush=True)

    torch.set_anomaly_enabled(True)

    # Diffusion process always begins from the top
    start = torch.zeros(args.train.batch_size, 1, args.model.n_embed, device=args.model.device, dtype=args.model.dtype)
    start[:,0,-1] = 1.0

    # Main loop
    for i in range(args.train.n_batches):

        # Setup optimizer
        lr = get_lr(i, args.train)
        for param_group in optim.param_groups:
            param_group['lr'] = lr

        optim.zero_grad()

        with torch.no_grad():

            # Get batch
            x, y = get_train_batch(train_data, args.model, args.train)

            # Get word vectors
            x = model.get_word_vecs(x)
            y = model.get_word_vecs(y)

            # Select a random timestep
            t = torch.rand(args.train.batch_size, 1, 1, device=args.model.device, dtype=args.model.dtype)

            # Compute groundtruth tangent by moving time t along geodesic and
            # then recomputing tangent from there
            v = model.manifold.log(start, y)
            v = v/torch.norm(v, dim=-1, keepdim=True)
            
            p = model.manifold.exp(start, t*v)
            p = model.manifold.proj(p + args.train.perturbation(p, t))

            v_true = model.manifold.log(p, y)

        # Compute model prediction
        v_pred = model.forward(p, x, t)
        loss = F.mse_loss(v_pred, v_true)*args.train.loss_scaling
        loss.backward()
        optim.step()

        print(f'batch = {i:08d}/{args.train.n_batches:08d}, loss = {loss.item():2.8f}, lr = {lr:2.8f}', flush=True)

        if (i+1) % args.train.checkpoint_interval_batches == 0:
            torch.save({
                'batch_idx': i,
                'model_config': args.model,
                'train_config': args.train,
                'model': model.state_dict(),
                'optim': optim.state_dict()
            }, os.path.join(args.train.checkpoint_dir, f'cpkt_{timestamp}_{i+1:08d}.pt'))