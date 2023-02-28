from dataclasses import dataclass
from model import *
from torch import Tensor
from train import TrainConfig, default_perturbation

import numpy as np
import torch

@dataclass
class GenConfig:

    # ODE Solver
    solver: str = 'forward_euler'

    # Number of timesteps
    nt: int = 100

    # Number of output tokens
    n_out = 64

def forward_euler_sampler(model: nn.Module, tokens: Tensor, config: GenConfig) -> Tensor:
    
    x = model.get_word_vecs(tokens)
    n_batch, n_tokens, n_embed = x.shape

    p = torch.zeros(n_batch, 1, n_embed, device=model.config.device, dtype=model.config.dtype)
    p[:,0,-1] = 1.0
    
    # Forward Euler on sphere
    steps = np.linspace(0, 1, config.nt)
    dt = 1/config.nt
    for t in steps:
        v = model.forward(p, x, t)
        p = model.manifold.exp(p, dt*v)
    
    # Compute closest vec
    p = p.repeat(1, model.word_vecs.shape[0], 1)
    ip = torch.einsum('bve,ve->bv', p, model.word_vecs) # TODO: make this generic!
    d = torch.arccos(torch.clamp(ip, -1.0+1e-6, 1.0-1e-6))
    token = torch.argmin(d, dim=1, keepdim=True)

    return token

if __name__ == '__main__':

    from simple_parsing import ArgumentParser
    import tiktoken

    parser = ArgumentParser()
    parser.add_arguments(GenConfig, 'gen')
    parser.add_argument('--checkpoint_path', type=str)

    args = parser.parse_args()
    
    print(f'Loading checkpoint {args.checkpoint_path}... ', end='')
    ckpt = torch.load(args.checkpoint_path)
    model_state = ckpt['model']
    model_config = ckpt['model_config']
    train_config = ckpt['train_config']
    print('done.')

    print('Setting up model... ', end='')
    model = Model(model_config)
    print('done.')

    tokenizer = tiktoken.get_encoding('gpt2')
    
    with torch.no_grad():
        while True:

            text = input('Prompt: ')
            toks = []
            toks.extend(tokenizer.encode(text))
            toks = torch.tensor(toks, dtype=torch.long).to(model_config.device).view(1, -1)

            print('Response: ', end='', flush=True)
            for i in range(args.gen.n_out):
                
                if toks.shape[1] > model_config.n_tokens:
                    toks = toks[:,-model_config.n_tokens:]
                
                if args.gen.solver == 'forward_euler':
                    next_token = forward_euler_sampler(model, toks, args.gen)
                    toks = torch.cat((toks, next_token), dim=1)

            print(tokenizer.decode(toks[0,:].tolist()), flush=True)
