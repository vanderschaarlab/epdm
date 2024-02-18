import os
import argparse
import json
import codecs

from utils import Trainer
from nn import UNet
from models import GaussianDiffusion


parser = argparse.ArgumentParser()

parser.add_argument('--in_memory', action='store_true', default=False)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--img_size', type=int, default=32)
parser.add_argument('--channel_dim', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--training_steps', type=int, default=100000)
parser.add_argument('--save_every', type=int, default=10000)
parser.add_argument('--test_bsz', type=int, default=10000)
parser.add_argument('--demo_samples', type=int, default=10000)
parser.add_argument('--test_samples', type=int, default=5000)
parser.add_argument('--worker_procs', type=int, default=10)

parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--denoising_iters', type=int, default=1000)
parser.add_argument('--bootstrap_steps', type=int, default=0)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--dim_mults', type=int, default=1)
parser.add_argument('--multi_nums', type=str, default=None)
parser.add_argument('--resnet_groups', type=int, default=8)
parser.add_argument('--attn_heads', type=int, default=4)
parser.add_argument('--head_dim', type=int, default=64)

args = parser.parse_args()
print(json.dumps(args.__dict__, indent=True, ensure_ascii=False), end='\n\n')

if args.multi_nums is None:
    dim_multipliers = tuple([2**i for i in range(0, args.dim_mults)])
else:
    dim_multipliers = [int(i) for i in args.multi_nums.split(',')]
model = UNet(
    dim=args.hidden_dim, 
    channels=args.channel_dim, 
    flash_attn=False,
    dim_mults=dim_multipliers, 
    full_attn=(False,) * len(dim_multipliers), 
    resnet_block_groups=args.resnet_groups,
    attn_dim_head=args.head_dim, 
    attn_heads=args.attn_heads
)
diffusion = GaussianDiffusion(
    model=model, 
    image_size=args.img_size, 
    timesteps=args.denoising_iters, 
    auto_normalize=True,
    bootstrap_steps=args.bootstrap_steps
)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
config_path = os.path.join(args.save_dir, 'config.json')
with codecs.open(config_path, 'w', 'utf-8') as fw:
    json.dump(args.__dict__, fw, indent=True, ensure_ascii=False)

trainer = Trainer(
    diffusion_model=diffusion, 
    folder=args.data_dir,
    results_folder=args.save_dir,
    train_batch_size=args.batch_size, 
    test_batch_size=args.test_bsz, 
    train_lr=args.learning_rate,
    train_num_steps=args.training_steps, 
    save_and_sample_every=args.save_every, 
    num_samples=args.demo_samples, 
    calculate_fid=False,
    num_fid_samples=args.test_samples,
    io_workers=args.worker_procs,
    in_memory=args.in_memory,
)
trainer.train()
