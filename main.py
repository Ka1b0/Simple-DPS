from functools import partial
import os
import argparse
import yaml

import torch
import torchvision
import matplotlib.pyplot as plt

from conditioners import get_conditioner
from operators import get_operator, get_noise_func
from unet import create_model
from diffusion import DDPM
from dataloader import get_dataset, get_dataloader
from utils import get_logger, set_seed

set_seed(0)


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', default='configs/model_config.yaml', type=str)
    parser.add_argument('--task_config', default='configs/base_config.yaml', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data', default='samples', type=str)
    parser.add_argument('--task_name', default='box-inpaint', type=str)
    parser.add_argument('--noise', default='gaussian', type=str)
    parser.add_argument('--conditioner', default='ps', type=str)
    parser.add_argument('--scale', default=0.3, type=float)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--batch_size', type=int, default=5)
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    config = {**load_yaml(args.model_config), **load_yaml(args.task_config)}
    config['data']['root'] = './data/{:}/'.format(args.data)
    config['operator']['name'] = args.task_name
    config['noise']['name'] = args.noise
    config['conditioner']['name'] = args.conditioner
    config['conditioner']['scale'] = args.scale
    # Load model
    model = create_model(**config['unet']).to(device)
    model.eval()

    # Prepare Operator and noise
    operator = get_operator(config)
    noise_func = get_noise_func(config)
    logger.info(f"Operation: {args.task_name} / Noise: {args.noise}")

    # Prepare conditioning method
    conditioner = get_conditioner(config, operator, noise_func)
    logger.info(f"Conditioning method : {conditioner.__class__.__name__}")
   
    # Load diffusion sampler
    ddpm = DDPM(config)
   
    # Working directory
    out_path = os.path.join(args.save_dir, config['data']['root'].split('/')[-2], 
                            '{:}_{:}'.format(args.task_name, args.noise))

    # Prepare dataloader
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**config['data'], transforms=transform)
    loader = get_dataloader(dataset, batch_size=args.batch_size, num_workers=0, train=False)

    for i, x in enumerate(loader):
        logger.info(f"Inference for image {i}")
        x = x.to(device)
        operator.preprocess(x, i)
        y_n = noise_func(operator(x))
        # Sampling
        x_T = torch.randn_like(x).requires_grad_()
        sample = ddpm.sample(x_T, y_n, model, conditioner)
        fname = str(i).zfill(5) + '.png'
        for k, imgs in {'input': y_n, 'label': x, args.conditioner: sample}.items():
            os.makedirs(os.path.join(out_path, k), exist_ok=True)
            for idx in range(imgs.shape[0]):
                img = imgs[idx]
                fname = '{:}.png'.format(str(i * args.batch_size + idx).zfill(5))
                torchvision.utils.save_image((img - img.min()) / (img.max() - img.min()), 
                                            os.path.join(out_path, k, fname))

if __name__ == '__main__':
    main()
